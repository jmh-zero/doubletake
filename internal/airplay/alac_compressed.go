package airplay

import (
	"encoding/binary"
	"math/bits"
)

// The screen-audio transport carries one explicitly sized 352-sample ALAC
// frame per RTP packet. This encoder covers that fixed 16-bit stereo shape.
const (
	alacScreenFrameSamples = 352
	alacStereoChannels     = 2
	alacSampleBits         = 16
	alacCodedChannelBits   = alacSampleBits + 1

	alacPredictionShift    = 4
	alacPredictionHalf     = 1 << (alacPredictionShift - 1)
	alacPredictionScale    = 1 << alacPredictionShift
	alacPredictionMinOrder = 1
	alacPredictionMaxOrder = 4

	alacStereoShift = 2

	alacHistoryStart         = 10
	alacHistoryMultiplier    = 40
	alacHistoryFractionBits  = 9
	alacHistoryZeroThreshold = 1 << alacHistoryFractionBits
	alacHistoryClamp         = 0xffff
	alacRiceLimit            = 14
	alacEntropyEscapePrefix  = 9
)

type alacEncoder struct {
	pcm       [alacStereoChannels][alacScreenFrameSamples]int32
	coded     [alacStereoChannels][alacScreenFrameSamples]int32
	residual  [alacStereoChannels][alacScreenFrameSamples]int32
	candidate [alacScreenFrameSamples]int32
	alternate [4096]byte
}

func (e *alacEncoder) Encode(out, pcm []byte) int {
	const pcmBytes = alacScreenFrameSamples * alacStereoChannels * (alacSampleBits / 8)
	if len(pcm) != pcmBytes {
		panic("ALAC encoder received a non-352-sample stereo PCM frame")
	}
	if len(out) < len(e.alternate) {
		panic("ALAC encoder output buffer is too small")
	}

	for sample := 0; sample < alacScreenFrameSamples; sample++ {
		offset := sample * alacStereoChannels * 2
		e.pcm[0][sample] = int32(int16(binary.LittleEndian.Uint16(pcm[offset:])))
		e.pcm[1][sample] = int32(int16(binary.LittleEndian.Uint16(pcm[offset+2:])))
	}

	encodedBytes := e.encodeStereoMode(out, 0)
	for weight := int32(1); weight <= 1<<alacStereoShift; weight++ {
		candidateBytes := e.encodeStereoMode(e.alternate[:], weight)
		if candidateBytes < encodedBytes {
			copy(out, e.alternate[:candidateBytes])
			encodedBytes = candidateBytes
		}
	}

	// Noise can cost more than ALAC's verbatim element. Keep the smaller valid
	// representation while ordinary audio uses the compressed form.
	const verbatimBytes = (3 + 4 + 12 + 4 + 32 +
		alacScreenFrameSamples*alacStereoChannels*alacSampleBits + 3 + 7) / 8
	if encodedBytes >= verbatimBytes {
		return encodeALACVerbatim(out, pcm, alacScreenFrameSamples, alacStereoChannels, alacSampleBits)
	}
	return encodedBytes
}

func (e *alacEncoder) encodeStereoMode(out []byte, weight int32) int {
	for sample := 0; sample < alacScreenFrameSamples; sample++ {
		left := e.pcm[0][sample]
		right := e.pcm[1][sample]
		if weight == 0 {
			e.coded[0][sample] = left
			e.coded[1][sample] = right
			continue
		}
		difference := left - right
		e.coded[0][sample] = right + ((difference * weight) >> alacStereoShift)
		e.coded[1][sample] = difference
	}

	var predictorOrders [alacStereoChannels]int
	for channel := range e.coded {
		predictorOrders[channel] = e.selectPredictor(
			e.coded[channel][:], e.residual[channel][:], alacCodedChannelBits)
	}

	var writer bitWriter
	writer.init(out)
	writer.write(1, 3) // TYPE_CPE
	writer.write(0, 4) // elementInstanceTag
	writer.write(0, 12)
	writer.write(8, 4) // explicit sample count, no shifted bytes, compressed
	writer.write(alacScreenFrameSamples, 32)
	writer.write(alacStereoShift, 8)
	writer.write(uint32(uint8(int8(weight))), 8)
	writeALACPredictorHeader(&writer, predictorOrders[0])
	writeALACPredictorHeader(&writer, predictorOrders[1])
	writeALACEntropyBlock(&writer, e.residual[0][:], alacCodedChannelBits)
	writeALACEntropyBlock(&writer, e.residual[1][:], alacCodedChannelBits)
	writer.write(7, 3) // TYPE_END
	return writer.flush()
}

func (e *alacEncoder) selectPredictor(samples, residuals []int32, channelBits uint32) int {
	bestOrder := alacPredictionMinOrder
	bestBits := int(^uint(0) >> 1)
	for order := alacPredictionMinOrder; order <= alacPredictionMaxOrder; order++ {
		alacPredictResiduals(samples, e.candidate[:], order)
		bits := alacEntropyBitCount(e.candidate[:], channelBits) + 16 + order*16
		if bits < bestBits {
			bestBits = bits
			bestOrder = order
			copy(residuals, e.candidate[:])
		}
	}
	return bestOrder
}

// alacEntropyBitCount follows the same Rice state transitions as the writer
// without serializing every candidate predictor. Only the selected candidate
// reaches the bit writer, keeping encoding work bounded on the audio send path.
func alacEntropyBitCount(residuals []int32, channelBits uint32) int {
	history := uint32(alacHistoryStart)
	zeroRunAdjustment := uint32(0)
	encodedBits := 0
	for sample := 0; sample < len(residuals); {
		parameter := alacRiceParameter(history)
		divisor := uint32(1<<parameter) - 1
		folded := alacFoldSigned(residuals[sample])
		symbol := folded - zeroRunAdjustment
		encodedBits += alacEntropyValueBitCount(symbol, divisor, parameter, channelBits, true)
		sample++

		history = alacHistoryMultiplier*(symbol+zeroRunAdjustment) + history -
			(alacHistoryMultiplier*history)>>alacHistoryFractionBits
		if symbol > alacHistoryClamp {
			history = alacHistoryClamp
		}
		zeroRunAdjustment = 0

		if history*4 >= alacHistoryZeroThreshold || sample == len(residuals) {
			continue
		}
		zeroParameter := uint32(bits.LeadingZeros32(history)-24) +
			(history+16)>>6
		zeroDivisor := (uint32(1<<zeroParameter) - 1) & (1<<alacRiceLimit - 1)
		run := uint32(0)
		for sample < len(residuals) && residuals[sample] == 0 {
			run++
			sample++
		}
		encodedBits += alacEntropyValueBitCount(run, zeroDivisor, zeroParameter, 16, false)
		if run < 0xffff {
			zeroRunAdjustment = 1
		}
		history = 0
	}
	return encodedBits
}

func alacEntropyValueBitCount(value, divisor, parameter, escapeBits uint32, omitUnitSuffix bool) int {
	prefix := value / divisor
	if prefix >= alacEntropyEscapePrefix {
		return alacEntropyEscapePrefix + int(escapeBits)
	}
	encodedBits := int(prefix) + 1
	if omitUnitSuffix && parameter == 1 {
		return encodedBits
	}
	if value-prefix*divisor == 0 {
		return encodedBits + int(parameter) - 1
	}
	return encodedBits + int(parameter)
}

func writeALACPredictorHeader(writer *bitWriter, order int) {
	writer.write(alacPredictionShift, 8) // mode=0, denominator shift=4
	writer.write(4<<5|uint32(order), 8)
	coefficients := alacPredictionCoefficients(order)
	for _, coefficient := range coefficients[:order] {
		writer.write(uint32(uint16(coefficient)), 16)
	}
}

// alacPredictResiduals starts with polynomial extrapolation from recent
// samples. Its coefficients follow each prediction error using only information
// carried in the residual stream, so the decoder stays in lockstep.
func alacPredictResiduals(samples, residuals []int32, order int) {
	if len(samples) == 0 {
		return
	}
	residuals[0] = samples[0]
	warmup := min(order, len(samples)-1)
	for sample := 1; sample <= warmup; sample++ {
		residuals[sample] = alacNarrowSigned(samples[sample]-samples[sample-1], alacCodedChannelBits)
	}
	coefficients := alacPredictionCoefficients(order)
	for sample := order + 1; sample < len(samples); sample++ {
		base := samples[sample-order-1]
		weightedDifference := int32(0)
		for tap, coefficient := range coefficients[:order] {
			weightedDifference += int32(coefficient) * (samples[sample-1-tap] - base)
		}
		prediction := base + (weightedDifference+alacPredictionHalf)>>alacPredictionShift
		error := alacNarrowSigned(samples[sample]-prediction, alacCodedChannelBits)
		residuals[sample] = error
		remaining := error
		for tap := order - 1; tap >= 0 && remaining != 0; tap-- {
			difference := base - samples[sample-1-tap]
			direction := alacSign(difference)
			if error > 0 {
				coefficients[tap] -= int16(direction)
				remaining -= int32(order-tap) *
					((direction * difference) >> alacPredictionShift)
				if remaining <= 0 {
					break
				}
			} else {
				coefficients[tap] += int16(direction)
				remaining -= int32(order-tap) *
					((-direction * difference) >> alacPredictionShift)
				if remaining >= 0 {
					break
				}
			}
		}
	}
}

func alacPredictionCoefficients(order int) [alacPredictionMaxOrder]int16 {
	var coefficients [alacPredictionMaxOrder]int16
	combination := 1
	for tap := 0; tap < order; tap++ {
		combination = combination * (order - tap) / (tap + 1)
		coefficient := combination * alacPredictionScale
		if tap&1 != 0 {
			coefficient = -coefficient
		}
		coefficients[tap] = int16(coefficient)
	}
	return coefficients
}

func alacNarrowSigned(value int32, width uint32) int32 {
	shift := 32 - width
	return value << shift >> shift
}

func alacSign(value int32) int32 {
	switch {
	case value > 0:
		return 1
	case value < 0:
		return -1
	default:
		return 0
	}
}

func alacFoldSigned(value int32) uint32 {
	return uint32(value)<<1 ^ uint32(value>>31)
}

// writeALACEntropyBlock writes the adaptive Rice stream described by the
// negotiated ALAC cookie (pb=40, mb=10, kb=14), including zero runs when the
// shared history falls below its threshold.
func writeALACEntropyBlock(writer *bitWriter, residuals []int32, channelBits uint32) {
	history := uint32(alacHistoryStart)
	zeroRunAdjustment := uint32(0)
	for sample := 0; sample < len(residuals); {
		parameter := alacRiceParameter(history)
		divisor := uint32(1<<parameter) - 1
		folded := alacFoldSigned(residuals[sample])
		symbol := folded - zeroRunAdjustment
		writeALACEntropyValue(writer, symbol, divisor, parameter, channelBits, true)
		sample++

		history = alacHistoryMultiplier*(symbol+zeroRunAdjustment) + history -
			(alacHistoryMultiplier*history)>>alacHistoryFractionBits
		if symbol > alacHistoryClamp {
			history = alacHistoryClamp
		}
		zeroRunAdjustment = 0

		if history*4 >= alacHistoryZeroThreshold || sample == len(residuals) {
			continue
		}
		zeroParameter := uint32(bits.LeadingZeros32(history)-24) +
			(history+16)>>6
		zeroDivisor := (uint32(1<<zeroParameter) - 1) & (1<<alacRiceLimit - 1)
		run := uint32(0)
		for sample < len(residuals) && residuals[sample] == 0 {
			run++
			sample++
		}
		writeALACEntropyValue(writer, run, zeroDivisor, zeroParameter, 16, false)
		if run < 0xffff {
			zeroRunAdjustment = 1
		}
		history = 0
	}
}

func alacRiceParameter(history uint32) uint32 {
	parameter := uint32(bits.Len32((history>>alacHistoryFractionBits)+3) - 1)
	if parameter > alacRiceLimit {
		return alacRiceLimit
	}
	return parameter
}

func writeALACEntropyValue(writer *bitWriter, value, divisor, parameter, escapeBits uint32, omitUnitSuffix bool) {
	prefix := value / divisor
	if prefix >= alacEntropyEscapePrefix {
		for bit := 0; bit < alacEntropyEscapePrefix; bit++ {
			writer.write(1, 1)
		}
		writer.write(value, escapeBits)
		return
	}
	for bit := uint32(0); bit < prefix; bit++ {
		writer.write(1, 1)
	}
	writer.write(0, 1)
	if omitUnitSuffix && parameter == 1 {
		return
	}
	remainder := value - prefix*divisor
	if remainder == 0 {
		writer.write(0, parameter-1)
		return
	}
	writer.write(remainder+1, parameter)
}
