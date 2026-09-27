package airplay

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestALACVerbatimEncoding(t *testing.T) {
	// Generate a 440 Hz sine wave frame: 352 samples, stereo, 16-bit, 44100 Hz
	const spf = 352
	const channels = 2
	const sampleRate = 44100
	const freq = 440.0

	pcm := make([]byte, spf*channels*2)
	for i := 0; i < spf; i++ {
		sample := int16(math.Sin(2*math.Pi*freq*float64(i)/sampleRate) * 16000)
		// S16LE format
		for ch := 0; ch < channels; ch++ {
			off := (i*channels + ch) * 2
			pcm[off] = byte(sample)
			pcm[off+1] = byte(sample >> 8)
		}
	}

	// Encode
	out := make([]byte, 4096)
	n := encodeALACVerbatim(out, pcm, spf, channels, 16)
	out = out[:n]

	t.Logf("ALAC frame: %d bytes", n)
	t.Logf("First 16 bytes (hex): %s", hex.EncodeToString(out[:min(16, n)]))

	// Verify header bits
	// Byte 0: 001 0000 0 = 0x20 (TYPE_CPE=1, elemTag=0, unused starts)
	if out[0] != 0x20 {
		t.Errorf("byte 0: got 0x%02x, want 0x20", out[0])
	}
	// Byte 1: 00000000 (unused continuation)
	if out[1] != 0x00 {
		t.Errorf("byte 1: got 0x%02x, want 0x00", out[1])
	}
	// Byte 2 contains packed flag/sample-count boundary bits; keep this check
	// loose and rely on the exact known-good prefix below.
	if out[2] == 0x00 {
		t.Errorf("byte 2 unexpectedly zero")
	}
	if n >= 8 {
		wantPrefix := "200012000002c0"
		gotPrefix := hex.EncodeToString(out[:7])
		if gotPrefix != wantPrefix {
			t.Errorf("header prefix: got %s, want %s", gotPrefix, wantPrefix)
		}
	}

	// Expected frame size: hasSize=1, with 32-bit numSamples field
	// = 23 header bits + 32 numSamples + 352*2*16 sample bits + 3 end bits
	// = 23 + 32 + 11264 + 3 = 11322 bits = 1415.25 → 1416 bytes
	expectedSize := (23 + 32 + spf*channels*16 + 3 + 7) / 8 // round up
	t.Logf("Expected size: %d bytes", expectedSize)
	if n != expectedSize {
		t.Errorf("frame size: got %d, want %d", n, expectedSize)
	}

	// Verify that encoded data has non-zero samples (sine wave, not silence)
	// Sample data starts at bit 55 (byte 6, bit 7)
	// Just check that bytes 7-20 aren't all zero
	allZero := true
	for i := 7; i < min(20, n); i++ {
		if out[i] != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("sample data appears to be all zeros (silence)")
	}

	// Test with all-zero PCM (silence) to verify encoding still works
	silentPCM := make([]byte, spf*channels*2)
	silentOut := make([]byte, 4096)
	sn := encodeALACVerbatim(silentOut, silentPCM, spf, channels, 16)
	t.Logf("Silent frame: %d bytes, first 16: %s", sn, hex.EncodeToString(silentOut[:min(16, sn)]))
}

func TestALACCompressedEncoding(t *testing.T) {
	pcm := testALACPCM()
	out := make([]byte, 4096)
	n := (&alacEncoder{}).Encode(out, pcm)
	out = out[:n]

	if n >= 1024 {
		t.Fatalf("compressed ALAC frame is %d bytes; RFC 2198 can only describe blocks up to 1023 bytes", n)
	}
	// A modern encrypted packet has 36 bytes of RTP, AEAD, and nonce overhead.
	// Keep enough room for the current frame, one redundant frame, and the
	// five-byte RFC 2198 header inside the 1472-byte UDP payload limit.
	if got, limit := 2*n+5, maximumAudioRTPDatagramBytes-36; got > limit {
		t.Fatalf("current plus one redundant ALAC frame uses %d bytes, payload limit is %d", got, limit)
	}
	if out[0] != 0x20 {
		t.Fatalf("first byte = 0x%02x, want stereo channel-pair element", out[0])
	}
	// The element flags begin after the 3-bit tag, 4-bit instance, and
	// 12 reserved bits. For a 352-sample packet they must say partial frame,
	// no shifted bytes, and compressed (1000 binary).
	flags := (uint16(out[2])<<8 | uint16(out[3])) >> 9 & 0xf
	if flags != 8 {
		t.Fatalf("element flags = 0x%x, want partial compressed frame (0x8)", flags)
	}
	reader := h264BitReader{data: out}
	if got := reader.readBits(3); got != 1 {
		t.Fatalf("element tag = %d, want stereo channel pair", got)
	}
	reader.readBits(4 + 12 + 4 + 32)
	if got := reader.readBits(8); got != alacStereoShift {
		t.Fatalf("stereo transform shift = %d, want %d", got, alacStereoShift)
	}
	if got := reader.readBits(8); got > 1<<alacStereoShift {
		t.Fatalf("stereo transform weight = %d, want 0..%d", got, 1<<alacStereoShift)
	}
	for channel := 0; channel < alacStereoChannels; channel++ {
		if got := reader.readBits(8); got != alacPredictionShift {
			t.Fatalf("channel %d predictor mode/shift = 0x%x, want 0x%x", channel, got, alacPredictionShift)
		}
		if got := reader.readBits(8) & 0x1f; got != alacPredictionOrder {
			t.Fatalf("channel %d predictor order = %d, want %d", channel, got, alacPredictionOrder)
		}
		for coefficientIndex, want := range alacPredictionCoefficients() {
			if got := int16(reader.readBits(16)); got != want {
				t.Fatalf("channel %d coefficient %d = %d, want %d", channel, coefficientIndex, got, want)
			}
		}
	}
	if reader.err {
		t.Fatal("compressed predictor header is truncated")
	}
	t.Logf("compressed 352-sample stereo ALAC frame: %d bytes", n)
}

func TestALACCompressedEncodingAllocatesNoPerFrameMemory(t *testing.T) {
	pcm := testALACPCM()
	out := make([]byte, 4096)
	encoder := &alacEncoder{}
	if allocations := testing.AllocsPerRun(100, func() {
		encoder.Encode(out, pcm)
	}); allocations != 0 {
		t.Fatalf("ALAC frame encoding allocated %.1f objects, want 0", allocations)
	}
}

func TestALACCompressedFrameDecodesLosslessly(t *testing.T) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		t.Skip("ffmpeg is unavailable for independent ALAC decode")
	}

	patterns := map[string][]byte{
		"music":   testALACPCM(),
		"silence": make([]byte, alacScreenFrameSamples*audioBytesPerSampleFrame),
		"noise":   testALACNoisePCM(),
	}
	for name, pcm := range patterns {
		t.Run(name, func(t *testing.T) {
			encoded := make([]byte, 4096)
			n := (&alacEncoder{}).Encode(encoded, pcm)
			encoded = encoded[:n]

			temporary := t.TempDir()
			cafPath := filepath.Join(temporary, "frame.caf")
			pcmPath := filepath.Join(temporary, "decoded.pcm")
			if err := os.WriteFile(cafPath, singleFrameALACCAF(encoded), 0o600); err != nil {
				t.Fatal(err)
			}
			command := exec.Command(ffmpeg, "-v", "error", "-y", "-i", cafPath,
				"-f", "s16le", "-acodec", "pcm_s16le", pcmPath)
			if output, err := command.CombinedOutput(); err != nil {
				t.Fatalf("ffmpeg rejected ALAC: %v\n%s", err, output)
			}
			decoded, err := os.ReadFile(pcmPath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(decoded, pcm) {
				t.Fatalf("decoded PCM differs: got %d bytes, want %d", len(decoded), len(pcm))
			}
		})
	}
}

func TestAudioCaptureSelectsALACRepresentationForTransport(t *testing.T) {
	pcm := testALACPCM()
	for _, test := range []struct {
		name       string
		compact    bool
		wantLength int
	}{
		{name: "legacy packet redundancy", compact: false, wantLength: 1416},
		{name: "RFC 2198 compound redundancy", compact: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			capture := &AudioCapture{
				pcmPipe:     io.NopCloser(bytes.NewReader(pcm)),
				waitCh:      make(chan struct{}),
				codec:       AudioCodecALAC,
				compactALAC: test.compact,
			}
			encoded := make([]byte, 8192)
			n, _, err := capture.readFramePosition(encoded)
			if err != nil {
				t.Fatal(err)
			}
			if test.wantLength != 0 && n != test.wantLength {
				t.Fatalf("encoded length = %d, want %d", n, test.wantLength)
			}
			if test.compact && n >= 1024 {
				t.Fatalf("compact encoded length = %d, want an RFC 2198-compatible block", n)
			}
		})
	}
}

func TestMirrorSessionUsesCompactALACOnlyForRFC2198(t *testing.T) {
	if (*MirrorSession)(nil).UsesCompactALAC() {
		t.Fatal("nil session selected compact ALAC")
	}
	legacy := &MirrorSession{audioStream: &AudioStream{}}
	if legacy.UsesCompactALAC() {
		t.Fatal("legacy packet redundancy selected compact ALAC")
	}
	compound := &MirrorSession{audioStream: &AudioStream{rfc2198: true, ct: byte(AudioCodecALAC)}}
	if !compound.UsesCompactALAC() {
		t.Fatal("RFC 2198 session did not select compact ALAC")
	}
}

func testALACPCM() []byte {
	const sampleRate = 44100
	pcm := make([]byte, alacScreenFrameSamples*audioBytesPerSampleFrame)
	for sampleIndex := 0; sampleIndex < alacScreenFrameSamples; sampleIndex++ {
		left := int16(math.Sin(2*math.Pi*440*float64(sampleIndex)/sampleRate) * 16000)
		right := int16(math.Sin(2*math.Pi*659.25*float64(sampleIndex)/sampleRate) * 12000)
		offset := sampleIndex * audioBytesPerSampleFrame
		binary.LittleEndian.PutUint16(pcm[offset:], uint16(left))
		binary.LittleEndian.PutUint16(pcm[offset+2:], uint16(right))
	}
	return pcm
}

func testALACNoisePCM() []byte {
	pcm := make([]byte, alacScreenFrameSamples*audioBytesPerSampleFrame)
	state := uint32(0x4d595df4)
	for offset := 0; offset < len(pcm); offset += 2 {
		state = state*1664525 + 1013904223
		binary.LittleEndian.PutUint16(pcm[offset:], uint16(state>>16))
	}
	return pcm
}

func singleFrameALACCAF(frame []byte) []byte {
	// CAF metadata for 44.1 kHz, stereo, 16-bit ALAC with the standard
	// 4096-sample configuration. The packet itself carries the partial-frame
	// count of 352 used by screen audio.
	prefix, err := hex.DecodeString(
		"6361666600010000" +
			"64657363000000000000002040e5888000000000616c61630000000000000000000010000000000200000000" +
			"6368616e000000000000000c006500020000000000000000" +
			"6b756b6900000000000000300000000c66726d61616c616300000024616c616300000000000010000010280a0e02000000004004001588800000ac44")
	if err != nil {
		panic(err)
	}

	var caf bytes.Buffer
	caf.Write(prefix)
	caf.WriteString("data")
	_ = binary.Write(&caf, binary.BigEndian, uint64(len(frame)+4))
	_ = binary.Write(&caf, binary.BigEndian, uint32(0))
	caf.Write(frame)

	packetSize := cafVariableInteger(uint64(len(frame)))
	caf.WriteString("pakt")
	_ = binary.Write(&caf, binary.BigEndian, uint64(24+len(packetSize)))
	_ = binary.Write(&caf, binary.BigEndian, uint64(1))
	_ = binary.Write(&caf, binary.BigEndian, uint64(alacScreenFrameSamples))
	_ = binary.Write(&caf, binary.BigEndian, uint32(0))
	_ = binary.Write(&caf, binary.BigEndian, uint32(0))
	caf.Write(packetSize)
	return caf.Bytes()
}

func cafVariableInteger(value uint64) []byte {
	var reversed [10]byte
	count := 0
	for {
		reversed[count] = byte(value & 0x7f)
		count++
		value >>= 7
		if value == 0 {
			break
		}
	}
	encoded := make([]byte, count)
	for index := 0; index < count; index++ {
		encoded[index] = reversed[count-1-index]
		if index+1 < count {
			encoded[index] |= 0x80
		}
	}
	return encoded
}
