//! WAV I/O fuzzing harness, mirroring `tests/fuzz_decoder.rs`'s pattern for the CLI's WAV parser
//! (`src/wav.rs`). Lower priority than the codec's own bitstream (WAV is a CLI-only convenience
//! layer, not part of the codec's threat model), but a real
//! parser of externally-supplied bytes should still never panic or hang on malformed input.
use fak::wav;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed ^ 0x9E3779B97F4A7C15) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_byte(&mut self) -> u8 { self.next_u64() as u8 }
    fn below(&mut self, bound: usize) -> usize { if bound == 0 { 0 } else { (self.next_u64() as usize) % bound } }
    fn bytes(&mut self, n: usize) -> Vec<u8> { (0..n).map(|_| self.next_byte()).collect() }
}

/// Hang-detection wall-clock budget per call: 5 s by default, overridable with `FAK_FUZZ_TIMEOUT_MS`.
/// Was 500 ms -- only ~10x the slowest legitimate input (50 ms locally, iteration 3449 of
/// `stream_decode_bit_flipped_known_length_stream...`), which spuriously failed under qemu (8 tests,
/// scattered iterations) and on the `macos-15-intel` CI runner (iteration 1669, 27.5 ms locally),
/// where 20 of these tests share a few vCPUs. A genuine hang
/// never returns, so a larger budget only delays reporting one; it cannot hide one.
static TIMEOUT: std::sync::LazyLock<Duration> = std::sync::LazyLock::new(|| {
    Duration::from_millis(std::env::var("FAK_FUZZ_TIMEOUT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(5000))
});

/// Writes `bytes` to a scratch file and runs `wav::read_wav` on it in a background thread with a
/// timeout, mirroring `fuzz_decoder.rs::decode_bounded`. `Ok(())` means it returned (parsed or
/// rejected -- either is fine) in time; `Err` means it panicked or hung.
fn read_wav_bounded(bytes: Vec<u8>, tag: &str, idx: u64) -> Result<(), &'static str> {
    let path = std::env::temp_dir().join(format!("nca_fuzz_wav_{tag}_{idx}.wav"));
    std::fs::write(&path, &bytes).expect("scratch write");
    let (tx, rx) = mpsc::channel();
    let p = path.clone();
    let handle = thread::Builder::new().spawn(move || {
        let _ = wav::read_wav(&p);
        let _ = tx.send(());
    }).expect("spawn");
    let result = match rx.recv_timeout(*TIMEOUT) {
        Ok(()) => { let _ = handle.join(); Ok(()) }
        Err(mpsc::RecvTimeoutError::Timeout) => Err("read_wav did not return within timeout (possible hang)"),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("read_wav panicked"),
    };
    let _ = std::fs::remove_file(&path);
    result
}

fn riff_header(chunk_size: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&chunk_size.to_le_bytes());
    b.extend_from_slice(b"WAVE");
    b
}

fn fmt_chunk(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    // Fuzzed inputs (huge channels/rate/bits) can make this overflow -- these are just adversarial
    // bytes being assembled for the reader to reject, not a real computation, so wrapping is fine.
    let byte_rate = rate.wrapping_mul(channels as u32).wrapping_mul((bits as u32 / 8).max(1));
    let block_align = channels.wrapping_mul((bits / 8).max(1));
    let mut body = Vec::new();
    body.extend_from_slice(&tag.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&rate.to_le_bytes());
    body.extend_from_slice(&byte_rate.to_le_bytes());
    body.extend_from_slice(&block_align.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    let mut b = Vec::new();
    b.extend_from_slice(b"fmt ");
    b.extend_from_slice(&(body.len() as u32).to_le_bytes());
    b.extend_from_slice(&body);
    b
}

#[test]
fn pure_random_bytes_never_panic_or_hang() {
    let mut rng = Rng::new(11);
    for i in 0..10_000u64 {
        let len = rng.below(1025);
        let bytes = rng.bytes(len);
        if let Err(reason) = read_wav_bounded(bytes, "rand", i) {
            panic!("iteration {i}, len {len}: {reason}");
        }
    }
}

#[test]
fn valid_riff_wave_with_random_body_never_panics_or_hangs() {
    let mut rng = Rng::new(12);
    for i in 0..8_000u64 {
        let mut bytes = riff_header(rng.next_u64() as u32);
        let body_len = rng.below(512);
        bytes.extend(rng.bytes(body_len));
        if let Err(reason) = read_wav_bounded(bytes, "riff", i) {
            panic!("iteration {i}, body_len {body_len}: {reason}");
        }
    }
}

#[test]
fn corrupted_fmt_and_huge_declared_sizes_never_panic_or_hang() {
    let mut rng = Rng::new(13);
    let bit_depths = [0u16, 1, 7, 8, 15, 16, 17, 23, 24, 25, 32, 65535];
    for i in 0..8_000u64 {
        let mut bytes = riff_header(0);
        let tag = if rng.below(4) == 0 { 0xFFFE } else { 1 }; // sometimes WAVE_FORMAT_EXTENSIBLE
        let channels = rng.next_byte() as u16;
        let rate = rng.next_u64() as u32;
        let bits = bit_depths[rng.below(bit_depths.len())];
        bytes.extend(fmt_chunk(tag, channels, rate, bits));
        // "data" chunk with a declared size deliberately including huge/adversarial values
        // (SS36: "enormous declared sizes") alongside small ones.
        let declared = match rng.below(4) {
            0 => 0u32,
            1 => 0xFFFF_FFFFu32,
            2 => rng.below(2048) as u32,
            _ => rng.next_u64() as u32,
        };
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&declared.to_le_bytes());
        let tail_len = rng.below(512);
        bytes.extend(rng.bytes(tail_len));
        if let Err(reason) = read_wav_bounded(bytes, "fmt", i) {
            panic!("iteration {i}: tag={tag} ch={channels} rate={rate} bits={bits} declared={declared}: {reason}");
        }
    }
}

#[test]
fn truncated_valid_wav_never_panics_or_hangs() {
    let w = wav::Wav {
        channels: vec![(0..3000i64).map(|i| ((i * 977) % 20001) - 10000).collect(), (0..3000i64).map(|i| -i).collect()],
        sample_rate: 44100,
        bits: 16,
        channel_mask: None,
        float_info: None,
    };
    let path = std::env::temp_dir().join("nca_fuzz_wav_valid_source.wav");
    wav::write_wav(&path, &w).unwrap();
    let full = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    for cut in (0..full.len()).step_by(37) {
        if let Err(reason) = read_wav_bounded(full[..cut].to_vec(), "trunc", cut as u64) {
            panic!("truncated to {cut}/{} bytes: {reason}", full.len());
        }
    }
}

#[test]
fn all_zero_and_all_one_bytes_never_panic_or_hang() {
    for len in [0usize, 1, 12, 44, 100, 1000] {
        if let Err(reason) = read_wav_bounded(vec![0u8; len], "zeros", len as u64) {
            panic!("{len} zero bytes: {reason}");
        }
        if let Err(reason) = read_wav_bounded(vec![0xFFu8; len], "ones", len as u64) {
            panic!("{len} 0xFF bytes: {reason}");
        }
    }
}
