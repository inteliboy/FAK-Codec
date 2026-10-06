//! Decoder fuzzing harness: the decoder must never crash, hang, read/write
//! out of bounds, or otherwise misbehave on arbitrary/corrupted/hostile input -- only ever return
//! `Ok` (a decode) or `Err` (a rejection). A deterministic, seeded, dependency-free
//! mutation/random-input harness on stable Rust, run as an ordinary `cargo test` on every platform
//! (Windows included, where libFuzzer's sanitizer story is poor --).
//! Coverage-guided fuzzing (cargo-fuzz + ASan, Linux/macOS + nightly) lives in `fuzz/` and
//! complements this rather than replacing it.
//!
//! Every property here runs the decoder in a background thread with a timeout: a panic shows up
//! as the channel disconnecting without a message, a hang shows up as the timeout firing (the
//! thread is deliberately leaked in that case -- there is no safe way to force-kill it, and it
//! doesn't need joining for the test process to exit).
use fak::{decoder, format::{StreamHeader, MODE_BLOCK_INDEPENDENT}};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// xorshift64* -- deterministic and seeded so any failure is exactly reproducible from the seed
/// printed in the assertion message, without pulling in an external RNG crate.
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

/// Runs `decoder::decode` on `bytes` in a background thread. Returns `Ok(())` if it returned
/// (decoded or rejected -- either is fine) within `TIMEOUT`; `Err(reason)` if it panicked or
/// didn't return in time.
fn decode_bounded(bytes: Vec<u8>) -> Result<(), &'static str> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new().spawn(move || {
        let _ = decoder::decode(&bytes);
        let _ = tx.send(());
    }).expect("spawn");
    match rx.recv_timeout(*TIMEOUT) {
        Ok(()) => { let _ = handle.join(); Ok(()) }
        Err(mpsc::RecvTimeoutError::Timeout) => Err("decoder did not return within timeout (possible hang)"),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("decoder panicked"),
    }
}

fn header_bytes(channels: u8, bits: u8, sample_rate: u32, total_frames: u64) -> Vec<u8> {
    StreamHeader { channels, bits_per_sample: bits, mode: MODE_BLOCK_INDEPENDENT, sample_rate, total_frames, pcm_hash: [0u8; 32] }.to_bytes()
}

/// Mirrors `decode_bounded` for `decoder::seek`:
/// a caller-supplied `target_frame` is untrusted input to this function just like the file bytes
/// are, so it gets the same never-panic-or-hang treatment.
fn seek_bounded(bytes: Vec<u8>, target_frame: u64) -> Result<(), &'static str> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new().spawn(move || {
        let _ = decoder::seek(&bytes, target_frame);
        let _ = tx.send(());
    }).expect("spawn");
    match rx.recv_timeout(*TIMEOUT) {
        Ok(()) => { let _ = handle.join(); Ok(()) }
        Err(mpsc::RecvTimeoutError::Timeout) => Err("seek did not return within timeout (possible hang)"),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("seek panicked"),
    }
}

/// Mirrors `decode_bounded` for `decoder::decode_stream`: a real, separate
/// hostile-input entry point (its own I/O-driven header/metadata/chunk parsing, not sharing
/// `decode`'s in-memory-slice code path) that deserves the same never-panic-or-hang treatment. Reads
/// through the actual `Read` trait (a slice, but read incrementally, not sliced directly) so this
/// exercises the real incremental I/O logic, not just the parsing.
fn stream_decode_bounded(bytes: Vec<u8>) -> Result<(), &'static str> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new().spawn(move || {
        let _ = decoder::decode_stream(bytes.as_slice(), |_| Ok(()));
        let _ = tx.send(());
    }).expect("spawn");
    match rx.recv_timeout(*TIMEOUT) {
        Ok(()) => { let _ = handle.join(); Ok(()) }
        Err(mpsc::RecvTimeoutError::Timeout) => Err("decode_stream did not return within timeout (possible hang)"),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err("decode_stream panicked"),
    }
}

#[test]
fn pure_random_bytes_never_panic_or_hang() {
    let mut rng = Rng::new(1);
    for i in 0..50_000u64 {
        let len = rng.below(2049);
        let bytes = rng.bytes(len);
        if let Err(reason) = decode_bounded(bytes.clone()) {
            panic!("iteration {i}, seed-derived input len {len}: {reason}\nbytes: {bytes:?}");
        }
    }
}

#[test]
fn valid_header_with_random_frame_bytes_never_panics_or_hangs() {
    let mut rng = Rng::new(2);
    let bit_depths = [8u8, 16, 24];
    for i in 0..20_000u64 {
        let channels = 1 + rng.below(6) as u8;
        let bits = bit_depths[rng.below(bit_depths.len())];
        let sample_rate = 1 + (rng.next_u64() as u32);
        // Deliberately include huge declared totals ("enormous declared sizes")
        // alongside small/realistic ones, so both the fast-reject and the plausible-but-corrupted
        // paths get exercised.
        let total_frames = match rng.below(4) {
            0 => 0,
            1 => rng.below(10_000) as u64,
            2 => u64::MAX,
            _ => (rng.next_u64() as u32) as u64,
        };
        let mut bytes = header_bytes(channels, bits, sample_rate, total_frames);
        let frame_len = rng.below(2049);
        bytes.extend(rng.bytes(frame_len));
        if let Err(reason) = decode_bounded(bytes.clone()) {
            panic!("iteration {i}: channels={channels} bits={bits} sr={sample_rate} total_frames={total_frames} frame_len={frame_len}: {reason}");
        }
    }
}

#[test]
fn corrupted_header_fields_never_panic_or_hang() {
    // Targeted: mutate individual header bytes (including the CRC itself) rather than random
    // whole-header noise, so both "CRC still happens to match" and "CRC correctly rejects" paths
    // both get real coverage, not just the statistically-dominant "bad CRC, rejected immediately" case.
    let mut rng = Rng::new(3);
    let base = header_bytes(2, 16, 44100, 1000);
    for i in 0..15_000u64 {
        let mut bytes = base.clone();
        let idx = rng.below(bytes.len());
        bytes[idx] = rng.next_byte();
        let extra = rng.below(512);
        bytes.extend(rng.bytes(extra));
        if let Err(reason) = decode_bounded(bytes.clone()) {
            panic!("iteration {i}, flipped header byte {idx}: {reason}\nbytes: {bytes:?}");
        }
    }
}

#[test]
fn bit_flipped_valid_stream_never_panics_or_hangs() {
    // Mutate an otherwise-valid encoded stream: this is the case most likely to slip past the
    // CRC (a corruption the checksum happens not to catch) and reach deep into subframe/predictor
    // parsing with attacker-adjacent-but-plausible bit patterns.
    use fak::encoder;
    let samples: Vec<i64> = (0..6000i64).map(|i| ((i * 977) % 20001) - 10000).collect();
    let valid = encoder::encode(&[samples.clone(), samples.iter().map(|&s| -s).collect()], 44100, 16).unwrap();
    let mut rng = Rng::new(4);
    for i in 0..20_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(4);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        if let Err(reason) = decode_bounded(bytes) {
            panic!("iteration {i}, {n_flips} bit flips in a valid stream: {reason}");
        }
    }
}

#[test]
fn truncated_valid_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..6000i64).map(|i| ((i * 631) % 15001) - 7500).collect();
    let valid = encoder::encode(&[samples.clone(), samples.iter().map(|&s| -s).collect()], 44100, 16).unwrap();
    for cut in 0..valid.len() {
        if let Err(reason) = decode_bounded(valid[..cut].to_vec()) {
            panic!("truncated to {cut}/{} bytes: {reason}", valid.len());
        }
    }
}

#[test]
fn all_zero_and_all_one_bytes_never_panic_or_hang() {
    for len in [0usize, 1, 16, 24, 100, 1000, 4096] {
        if let Err(reason) = decode_bounded(vec![0u8; len]) {
            panic!("{len} zero bytes: {reason}");
        }
        if let Err(reason) = decode_bounded(vec![0xFFu8; len]) {
            panic!("{len} 0xFF bytes: {reason}");
        }
    }
}

/// `decoder::seek` is a second, separate entry point
/// into hostile file bytes, with its own chunk-lookup arithmetic `decode`/`decode_full` never
/// exercise -- gets the same never-panic-or-hang fuzzing as the rest of this module, both on random
/// bytes and on a real valid multi-chunk stream with random target frames, bit flips, and truncation.
#[test]
fn seek_pure_random_bytes_never_panic_or_hang() {
    let mut rng = Rng::new(8);
    for i in 0..20_000u64 {
        let len = rng.below(2049);
        let bytes = rng.bytes(len);
        let target = rng.next_u64();
        if let Err(reason) = seek_bounded(bytes.clone(), target) {
            panic!("iteration {i}, len {len}, target {target}: {reason}\nbytes: {bytes:?}");
        }
    }
}

#[test]
fn seek_random_target_on_valid_multichunk_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..30_000i64).map(|i| ((i * 977) % 20001) - 10000).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, 5000, 1, Some(fak::format::DEFAULT_FEC_GROUP), &Default::default(),
    ).unwrap();
    let mut rng = Rng::new(9);
    for i in 0..10_000u64 {
        let target = match rng.below(4) {
            0 => rng.below(30_000) as u64, // in range
            1 => u64::MAX,
            2 => 30_000, // exactly one past the end
            _ => rng.next_u64(),
        };
        if let Err(reason) = seek_bounded(valid.clone(), target) {
            panic!("iteration {i}, target {target}: {reason}");
        }
    }
}

#[test]
fn seek_bit_flipped_valid_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..20_000i64).map(|i| ((i * 631) % 15001) - 7500).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(fak::format::DEFAULT_FEC_GROUP), &Default::default(),
    ).unwrap();
    let mut rng = Rng::new(10);
    for i in 0..10_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(4);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        let target = rng.below(20_000) as u64;
        if let Err(reason) = seek_bounded(bytes, target) {
            panic!("iteration {i}, {n_flips} bit flips, target {target}: {reason}");
        }
    }
}

#[test]
fn seek_truncated_valid_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..20_000i64).map(|i| ((i * 631) % 15001) - 7500).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, 4000, 1, Some(fak::format::DEFAULT_FEC_GROUP), &Default::default(),
    ).unwrap();
    for cut in (0..valid.len()).step_by(11) {
        if let Err(reason) = seek_bounded(valid[..cut].to_vec(), 10_000) {
            panic!("truncated to {cut}/{} bytes: {reason}", valid.len());
        }
    }
}

#[test]
fn stream_decode_pure_random_bytes_never_panic_or_hang() {
    let mut rng = Rng::new(11);
    for i in 0..20_000u64 {
        let len = rng.below(2049);
        let bytes = rng.bytes(len);
        if let Err(reason) = stream_decode_bounded(bytes.clone()) {
            panic!("iteration {i}, len {len}: {reason}\nbytes: {bytes:?}");
        }
    }
}

#[test]
fn stream_decode_bit_flipped_known_length_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..20_000i64).map(|i| ((i * 977) % 20001) - 10000).collect();
    // FEC disabled: decode_stream never understands FEC parity blocks (file-based
    // path only), so a real FEC-enabled file is outside decode_stream's actual input domain.
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, 4000, 1, None, &Default::default(),
    ).unwrap();
    let mut rng = Rng::new(12);
    for i in 0..10_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(4);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        if let Err(reason) = stream_decode_bounded(bytes) {
            panic!("iteration {i}, {n_flips} bit flips: {reason}");
        }
    }
}

#[test]
fn stream_decode_truncated_known_length_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..20_000i64).map(|i| ((i * 631) % 15001) - 7500).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, 4000, 1, None, &Default::default(),
    ).unwrap();
    for cut in (0..valid.len()).step_by(11) {
        if let Err(reason) = stream_decode_bounded(valid[..cut].to_vec()) {
            panic!("truncated to {cut}/{} bytes: {reason}", valid.len());
        }
    }
}

/// A `TOTAL_FRAMES_UNKNOWN` stream (the genuinely unbounded/live case, `encoder::StreamEncoder`)
/// exercises a real, different code path from the known-length tests above -- its own stopping
/// condition (EOF, not a declared total) and its own per-chunk sanity bound (no `data.len()` to
/// fall back on the way the in-memory `locate_chunks` path has). Bit-flipped and truncated variants
/// of a real streamed-mode encode must still never panic or hang.
#[test]
fn stream_decode_unknown_length_stream_never_panics_or_hangs() {
    use fak::encoder::StreamEncoder;
    let samples: Vec<i64> = (0..20_000i64).map(|i| ((i * 977) % 20001) - 10000).collect();
    let mut valid = Vec::new();
    {
        let mut enc = StreamEncoder::new(&mut valid, 1, 44100, 16, fak::format::MODE_BLOCK_INDEPENDENT, &Default::default()).unwrap();
        for chunk in samples.chunks(4000) { enc.push_chunk(&[chunk.to_vec()]).unwrap(); }
        enc.finish().unwrap();
    }
    let mut rng = Rng::new(13);
    for i in 0..10_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(4);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        if let Err(reason) = stream_decode_bounded(bytes) {
            panic!("iteration {i}, {n_flips} bit flips (unknown-length stream): {reason}");
        }
    }
    for cut in (0..valid.len()).step_by(11) {
        if let Err(reason) = stream_decode_bounded(valid[..cut].to_vec()) {
            panic!("truncated to {cut}/{} bytes (unknown-length stream): {reason}", valid.len());
        }
    }
}

///  (FEC): the decoder's own doc comment for `recover_chunk`
/// promises corrupted-then-repaired inputs are handled as safely as corrupted-and-rejected ones --
/// this fuzzes an FEC-enabled file specifically (a small group size, so a run of random bit flips
/// is likely to land more than one hit in the same group at least some of the time, exercising both
/// the successful-recovery and the safe-multi-loss-failure paths, plus corruption landing inside a
/// parity block's own table/payload rather than a data chunk). `decode_bounded`'s contract already
/// treats a successful recovery (`Ok`) and a correctly-rejected unrecoverable corruption (`Err`)
/// as equally valid outcomes -- only a panic or a hang is a failure here.
#[test]
fn fec_bit_flipped_valid_stream_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..30_000i64).map(|i| ((i * 977) % 20001) - 10000).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, MODE_BLOCK_INDEPENDENT, 3000, 1, Some(3), &Default::default(),
    ).unwrap();
    let mut rng = Rng::new(14);
    for i in 0..20_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(6);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        if let Err(reason) = decode_bounded(bytes) {
            panic!("iteration {i}, {n_flips} bit flips in an FEC-enabled stream: {reason}");
        }
    }
}

/// Same corpus of FEC-enabled corruption, through `seek` instead of `decode_full` -- its own
/// CRC-check-then-recover call site (not shared code with `decode_full`'s), so it needs its own
/// direct fuzz coverage too.
#[test]
fn fec_bit_flipped_valid_stream_via_seek_never_panics_or_hangs() {
    use fak::encoder;
    let samples: Vec<i64> = (0..30_000i64).map(|i| ((i * 631) % 15001) - 7500).collect();
    let valid = encoder::encode_chunked(
        &[samples.clone(), samples.iter().map(|&s| -s).collect()],
        44100, 16, MODE_BLOCK_INDEPENDENT, 3000, 1, Some(3), &Default::default(),
    ).unwrap();
    let mut rng = Rng::new(15);
    for i in 0..20_000u64 {
        let mut bytes = valid.clone();
        let n_flips = 1 + rng.below(6);
        for _ in 0..n_flips {
            let idx = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[idx] ^= 1 << bit;
        }
        let target = rng.below(30_000) as u64;
        if let Err(reason) = seek_bounded(bytes, target) {
            panic!("iteration {i}, {n_flips} bit flips, target {target} (FEC-enabled stream): {reason}");
        }
    }
}
