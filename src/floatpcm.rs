//! (b): lossless 32-bit float PCM, by exact fixed-point grid mapping.
//!
//! Most real 32-bit float audio (DAW/mastering-tool exports) is actually integer-precision content
//! normalized into float: every sample's exact real value equals `k / 2^s` for one file-wide integer
//! scale `s` and per-sample integer `k`. When that holds, the file can be losslessly reduced to
//! plain integer PCM -- the *entire* existing encoder/decoder/predictor stack runs unchanged, no
//! bitstream changes at all. A per-sample value that is not on that grid (`NaN`, `Inf`, `-0.0`
//! distinct from `+0.0`, or a genuine non-grid float such as real DSP/dither output) is stored
//! verbatim as a bit-pattern exception instead (`FloatInfo::exceptions`) -- correctness never
//! depends on the grid hypothesis holding, only compression efficiency does. A file that is *not*
//! grid-aligned at all (dense dithered float) still round-trips bit-exactly, just with one exception
//! per sample -- no better than storing it raw. A real WavPack-style dual int+exponent/mantissa
//! entropy scheme for that case is a separate, larger effort (not started (b)
//! status note).
//!
//! All exactness checks are done on the float's raw bit pattern with integer arithmetic -- never by
//! comparing floating-point values -- so there is no rounding ambiguity about what "exact" means.

/// Upper bound on the file-wide scale exponent `s` this module will choose (`k / 2^s`). Audio
/// content normalized to a float range needs at most `s` in the low 30s (full-scale int32 use
/// `s == 31`); values that genuinely need more precision than this fall back to per-sample
/// exceptions rather than growing `s` (and therefore the shift arithmetic, and the exception's own
/// cost when a single outlier sample would otherwise force deep precision on every other sample).
pub const MAX_SCALE_EXP: u32 = 31;

/// Bit pattern of `-0.0`. Numerically equal to `+0.0` (so it collapses to grid value `k == 0` like
/// `+0.0` under any scale) but not bit-identical to it, and the generic reconstruction path
/// (`reconstruct`) always regenerates the canonical `+0.0` bits for `k == 0` -- so `-0.0` can never
/// be produced by the grid path and always needs its own exception entry.
const NEG_ZERO_BITS: u32 = 0x8000_0000;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FloatInfo {
    /// The file-wide scale: a grid sample's exact real value is `k / 2^scale_exp`.
    pub scale_exp: u8,
    /// Samples that are not exactly on the grid at `scale_exp` (or are `NaN`/`Inf`/`-0.0`), stored
    /// as their raw bit pattern. The corresponding entry in the mapped integer PCM is a placeholder
    /// `0` -- ignored on decode, present only so the predictor pipeline sees a bounded value.
    pub exceptions: Vec<FloatException>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatException {
    pub channel: u32,
    /// Sample-frame index within the channel (0-based, matches `StreamHeader::total_frames`'s unit).
    pub index: u64,
    pub bits: u32,
}

/// Decomposes a finite float32 bit pattern into `(mantissa, exp2)` such that the exact real value is
/// `mantissa * 2^exp2` (mantissa carries the sign; not normalized -- may have trailing zero bits).
/// `None` for `Inf`/`NaN` (exponent field `0xFF`), which have no real value to decompose.
fn decompose(bits: u32) -> Option<(i64, i32)> {
    let exp_field = (bits >> 23) & 0xFF;
    if exp_field == 0xFF { return None; }
    let mantissa_field = (bits & 0x007F_FFFF) as i64;
    let sign: i64 = if bits & 0x8000_0000 != 0 { -1 } else { 1 };
    if exp_field == 0 {
        // Subnormal (includes zero): value = mantissa_field * 2^-149, no implicit leading bit.
        Some((sign * mantissa_field, -149))
    } else {
        // Normal: value = (2^23 | mantissa_field) * 2^(exp_field - 127 - 23).
        Some((sign * (0x0080_0000 | mantissa_field), exp_field as i32 - 150))
    }
}

/// The minimal scale `s >= 0` at which this (already-decomposed) value is exactly `k / 2^s` for some
/// integer `k`, ignoring container width. `m == 0` (zero magnitude) never needs any scale.
fn min_scale_for(m: i64, e: i32) -> u32 {
    if m == 0 { return 0; }
    let tz = m.trailing_zeros() as i64; // magnitude's trailing-zero count; correct for negative m too
    (-(e as i64) - tz).max(0) as u32
}

/// Guard against shift amounts that would overflow the i128 arithmetic below. No real audio grid
/// value needs a shift anywhere close to this; anything that would is treated as "not on the grid"
/// rather than risking overflow.
const MAX_SHIFT: i64 = 100;

/// Exact `k` such that the finite value's real value equals `k / 2^scale_exp`, or `None` if it is
/// not exactly representable at this scale (or is `NaN`/`Inf`/`-0.0`). No width bound here --
/// callers that need one check it themselves.
fn exact_k(bits: u32, scale_exp: u32) -> Option<i64> {
    if bits == NEG_ZERO_BITS { return None; }
    let (m, e) = decompose(bits)?;
    if m == 0 { return Some(0); }
    let shift = e as i64 + scale_exp as i64;
    if shift.abs() > MAX_SHIFT { return None; }
    let k128: i128 = if shift >= 0 {
        (m as i128) << shift
    } else {
        let s = (-shift) as u32;
        if (m as i128) & ((1i128 << s) - 1) != 0 { return None; } // would lose bits: not exact
        (m as i128) >> s
    };
    i64::try_from(k128).ok()
}

/// Reconstructs the exact float32 bit pattern for a grid value `k / 2^scale_exp`. Safe (bit-exact,
/// no rounding) precisely because every `k` this is ever called with came from `exact_k` succeeding
/// at the same `scale_exp` -- so the value is guaranteed exactly representable in binary32, and
/// converting an exactly-representable value through wider (f64) precision and back never rounds.
fn reconstruct(k: i64, scale_exp: u8) -> u32 {
    if k == 0 { return 0; } // canonical +0.0
    let v = k as f64 / (2f64).powi(scale_exp as i32);
    (v as f32).to_bits()
}

/// Smallest of {8, 16, 24, 32} whose signed range holds every value in `ks`.
fn min_width(ks: &[i64]) -> u8 {
    let max_abs = ks.iter().fold(0i64, |a, &k| a.max(k.unsigned_abs().min(i64::MAX as u64) as i64));
    for bits in [8u8, 16, 24] {
        if max_abs < (1i64 << (bits - 1)) { return bits; }
    }
    32
}

/// Maps raw float32 bit patterns (one `Vec<u32>` per channel, equal lengths) to integer PCM plus the
/// side information needed to invert it exactly. Returns `(channels, bits_per_sample, info)`.
pub fn map_to_pcm(channels_bits: &[Vec<u32>]) -> (Vec<Vec<i64>>, u8, FloatInfo) {
    let mut need = 0u32;
    for ch in channels_bits {
        for &bits in ch {
            if bits == NEG_ZERO_BITS { continue; }
            if let Some((m, e)) = decompose(bits) {
                need = need.max(min_scale_for(m, e));
            }
        }
    }
    let scale_exp = need.min(MAX_SCALE_EXP);

    let mut mapped: Vec<Vec<i64>> = channels_bits.iter().map(|ch| Vec::with_capacity(ch.len())).collect();
    let mut exceptions = Vec::new();
    for (ci, ch) in channels_bits.iter().enumerate() {
        for (i, &bits) in ch.iter().enumerate() {
            match exact_k(bits, scale_exp) {
                Some(k) if k.unsigned_abs() <= i32::MAX as u64 => mapped[ci].push(k),
                _ => {
                    mapped[ci].push(0);
                    exceptions.push(FloatException { channel: ci as u32, index: i as u64, bits });
                }
            }
        }
    }
    let all_ks: Vec<i64> = mapped.iter().flatten().copied().collect();
    let width = min_width(&all_ks);
    (mapped, width, FloatInfo { scale_exp: scale_exp as u8, exceptions })
}

/// Inverts `map_to_pcm`: decoded integer PCM plus the `FloatInfo` that was stored -> exact original
/// float32 bit patterns. `channels` may be owned `Vec`s or borrowed slices (a decoder can hand over
/// one chunk's or one seek range's worth without copying) -- pair with `slice_info` when `channels`
/// doesn't start at the stream's first sample-frame, since exception indices are absolute over the
/// whole stream.
pub fn unmap_from_pcm<C: AsRef<[i64]>>(channels: &[C], info: &FloatInfo) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = channels.iter().map(|ch| ch.as_ref().iter().map(|&k| reconstruct(k, info.scale_exp)).collect()).collect();
    for e in &info.exceptions {
        if let Some(ch) = out.get_mut(e.channel as usize) {
            if let Some(slot) = ch.get_mut(e.index as usize) { *slot = e.bits; }
        }
    }
    out
}

/// Slices `info` down to the exceptions whose (whole-stream-absolute) sample-frame index falls
/// within `[start, start + len)`, re-based to be relative to `start` -- for reconstructing just one
/// contiguous piece of a larger float stream (one decoded chunk, one seek range) with
/// `unmap_from_pcm`, rather than the whole thing at once.
pub fn slice_info(info: &FloatInfo, start: u64, len: u64) -> FloatInfo {
    let end = start + len;
    FloatInfo {
        scale_exp: info.scale_exp,
        exceptions: info.exceptions.iter()
            .filter(|e| e.index >= start && e.index < end)
            .map(|e| FloatException { channel: e.channel, index: e.index - start, bits: e.bits })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(channels_bits: &[Vec<u32>]) {
        let (mapped, bits, info) = map_to_pcm(channels_bits);
        assert!(bits == 8 || bits == 16 || bits == 24 || bits == 32);
        let back = unmap_from_pcm(&mapped, &info);
        assert_eq!(&back, channels_bits);
    }

    #[test]
    fn plus_and_minus_zero_round_trip_distinctly() {
        roundtrip(&[vec![0x0000_0000, 0x8000_0000, 0x0000_0000]]);
    }

    #[test]
    fn nan_payloads_and_infinities_round_trip_bit_exact() {
        let vals = vec![
            f32::NAN.to_bits(),
            0x7FC0_0000, // canonical quiet NaN
            0x7F80_0001, // signaling NaN, payload 1
            0xFFC0_1234, // negative NaN with an arbitrary payload
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
        ];
        roundtrip(&[vals]);
    }

    #[test]
    fn denormals_round_trip() {
        let vals: Vec<u32> = (0u32..20).chain((0u32..20).map(|m| m | 0x8000_0000)).collect();
        roundtrip(&[vals]);
    }

    #[test]
    fn int16_normalized_grid_has_zero_exceptions() {
        // Common DAW convention: original int16 samples divided by 32768.
        let raw: Vec<i32> = vec![0, 1, -1, 32767, -32768, 12345, -12345, 100, -100];
        let bits: Vec<u32> = raw.iter().map(|&s| (s as f32 / 32768.0).to_bits()).collect();
        let (_, _, info) = map_to_pcm(&[bits.clone()]);
        assert_eq!(info.exceptions.len(), 0, "clean int16-normalized content should need no exceptions");
        assert_eq!(info.scale_exp, 15);
        roundtrip(&[bits]);
    }

    #[test]
    fn int24_normalized_grid_has_zero_exceptions() {
        let raw: Vec<i32> = vec![0, 1, -1, 8_388_607, -8_388_608, 1_234_567, -1_234_567];
        let bits: Vec<u32> = raw.iter().map(|&s| (s as f32 / 8_388_608.0).to_bits()).collect();
        let (_, _, info) = map_to_pcm(&[bits.clone()]);
        assert_eq!(info.exceptions.len(), 0);
        assert_eq!(info.scale_exp, 23);
        roundtrip(&[bits]);
    }

    #[test]
    fn full_scale_int32_normalized_grid_is_mostly_exact() {
        // `i32::MAX / 2^31` and `i32::MIN / 2^31` are the two full-scale extremes; float32's 24-bit
        // mantissa can't hold `i32::MAX`'s value exactly, so the division itself rounds `i32::MAX /
        // 2^31` up to exactly `1.0` -- which needs `2^31`, one bit past a 32-bit signed container's
        // range at scale_exp 31. That's a real, correct precision loss already baked into the source
        // float32 file (not an artifact of this module), and the encoder must (and does) fall back
        // to an exception for exactly those two boundary samples rather than mis-widening the scale
        // for everything else; every mid-range sample still needs zero exceptions.
        let raw: Vec<i64> = vec![0, 1, -1, i32::MAX as i64, i32::MIN as i64, 1_000_000_000, -1_000_000_000];
        let bits: Vec<u32> = raw.iter().map(|&s| (s as f64 / 2147483648.0) as f32).map(f32::to_bits).collect();
        let (_, _, info) = map_to_pcm(&[bits.clone()]);
        assert_eq!(info.exceptions.len(), 2);
        assert!(info.exceptions.iter().all(|e| e.index == 3 || e.index == 4), "only the +-full-scale samples should need an exception");
        roundtrip(&[bits]);
    }

    #[test]
    fn one_wild_outlier_becomes_an_exception_not_a_global_precision_blowup() {
        // A clean int16-normalized channel, plus one sample carrying deep subnormal precision that
        // would force scale_exp far past MAX_SCALE_EXP if it set the global scale.
        let mut bits: Vec<u32> = (0i32..2000).map(|s| (((s % 4000) - 2000) as f32 / 32768.0).to_bits()).collect();
        bits[500] = 0x0000_0001; // smallest positive subnormal: needs scale_exp == 149
        let (mapped, _, info) = map_to_pcm(&[bits.clone()]);
        assert_eq!(info.exceptions.len(), 1);
        assert_eq!(info.exceptions[0].index, 500);
        assert_eq!(info.exceptions[0].bits, 0x0000_0001);
        assert!(info.scale_exp as u32 <= MAX_SCALE_EXP);
        assert_eq!(mapped[0][500], 0); // placeholder
        roundtrip(&[bits]);
    }

    #[test]
    fn genuinely_non_grid_dithered_float_still_round_trips() {
        // Deterministic PRNG standing in for real DSP/dither output: essentially never lands on any
        // single global grid, so this exercises "every sample is an exception."
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let bits: Vec<u32> = (0..500).map(|_| {
            let raw = next() as u32;
            // Avoid accidentally generating Inf/NaN (exponent field 0xFF) for this test, which is
            // about the non-grid-finite-value path specifically.
            if (raw >> 23) & 0xFF == 0xFF { raw & 0x7F7F_FFFF } else { raw }
        }).collect();
        roundtrip(&[bits]);
    }

    #[test]
    fn multi_channel_exceptions_track_the_right_channel_and_index() {
        let ch0: Vec<u32> = vec![0.0f32, 0.5, -0.5].into_iter().map(f32::to_bits).collect();
        let ch1: Vec<u32> = vec![0.25f32, f32::NAN, -0.25].into_iter().map(f32::to_bits).collect();
        let (_, _, info) = map_to_pcm(&[ch0.clone(), ch1.clone()]);
        assert_eq!(info.exceptions.len(), 1);
        assert_eq!(info.exceptions[0].channel, 1);
        assert_eq!(info.exceptions[0].index, 1);
        roundtrip(&[ch0, ch1]);
    }

    #[test]
    fn empty_channels_do_not_panic() {
        roundtrip(&[vec![], vec![]]);
    }

    #[test]
    fn random_bit_patterns_always_round_trip() {
        let mut st = 0xD1B54A32D192ED03u64;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        for _ in 0..50 {
            let bits: Vec<u32> = (0..200).map(|_| next() as u32).collect();
            roundtrip(&[bits]);
        }
    }

    #[test]
    fn slice_info_rebases_and_filters_exceptions_for_a_sub_range() {
        let raw: Vec<u32> = vec![0.0f32, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0].into_iter().map(f32::to_bits).collect();
        let mut bits = raw.clone();
        bits[1] = f32::NAN.to_bits(); // absolute index 1
        bits[6] = 0x8000_0000; // absolute index 6 (-0.0)
        let (mapped, _, info) = map_to_pcm(&[bits.clone()]);
        assert_eq!(info.exceptions.len(), 2);

        // A chunk covering absolute frames [4..8): only the index-6 exception should survive,
        // rebased to local index 2.
        let sliced = slice_info(&info, 4, 4);
        assert_eq!(sliced.exceptions.len(), 1);
        assert_eq!(sliced.exceptions[0].index, 2);
        assert_eq!(sliced.exceptions[0].bits, 0x8000_0000);
        let part = [mapped[0][4..8].to_vec()];
        let back = unmap_from_pcm(&part, &sliced);
        assert_eq!(back[0], bits[4..8]);

        // A chunk covering [0..4): only the index-1 exception survives, rebased to local index 1.
        let sliced2 = slice_info(&info, 0, 4);
        assert_eq!(sliced2.exceptions.len(), 1);
        assert_eq!(sliced2.exceptions[0].index, 1);
        let part2 = [mapped[0][0..4].to_vec()];
        let back2 = unmap_from_pcm(&part2, &sliced2);
        assert_eq!(back2[0], bits[0..4]);
    }
}
