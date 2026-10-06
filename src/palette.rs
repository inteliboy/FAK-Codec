//! Palette (low-cardinality alphabet) coding: when a block uses only a handful of distinct raw
//! sample values, code each sample as a small fixed-width index into a value table instead of via
//! prediction + Rice. This exists for a real, reproducible gap the other predictors can't close:
//! i.i.d. data drawn from a small alphabet (e.g. a fair coin flip between two fixed extreme
//! values, sample-to-sample independent) has *no* predictive structure to exploit -- fixed/LPC
//! prediction and Rice coding of the residual both fail, costing the full bit depth per sample --
//! but the raw values themselves repeat from a tiny set, which an index code captures directly.
//! Found via a WavPack comparison: WavPack got a pathological
//! synthetic test file from 16 bits/sample (us, FLAC) down to ~3; a palette is the direct fix.

pub const MAX_PALETTE: usize = 16;

/// Bits needed to index `count` distinct values (>=1).
fn index_bits(count: usize) -> u32 {
    debug_assert!((1..=MAX_PALETTE).contains(&count));
    if count <= 1 { 0 } else { (usize::BITS - (count - 1).leading_zeros()).max(1) }
}

/// If `samples` uses at most `MAX_PALETTE` distinct values (and more than one, since a single
/// distinct value is the cheaper Constant case), returns the palette (first-seen order, so
/// encoding is deterministic) and the per-sample index stream. `None` otherwise.
pub fn build(samples: &[i64]) -> Option<(Vec<i64>, Vec<u32>)> {
    let mut palette: Vec<i64> = Vec::new();
    let mut idx = Vec::with_capacity(samples.len());
    for &s in samples {
        let pos = match palette.iter().position(|&v| v == s) {
            Some(p) => p,
            None => {
                if palette.len() >= MAX_PALETTE { return None; }
                palette.push(s);
                palette.len() - 1
            }
        };
        idx.push(pos as u32);
    }
    if palette.len() <= 1 { return None; }
    Some((palette, idx))
}

/// Total bits for the palette table + index stream (not counting the subframe-type selector).
pub fn cost_bits(palette_len: usize, n: usize, bits_eff: u32) -> u64 {
    4 + palette_len as u64 * bits_eff as u64 + n as u64 * index_bits(palette_len) as u64
}

pub fn index_width(palette_len: usize) -> u32 { index_bits(palette_len) }

/// Header fields for the run-length-coded index stream (`SubframeType::PaletteRle`): the fixed
/// width used for every run-length field, and how many runs there are. Both are stored once per
/// subframe rather than per run, since (unlike Rice's per-partition k) one width covering the
/// longest run in the block is enough to make a real difference on genuinely run-structured data
/// (e.g. a slow periodic square wave -- long runs of a repeated index, where
/// paying `index_bits` per *sample* is wasteful compared to paying it once per *run*) without the
/// complexity of an adaptive/partitioned scheme. Falls back to costing the same as flat `Palette`
/// (or worse, by the small fixed header) on data with no run structure -- the encoder always
/// compares both costs and picks whichever is actually smaller (`encoder.rs`), so this is never a
/// regression, only sometimes a missed opportunity to save more than it does.
pub struct RleParams { pub run_len_bits: u32, pub num_runs: usize }

/// Collapses an index stream into (index, run_length) pairs. `idx` must be non-empty (checked by
/// `build`'s own non-empty-input contract upstream; an empty slice just yields an empty result).
pub fn runs_from_indices(idx: &[u32]) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let mut it = idx.iter();
    if let Some(&first) = it.next() {
        let (mut cur, mut len) = (first, 1u32);
        for &v in it {
            if v == cur { len += 1; } else { runs.push((cur, len)); cur = v; len = 1; }
        }
        runs.push((cur, len));
    }
    runs
}

pub fn rle_params(runs: &[(u32, u32)]) -> RleParams {
    let max_len = runs.iter().map(|&(_, l)| l).max().unwrap_or(1);
    let run_len_bits = (u32::BITS - (max_len - 1).leading_zeros()).max(1);
    RleParams { run_len_bits, num_runs: runs.len() }
}

/// Total bits for the palette table + run-length-coded index stream (not counting the
/// subframe-type selector). Mirrors `cost_bits`'s contract (same table cost), differing only in
/// how the index stream itself is priced: `RLE_HEADER_BITS` (run_len_bits field + num_runs field)
/// once, then `index_bits + run_len_bits` per *run* instead of `index_bits` per *sample*.
const RUN_LEN_BITS_FIELD: u64 = 5; // covers run_len_bits up to 31, comfortably more than MAX_FRAME_FRAMES needs
const NUM_RUNS_FIELD: u64 = 20; // num_runs-1, covers up to 2^20 runs -- matches format::MAX_FRAME_FRAMES
pub fn rle_cost_bits(palette_len: usize, runs: &[(u32, u32)], bits_eff: u32) -> u64 {
    let p = rle_params(runs);
    let iw = index_bits(palette_len) as u64;
    4 + palette_len as u64 * bits_eff as u64 + RUN_LEN_BITS_FIELD + NUM_RUNS_FIELD
        + p.num_runs as u64 * (iw + p.run_len_bits as u64)
}

#[cfg(test)]
mod rle_tests {
    use super::*;

    #[test]
    fn runs_from_indices_collapses_repeats() {
        assert_eq!(runs_from_indices(&[0, 0, 0, 1, 1, 0, 0, 0, 0]), vec![(0, 3), (1, 2), (0, 4)]);
        assert_eq!(runs_from_indices(&[]), vec![]);
        assert_eq!(runs_from_indices(&[3]), vec![(3, 1)]);
        assert_eq!(runs_from_indices(&[1, 2, 3]), vec![(1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn rle_beats_flat_on_long_runs_loses_on_no_structure() {
        // A long-run signal (mimics square100: two values, long runs): RLE should cost far less
        // than flat per-sample indexing.
        let n = 4096;
        let idx: Vec<u32> = (0..n).map(|i| if (i / 220) % 2 == 0 { 0 } else { 1 }).collect();
        let runs = runs_from_indices(&idx);
        let flat = cost_bits(2, idx.len(), 16);
        let rle = rle_cost_bits(2, &runs, 16);
        assert!(rle < flat / 4, "rle={rle} should be well under a quarter of flat={flat} on long-run data");

        // Alternating with no run structure at all (worst case for RLE: every run has length 1):
        // RLE must cost *more* than flat here (extra per-run overhead with no runs to amortize it
        // over), which is fine -- the encoder picks whichever is smaller, this just confirms RLE
        // isn't free and the comparison is a real one.
        let idx: Vec<u32> = (0..n).map(|i| i % 2).collect();
        let runs = runs_from_indices(&idx);
        let flat = cost_bits(2, idx.len(), 16);
        let rle = rle_cost_bits(2, &runs, 16);
        assert!(rle > flat, "rle={rle} should cost more than flat={flat} when every run has length 1");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_finds_small_alphabets() {
        let (pal, idx) = build(&[5, -5, 5, 5, -5, -5]).unwrap();
        assert_eq!(pal, vec![5, -5]);
        assert_eq!(idx, vec![0, 1, 0, 0, 1, 1]);
    }

    #[test]
    fn rejects_constant_and_high_cardinality() {
        assert!(build(&[7, 7, 7, 7]).is_none()); // single value: Constant handles this instead
        let varied: Vec<i64> = (0..100).collect(); // 100 distinct values, far over MAX_PALETTE
        assert!(build(&varied).is_none());
        assert!(build(&[]).is_none());
    }

    #[test]
    fn index_bits_matches_palette_size() {
        assert_eq!(index_bits(2), 1);
        assert_eq!(index_bits(3), 2);
        assert_eq!(index_bits(4), 2);
        assert_eq!(index_bits(5), 3);
        assert_eq!(index_bits(16), 4);
    }
}
