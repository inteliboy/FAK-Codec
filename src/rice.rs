//! Partitioned Rice/Golomb coding of predictor residuals.
//!
//! Residuals are zigzag-mapped to non-negative integers, then split into fixed-size partitions.
//! Each partition picks its own code -- a Rice parameter or a recursive Golomb-Rice one (format
//! v16) -- signalled as the change from the previous partition's (v15), or escapes to
//! fixed-width raw storage when that is cheaper or when a residual is too large for any sane k --
//! this bounds per-value cost and keeps the decoder safe against pathological/adversarial residual
//! distributions. The partition size itself is chosen per subframe from a
//! small candidate set (see `PARTITION_SIZES`) by real cost, not fixed: a residual stream that is
//! mostly zero with rare huge spikes (e.g. a near-perfectly-predicted square wave, where the
//! predictor is exact except at each edge) wants small partitions so an outlier's Rice parameter
//! doesn't drag up the cost of thousands of unrelated zeros sharing its partition; a smooth noisy
//! residual wants large partitions to amortize its parameter code. Measured effect: this took one
//! adversarial synthetic file from +137% vs FLAC `-8` to parity.
use crate::bitio::{BitReader, BitReaderError, BitWriter};

pub const PARTITION_SIZES: [usize; 6] = [32, 64, 128, 256, 512, 1024];
/// Parameter code for "escape to fixed-width raw storage".
const ESCAPE: u32 = 63;
const MAX_K: u32 = 30;
/// Largest parameter index (format v16): see [`param`].
const MAX_PARAM: u32 = 2 * MAX_K + 1;

/// Parameter index -> `(k, rgr)`: index `2k` is Rice parameter `k`, `2k + 1` the recursive
/// Golomb-Rice code with parameter `k` (`BitReader::read_rgr`), whose scale sits between Rice `k`
/// and `k + 1`: its first `3 * 2^k` values cost `k + 2` bits, then one more bit per `2^k`.
#[inline(always)]
fn param(i: u32) -> (u32, bool) { (i >> 1, i & 1 == 1) }

/// Bits of one code for zigzag value `z` under parameter index `i` ([`param`]).
#[inline(always)]
fn code_len(z: u64, i: u32) -> u64 {
    let (k, rgr) = param(i);
    if !rgr { return (z >> k) + 1 + k as u64; }
    ((z >> k) + k as u64).max(k as u64 + 2)
}

/// Cost of signalling parameter code `p` (an index or `ESCAPE`) after the previous partitions
/// (format v15): the first partition of a stream, or any after only escapes, stores 6 raw bits;
/// later ones code the change from the last non-escape index, `0` (same), `10s` (+-1), `110s`
/// (+-2), else `111` + 6 raw bits -- AAC codes its scale factors, HEVC its Rice parameters, the
/// same way, since neighbouring partitions' parameters are nearly always equal or adjacent.
#[inline(always)]
fn sel_len(p: u32, prev: Option<u32>) -> u64 {
    match prev {
        None => 6,
        Some(_) if p == ESCAPE => 9,
        Some(q) => match p.abs_diff(q) { 0 => 1, 1 => 3, 2 => 4, _ => 9 },
    }
}

/// One partition's choice: a parameter index, or `ESCAPE` with a raw width.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Part { p: u32, width: u32 }
/// No legitimate residual from this codec's predictors ever needs more than this many bits
/// (widened from 40, which was sized for 25-bit samples/order-32 LPC, to keep the same ~15-bit
/// headroom above the widest supported container -- 32-bit samples, 33-bit widened side channel;
/// still well inside the escape width field's own 6-bit (0..=63) range, `write_bits(width, 6)`
/// below); a wider escape width can only come from a corrupted or hostile stream, so the decoder
/// rejects it outright rather than trusting it.
const MAX_ESCAPE_WIDTH: u32 = 56;

pub fn zigzag(v: i64) -> u64 { ((v << 1) ^ (v >> 63)) as u64 }
pub fn unzigzag(u: u64) -> i64 { ((u >> 1) as i64) ^ -((u & 1) as i64) }

fn bits_needed(max_val: u64) -> u32 { if max_val == 0 { 0 } else { 64 - max_val.leading_zeros() } }

/// The parameter indices worth pricing for a partition: around the Rice parameter the mean
/// suggests (`floor(log2(mean))`), one and a half steps down, one up (indices
/// `i0 - 3 ..= i0 + 2`). The window used to be `i0 - 4 ..= i0 + 3`; on 3 sets (34 real and synthetic
/// files, every level tried) the narrower one gave byte-identical files, and `i0 - 2 ..= i0 + 1`
/// differed by 0.0003%.
const CANDIDATES_BELOW: u32 = 3;
const CANDIDATES_ABOVE: u32 = 2;
#[inline(always)]
fn candidates(zz: &[u64]) -> std::ops::RangeInclusive<u32> {
    candidates_of(zz.iter().map(|&v| v as u128).sum(), zz.len())
}

/// [`candidates`] from a partition's sum and length.
#[inline(always)]
fn candidates_of(sum: u128, len: usize) -> std::ops::RangeInclusive<u32> { candidates_of_f(sum as f64, len) }

/// [`candidates_of`] from the sum already as an `f64` (the same rounding as `u128 as f64` for a sum
/// that fits a `u64`).
#[inline(always)]
fn candidates_of_f(sum: f64, len: usize) -> std::ops::RangeInclusive<u32> {
    let mean = sum / len.max(1) as f64;
    let k0 = if mean < 1.0 { 0 } else { crate::detmath::floor_log2(mean).clamp(0, MAX_K as i32) as u32 };
    let i0 = 2 * k0;
    i0.saturating_sub(CANDIDATES_BELOW)..=(i0 + CANDIDATES_ABOVE).min(MAX_PARAM)
}

/// Choose every partition's code for partition length `part_size`, greedily in stream order: each
/// takes whichever of its candidate parameters and the escape is cheapest including its own
/// signalling after the previous choice. Returns the choices and the exact total bits, including
/// the 3-bit partition-size field -- exactly what [`encode`] writes.
#[inline(always)]
fn plan_parts(zz: &[u64], part_size: usize) -> (Vec<Part>, u64) {
    let mut parts = Vec::with_capacity(zz.len().div_ceil(part_size));
    let mut total = 3u64;
    let mut prev: Option<u32> = None;
    for chunk in zz.chunks(part_size) {
        let width = bits_needed(*chunk.iter().max().unwrap_or(&0));
        let mut best = (sel_len(ESCAPE, prev) + 6 + width as u64 * chunk.len() as u64, Part { p: ESCAPE, width });
        for i in candidates(chunk) {
            let bits = sel_len(i, prev) + chunk.iter().map(|&z| code_len(z, i)).sum::<u64>();
            if bits < best.0 { best = (bits, Part { p: i, width: 0 }); }
        }
        total += best.0;
        if best.1.p != ESCAPE { prev = Some(best.1.p); }
        parts.push(best.1);
    }
    (parts, total)
}

#[cfg(test)]
fn cost_bits_at(res: &[i64], part_size: usize) -> u64 {
    let zz: Vec<u64> = res.iter().map(|&e| zigzag(e)).collect();
    plan_parts(&zz, part_size).1
}

/// The cheapest partition size for this residual stream, by real cost (index into
/// `PARTITION_SIZES`, the size itself, and its total cost including the 3-bit selector).
fn best_partition_size(res: &[i64]) -> (usize, usize, u64) {
    let p = best_plan_impl(res, false);
    (p.idx, p.size, p.bits)
}

/// The cheapest partition size for a residual stream together with every partition's code, so the
/// writer ([`encode_sized`]) does not have to plan them again.
pub struct PartitionPlan { idx: usize, size: usize, bits: u64, zz: Vec<u64>, parts: Vec<Part> }

/// Read-only view of a winning Rice plan for isolated entropy-coder experiments.
/// Available only when the research tap feature is enabled; it does not affect planning or SIMD.
#[cfg(feature = "research-tap")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct ResearchPart { pub param: u32, pub width: u32 }

impl PartitionPlan {
    /// Exact bits [`encode_sized`] writes for the residual this plan was made from.
    pub fn bits(&self) -> u64 { self.bits }

    /// Partition choices and zigzagged values, without re-running the Rice planner.
    /// `idx`, `size`, and the choices are exactly those selected by the production path (including
    /// its SIMD implementation when enabled).
    #[cfg(feature = "research-tap")]
    #[doc(hidden)]
    pub fn research_view(&self) -> (usize, usize, Vec<ResearchPart>, &[u64]) {
        (self.idx, self.size,
            self.parts.iter().map(|p| ResearchPart { param: p.p, width: p.width }).collect(),
            &self.zz)
    }
}

fn best_plan_impl(res: &[i64], want_parts: bool) -> PartitionPlan {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
            return unsafe { best_plan_avx2(res, want_parts) };
        }
    }
    best_plan_body(res, want_parts)
}

/// [`best_plan_impl`] compiled for AVX2: the same integer code, vectorized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn best_plan_avx2(res: &[i64], want_parts: bool) -> PartitionPlan { best_plan_body(res, want_parts) }

/// Samples per block of the cost tables in [`best_partition_size_body`]: the smallest partition size,
/// so every partition of every candidate size is a whole number of blocks (the last may be short).
const BLOCK: usize = PARTITION_SIZES[0];
/// Widest parameter-index range the block tables are built for (`hi - lo + 1`); a residual whose
/// partitions want a wider spread (spikes over silence) takes [`best_partition_size_direct`], which
/// prices only the parameters each partition needs.
const MAX_TABLE_WIDTH: u32 = 24;

/// The cheapest partition size for this residual stream, by real cost: the same result as running
/// [`plan_parts`] for every size ([`best_partition_size_direct`], the reference the tests compare
/// against) but pricing each parameter once per 32-sample block instead of once per size. Every
/// partition is a whole number of blocks, its cost under parameter `i` is the sum of its blocks'
/// costs (integers, so the sums are exact and the greedy choice is unchanged), and the escape width
/// comes from the blocks' maxima. Sizes are tried in ascending order and only a strictly cheaper
/// one replaces the best, as `min_by_key` keeps the first minimum.
#[inline(always)]
fn best_plan_body(res: &[i64], want_parts: bool) -> PartitionPlan {
    let zz: Vec<u64> = res.iter().map(|&e| zigzag(e)).collect();
    let (idx, size, bits, parts) = best_partition_size_from(&zz, want_parts);
    PartitionPlan { idx, size, bits, zz, parts }
}

/// [`best_plan_body`] on already-zigzagged values; `parts` (empty unless `want_parts`) are the
/// winning size's partition codes.
#[inline(always)]
fn best_partition_size_from(zz: &[u64], want_parts: bool) -> (usize, usize, u64, Vec<Part>) {
    if zz.is_empty() { return best_partition_size_zz(zz, want_parts); }
    let n = zz.len();
    let nb = n.div_ceil(BLOCK);
    // Per size (level `l` = `PARTITION_SIZES[l]`): every partition's sum, maximum and the parameter
    // range it will price, from the blocks' sums and maxima by pairwise merging.
    // A partition's escape width is the bit length of its maximum, which is that of the OR of its values.
    let mut ors: Vec<Vec<u64>> = Vec::with_capacity(PARTITION_SIZES.len());
    ors.push(zz.chunks(BLOCK).map(|b| b.iter().fold(0u64, |a, &v| a | v)).collect());
    // Values under 2^53 keep every partition's sum (at most 1024 values) inside a u64, and those
    // sums exact as f64; anything larger takes the reference planner.
    if ors[0].iter().fold(0u64, |a, &v| a | v) >= 1 << 53 { return best_partition_size_zz(zz, want_parts); }
    let mut sums: Vec<Vec<u64>> = Vec::with_capacity(PARTITION_SIZES.len());
    sums.push(zz.chunks(BLOCK).map(|b| b.iter().sum::<u64>()).collect());
    for l in 1..PARTITION_SIZES.len() {
        let s: Vec<u64> = sums[l - 1].chunks(2).map(|c| c.iter().sum()).collect();
        let m: Vec<u64> = ors[l - 1].chunks(2).map(|c| c.iter().fold(0u64, |a, &v| a | v)).collect();
        sums.push(s);
        ors.push(m);
    }
    let len_of = |l: usize, j: usize| PARTITION_SIZES[l].min(n - j * PARTITION_SIZES[l]);
    let ranges: Vec<Vec<std::ops::RangeInclusive<u32>>> = sums.iter().enumerate()
        .map(|(l, s)| s.iter().enumerate().map(|(j, &sum)| candidates_of_f(sum as f64, len_of(l, j))).collect()).collect();
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    for r in ranges.iter().flatten() { lo = lo.min(*r.start()); hi = hi.max(*r.end()); }
    if hi - lo + 1 > MAX_TABLE_WIDTH { return best_partition_size_zz(zz, want_parts); }
    // `pref[row][b]` = cost of blocks `0..b` under parameter `lo + row` (prefix sums, so a partition's
    // cost is a difference).
    let w = (hi - lo + 1) as usize;
    let stride = nb + 1;
    let mut pref = vec![0u64; w * stride];
    for (row, i) in pref.chunks_mut(stride).zip(lo..=hi) {
        let (k, rgr) = param(i);
        let mut run = 0u64;
        if !rgr {
            for (t, b) in row[1..].iter_mut().zip(zz.chunks(BLOCK)) { run += b.iter().map(|&z| (z >> k) + 1 + k as u64).sum::<u64>(); *t = run; }
        } else {
            for (t, b) in row[1..].iter_mut().zip(zz.chunks(BLOCK)) { run += b.iter().map(|&z| ((z >> k) + k as u64).max(k as u64 + 2)).sum::<u64>(); *t = run; }
        }
    }
    let mut best: Option<(usize, usize, u64, Vec<Part>)> = None;
    for (l, &sz) in PARTITION_SIZES.iter().enumerate() {
        let per = sz / BLOCK;
        let mut total = 3u64;
        let mut prev: Option<u32> = None;
        let mut parts: Vec<Part> = Vec::new();
        for (j, range) in ranges[l].iter().enumerate() {
            let (b0, b1) = (j * per, ((j + 1) * per).min(nb));
            let width = bits_needed(ors[l][j]);
            let mut cheapest = sel_len(ESCAPE, prev) + 6 + width as u64 * len_of(l, j) as u64;
            let mut pick = ESCAPE;
            for i in range.clone() {
                let row = &pref[(i - lo) as usize * stride..][..stride];
                let bits = sel_len(i, prev) + (row[b1] - row[b0]);
                if bits < cheapest { cheapest = bits; pick = i; }
            }
            total += cheapest;
            if pick != ESCAPE { prev = Some(pick); }
            if want_parts { parts.push(Part { p: pick, width: if pick == ESCAPE { width } else { 0 } }); }
        }
        if best.as_ref().is_none_or(|b| total < b.2) { best = Some((l, sz, total, parts)); }
    }
    best.unwrap()
}

/// The reference form of [`best_partition_size_from`]: every size planned from scratch. Also the
/// fallback for residuals whose partitions want a wide spread of parameters.
#[inline(always)]
fn best_partition_size_zz(zz: &[u64], want_parts: bool) -> (usize, usize, u64, Vec<Part>) {
    PARTITION_SIZES.iter().enumerate()
        .map(|(i, &sz)| { let (parts, cost) = plan_parts(zz, sz); (i, sz, cost, if want_parts { parts } else { Vec::new() }) })
        .min_by_key(|&(_, _, cost, _)| cost)
        .unwrap()
}

#[cfg(test)]
fn best_partition_size_direct(res: &[i64]) -> (usize, usize, u64) {
    let zz: Vec<u64> = res.iter().map(|&e| zigzag(e)).collect();
    let (i, s, c, _) = best_partition_size_zz(&zz, false);
    (i, s, c)
}

pub fn cost_bits(res: &[i64]) -> u64 { best_partition_size(res).2 }

/// Partition length `estimate_bits` costs separately.
pub const ESTIMATE_PART: usize = 256;

/// A cheap approximation of `cost_bits`, used only to *rank* predictor/order candidates during the
/// encoder's search (the real size comes from `encode`, once, for the winner): per 256-sample
/// partition, one Rice parameter from the partition's mean (no per-k search, no partition-size
/// search) plus its 5-bit selector; an all-zero partition costs what `encode` really writes for it
/// (a zero-width escape). Before this was a single k over the whole residual, which
/// misranked candidates often enough that *more* candidates made output larger; per partition it tracks `cost_bits` closely at similar cost.
pub fn estimate_bits(res: &[i64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::avx2_enabled() {
            // Safety: feature confirmed by `is_x86_feature_detected!("avx2")`.
            return unsafe { estimate_bits_avx2(res) };
        }
    }
    estimate_bits_body(res)
}

/// [`estimate_bits`] compiled for AVX2: the same integer code, auto-vectorized.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn estimate_bits_avx2(res: &[i64]) -> u64 { estimate_bits_body(res) }

#[inline(always)]
fn estimate_bits_body(res: &[i64]) -> u64 {
    if res.is_empty() { return 0; }
    res.chunks(ESTIMATE_PART).map(|c| {
        let sum: u64 = c.iter().map(|&e| zigzag(e)).sum();
        if sum == 0 { return 11; }
        let mean = sum as f64 / c.len() as f64;
        let k = if mean < 1.0 { 0 } else { crate::detmath::floor_log2(mean).clamp(0, MAX_K as i32) as u32 };
        5 + c.iter().map(|&e| (zigzag(e) >> k) + 1 + k as u64).sum::<u64>()
    }).sum::<u64>() + 3
}

/// Research tap (feature `research-tap`, never in normal builds): every residual block the encoder
/// writes, for pricing alternative entropy coders offline (`examples/residual_coders.rs`).
#[cfg(feature = "research-tap")]
#[doc(hidden)]
pub static TAP: std::sync::Mutex<Vec<Vec<i64>>> = std::sync::Mutex::new(Vec::new());
#[cfg(feature = "research-tap")]
pub static TAP_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Research tap, decode side: the partition-size index and every partition's selector (31 =
/// escape) of the residual streams the decoder reads (`decoder::TAP`).
#[cfg(feature = "research-tap")]
#[doc(hidden)]
pub static DTAP: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());
/// Disable research-only decoder tracing while measuring the production decode loop.
#[cfg(feature = "research-tap")]
#[doc(hidden)]
pub static DTAP_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn encode(w: &mut BitWriter, res: &[i64]) { encode_sized(w, res, &best_plan_impl(res, true)) }

/// The cheapest partition size for `res` as `(index, size, total bits)`: what [`cost_bits`] returns
/// the last element of, kept so a caller that priced a residual and then writes it (the encoder's
/// `write_subframe`) can hand the choice to [`encode_sized`] instead of searching twice.
pub fn best_partition(res: &[i64]) -> PartitionPlan { best_plan_impl(res, true) }

/// [`encode`] with the partitioning already chosen by [`best_partition`] *for this same `res`*.
pub fn encode_sized(w: &mut BitWriter, res: &[i64], plan: &PartitionPlan) {
    #[cfg(feature = "research-tap")]
    if TAP_ENABLED.load(std::sync::atomic::Ordering::Relaxed) { TAP.lock().unwrap().push(res.to_vec()); }
    let _ = res;
    w.write_bits(plan.idx as u64, 3);
    let (zz, part_size) = (&plan.zz, plan.size);
    let mut prev: Option<u32> = None;
    for (chunk, part) in zz.chunks(part_size).zip(plan.parts.iter().copied()) {
        write_sel(w, part.p, prev);
        if part.p == ESCAPE {
            w.write_bits(part.width as u64, 6);
            for &z in chunk { w.write_bits64(z, part.width); }
            continue;
        }
        prev = Some(part.p);
        let (k, rgr) = param(part.p);
        if rgr {
            for &z in chunk { w.write_rgr(z, k); }
        } else {
            for &z in chunk { w.write_rice(z, k); }
        }
    }
}

/// Writes parameter code `p` after `prev` exactly as [`sel_len`] prices it.
fn write_sel(w: &mut BitWriter, p: u32, prev: Option<u32>) {
    let Some(q) = prev else { return w.write_bits(p as u64, 6) };
    let neg = (p < q) as u64;
    match if p == ESCAPE { 3 } else { p.abs_diff(q) } {
        0 => w.write_bits(0, 1),
        1 => w.write_bits(0b100 | neg, 3),
        2 => w.write_bits(0b1100 | neg, 4),
        _ => { w.write_bits(0b111, 3); w.write_bits(p as u64, 6); }
    }
}

/// Reads a parameter code written by [`write_sel`], validating it: an index above `MAX_PARAM`
/// (other than `ESCAPE`) or a change that leaves `0..=MAX_PARAM` is a corrupted stream.
#[inline(always)]
fn read_sel(r: &mut BitReader, prev: Option<u32>) -> Result<u32, BitReaderError> {
    let raw = |r: &mut BitReader| -> Result<u32, BitReaderError> {
        let p = r.read_bits(6)? as u32;
        if p > MAX_PARAM && p != ESCAPE { return Err(BitReaderError("invalid Golomb parameter (corrupted stream?)")); }
        Ok(p)
    };
    let Some(q) = prev else { return raw(r) };
    if r.read_bits(1)? == 0 { return Ok(q); }
    let step = if r.read_bits(1)? == 0 { 1 } else if r.read_bits(1)? == 0 { 2 } else { return raw(r) };
    let neg = r.read_bits(1)? == 1;
    let p = if neg { q.checked_sub(step) } else { Some(q + step) };
    match p { Some(p) if p <= MAX_PARAM => Ok(p), _ => Err(BitReaderError("Golomb parameter change out of range (corrupted stream?)")) }
}

/// Decode exactly `n` residuals. `n` must come from data already trusted (the frame/subframe
/// header, whose own fields are range-checked before this is called) so partition counts and
/// sizes are bounded and cannot be inflated by a malicious partition-parameter field.
pub fn decode(r: &mut BitReader, n: usize) -> Result<Vec<i64>, BitReaderError> {
    #[cfg(target_arch = "x86_64")]
    {
        if lzcnt_enabled() {
            // Safety: features confirmed by `lzcnt_enabled`.
            return unsafe { decode_lzcnt(r, n) };
        }
    }
    decode_body(r, n)
}

/// `FAK_DISABLE_LZCNT=1` forces the portable bit-count code in the Rice decoder.
#[cfg(target_arch = "x86_64")]
fn lzcnt_enabled() -> bool {
    use std::sync::OnceLock;
    static LZCNT: OnceLock<bool> = OnceLock::new();
    *LZCNT.get_or_init(|| is_x86_feature_detected!("lzcnt") && is_x86_feature_detected!("bmi2")
        && std::env::var_os("FAK_DISABLE_LZCNT").is_none_or(|v| v != "1"))
}

/// The Rice decoder implementation in use on this machine.
pub fn decode_kernel() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if lzcnt_enabled() { return "lzcnt+bmi2"; }
    "portable"
}

/// [`decode`] compiled with `lzcnt`/`bmi2`: the Rice quotient is a leading-zero
/// count, which baseline x86-64 builds from `bsr` plus a zero-input fix-up. Same code, same
/// results (a compiler instruction-selection change only).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "lzcnt,bmi1,bmi2")]
unsafe fn decode_lzcnt(r: &mut BitReader, n: usize) -> Result<Vec<i64>, BitReaderError> { decode_body(r, n) }

#[inline(always)]
fn decode_body(r: &mut BitReader, n: usize) -> Result<Vec<i64>, BitReaderError> {
    let idx = r.read_bits(3)? as usize;
    if idx >= PARTITION_SIZES.len() { return Err(BitReaderError("invalid partition-size index")); }
    let part_size = PARTITION_SIZES[idx];
    #[cfg(feature = "research-tap")]
    if DTAP_ENABLED.load(std::sync::atomic::Ordering::Relaxed) { DTAP.lock().unwrap().push(idx as u8); }
    let mut out = Vec::with_capacity(n);
    let mut remaining = n;
    let mut prev: Option<u32> = None;
    while remaining > 0 {
        let take = remaining.min(part_size);
        let sel = read_sel(r, prev)?;
        #[cfg(feature = "research-tap")]
        if DTAP_ENABLED.load(std::sync::atomic::Ordering::Relaxed) { DTAP.lock().unwrap().push(sel as u8); }
        if sel == ESCAPE {
            let width = r.read_bits(6)? as u32;
            if width > MAX_ESCAPE_WIDTH { return Err(BitReaderError("escape width exceeds sane bound (corrupted stream?)")); }
            for _ in 0..take {
                let z = r.read_bits64(width)?;
                out.push(unzigzag(z));
            }
        } else {
            prev = Some(sel);
            match param(sel) {
                (k, false) => r.read_golomb_into::<false>(k, 1 << 32, take, &mut out, unzigzag)?,
                (k, true) => r.read_golomb_into::<true>(k, 1 << 32, take, &mut out, unzigzag)?,
            }
        }
        remaining -= take;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(res: &[i64]) {
        let mut w = BitWriter::new();
        encode(&mut w, res);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let back = decode(&mut r, res.len()).unwrap();
        assert_eq!(back, res);
    }

    #[test]
    fn best_partition_size_matches_per_partition_reference() {
        let mut st = 0x853C_49E6_748F_EA9Bu64;
        for trial in 0..400usize {
            let n = 1 + trial * 7 % 3000;
            let scale = 1i64 << (trial % 40);
            let res: Vec<i64> = (0..n).map(|_| { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st as i64).rem_euclid(2 * scale) - scale }).collect();
            let want = PARTITION_SIZES.iter().enumerate().map(|(i, &sz)| (i, sz, cost_bits_at(&res, sz))).min_by_key(|&(_, _, c)| c).unwrap();
            assert_eq!(best_partition_size(&res), want, "n={n}");
            assert_eq!(best_partition_size(&res), want);
            assert_eq!(best_partition_size_direct(&res), want);
        }
    }

    /// Residuals whose scale changes by orders of magnitude inside one subframe (spikes over
    /// silence, a decaying tail): the parameter spread exceeds `MAX_TABLE_WIDTH` for some and not
    /// others, so both the block-table path and the direct fallback are checked against the reference.
    #[test]
    fn best_partition_size_matches_reference_on_mixed_scales() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |m: u64| { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st % m };
        let (mut tabled, mut direct) = (0, 0);
        for trial in 0..600usize {
            let n = 1 + rnd(5000) as usize;
            let mut res = Vec::with_capacity(n);
            while res.len() < n {
                let seg = (1 + rnd(700) as usize).min(n - res.len());
                let bits = rnd(if trial % 3 == 0 { 30 } else { 12 });
                let scale = 1i64 << bits;
                for _ in 0..seg { res.push(if trial % 5 == 0 && rnd(97) != 0 { 0 } else { rnd(2 * scale as u64) as i64 - scale }); }
            }
            let want = best_partition_size_direct(&res);
            assert_eq!(best_partition_size(&res), want, "trial {trial} n={n}");
            let zz: Vec<u64> = res.iter().map(|&e| zigzag(e)).collect();
            let sums: Vec<u128> = zz.chunks(BLOCK).map(|b| b.iter().map(|&v| v as u128).sum()).collect();
            let (mut lo, mut hi) = (u32::MAX, 0u32);
            for &sz in &PARTITION_SIZES {
                for (c, chunk) in zz.chunks(sz).enumerate() {
                    let per = sz / BLOCK;
                    let r = candidates_of(sums[c * per..((c + 1) * per).min(sums.len())].iter().sum(), chunk.len());
                    lo = lo.min(*r.start()); hi = hi.max(*r.end());
                }
            }
            if hi - lo + 1 > MAX_TABLE_WIDTH { direct += 1 } else { tabled += 1 }
        }
        assert!(tabled > 50 && direct > 50, "both paths exercised: {tabled} tabled, {direct} direct");
    }

    #[test]
    fn zigzag_roundtrip() {
        for v in [0i64, 1, -1, 2, -2, 12345, -12345, i32::MAX as i64, i32::MIN as i64] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn roundtrip_various_distributions() {
        roundtrip(&[]);
        roundtrip(&[0; 500]);
        roundtrip(&(0..1000).map(|i| if i % 2 == 0 { i } else { -i }).collect::<Vec<_>>());
        roundtrip(&(0..2000).map(|i: i64| ((i * 7919) % 4001) - 2000).collect::<Vec<_>>());
        // Adversarial: huge outliers mixed with tiny values, forcing the escape path.
        roundtrip(&[0, 1, -1, 1 << 27, -(1 << 27), 0, 2, -2, i32::MAX as i64, i32::MIN as i64]);
        roundtrip(&vec![i32::MAX as i64; 300]);
        roundtrip(&vec![i32::MIN as i64; 300]);
    }

    #[test]
    fn cost_bits_matches_actual_encoded_length() {
        let res: Vec<i64> = (0..3000).map(|i: i64| ((i * 12345) % 777) - 388).collect();
        let mut w = BitWriter::new();
        encode(&mut w, &res);
        let actual = w.bit_len();
        let estimated = cost_bits(&res);
        assert_eq!(actual, estimated);
    }

    #[test]
    fn a_plan_prices_and_writes_what_encode_writes() {
        // `best_partition` -> `encode_sized` must produce the bytes `encode` does, and the plan's bits
        // are the bits written, over residuals of very different scales and shapes (including ones
        // whose partitions want a wide spread of parameters, which take the reference planner).
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let mut cases: Vec<Vec<i64>> = Vec::new();
        for scale in [1i64, 7, 300, 20_000, 3_000_000] {
            for n in [1usize, 31, 32, 33, 1000, 4096, 5000] {
                cases.push((0..n).map(|_| (next() % (2 * scale as u64 + 1)) as i64 - scale).collect());
            }
        }
        // Spikes over silence, and a scale that jumps by orders of magnitude.
        cases.push((0..4096).map(|i| if i % 700 == 0 { 1 << 30 } else { 0 }).collect());
        cases.push((0..4096).map(|i| if i < 2048 { (next() % 5) as i64 - 2 } else { (next() % 2_000_001) as i64 - 1_000_000 }).collect());
        for (i, res) in cases.iter().enumerate() {
            let plan = best_partition(res);
            let mut a = BitWriter::new();
            encode_sized(&mut a, res, &plan);
            let mut b = BitWriter::new();
            encode(&mut b, res);
            assert_eq!(a.bit_len(), plan.bits(), "case {i}: bits written vs planned");
            assert_eq!(plan.bits(), cost_bits(res), "case {i}: plan vs cost_bits");
            assert_eq!(a.finish(), b.finish(), "case {i}: encode_sized vs encode");
        }
    }

    #[test]
    fn estimate_bits_ranks_candidates_like_cost_bits() {
        let quiet: Vec<i64> = (0..2000).map(|i: i64| (i % 5) - 2).collect();
        let loud: Vec<i64> = (0..2000).map(|i: i64| ((i * 977) % 40001) - 20000).collect();
        assert!(estimate_bits(&quiet) < estimate_bits(&loud));
        assert!(cost_bits(&quiet) < cost_bits(&loud));
        assert_eq!(estimate_bits(&[]), 0);
    }

    #[test]
    fn small_partitions_win_on_sparse_spikes() {
        // Mostly-zero residual with rare huge spikes (what a near-perfect predictor leaves behind
        // on a square wave): small partitions should isolate the spikes and beat one fixed at 256.
        let mut res = vec![0i64; 4096];
        for i in (0..4096).step_by(220) { res[i] = 52426; }
        let (_, chosen_size, cost_chosen) = best_partition_size(&res);
        let cost_256 = cost_bits_at(&res, 256);
        assert!(cost_chosen <= cost_256, "chosen size {chosen_size} cost {cost_chosen} should be <= fixed-256 cost {cost_256}");
    }

    #[test]
    fn decoder_rejects_invalid_partition_index() {
        let mut w = BitWriter::new();
        w.write_bits(7, 3); // PARTITION_SIZES has 6 entries (indices 0..=5); 7 is invalid
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(decode(&mut r, 10).is_err());
    }

    /// Format v15/v16 parameter codes: every index round-trips through `write_sel`/`read_sel` from
    /// every previous index at the cost `sel_len` charges, and invalid ones are rejected.
    #[test]
    fn parameter_codes_roundtrip_and_reject_invalid() {
        for prev in [None].into_iter().chain((0..=MAX_PARAM).map(Some)) {
            for p in (0..=MAX_PARAM).chain([ESCAPE]) {
                let mut w = BitWriter::new();
                write_sel(&mut w, p, prev);
                assert_eq!(w.bit_len(), sel_len(p, prev), "p {p} prev {prev:?}");
                let bytes = w.finish();
                assert_eq!(read_sel(&mut BitReader::new(&bytes), prev).unwrap(), p);
            }
        }
        let bad = |bits: &[(u64, u32)], prev: Option<u32>| {
            let mut w = BitWriter::new();
            for &(v, n) in bits { w.write_bits(v, n); }
            w.write_bits(0, 16);
            let bytes = w.finish();
            read_sel(&mut BitReader::new(&bytes), prev).is_err()
        };
        assert!(bad(&[(62, 6)], None));
        assert!(bad(&[(0b111, 3), (62, 6)], Some(5)));
        assert!(bad(&[(0b101, 3)], Some(0)), "0 - 1");
        assert!(bad(&[(0b1101, 4)], Some(1)), "1 - 2");
        assert!(bad(&[(0b1100, 4)], Some(MAX_PARAM - 1)), "past the largest index");
        // Every parameter index codes and decodes a stream exactly.
        for i in 0..=MAX_PARAM.min(44) {
            let (k, rgr) = param(i);
            let m = if rgr { 3u64 << k } else { 1u64 << k };
            let res: Vec<i64> = (0..300i64).map(|j| ((j * 7919) % (4 * m as i64 + 3)) - 2 * m as i64).collect();
            let zz: Vec<u64> = res.iter().map(|&e| zigzag(e)).collect();
            let mut w = BitWriter::new();
            write_sel(&mut w, i, None);
            for &z in &zz { if rgr { w.write_rgr(z, k) } else { w.write_unary(z >> k); if k > 0 { w.write_bits(z & ((1 << k) - 1), k); } } }
            assert_eq!(w.bit_len(), 6 + zz.iter().map(|&z| code_len(z, i)).sum::<u64>(), "index {i}");
        }
    }

    #[test]
    fn decoder_rejects_oversized_escape_width() {
        let mut w = BitWriter::new();
        w.write_bits(0, 3); // valid partition-size index
        w.write_bits(ESCAPE as u64, 6); // first partition: raw 6-bit parameter code
        w.write_bits(63, 6); // width far beyond MAX_ESCAPE_WIDTH -- only reachable via a hostile stream
        w.write_bits64(0, 63);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert!(decode(&mut r, 1).is_err());
    }

    #[test]
    fn decoder_rejects_truncated_stream_without_panicking() {
        let res: Vec<i64> = (0..300).collect();
        let mut w = BitWriter::new();
        encode(&mut w, &res);
        let mut bytes = w.finish();
        bytes.truncate(bytes.len() / 2);
        let mut r = BitReader::new(&bytes);
        assert!(decode(&mut r, res.len()).is_err());
    }
}
