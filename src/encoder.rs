//! Reference encoder: fixed predictors (order 0-4) + stereo decorrelation + partitioned Rice,
//! block-independent so every frame decodes without any state from earlier frames.
use crate::bitio::BitWriter;
use crate::crc::crc32;
use crate::crossch::{self, CrossParams};
use crate::format::{
    default_chunk_frames, write_chunk_header, ChunkEntry, ParityBuilder, FormatError, StreamHeader,
    SubframeType, DEFAULT_BLOCK_SIZE, HISTORY_LEN, MAX_CHUNK_FRAMES,
    MAX_FRAME_FRAMES, MODE_BLOCK_INDEPENDENT,
};
use crate::metadata::{self, Metadata};
use crate::parallel;
use crate::sha256;
use crate::stage2;
use crate::ltp;
use crate::lpc::{self, QuantizedLpc, MAX_PRECISION, MIN_PRECISION};
use crate::palette;
use crate::predictors::{self, MAX_ORDER};
use crate::rice;
use crate::stereo::{self, StereoMode};
use crate::valuemap;

/// Orders tried for LPC (a coarse search over the one Levinson-Durbin pass per block, not every
/// order up to MAX_ORDER -- cost-comparing real quantized-integer residuals is the expensive
/// part, this keeps it to a handful of candidates per block).
///
/// Every order since (was 1,2,4,6,8,12,16,24,32): the analytic ranking still shortlists
/// 3 per window, so this costs little, and with the per-partition `rice::estimate_bits` it was
/// smaller on every real corpus file.
const LPC_ORDER_CANDIDATES: &[usize] = &[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
];

#[derive(Clone)]
struct SubframePlan {
    kind: SubframeType, order: u8, wasted: u32, bits: u64,
    lpc: Option<QuantizedLpc>,
    palette: Option<(Vec<i64>, Vec<u32>)>,
    /// Cross-channel term (written as subframe type `Cross` around the Fixed/LPC `kind`).
    cross: Option<CrossParams>,
}

/// Largest k such that every sample is a multiple of 2^k (FLAC calls this "wasted bits": a
/// common trailing-zero run, e.g. audio that originated at a lower bit depth and was left-shifted
/// into a wider container). Shifting it out before prediction/entropy coding is lossless -- the
/// decoder shifts back in -- and removes bits that carry no information. Capped so at least 1 bit
/// of headroom remains (an all-zero subframe is caught by the CONSTANT case instead).
fn wasted_bits(samples: &[i64], bits_eff: u32) -> u32 {
    if samples.is_empty() || bits_eff == 0 { return 0; }
    let k = samples.iter().map(|&s| if s == 0 { 63 } else { s.trailing_zeros() }).min().unwrap_or(0);
    k.min(bits_eff - 1).min(30)
}

/// The sample sequence an order-`order` predictor runs over, and whether its warmup is sent
/// verbatim. Format v9: when the preceding frames of the same chunk
/// hold at least `order` samples, the warmup is those samples (`hist`, already in this subframe's
/// channel representation and wasted-bits domain) and every one of the frame's `n` samples gets a
/// Rice-coded residual; otherwise (a chunk's first frame) the first `order` samples are sent
/// verbatim, as before v9.
fn warmup_source<'a>(hist: &[i64], shifted: &'a [i64], ext: &'a [i64], order: usize) -> (&'a [i64], bool) {
    if order > 0 && hist.len() >= order { (&ext[hist.len() - order..], false) } else { (shifted, true) }
}

/// A subframe's chosen plan plus what [`refine_precision`] and [`write_subframe`] need: `ext`, the
/// history followed by the samples, both in the wasted-bits domain (`hist_len` of the former), and
/// (for an LPC plan) the winning candidate's float coefficients, kept so they can be requantized at
/// other precisions.
#[derive(Clone)]
struct Planned {
    plan: SubframePlan, ext: std::rc::Rc<Vec<i64>>, hist_len: usize, eff_bits: u32, lpc_float: Option<Vec<f64>>, cross_src: std::rc::Rc<Vec<i64>>,
    /// [`coded_residual`] of `plan`, computed once for the cross-channel search (which reads each
    /// plan's residual from several stereo modes and windows). Reset whenever `plan` changes.
    resid: std::cell::OnceCell<Option<(Vec<i64>, usize)>>,
}

impl Planned {
    fn hist(&self) -> &[i64] { &self.ext[..self.hist_len] }
    fn shifted(&self) -> &[i64] { &self.ext[self.hist_len..] }
}

/// Estimated bits for an LPC subframe with coefficients `q` (the same `estimate_bits` ranking cost
/// the candidate search uses).
fn lpc_bits(q: &QuantizedLpc, hist: &[i64], shifted: &[i64], ext: &[i64], eff_bits: u32) -> u64 {
    let order = q.coeffs.len();
    let (src, verbatim) = warmup_source(hist, shifted, ext, order);
    let res = lpc::residuals_estimate(q, src);
    let warmup_bits = if verbatim { order as u64 * eff_bits as u64 } else { 0 };
    // warmup + coeffs + shift and precision fields
    warmup_bits + lpc::coeff_bits(&q.coeffs) + 9 + rice::estimate_bits(&res)
}

/// Estimated bits for an LPC subframe as [`lpc_bits`], but from every `stride`-th 256-sample partition of
/// the residual only, scaled up: ranks candidates at a fraction of the cost (`FAK_SUB`).
fn lpc_bits_sampled(q: &QuantizedLpc, hist: &[i64], shifted: &[i64], ext: &[i64], eff_bits: u32, stride: usize) -> u64 {
    let order = q.coeffs.len();
    let (src, verbatim) = warmup_source(hist, shifted, ext, order);
    let n_out = src.len().saturating_sub(order);
    let (mut bits, mut seen) = (0u64, 0usize);
    let (mut t, mut part) = (0usize, 0usize);
    while t < n_out {
        let len = rice::ESTIMATE_PART.min(n_out - t);
        if part % stride == 0 {
            let res = lpc::residuals_estimate(q, &src[t..t + order + len]);
            bits += rice::estimate_bits(&res).saturating_sub(3);
            seen += len;
        }
        t += len;
        part += 1;
    }
    let scaled = if seen == 0 { 0 } else { (bits as f64 * n_out as f64 / seen as f64) as u64 };
    let warmup_bits = if verbatim { order as u64 * eff_bits as u64 } else { 0 };
    warmup_bits + lpc::coeff_bits(&q.coeffs) + 9 + scaled
}

/// Candidates are ranked on every `FAK_SUB`-th partition (1: every candidate exactly on all of them)
/// and the best `FAK_SUB_TOP` are then costed exactly. 3 and 2 cost +0.002..0.004% on three sets for
/// 3-4% less encode time (4 and 3: the same size); 2 and 2 or fewer exact costings lost more.
fn sub_stride() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_SUB").ok().and_then(|v| v.parse().ok()).unwrap_or(3))
}
fn sub_top() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_SUB_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// A subframe's search inputs, computed before any candidate is costed.
enum Analysis {
    /// Needs no search (empty or constant).
    Done(Planned),
    Search(Search),
}

/// `ext`: history then samples in the wasted-bits domain (`hist_len` of the former); `floats` the
/// LPC candidates, `quant` each quantized at `PRECISION` (None where that fails).
struct Search { wasted: u32, eff_bits: u32, ext: Vec<i64>, hist_len: usize, floats: Vec<Vec<f64>>, quant: Vec<Option<QuantizedLpc>>, best: usize, est: f64, model: Vec<lpc::CandModel>, autocorrs: Vec<Vec<f64>> }

fn analyze(samples: &[i64], history: &[i64], bits_eff: u32) -> Analysis {
    let g24 = crate::prof::span(crate::prof::Phase::AnalyzePrep);
    if samples.is_empty() {
        let plan = SubframePlan { kind: SubframeType::Verbatim, order: 0, wasted: 0, bits: 0, lpc: None, palette: None, cross: None };
        return Analysis::Done(Planned { plan, ext: Default::default(), hist_len: 0, eff_bits: bits_eff, lpc_float: None, cross_src: Default::default(), resid: Default::default() });
    }
    let wasted = wasted_bits(samples, bits_eff);
    let eff_bits = bits_eff - wasted;
    // History and samples, both shifted into the wasted-bits domain. Arithmetic shift (floor),
    // applied identically by the decoder: the history only has to be the *same* on both sides,
    // not divisible by `2^wasted`, since it is used for prediction only and never emitted.
    let ext: Vec<i64> = history.iter().chain(samples).map(|&s| s >> wasted).collect();
    let hist_len = history.len();
    let shifted = &ext[hist_len..];

    if shifted.iter().all(|&s| s == shifted[0]) {
        let plan = SubframePlan { kind: SubframeType::Constant, order: 0, wasted, bits: eff_bits as u64, lpc: None, palette: None, cross: None };
        return Analysis::Done(Planned { plan, ext: ext.into(), hist_len, eff_bits, lpc_float: None, cross_src: Default::default(), resid: Default::default() });
    }
    drop(g24);
    let lpc::CandidateSet { floats, model, autocorrs, best_pos: best, best_est: est } = lpc::candidates_full(shifted, LPC_ORDER_CANDIDATES, eff_bits, hist_len);
    let _g = crate::prof::span(crate::prof::Phase::Levinson);
    let quant = floats.iter().map(|a| lpc::quantize(a)).collect();
    Analysis::Search(Search { wasted, eff_bits, ext, hist_len, floats, quant, best, est, model, autocorrs })
}

/// Cost charged to a subframe whose search was skipped (`prune_stereo`): no stereo mode using it can win.
const SKIPPED_BITS: u64 = 1 << 60;

/// Percent by which a stereo mode's analytic estimate may exceed the best mode's and still get its
/// subframes searched (`FAK_SM` overrides `default`; negative = search all four, the exhaustive
/// behaviour). Only `Fast` prunes: with cross-channel prediction the mode that wins is often not
/// the one the plain estimate ranks first (+0.14..0.18% at `normal` for pruning to the best mode).
fn stereo_margin(default: f64) -> f64 {
    std::env::var("FAK_SM").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Of the four subframes of a stereo pair (L, R, M, S), keeps the searches only the modes whose
/// cost (one candidate) is within `stereo_margin` of the best mode need; the others become
/// [`SKIPPED_BITS`] placeholders.
fn prune_stereo(analyses: Vec<Analysis>, default_margin: f64) -> Vec<Analysis> {
    let margin = stereo_margin(default_margin);
    if margin < 0.0 || analyses.len() != 4 { return analyses; }
    // Two independent rankings of the four modes, and the union of the subframes their winners
    // use is searched. (a) The Levinson model's estimate: smooth, and better than the exhaustive
    // search's own cruder cost on real music, but wrong on clipped or repetitive signals (+8% on
    // one synthetic file). (b) The real Rice cost of each subframe's one best analytic candidate.
    // Where they agree only that mode's two subframes are searched.
    let mut lev = [0f64; 4];
    let mut real = [0f64; 4];
    for (i, a) in analyses.iter().enumerate() {
        match a {
            Analysis::Search(s) => {
                let Some(q) = s.quant.get(s.best).and_then(|q| q.as_ref()) else { return analyses };
                let (hist, shifted) = s.ext.split_at(s.hist_len);
                real[i] = lpc_bits(q, hist, shifted, &s.ext, s.eff_bits) as f64;
                lev[i] = s.est + 1.94 * shifted.len() as f64;
            }
            Analysis::Done(p) => { lev[i] = p.plan.bits as f64; real[i] = lev[i]; }
        }
    }
    if lev.iter().any(|e| !e.is_finite()) { return analyses; }
    // Subframes of each mode: LeftRight, MidSide, LeftSide, SideRight (L=0, R=1, M=2, S=3).
    const MODES: [[usize; 2]; 4] = [[0, 1], [2, 3], [0, 3], [3, 1]];
    let mut needed = [false; 4];
    for est in [&lev, &real] {
        let cost: Vec<f64> = MODES.iter().map(|m| est[m[0]] + est[m[1]]).collect();
        let best = cost.iter().copied().fold(f64::INFINITY, f64::min);
        for (m, c) in MODES.iter().zip(&cost) {
            if *c <= best * (1.0 + margin / 100.0) { for &i in m { needed[i] = true; } }
        }
    }
    analyses.into_iter().zip(needed).map(|(a, need)| match a {
        Analysis::Search(s) if !need => {
            let plan = SubframePlan { kind: SubframeType::Verbatim, order: 0, wasted: s.wasted, bits: SKIPPED_BITS, lpc: None, palette: None, cross: None };
            Analysis::Done(Planned { plan, ext: s.ext.into(), hist_len: s.hist_len, eff_bits: s.eff_bits, lpc_float: None, cross_src: Default::default(), resid: Default::default() })
        }
        a => a,
    }).collect()
}

/// The subframes of the frame `start..end` (history from `chunk_start`), analyzed: L, R, M, S for
/// a pair (S one bit wider), else each channel in order.
fn analyze_frame(channels: &[Vec<i64>], chunk_start: usize, start: usize, end: usize, bits_per_sample: u8) -> Vec<Analysis> {
    let h0 = start.saturating_sub(HISTORY_LEN).max(chunk_start);
    let base = bits_per_sample as u32;
    if channels.len() == 2 {
        let (l, r) = (&channels[0][start..end], &channels[1][start..end]);
        let (hl, hr) = (&channels[0][h0..start], &channels[1][h0..start]);
        let g25 = crate::prof::span(crate::prof::Phase::MidSide);
        let (m, hm, s, hs) = (stereo::mid(l, r), stereo::mid(hl, hr), stereo::side(l, r), stereo::side(hl, hr));
        drop(g25);
        vec![analyze(l, hl, base), analyze(r, hr, base), analyze(&m, &hm, base), analyze(&s, &hs, base + 1)]
    } else {
        channels.iter().map(|c| analyze(&c[start..end], &c[h0..start], base)).collect()
    }
}

/// Plans a frame's analyzed subframes; `warm[i]` is subframe `i`'s precision state.
fn plan_subframes(analyses: Vec<Analysis>, warm: &[u32], second_opinions: usize) -> Vec<Planned> {
    analyses.into_iter().enumerate().map(|(i, a)| match a {
        Analysis::Done(p) => p,
        Analysis::Search(s) => select(s, warm[i], second_opinions),
    }).collect()
}

/// The candidate search proper, on an [`analyze`]d subframe. `warm`: the coefficient precision
/// this channel slot settled on in the previous frame (`refine_precision`'s start).
/// `second_opinions`: how many of the best candidates (by their cost at `PRECISION`) are also
/// costed quantized at `warm` -- see the LPC loop below.
fn select(a: Search, warm: u32, second_opinions: usize) -> Planned {
    let Search { wasted, eff_bits, ext, hist_len, floats, quant, best: _, est: _, model, autocorrs } = a;
    let (hist, shifted) = ext.split_at(hist_len);
    let n = shifted.len() as u64;
    // Candidate search uses `estimate_bits` (cheap, single-pass, no partition-size search) purely
    // to rank ~14 candidates per subframe; it was the dominant encode-time cost (full `cost_bits`
    // does a 6-way partition-size × per-partition-k search per candidate). The real size is
    // computed once, for whichever candidate wins, by `write_subframe` -> `rice::encode`.
    let verbatim_bits = n * eff_bits as u64;
    let mut best = SubframePlan { kind: SubframeType::Verbatim, order: 0, wasted, bits: verbatim_bits, lpc: None, palette: None, cross: None };
    let g3 = crate::prof::span(crate::prof::Phase::FixedOrders);
    for order in 0..=MAX_ORDER.min(shifted.len().saturating_sub(1)) as u8 {
        let (src, verbatim) = warmup_source(hist, shifted, &ext, order as usize);
        let res = predictors::residuals(order, src);
        let warmup_bits = if verbatim { order as u64 * eff_bits as u64 } else { 0 };
        let bits = warmup_bits + rice::estimate_bits(&res);
        if bits < best.bits { best = SubframePlan { kind: SubframeType::Fixed, order, wasted, bits, lpc: None, palette: None, cross: None }; }
    }
    // Candidates are compared at `PRECISION` (14 bits), but `refine_precision` later moves the
    // winner to whatever precision is cheapest -- often far lower (5-10 bits on some content) --
    // so a comparison at 14 bits alone can pick the wrong candidate. The best `second_opinions`
    // are also costed at `warm`, where the previous frame's refinement ended up, and the best of
    // both is kept. Costing at `warm` *only* was measured unstable (many files larger, up to +5%:
    // a precision that suited the last frame can mislead this one); both together were smaller on
    // every file.
    drop(g3);
    let g4 = crate::prof::span(crate::prof::Phase::LpcCosting);
    let mut scored: Vec<(u64, usize, QuantizedLpc)> = if sub_stride() > 1 && shifted.len() >= 2 * rice::ESTIMATE_PART * sub_stride() {
        let mut ranked: Vec<(u64, usize, QuantizedLpc)> = quant.into_iter().enumerate()
            .filter_map(|(i, q)| q.map(|q| (lpc_bits_sampled(&q, hist, shifted, &ext, eff_bits, sub_stride()), i, q)))
            .collect();
        ranked.sort_by_key(|r| (r.0, r.1));
        ranked.into_iter().take(sub_top()).map(|(_, i, q)| (lpc_bits(&q, hist, shifted, &ext, eff_bits), i, q)).collect()
    } else {
        quant.into_iter().enumerate()
            .filter_map(|(i, q)| q.map(|q| (lpc_bits(&q, hist, shifted, &ext, eff_bits), i, q)))
            .collect()
    };
    drop(g4);
    let g13 = crate::prof::span(crate::prof::Phase::Precision);
    let use_model = precision_model();
    if use_model {
        // Precision by the excess-power model: the best few candidates (exactly costed at `PRECISION`)
        // are priced at every precision from the autocorrelation the coefficients came from, and the
        // best (candidate, precision) is then costed exactly once.
        let g5 = crate::prof::span(crate::prof::Phase::Precision);
        let mut by_cost: Vec<usize> = (0..scored.len()).collect();
        by_cost.sort_by_key(|&j| (scored[j].0, j));
        let n_coded = shifted.len() as f64;
        let mut pick: Option<(f64, usize, u32)> = None; // predicted bits, index into `scored`, precision
        for &j in by_cost.iter().take(model_top()) {
            let (bits_ref, i, q_ref) = (&scored[j].0, scored[j].1, &scored[j].2);
            let m = model[i];
            if let Some((prec, delta)) = lpc::best_precision(&floats[i], &autocorrs[m.window], m.err, q_ref, n_coded) {
                let pred = *bits_ref as f64 + delta;
                if pick.is_none_or(|p| pred < p.0) { pick = Some((pred, j, prec)); }
            }
        }
        if let Some((pred, j, prec)) = pick {
            if pred < scored[j].0 as f64 {
                let i = scored[j].1;
                if let Some(q) = lpc::quantize_at(&floats[i], prec) {
                    let bits = lpc_bits(&q, hist, shifted, &ext, eff_bits);
                    scored.push((bits, i, q));
                }
            }
        }
        drop(g5);
    } else if second_opinions > 0 && warm != lpc::PRECISION {
        let mut by_cost: Vec<usize> = (0..scored.len()).collect();
        by_cost.sort_by_key(|&j| (scored[j].0, j));
        let extra: Vec<(u64, usize, QuantizedLpc)> = by_cost.iter().take(second_opinions).filter_map(|&j| {
            let i = scored[j].1;
            lpc::quantize_at(&floats[i], warm).map(|q| (lpc_bits(&q, hist, shifted, &ext, eff_bits), i, q))
        }).collect();
        scored.extend(extra);
    }
    drop(g13);
    let mut lpc_float: Option<&Vec<f64>> = None;
    for (bits, i, q) in scored {
        if bits < best.bits {
            best = SubframePlan { kind: SubframeType::Lpc, order: 0, wasted, bits, lpc: Some(q), palette: None, cross: None };
            lpc_float = Some(&floats[i]);
        }
    }
    // Palette: i.i.d./few-valued blocks have no predictive structure for Fixed/LPC to exploit and
    // cost Rice-coded residuals the full bit depth, but the raw alphabet itself may be tiny.
    // PaletteRle (same underlying table+index stream): when the *same* index
    // repeats for many samples in a row (e.g. a slow periodic square wave), paying `index_bits`
    // once per sample is wasteful -- run-length-coding the index stream instead only pays it once
    // per run. Real per-block cost decides between flat and run-length, not a heuristic: on data
    // with no run structure RLE's small fixed header makes it strictly worse, so it only wins when
    // it actually should.
    let g23 = crate::prof::span(crate::prof::Phase::Palette);
    let pb = palette::build(shifted);
    drop(g23);
    if let Some((pal, idx)) = pb {
        let flat_bits = palette::cost_bits(pal.len(), idx.len(), eff_bits);
        let runs = palette::runs_from_indices(&idx);
        let rle_bits = palette::rle_cost_bits(pal.len(), &runs, eff_bits);
        let (kind, bits) = if rle_bits < flat_bits { (SubframeType::PaletteRle, rle_bits) } else { (SubframeType::Palette, flat_bits) };
        if bits < best.bits {
            best = SubframePlan { kind, order: 0, wasted, bits, lpc: None, palette: Some((pal, idx)), cross: None };
        }
    }
    let lpc_float = if best.kind == SubframeType::Lpc { lpc_float.cloned() } else { None };
    Planned { plan: best, ext: ext.into(), hist_len, eff_bits, lpc_float, cross_src: Default::default(), resid: Default::default() }
}

/// The LPC coefficient precision is chosen by the excess-power model (`select`); `FAK_PREC=walk` restores
/// the greedy walk of exactly costed precisions (`refine_precision`) and `FAK_PREC=model+walk`
/// does both (for testing).
fn precision_model() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("FAK_PREC").is_ok_and(|v| v == "walk"))
}
fn precision_refine() -> bool { std::env::var("FAK_PREC").is_ok_and(|v| v == "model+walk") }
fn model_top() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_MODEL_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}

/// Coefficient precision refinement: requantize the winning LPC candidate at other
/// precisions and keep whichever wins. A greedy walk, not an exhaustive search: it starts at
/// `*start` -- the precision the same channel slot settled on in the previous frame of this chunk,
/// `PRECISION` for a chunk's first frame -- and steps one bit at a time in whichever direction
/// lowers the cost, stopping at the first step that doesn't. The best precision is strongly
/// content-dependent (from 5 bits on a 192 kHz file to 15-16 on a 384 kHz one, measured) but
/// stable from frame to frame, so a warm start usually needs only the two neighbouring
/// evaluations. `start` is chunk-local state, so output stays identical for every thread count.
fn refine_precision(p: &mut Planned, start: &mut u32) {
    if precision_model() && !precision_refine() { return; }
    let (Some(a), Some(q0)) = (p.lpc_float.as_ref(), p.plan.lpc.as_ref()) else { return };
    let _g = crate::prof::span(crate::prof::Phase::Precision);
    let eval = |prec: u32| lpc::quantize_at(a, prec).map(|q| { let b = lpc_bits(&q, p.hist(), p.shifted(), &p.ext, p.eff_bits); (q, b) });
    let s0 = (*start).clamp(MIN_PRECISION, MAX_PRECISION);
    let mut best = if s0 == q0.precision { Some((q0.clone(), p.plan.bits)) } else { eval(s0) };
    let Some(mut best_bits) = best.as_ref().map(|(_, b)| *b) else { return };
    for dir in [-1i32, 1] {
        let mut prec = s0 as i32 + dir;
        let mut moved = false;
        while (MIN_PRECISION as i32..=MAX_PRECISION as i32).contains(&prec) {
            let Some((q, bits)) = eval(prec as u32) else { break };
            if bits >= best_bits { break; }
            best_bits = bits;
            best = Some((q, bits));
            moved = true;
            prec += dir;
        }
        if moved { break; }
    }
    let (q, bits) = best.expect("set above");
    *start = q.precision;
    if bits < p.plan.bits { p.plan.lpc = Some(q); p.plan.bits = bits; p.resid = Default::default(); }
}

/// Cross-channel search windows, as (first lag, tap count) around the current sample: the
/// narrow one is always tried; the wide one only when the narrow one already saves at least
/// `CROSS_WIDE_TRIGGER` of the subframe's bits (strong inter-channel delay structure), and it is
/// kept only if it saves a further `CROSS_MIN_GAIN`. Every cross-channel subframe costs a decode
/// pass proportional to its taps, so a candidate must save at least `CROSS_MIN_GAIN` (per mille)
/// of the subframe's estimated bits to be used at all.
const CROSS_NARROW: (i32, usize) = (-2, 5);
const CROSS_WIDE_DEFAULT: (i32, usize) = (-5, 11);
fn cross_wide() -> (i32, usize) {
    static V: std::sync::OnceLock<(i32, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_XW").ok().and_then(|v| { let (a, b) = v.split_once(',')?; Some((a.parse().ok()?, b.parse().ok()?)) }).unwrap_or(CROSS_WIDE_DEFAULT))
}
fn cross_trigger() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_XTRIG").ok().and_then(|v| v.parse().ok()).unwrap_or(CROSS_WIDE_TRIGGER))
}
const CROSS_MIN_GAIN: u64 = 10;
const CROSS_WIDE_TRIGGER: u64 = 30;
/// Precision the taps are quantized to (10-14 measured within 0.02% of each other).
const CROSS_PRECISION: u32 = 12;
/// Own order the joint (samples-source) fit uses when the target is not already an LPC plan.
const CROSS_DEFAULT_ORDER: usize = 8;
/// Fewer fitted positions than this and the cross-channel candidates are not tried.
const CROSS_MIN_FIT: usize = 64;

/// The residual of a Fixed/LPC plan's own predictor (for a cross-channel plan: before the
/// cross-channel term is taken out) and the frame position it starts at -- exactly what the
/// decoder's `SubOut` holds for this subframe. `None` for other subframe types.
fn coded_residual(p: &Planned) -> Option<(Vec<i64>, usize)> {
    let order = match p.plan.kind {
        SubframeType::Lpc => p.plan.lpc.as_ref()?.coeffs.len(),
        SubframeType::Fixed => p.plan.order as usize,
        _ => return None,
    };
    let (src, verbatim) = warmup_source(p.hist(), p.shifted(), &p.ext, order);
    let res = match p.plan.kind { SubframeType::Lpc => lpc::residuals(p.plan.lpc.as_ref()?, src), _ => predictors::residuals(order as u8, src) };
    Some((res, if verbatim { order } else { 0 }))
}

/// [`coded_residual`], cached in the plan.
fn cached_residual(p: &Planned) -> Option<&(Vec<i64>, usize)> { p.resid.get_or_init(|| coded_residual(p)).as_ref() }

fn subtract_terms(c: &CrossParams, src: &[i64], start: usize, res: &mut [i64]) {
    crossch::apply(c, src, start, res, true).expect("sources are checked against SOURCE_BOUND before use");
}

/// Frame-length signal of `p`'s predictor residual, zero where it has none (the decoder's
/// `SubOut::residual_signal`).
fn residual_signal(p: &Planned) -> Vec<i64> {
    let n = p.shifted().len();
    let mut v = vec![0i64; n];
    if let Some((res, start)) = cached_residual(p) { v[*start..].copy_from_slice(res); }
    v
}

/// `p`'s decoded samples with wasted bits restored (the decoder's `SubOut::samples`).
fn restored_samples(p: &Planned) -> Vec<i64> { p.shifted().iter().map(|&v| v << p.plan.wasted).collect() }

/// Best cross-channel variant of `b` referencing `a` (subframe index `ref_idx` in the frame), if
/// either candidate beats `b`'s own plan:
/// - *residual source*: taps over `a`'s coded residual, fitted to `b`'s residual; `b`'s own
///   predictor unchanged;
/// - *samples source*: `b`'s own LPC re-solved jointly with taps over `a`'s samples.
fn plan_cross(a: &Planned, b: &Planned, ref_idx: u8) -> Option<Planned> {
    let _g = crate::prof::span(crate::prof::Phase::CrossSearch);
    if !matches!(b.plan.kind, SubframeType::Fixed | SubframeType::Lpc) || b.plan.cross.is_some() { return None; }
    let step = |bits: u64, per_mille: u64| bits.saturating_sub(b.plan.bits * per_mille / 1000);
    let narrow = cross_window(a, b, ref_idx, CROSS_NARROW, step(b.plan.bits, CROSS_MIN_GAIN))?;
    if narrow.plan.bits <= step(b.plan.bits, cross_trigger()) {
        if let Some(wide) = cross_window(a, b, ref_idx, cross_wide(), step(narrow.plan.bits, CROSS_MIN_GAIN)) { return Some(wide); }
    }
    Some(narrow)
}

/// The better of the two cross-channel candidates for one window, if it costs fewer than `limit`
/// estimated bits.
fn cross_window(a: &Planned, b: &Planned, ref_idx: u8, (lag0, taps): (i32, usize), limit: u64) -> Option<Planned> {
    let n = b.shifted().len();
    let xprec = CROSS_PRECISION;
    let t_lo = (-lag0).max(0) as usize;
    let t_hi = n.saturating_sub((lag0 + taps as i32 - 1).max(0) as usize);
    let mut best: Option<Planned> = None;
    let mut best_bits = limit;
    let cross_feats: Vec<(usize, isize)> = (0..taps).map(|k| (0usize, lag0 as isize + k as isize)).collect();

    // Residual source.
    let g19 = crate::prof::span(crate::prof::Phase::CrossPrep);
    if let Some((eb, start)) = cached_residual(b).cloned() {
        let src = residual_signal(a);
        let (t0, t1) = (t_lo.max(start), t_hi);
        if t1 >= t0 + CROSS_MIN_FIT && crossch::source_in_bounds(&src) {
            let mut target = vec![0i64; n];
            target[start..].copy_from_slice(&eb);
            drop(g19);
            let g16 = crate::prof::span(crate::prof::Phase::CrossNormalEq);
            let (r, rhs) = crossch::normal_equations(&[&src, &target], &cross_feats, (1, 0), t0, t1);
            drop(g16);
            let g17 = crate::prof::span(crate::prof::Phase::CrossSolve);
            let sol = crossch::solve_spd(r, &rhs).and_then(|c| crossch::quantize(&c, xprec));
            drop(g17);
            let _g18 = crate::prof::span(crate::prof::Phase::CrossResidual);
            if let Some((coeffs, shift)) = sol {
                let cp = CrossParams { ref_idx, source: crossch::Source::Residual, lag0, coeffs, shift, precision: xprec };
                let mut res = eb.clone();
                subtract_terms(&cp, &src, start, &mut res);
                let bits = b.plan.bits - rice::estimate_bits(&eb) + 3 + cp.side_bits() + rice::estimate_bits(&res);
                if bits < best_bits {
                    best_bits = bits;
                    let mut v = b.clone();
                    v.plan.bits = bits;
                    v.plan.cross = Some(cp);
                    v.cross_src = src.into();
                    v.resid = Default::default();
                    best = Some(v);
                }
            }
        }
    }

    // Samples source, own LPC re-solved jointly.
    let own = match (&b.plan.lpc, b.plan.kind) { (Some(q), SubframeType::Lpc) => q.coeffs.len(), _ => CROSS_DEFAULT_ORDER };

    let h = b.hist_len;
    if h >= own && t_hi >= t_lo + CROSS_MIN_FIT {
        let src = restored_samples(a);
        if crossch::source_in_bounds(&src) {
            // Signal 0: `b.ext` (history + frame), frame position t at index t + h; signal 1: `src`.
            let mut feats: Vec<(usize, isize)> = (0..own).map(|j| (0usize, h as isize - 1 - j as isize)).collect();
            feats.extend(cross_feats.iter().map(|&(_, o)| (1usize, o)));
            let g16 = crate::prof::span(crate::prof::Phase::CrossNormalEq);
            let (r, rhs) = crossch::normal_equations(&[&b.ext, &src], &feats, (0, h as isize), t_lo, t_hi);
            drop(g16);
            let prec = b.plan.lpc.as_ref().map(|q| q.precision).unwrap_or(lpc::PRECISION);
            let g17 = crate::prof::span(crate::prof::Phase::CrossSolve);
            let solved = crossch::solve_spd(r, &rhs);
            drop(g17);
            let _g18 = crate::prof::span(crate::prof::Phase::CrossResidual);
            if let Some(c) = solved {
                if let (Some(q), Some((coeffs, shift))) = (lpc::quantize_at(&c[..own], prec), crossch::quantize(&c[own..], xprec)) {
                    let cp = CrossParams { ref_idx, source: crossch::Source::Samples, lag0, coeffs, shift, precision: xprec };
                    let mut res = lpc::residuals(&q, &b.ext[h - own..]);
                    subtract_terms(&cp, &src, 0, &mut res);
                    let bits = lpc::coeff_bits(&q.coeffs) + 9 + 3 + cp.side_bits() + rice::estimate_bits(&res);
                    if bits < best_bits {
                        let mut v = b.clone();
                        v.plan = SubframePlan { kind: SubframeType::Lpc, order: 0, wasted: b.plan.wasted, bits, lpc: Some(q), palette: None, cross: Some(cp) };
                        v.lpc_float = None;
                        v.cross_src = src.into();
                        v.resid = Default::default();
                        best = Some(v);
                    }
                }
            }
        }
    }
    best
}

/// Squared correlation of two plans' coded residuals: ranks which earlier channel a multichannel
/// subframe references (one cross-channel fit per channel instead of one per earlier channel).
fn residual_affinity(a: &[i64], b: &[i64]) -> f64 {
    let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) { let (x, y) = (x as f64, y as f64); xy += x * y; xx += x * x; yy += y * y; }
    if xx > 0.0 && yy > 0.0 { xy * xy / (xx * yy) } else { 0.0 }
}
/// Skip cross-channel planning for a pair whose coded residuals have squared zero-lag correlation
/// under this (default 0.05; `FAK_CROSS_SCREEN=0` disables). Encoder-only: the format is unchanged.
fn cross_screen() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_CROSS_SCREEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05))
}
/// Back-off for the long-term prediction search, one per subframe position and block size, reset at
/// each chunk. Music uses LTP in a few percent of subframes but periodic signals use it in runs, so a
/// search that finds nothing is not repeated for the next `skip` frames, doubling up to the cap
/// (`FAK_LTP_CAP`, frames). Encoder-only: the format never sees it.
#[derive(Clone, Copy, Default)]
struct LtpGate { skip: u32, next: u32 }
impl LtpGate {
    fn open(&mut self, lt: Option<ltp::Search>) -> Option<ltp::Search> {
        if self.skip > 0 { self.skip -= 1; None } else { lt }
    }
    fn record(&mut self, used: bool, cap: u32) {
        if used { self.next = 0; } else { self.next = (self.next * 2 + 1).min(cap); self.skip = self.next; }
    }
}
fn ltp_cap(_e: Effort) -> u32 {
    static V: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    V.get_or_init(|| std::env::var("FAK_LTP_CAP").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(3)
}
fn write_subframe(w: &mut BitWriter, p: &Planned, bits_eff: u32, stage2: &[(usize, u32)], lt: Option<ltp::Search>, load: u64, gate: &mut LtpGate, cap: u32, mut carry: Option<&mut stage2::Carried>) {
    let had_lt = lt.is_some();
    let lt = gate.open(lt);
    let outcome = std::cell::Cell::new(None::<bool>);
    let (plan, shifted, hist, ext) = (&p.plan, p.shifted(), p.hist(), &p.ext[..]);
    let eff_bits = bits_eff - plan.wasted;
    w.write_bits(plan.wasted as u64, 5);
    if plan.cross.is_some() {
        w.write_bits(SubframeType::Cross as u64, 3);
    }
    w.write_bits(plan.kind as u64, 3);
    // Residual writer shared by Fixed/LPC: the cross-channel fields (when present) sit between the
    // warmup and the residual, and the residual is net of the cross-channel term.
    let mut write_res = |w: &mut BitWriter, mut res: Vec<i64>, start: usize| {
        if let Some(c) = &plan.cross {
            c.write(w);
            subtract_terms(c, &p.cross_src, start, &mut res);
        }
        // Stage 2: each candidate filter is really run and costed; kept only if the
        // subframe gets smaller including its 13-bit header (else 1 flag bit says "off").
        let mut best: Option<(stage2::Params, Vec<i64>, u64)> = None;
        let mut plain = 0;
        let mut pick = None;
        let lt_cfg = lt.filter(|_| !res.is_empty());
        let price = |r: &[i64], base: u64| lt_cfg.and_then(|s| ltp::search(r, &s, base, rice::cost_bits));
        // Long-term prediction's result on whatever the chosen stage-2 branch leaves (None: off).
        let mut lt_res: Option<(ltp::Params, Vec<i64>, u64)> = None;
        let mut lt_done = false;
        if !res.is_empty() && res.iter().all(|e| e.abs() <= stage2::MAX_RESIDUAL) {
            let g10 = crate::prof::span(crate::prof::Phase::RiceCoding);
            let p = rice::best_partition(&res);
            drop(g10);
            plain = p.bits();
            pick = Some(p);
            let g8 = crate::prof::span(crate::prof::Phase::Stage2);
            let all: Vec<stage2::Params> = stage2.iter().map(|&(taps, k)| stage2::Params::for_block(&res, taps, k, stage2_target())).collect();
            // The 256-tap filters are the dear ones (most of stage 2's encode time) and, under the
            // decode-cost gate, almost never used: they run only when the cheaper filters already
            // save `stage2_screen()` bits per sample by the cheap estimate.
            let (small, large): (Vec<_>, Vec<_>) = all.into_iter().partition(|p| p.taps < LARGE_TAPS);
            let run = |ps: Vec<stage2::Params>| -> Vec<(stage2::Params, Vec<i64>, u64)> {
                if ps.is_empty() { return Vec::new(); }
                let outs = stage2::forward_multi(&ps, &res);
                ps.into_iter().zip(outs).map(|(p, c)| { let e = rice::estimate_bits(&c); (p, c, e) }).collect()
            };
            let mut cands = run(small);
            let screen_ok = match cands.iter().map(|c| c.2).min() {
                None => true,
                Some(best_small) => {
                    let saved = rice::estimate_bits(&res).saturating_sub(best_small) as f64;
                    saved >= stage2_screen() * res.len() as f64
                }
            };
            if screen_ok { cands.extend(run(large)); }
            // Exactly costing a candidate is the dearer step: only those whose cheap estimate is
            // within `STAGE2_EXACT_SLACK` (per mille) of the best estimate are costed exactly.
            // Decode-cost gate: a filter must save `gate` bits per sample per 256 taps, since the
            // decoder pays time in proportion to taps x samples. A candidate that fails it by its
            // estimate (1% slack) must not set the cut below, or it would keep out a cheaper filter
            // that passes.
            let (gate, fixed) = stage2_cost(bits_eff, load);
            let need_of = |taps: usize| (gate * ((taps as f64 + fixed) / 256.0) * res.len() as f64) as u64;
            let plain_est = rice::estimate_bits(&res);
            cands.retain(|c| c.2 + need_of(c.0.taps) < plain_est + plain_est / 100);
            let cut = cands.iter().map(|c| c.2).min().map_or(0, |m| m + m * stage2_slack() / 1000);
            for (sp, coded, est) in cands {
                if est > cut { continue; }
                let bits = rice::cost_bits(&coded) + stage2::Params::HEADER_BITS - 1;
                let need = need_of(sp.taps);
                if bits + need < plain && best.as_ref().is_none_or(|b| bits < b.2) { best = Some((sp, coded, bits)); }
            }
            // Long-term prediction runs on what stage 2 leaves, so the two are decided together:
            // stage 2 beating the plain residual says nothing about beating plain residual + LTP
            // (on a periodic signal, LTP alone can be far cheaper than stage 2 then LTP; 
            // research follow-up). Price both whole pipelines, keep the cheaper.
            drop(g8);
            let _g9 = crate::prof::span(crate::prof::Phase::Ltp);
            if let (Some(b), true) = (&best, lt_cfg.is_some()) {
                let s2_cost = b.2 + 1 - stage2::Params::HEADER_BITS;
                let skip_plain = s2_cost + (plain as f64 * stage2_ltp_skip()) as u64 <= plain;
                let (lt_plain, lt_s2) = (if skip_plain { None } else { price(&res, plain) }, price(&b.1, s2_cost));
                let total_plain = 1 + lt_plain.as_ref().map_or(plain + 1, |l| l.2);
                let total_s2 = stage2::Params::HEADER_BITS + lt_s2.as_ref().map_or(s2_cost + 1, |l| l.2);
                if !skip_plain && total_plain <= total_s2 { best = None; lt_res = lt_plain; } else { lt_res = lt_s2; }
                lt_done = true;
            }
        }
        // Carried long filter (H140): competes with the best per-subframe filter and plain; its state
        // advances on every in-range subframe either way (forward == advance), as the decoder does.
        let mut carried_s = None;
        if let Some(c) = carry.as_deref_mut() {
            if !res.is_empty() && res.iter().all(|e| e.abs() <= stage2::MAX_RESIDUAL) {
                let s = stage2::Params::for_block(&res, c.taps(), c.k(), stage2::Carried::TARGET).s;
                let mut t = c.clone();
                let mut out = res.clone();
                t.forward(&mut out, s);
                let cb = rice::cost_bits(&out);
                // carried: flag + 6-bit shift; otherwise flag + stage-2 enable bit (+12 header bits).
                if cb + 5 < best.as_ref().map_or(plain, |b| b.2.min(plain)) {
                    carried_s = Some(s);
                    best = None; pick = None; lt_res = None; lt_done = false;
                    res = out; plain = cb;
                }
                *c = t;
            }
        }
        if carry.is_some() { w.write_bits(carried_s.is_some() as u64, 1); }
        match carried_s {
            Some(s) => stage2::Carried::write_s(s, w),
            None => stage2::Params::write(best.as_ref().map(|b| &b.0), w),
        }
        // `pick` describes `res` as it is now; stage 2 replacing it makes it stale.
        let pick = if best.is_some() { None } else { pick };
        let (res, plain) = match best { Some(b) => (b.1, b.2 + 1 - stage2::Params::HEADER_BITS), None => (res, plain) };
        // Long-term prediction on whatever stage 2 left, kept only if it pays for its header.
        let g9 = crate::prof::span(crate::prof::Phase::Ltp);
        let lt = if lt_done { lt_res } else { price(&res, if plain > 0 { plain } else { rice::cost_bits(&res) }) };
        drop(g9);
        if lt_cfg.is_some() { outcome.set(Some(lt.is_some())); }
        ltp::Params::write(lt.as_ref().map(|b| &b.0), w);
        let _g10 = crate::prof::span(crate::prof::Phase::RiceCoding);
        match (&lt, pick) {
            (None, Some(p)) => rice::encode_sized(w, &res, &p),
            (l, _) => rice::encode(w, l.as_ref().map_or(&res, |b| &b.1)),
        }
    };
    match plan.kind {
        SubframeType::Constant => w.write_signed(shifted[0], eff_bits),
        SubframeType::Verbatim => { for &s in shifted { w.write_signed(s, eff_bits); } }
        SubframeType::Fixed => {
            w.write_bits(plan.order as u64, 3);
            let o = plan.order as usize;
            let (src, verbatim) = warmup_source(hist, shifted, ext, o);
            if verbatim { for &s in &shifted[..o] { w.write_signed(s, eff_bits); } }
            let g7 = crate::prof::span(crate::prof::Phase::FinalResidual);
            let rs = predictors::residuals(plan.order, src);
            drop(g7);
            write_res(w, rs, if verbatim { o } else { 0 });
        }
        SubframeType::Lpc => {
            let q = plan.lpc.as_ref().expect("Lpc plan always carries coefficients");
            let order = q.coeffs.len();
            w.write_bits((order - 1) as u64, 5); // order 1..=32 stored as order-1 in 5 bits
            w.write_bits(q.shift as u64, 5);
            w.write_bits((q.precision - 1) as u64, 4); // precision 1..=16 stored as precision-1
            lpc::write_coeffs(w, &q.coeffs);
            let (src, verbatim) = warmup_source(hist, shifted, ext, order);
            if verbatim { for &s in &shifted[..order] { w.write_signed(s, eff_bits); } }
            let g7 = crate::prof::span(crate::prof::Phase::FinalResidual);
            let rs = lpc::residuals(q, src);
            drop(g7);
            write_res(w, rs, if verbatim { order } else { 0 });
        }
        SubframeType::Palette => {
            let (pal, idx) = plan.palette.as_ref().expect("Palette plan always carries a table");
            w.write_bits((pal.len() - 2) as u64, 4); // count 2..=16 stored as count-2 in 4 bits
            for &v in pal { w.write_signed(v, eff_bits); }
            let iw = palette::index_width(pal.len());
            for &i in idx { w.write_bits(i as u64, iw); }
        }
        SubframeType::Cross => unreachable!("plans carry the inner kind"),
        SubframeType::PaletteRle => {
            let (pal, idx) = plan.palette.as_ref().expect("PaletteRle plan always carries a table");
            w.write_bits((pal.len() - 2) as u64, 4);
            for &v in pal { w.write_signed(v, eff_bits); }
            let runs = palette::runs_from_indices(idx);
            let p = palette::rle_params(&runs);
            w.write_bits(p.run_len_bits as u64, 5);
            w.write_bits((p.num_runs - 1) as u64, 20);
            let iw = palette::index_width(pal.len());
            for &(idx_val, len) in &runs {
                w.write_bits(idx_val as u64, iw);
                w.write_bits((len - 1) as u64, p.run_len_bits);
            }
        }
    }
    if had_lt { if let Some(used) = outcome.get() { gate.record(used, cap); } }
}

/// Block-independent mode with the default chunking and every available thread.
/// `channels[c]` is the full sample sequence for channel `c` (as read from PCM, sign-extended to
/// i64). All channels must have the same length. `bits_per_sample` must be 8, 16, 24, or 32.
///
/// FEC is opt-in, not on by default: measured real cost against FLAC `-8` on real corpus
/// content showed default-on FEC (group 16) turned FAK's established compression parity into a
/// consistent ~6-7% *deficit*, a real regression of this project's primary objective -- reverted to
/// opt-in (`encode_chunked`'s `fec_group: Some(...)`) once that was measured, not left as a silent
/// default cost. `fak encode --fec`/`--fec-group N` opts in explicitly.
pub fn encode(channels: &[Vec<i64>], sample_rate: u32, bits_per_sample: u8) -> Result<Vec<u8>, FormatError> {
    encode_chunked(channels, sample_rate, bits_per_sample, MODE_BLOCK_INDEPENDENT, default_chunk_frames(sample_rate), parallel::default_threads(), None, &Metadata::default())
}

/// Splits the stream into chunks of `chunk_frames` sample-frames (the last may be shorter), encodes
/// them on up to `threads` workers, and writes header + metadata block + self-delimited chunks
/// (each with its own inline sync+frames+bytes+crc header immediately before its payload).
/// Output is identical for every `threads` value: chunks are encoded independently and emitted in
/// order. `metadata` (`src/metadata.rs`) may be empty -- an empty block is still
/// written, so the format never needs a separate presence flag for it.
///
/// `fec_group`: `Some(group_size)` writes one XOR-parity block after every `group_size`
/// data chunks, letting `decoder::decode_full`/`seek` reconstruct any *one* damaged chunk per group
/// from the others plus the parity block, instead of only detecting and rejecting the corruption.
/// `None` disables FEC entirely (the exact v7 behavior). This is the file-based path only --
/// [`StreamEncoder`] never writes parity blocks (see the `VERSION` doc comment in `format.rs`).
pub fn encode_chunked(
    channels: &[Vec<i64>], sample_rate: u32, bits_per_sample: u8, mode: u8, chunk_frames: usize, threads: usize,
    fec_group: Option<usize>, metadata: &Metadata,
) -> Result<Vec<u8>, FormatError> {
    encode_chunked_effort(channels, sample_rate, bits_per_sample, mode, chunk_frames, threads, fec_group, metadata, Effort::default())
}

/// [`encode_chunked`] with an explicit block-mode [`Effort`].
#[allow(clippy::too_many_arguments)]
pub fn encode_chunked_effort(
    channels: &[Vec<i64>], sample_rate: u32, bits_per_sample: u8, mode: u8, chunk_frames: usize, threads: usize,
    fec_group: Option<usize>, metadata: &Metadata, effort: Effort,
) -> Result<Vec<u8>, FormatError> {
    let mut out = Vec::new();
    encode_chunked_to(&mut out, channels, sample_rate, bits_per_sample, mode, chunk_frames, threads, fec_group, metadata, effort)?;
    Ok(out)
}

/// [`encode_chunked_effort`] written straight to `w`: chunks are encoded in
/// order-preserving parallel batches and written as each batch finishes, so memory is the input
/// samples plus one batch of payloads instead of every payload plus a second, whole-file copy of
/// them. A batch always holds whole FEC groups, so parity blocks see their complete group. Output
/// bytes are identical to [`encode_chunked_effort`] for any `threads`.
#[allow(clippy::too_many_arguments)]
pub fn encode_chunked_to<W: std::io::Write>(
    w: &mut W, channels: &[Vec<i64>], sample_rate: u32, bits_per_sample: u8, mode: u8, chunk_frames: usize, threads: usize,
    fec_group: Option<usize>, metadata: &Metadata, effort: Effort,
) -> Result<(), FormatError> {
    encode_chunked_to_progress(w, channels, sample_rate, bits_per_sample, mode, chunk_frames, threads, fec_group, metadata, effort, &mut |_, _| true)
}

/// [`encode_chunked_to`] that reports progress: `progress(frames_done, frames_total)` is called after
/// each batch of chunks is written, and returning `false` cancels the encode (an error "cancelled";
/// what was already written to `w` is then a truncated stream the caller must discard). Output
/// bytes are identical to [`encode_chunked_to`].
#[allow(clippy::too_many_arguments)]
pub fn encode_chunked_to_progress<W: std::io::Write>(
    w: &mut W, channels: &[Vec<i64>], sample_rate: u32, bits_per_sample: u8, mode: u8, chunk_frames: usize, threads: usize,
    fec_group: Option<usize>, metadata: &Metadata, effort: Effort, progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), FormatError> {
    if let Some(g) = fec_group {
        if g == 0 { return Err(FormatError(format!("invalid FEC group size {g}"))); }
    }
    if channels.is_empty() { return Err(FormatError("no channels".into())); }
    let n = channels[0].len();
    if channels.iter().any(|c| c.len() != n) { return Err(FormatError("channel length mismatch".into())); }
    if channels.len() > u8::MAX as usize { return Err(FormatError("too many channels".into())); }
    if mode != MODE_BLOCK_INDEPENDENT { return Err(FormatError(format!("unknown mode {mode}"))); }
    if chunk_frames == 0 || chunk_frames > MAX_CHUNK_FRAMES as usize { return Err(FormatError(format!("invalid chunk length {chunk_frames}"))); }
    let nch = channels.len();
    let ranges: Vec<(usize, usize)> = (0..n).step_by(chunk_frames).map(|s| (s, (s + chunk_frames).min(n))).collect();
    // Workers run continuously, a window of chunks ahead of the writer (no barrier between batches),
    // while the whole-stream PCM hash is computed on another thread. The header carries it and
    // goes first, so nothing is written until it is done; the workers are already encoding by then.
    let window = threads.max(1) * 4;
    let mut writer = OrderedWriter::new(fec_group, Some(ranges.len()));
    crate::simd::choose_kernels();
    std::thread::scope(|scope| -> Result<(), FormatError> {
        let hash = scope.spawn(|| sha256::pcm_digest(channels, bits_per_sample));
        let mut hash = Some(hash);
        let mut header_written = false;
        parallel::ordered_pipeline(ranges.len(), threads, window,
            |i| encode_block_chunk(channels, ranges[i].0, ranges[i].1, bits_per_sample, effort, sample_rate as u64 * nch as u64),
            |i, payload| -> Result<(), FormatError> {
                if !header_written {
                    let pcm_hash = hash.take().expect("hashed once").join().map_err(|_| FormatError("hash thread failed".into()))?;
                    let header = StreamHeader { channels: nch as u8, bits_per_sample, mode, sample_rate, total_frames: n as u64, pcm_hash };
                    w.write_all(&header.to_bytes()).map_err(io_err)?;
                    w.write_all(&metadata::write_block(metadata)).map_err(io_err)?;
                    header_written = true;
                }
                if !progress(ranges[i].0 as u64, n as u64) { return Err(FormatError("cancelled".into())); }
                writer.push(w, (ranges[i].1 - ranges[i].0) as u32, payload)?;
                if !progress(ranges[i].1 as u64, n as u64) { return Err(FormatError("cancelled".into())); }
                Ok(())
            })?;
        if !header_written {
            // No chunks (an empty stream): the header still goes out.
            let pcm_hash = hash.take().expect("hashed once").join().map_err(|_| FormatError("hash thread failed".into()))?;
            let header = StreamHeader { channels: nch as u8, bits_per_sample, mode, sample_rate, total_frames: n as u64, pcm_hash };
            w.write_all(&header.to_bytes()).map_err(io_err)?;
            w.write_all(&metadata::write_block(metadata)).map_err(io_err)?;
        }
        writer.finish(w)
    })
}

/// Writes finished chunks in order; with FEC, a Reed-Solomon parity block after every `group` of them
/// and after the last one. Chunks are written as they arrive and folded into the running
/// shards, so memory is `m` shards of one chunk's length, however large the group.
struct OrderedWriter { group: usize, m: usize, cur: Option<ParityBuilder> }

impl OrderedWriter {
    /// `fec_group`: `None` for no FEC, else the requested group (`FEC_AUTO` = whole file);
    /// `expected_chunks` sizes the shard count (about 1% of the group) when the total is known.
    fn new(fec_group: Option<usize>, expected_chunks: Option<usize>) -> Self {
        match fec_group {
            None => OrderedWriter { group: 0, m: 0, cur: None },
            Some(g) => {
                let group = crate::format::effective_group(g);
                let est = expected_chunks.map_or(group.min(400), |e| e.clamp(1, group));
                OrderedWriter { group, m: crate::format::auto_parity(est), cur: None }
            }
        }
    }

    fn push<W: std::io::Write>(&mut self, w: &mut W, frames: u32, payload: Vec<u8>) -> Result<(), FormatError> {
        let bytes = u32::try_from(payload.len()).map_err(|_| FormatError("chunk payload exceeds 4 GiB".into()))?;
        let entry = ChunkEntry { frames, bytes, crc: crc32(&payload) };
        w.write_all(&write_chunk_header(entry.frames, entry.bytes, entry.crc)).map_err(io_err)?;
        w.write_all(&payload).map_err(io_err)?;
        if self.group == 0 { return Ok(()); }
        let m = self.m;
        let cur = self.cur.get_or_insert_with(|| ParityBuilder::new(m));
        cur.push(entry, &payload);
        if cur.count() == self.group { self.finish(w)?; }
        Ok(())
    }

    /// Writes the parity block for what has been pushed since the last one (a partial group at the end).
    fn finish<W: std::io::Write>(&mut self, w: &mut W) -> Result<(), FormatError> {
        if let Some(b) = self.cur.take() { w.write_all(&b.finish()?).map_err(io_err)?; }
        Ok(())
    }
}

fn io_err(e: std::io::Error) -> FormatError { FormatError(format!("I/O error: {e}")) }

/// Encoder for a seekable sink that never holds the whole audio: the caller pushes chunks in order
/// ([`Self::push_chunk`], all but the last exactly the same length) and memory stays at a window of
/// chunks (`threads * 2`, rounded up to whole FEC groups) plus their payloads, instead of every
/// sample as 8-byte integers (a 275 MB WAV needed 733 MB). The stream header sits ahead of the
/// chunks but holds the whole-stream PCM hash and length, so it is written as a placeholder and
/// rewritten in place by [`Self::finish`]. Output bytes are identical to [`encode_chunked_to`] for
/// the same audio and chunk length (test `file_encoder_matches_whole_file_encode`).
///
/// With more than one thread the chunks go to a persistent pool of workers as they arrive: the
/// calling thread keeps reading, hashing and writing (finished chunks, in order) while the workers
/// encode, so there is no barrier between batches and the serial work overlaps the parallel work.
pub struct FileEncoder<W: std::io::Write + std::io::Seek> {
    w: W, channels: usize, sample_rate: u32, bits_per_sample: u8, mode: u8, effort: Effort,
    threads: usize, window: usize,
    hasher: Option<sha256::PcmHasher>, frames: u64, chunk_len: usize, ended: bool,
    pool: Option<EncodePool>,
    /// Index the next pushed chunk gets, and the next one to be written.
    next_index: usize, next_write: usize,
    /// Frames of the chunks from `next_write` on, in order.
    frames_in_flight: std::collections::VecDeque<u32>,
    /// Encoded chunks that finished before their turn.
    ready: std::collections::HashMap<usize, Vec<u8>>,
    writer: OrderedWriter,
    fec_group: Option<usize>,
}

/// The worker threads of a [`FileEncoder`]: jobs in, `(index, payload)` out, plus a thread that
/// hashes the PCM in order (a SHA-256 stream cannot be split across threads, but it need not sit on
/// the calling thread's critical path either).
struct EncodePool {
    jobs: Option<std::sync::mpsc::SyncSender<(usize, std::sync::Arc<Vec<Vec<i64>>>)>>,
    results: std::sync::mpsc::Receiver<(usize, Option<Vec<u8>>)>,
    workers: Vec<std::thread::JoinHandle<()>>,
    hash_tx: Option<std::sync::mpsc::SyncSender<std::sync::Arc<Vec<Vec<i64>>>>>,
    hasher: Option<std::thread::JoinHandle<sha256::PcmHasher>>,
}

impl EncodePool {
    fn new(threads: usize, window: usize, bits: u8, effort: Effort, load: u64, hasher: sha256::PcmHasher) -> Self {
        use std::sync::{mpsc, Arc, Mutex};
        crate::simd::choose_kernels();
        let (jobs, job_rx) = mpsc::sync_channel::<(usize, Arc<Vec<Vec<i64>>>)>(window);
        let (hash_tx, hash_rx) = mpsc::sync_channel::<Arc<Vec<Vec<i64>>>>(window);
        let hash_thread = std::thread::spawn(move || {
            let mut h = hasher;
            while let Ok(chunk) = hash_rx.recv() { h.update(&chunk); }
            h
        });
        let (res_tx, results) = mpsc::channel();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = (0..threads).map(|_| {
            let (rx, tx) = (job_rx.clone(), res_tx.clone());
            std::thread::spawn(move || loop {
                let job = rx.lock().map(|g| g.recv());
                let Ok(Ok((index, chunk))) = job else { break };
                // A panic in the encoder becomes an error on the calling thread (`None`), not a lost chunk.
                let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode_block_chunk(&chunk, 0, chunk[0].len(), bits, effort, load))).ok();
                drop(chunk);
                if tx.send((index, payload)).is_err() { break; }
            })
        }).collect();
        EncodePool { jobs: Some(jobs), results, workers, hash_tx: Some(hash_tx), hasher: Some(hash_thread) }
    }

    /// Stops accepting jobs and waits for the workers and the hasher; returns the hasher, if it ran.
    fn shut_down(&mut self) -> Option<sha256::PcmHasher> {
        self.jobs = None;
        self.hash_tx = None;
        for w in self.workers.drain(..) { let _ = w.join(); }
        self.hasher.take().and_then(|h| h.join().ok())
    }
}

/// A dropped encoder (an error, a cancelled conversion) still stops and joins its workers.
impl Drop for EncodePool {
    fn drop(&mut self) { let _ = self.shut_down(); }
}

impl<W: std::io::Write + std::io::Seek> FileEncoder<W> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mut w: W, channels: usize, sample_rate: u32, bits_per_sample: u8, mode: u8, threads: usize,
        fec_group: Option<usize>, metadata: &Metadata, effort: Effort,
    ) -> Result<Self, FormatError> {
        if let Some(g) = fec_group {
            if g == 0 { return Err(FormatError(format!("invalid FEC group size {g}"))); }
        }
        if channels == 0 { return Err(FormatError("no channels".into())); }
        if channels > u8::MAX as usize { return Err(FormatError("too many channels".into())); }
        if mode != MODE_BLOCK_INDEPENDENT { return Err(FormatError(format!("unknown mode {mode}"))); }
        let window = threads.max(1) * 2;
        let placeholder = StreamHeader { channels: channels as u8, bits_per_sample, mode, sample_rate, total_frames: 0, pcm_hash: [0; crate::format::PCM_HASH_LEN] };
        w.write_all(&placeholder.to_bytes()).map_err(io_err)?;
        w.write_all(&metadata::write_block(metadata)).map_err(io_err)?;
        Ok(FileEncoder {
            w, channels, sample_rate, bits_per_sample, mode, effort, threads: threads.max(1), window,
            hasher: Some(sha256::PcmHasher::new(bits_per_sample)), frames: 0, chunk_len: 0, ended: false,
            pool: None, next_index: 0, next_write: 0, frames_in_flight: Default::default(),
            ready: Default::default(), writer: OrderedWriter::new(fec_group, None), fec_group,
        })
    }

    /// Tells the encoder how many chunks the file will have, before the first chunk is pushed, so a
    /// whole-file FEC block is sized to about 1% of them (without it, as if there were 400).
    pub fn set_expected_chunks(&mut self, chunks: usize) {
        if self.next_index == 0 { self.writer = OrderedWriter::new(self.fec_group, Some(chunks)); }
    }

    /// Adds the next chunk (`0 < frames <= MAX_CHUNK_FRAMES`, one `Vec` per channel). Every chunk
    /// but the last must have the same length as the first.
    pub fn push_chunk(&mut self, chunk: Vec<Vec<i64>>) -> Result<(), FormatError> {
        if chunk.len() != self.channels { return Err(FormatError(format!("expected {} channels, got {}", self.channels, chunk.len()))); }
        let n = chunk[0].len();
        if chunk.iter().any(|c| c.len() != n) { return Err(FormatError("channel length mismatch".into())); }
        if n == 0 || n > MAX_CHUNK_FRAMES as usize { return Err(FormatError(format!("invalid chunk length {n}"))); }
        if self.ended { return Err(FormatError("chunk after a shorter chunk".into())); }
        if self.chunk_len == 0 { self.chunk_len = n; }
        if n > self.chunk_len { return Err(FormatError("chunk longer than the first".into())); }
        self.ended = n < self.chunk_len;
        self.frames += n as u64;
        let index = self.next_index;
        self.next_index += 1;
        self.frames_in_flight.push_back(n as u32);
        if self.threads <= 1 {
            self.hasher.as_mut().expect("hashing inline without a pool").update(&chunk);
            let payload = encode_block_chunk(&chunk, 0, n, self.bits_per_sample, self.effort, self.sample_rate as u64 * self.channels as u64);
            self.ready.insert(index, payload);
            return self.write_ready();
        }
        let (threads, window, bits, effort) = (self.threads, self.window, self.bits_per_sample, self.effort);
        if self.pool.is_none() {
            let hasher = self.hasher.take().expect("hasher present until the pool takes it");
            self.pool = Some(EncodePool::new(threads, window, bits, effort, self.sample_rate as u64 * self.channels as u64, hasher));
        }
        let pool = self.pool.as_ref().expect("just created");
        let chunk = std::sync::Arc::new(chunk);
        pool.hash_tx.as_ref().expect("pool accepts chunks until finish").send(chunk.clone())
            .map_err(|_| FormatError("encoder hasher failed".into()))?;
        pool.jobs.as_ref().expect("pool accepts jobs until finish").send((index, chunk))
            .map_err(|_| FormatError("encoder worker failed".into()))?;
        // Take what has finished, and wait while too many chunks are in flight.
        self.collect(false)?;
        while self.next_index - self.next_write >= self.window { self.collect(true)?; }
        Ok(())
    }

    /// Receives finished chunks (blocking for one if `block`) and writes every one now in order.
    fn collect(&mut self, block: bool) -> Result<(), FormatError> {
        let Some(pool) = self.pool.as_ref() else { return Ok(()) };
        let failed = || FormatError("encoder worker failed".into());
        if block {
            let (i, p) = pool.results.recv().map_err(|_| failed())?;
            self.ready.insert(i, p.ok_or_else(failed)?);
        }
        while let Ok((i, p)) = pool.results.try_recv() { self.ready.insert(i, p.ok_or_else(failed)?); }
        self.write_ready()
    }

    /// Writes the finished chunks that are next in order (a whole FEC group at a time with FEC).
    fn write_ready(&mut self) -> Result<(), FormatError> {
        while let Some(payload) = self.ready.remove(&self.next_write) {
            let frames = self.frames_in_flight.pop_front().expect("a chunk in flight per index");
            self.next_write += 1;
            self.writer.push(&mut self.w, frames, payload)?;
        }
        Ok(())
    }

    /// Frames pushed so far.
    pub fn frames(&self) -> u64 { self.frames }

    /// Encodes what is pending, writes the real header (length and PCM hash) over the placeholder and
    /// returns the sink positioned at its end.
    pub fn finish(mut self) -> Result<W, FormatError> {
        use std::io::SeekFrom;
        if let Some(pool) = self.pool.as_mut() { pool.jobs = None; }
        while self.next_write < self.next_index { self.collect(true)?; }
        if let Some(mut pool) = self.pool.take() {
            self.hasher = Some(pool.shut_down().ok_or_else(|| FormatError("encoder hasher failed".into()))?);
        }
        self.writer.finish(&mut self.w)?;
        let header = StreamHeader {
            channels: self.channels as u8, bits_per_sample: self.bits_per_sample, mode: self.mode,
            sample_rate: self.sample_rate, total_frames: self.frames, pcm_hash: self.hasher.take().expect("hasher returned by the pool or never moved").finalize(),
        };
        self.w.seek(SeekFrom::Start(0)).map_err(io_err)?;
        self.w.write_all(&header.to_bytes()).map_err(io_err)?;
        self.w.seek(SeekFrom::End(0)).map_err(io_err)?;
        self.w.flush().map_err(io_err)?;
        Ok(self.w)
    }
}

/// Incremental encoder for a genuinely unbounded/live source -- a microphone feed, a network capture, anything where the total length isn't known when
/// encoding starts. Writes the stream header immediately, with `total_frames = TOTAL_FRAMES_UNKNOWN`
/// since the real total can't be known yet, then the metadata block, then flushes each chunk's own
/// self-delimited header + payload to the sink as soon as it's encoded via [`Self::push_chunk`] --
/// never buffering more than one chunk in memory, and never seeking backward on the sink (a real
/// pipe or socket is a valid target; [`encode_chunked`] cannot do this, since it always writes a
/// known, exact `total_frames` computed from the whole input up front).
///
/// No whole-stream `pcm_hash` is written (it would require having seen every sample already, which
/// defeats the point) -- `decoder::verify` refuses a `TOTAL_FRAMES_UNKNOWN` stream outright rather
/// than comparing against a meaningless all-zero hash.
pub struct StreamEncoder<W: std::io::Write> {
    w: W,
    bits_per_sample: u8,
    channels: usize,
    sample_rate: u32,
    effort: Effort,
}

impl<W: std::io::Write> StreamEncoder<W> {
    /// Writes the header and metadata block immediately. `channels` is the channel *count* (not
    /// sample data -- that comes chunk by chunk via `push_chunk`).
    pub fn new(mut w: W, channels: usize, sample_rate: u32, bits_per_sample: u8, mode: u8, metadata: &Metadata) -> Result<Self, FormatError> {
        if channels == 0 { return Err(FormatError("no channels".into())); }
        if channels > u8::MAX as usize { return Err(FormatError("too many channels".into())); }
        if mode != MODE_BLOCK_INDEPENDENT { return Err(FormatError(format!("unknown mode {mode}"))); }
        let header = StreamHeader {
            channels: channels as u8, bits_per_sample, mode, sample_rate,
            total_frames: crate::format::TOTAL_FRAMES_UNKNOWN, pcm_hash: [0u8; crate::format::PCM_HASH_LEN],
        };
        w.write_all(&header.to_bytes()).map_err(io_err)?;
        w.write_all(&metadata::write_block(metadata)).map_err(io_err)?;
        Ok(StreamEncoder { w, bits_per_sample, channels, sample_rate, effort: Effort::default() })
    }

    /// Sets the block-mode [`Effort`] for the chunks pushed after this (default `Normal`).
    pub fn with_effort(mut self, effort: Effort) -> Self { self.effort = effort; self }

    /// Encodes one chunk and flushes it immediately. `chunk_channels[c]` must have the same length
    /// across every channel and there must be exactly as many channels as given to [`Self::new`].
    /// `0 < frames <= MAX_CHUNK_FRAMES`, the same bound the decoder enforces on the other end.
    pub fn push_chunk(&mut self, chunk_channels: &[Vec<i64>]) -> Result<(), FormatError> {
        if chunk_channels.len() != self.channels {
            return Err(FormatError(format!("expected {} channels, got {}", self.channels, chunk_channels.len())));
        }
        let n = chunk_channels[0].len();
        if chunk_channels.iter().any(|c| c.len() != n) { return Err(FormatError("channel length mismatch".into())); }
        if n == 0 || n > MAX_CHUNK_FRAMES as usize { return Err(FormatError(format!("invalid chunk length {n}"))); }
        let payload = encode_block_chunk(chunk_channels, 0, n, self.bits_per_sample, self.effort, self.sample_rate as u64 * self.channels as u64);
        let bytes = u32::try_from(payload.len()).map_err(|_| FormatError("chunk payload exceeds 4 GiB".into()))?;
        self.w.write_all(&write_chunk_header(n as u32, bytes, crc32(&payload))).map_err(io_err)?;
        self.w.write_all(&payload).map_err(io_err)?;
        Ok(())
    }

    /// Flushes the underlying sink and returns it. There is nothing to patch -- `total_frames` was
    /// never known and stays the sentinel; the stream's real length is only ever recoverable by a
    /// reader that counts the chunks it actually decodes (`decoder::decode_stream`).
    pub fn finish(mut self) -> Result<W, FormatError> {
        self.w.flush().map_err(io_err)?;
        Ok(self.w)
    }
}

/// Block-mode encoder effort: how the frame lengths within each chunk are chosen. The
/// decoder doesn't care -- `frame_frames` has always been per frame -- so this is encoder-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    /// Every frame `DEFAULT_BLOCK_SIZE` long, no second-opinion LPC costing. Fastest.
    Fast,
    /// Variable frame length chosen by an analytic cost estimate, then only the chosen frames are
    /// encoded; 3 LPC candidates also costed at the warm-start precision. ~1.3x `Fast`'s encode
    /// time.
    #[default]
    Normal,
    /// Variable frame length chosen by really encoding every candidate frame, and every LPC
    /// candidate also costed at the warm-start precision: ~6x `Fast`'s encode time.
    Max,
    /// `Max`, plus the stage-2 adaptive filter on every subframe where it pays: about 1%
    /// smaller than `Max` but about 2x the decode time, so it is
    /// kept out of the levels that promise fast decode.
    Insane,
}

impl Effort {
    /// `select`'s `second_opinions`: LPC candidates also costed at the warm-start
    /// precision (measured,: 3 keeps encode time where it was before 
    /// speedups for -0.20%; all of them -0.27% for ~10% more).
    /// Whether the cross-channel candidates are searched: not at `Fast`, whose point is
    /// encode speed (the search costs ~30%). `FAK_NO_CROSS=1` turns them off at every effort (for
    /// testing; the output is then a valid stream without `Cross` subframes).
    fn cross(self) -> bool { self != Effort::Fast && !matches!(std::env::var("FAK_NO_CROSS").as_deref(), Ok("1")) }

    /// Stage-2 filter settings (taps, k) tried on each coded subframe; the smallest wins,
    /// or the stage stays off. Only at `Insane`: it roughly doubles decode time.
    /// `FAK_NO_STAGE2=1` turns it off (for testing).
    fn stage2(self) -> &'static [(usize, u32)] {
        if matches!(std::env::var("FAK_NO_STAGE2").as_deref(), Ok("1")) { return &[]; }
        match self {
            // `FAK_STAGE2_SET="taps:k,taps:k,..."` replaces the candidates (for testing);
            // `FAK_STAGE2_SET_<FAST|NORMAL|MAX>` enables a set at those levels.
            Effort::Insane => stage2_set("FAK_STAGE2_SET", &[(256, 6), (256, 5), (128, 5), (32, 4)]),
            Effort::Max => stage2_set("FAK_STAGE2_SET_MAX", &[(128, 5), (32, 4)]),
            Effort::Normal => stage2_set("FAK_STAGE2_SET_NORMAL", &[(32, 4)]),
            Effort::Fast => stage2_set("FAK_STAGE2_SET_FAST", &[(16, 3)]),
        }
    }

    /// Long-term prediction search on each coded subframe. Not at `Fast`, whose point is
    /// encode speed (the search cost it +43% for -0.33%); its files then decode at v16's
    /// speed. `FAK_NO_LTP=1` turns it off at every effort, `FAK_LTP_MIN_GAIN` / `FAK_LTP_EXACT`
    /// override the threshold and exact-pricing count (for testing).
    fn ltp(self) -> Option<ltp::Search> {
        if self == Effort::Fast || matches!(std::env::var("FAK_NO_LTP").as_deref(), Ok("1")) { return None; }
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        Some(ltp::Search {
            candidates: 6,
            tap_counts: &ltp::TAP_COUNTS,
            // Candidates run and priced exactly: 1 at `Normal` (3 is 0.004% smaller for about 8%
            // more encode time, measured at `fast`), 3 above.
            exact: env("FAK_LTP_EXACT").map_or(if self == Effort::Normal { 1 } else { 3 }, |v| v as usize),
            // `Normal` and `Max`, which promise fast decode, use the stage only where it saves >= 0.05 bits per
            // value: 73% of its saving for about a third of its decode cost. `Insane`
            // already trades decode time for size and takes every saving.
            min_gain: env("FAK_LTP_MIN_GAIN").unwrap_or(if self == Effort::Insane { 0.0 } else { LTP_MIN_GAIN }),
        })
    }

    fn second_opinions(self) -> usize {
        if let Some(v) = std::env::var("FAK_SECOND").ok().and_then(|v| v.parse().ok()) { return v; }
        match self { Effort::Fast => 0, Effort::Normal | Effort::Max | Effort::Insane => 3 }
    }
}

/// Least saving (bits per residual value) for which long-term prediction is used at the levels
/// that promise fast decode (0 / 0.02 / 0.05 measured).
const LTP_MIN_GAIN: f64 = 0.05;

/// Stage-2 input scale: residual blocks are scaled so their mean magnitude is about 2^9, where
/// the 16-bit filter adapts best (measured).
fn stage2_set(var: &'static str, default: &'static [(usize, u32)]) -> &'static [(usize, u32)] {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<(&'static str, &'static [(usize, u32)])>>> = OnceLock::new();
    let mut c = CACHE.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some(&(_, v)) = c.iter().find(|(k, _)| *k == var) { return v; }
    let off = std::env::var(var).map_or(false, |v| v == "none");
    let parsed = std::env::var(var).ok().map(|v| v.split(',').filter_map(|c| {
        let (t, k) = c.split_once(':')?;
        let (t, k) = (t.parse().ok()?, k.parse().ok()?);
        (stage2::TAPS.contains(&t) && (1..=15).contains(&k)).then_some((t, k))
    }).collect::<Vec<_>>());
    let v: &'static [(usize, u32)] = match parsed.filter(|p| !p.is_empty()) {
        _ if off => &[],
        Some(p) => Box::leak(p.into_boxed_slice()),
        None => default,
    };
    c.push((var, v));
    v
}
fn stage2_ltp_skip() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_S2_LTP_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.01))
}
const STAGE2_TARGET: u32 = 9;
/// Per mille (`FAK_S2_SLACK` overrides): 5 costs +0.0001% size, 0 costs +0.003% (measured).
const STAGE2_EXACT_SLACK: u64 = 5;
fn stage2_slack() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_S2_SLACK").ok().and_then(|v| v.parse().ok()).unwrap_or(STAGE2_EXACT_SLACK))
}
/// `FAK_STAGE2_TARGET` overrides it (for testing).
/// What a stage-2 filter must save to be used, as bits per sample: `gate * (taps + fixed) / 256`. The
/// decoder pays time in proportion to taps x samples plus a fixed cost per subframe (`fixed`, in taps'
/// worth), so this is the encoder-side knob that trades size for decode speed with no format change.
/// FLAC decodes 16-bit audio much faster than wider audio (its 32-bit predictor path), so the two are
/// calibrated apart, both against FLAC -8 on 18 real files ("Encode-speed pass
/// on x86-64"): with (0.5, 0) `insane` decoded at 0.88x FLAC on the 16-bit files and 1.41x on the
/// 24-bit ones; these values give about 1.06x and 1.47x. `FAK_S2_GATE` / `FAK_S2_FIXED` override both.
const STAGE2_GATE_NARROW: f64 = 0.4;
const STAGE2_FIXED_NARROW: f64 = 56.0;
const STAGE2_GATE_WIDE: f64 = 0.5;
const STAGE2_FIXED_WIDE: f64 = 16.0;
/// At most this gate applies to a stream that is cheap to decode in real time (`load` = samples per
/// second over all channels at or below `LOAD_LOW`): the decoder then runs thousands of times real
/// time, so the FLAC-parity gate above would only throw size away (`FAK_S2_LOW` overrides the gate,
/// 0 turns it off; the fixed term is 0). Between `LOAD_LOW` and `LOAD_HIGH` (more than a 192 kHz
/// stereo stream, e.g. 6 or 8 channels at 96-192 kHz) it rises linearly to the parity values, which
/// keeps the worst case (a 256-tap filter on every sample, ~20 ns per sample) above 50x real time.
const STAGE2_GATE_LOW: f64 = 0.02;
const LOAD_LOW: u64 = 500_000;
const LOAD_HIGH: u64 = 1_200_000;
/// (gate, fixed) for a container of `bits_eff` bits (16, or 17 for a widened side channel, and
/// narrower are the narrow class) in a stream of `load` samples per second.
fn stage2_cost(bits_eff: u32, load: u64) -> (f64, f64) {
    static ENV: std::sync::OnceLock<(Option<f64>, Option<f64>, Option<f64>)> = std::sync::OnceLock::new();
    let (g, f, low) = *ENV.get_or_init(|| {
        let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse().ok());
        (get("FAK_S2_GATE"), get("FAK_S2_FIXED"), get("FAK_S2_LOW"))
    });
    let narrow = bits_eff <= 17;
    let (hg, hf) = if narrow { (STAGE2_GATE_NARROW, STAGE2_FIXED_NARROW) } else { (STAGE2_GATE_WIDE, STAGE2_FIXED_WIDE) };
    let t = (load.saturating_sub(LOAD_LOW) as f64 / (LOAD_HIGH - LOAD_LOW) as f64).min(1.0);
    let lg = low.unwrap_or(STAGE2_GATE_LOW);
    (g.unwrap_or(lg + t * (hg - lg)), f.unwrap_or(t * hf))
}
/// Filters of at least this many taps are the screened ones.
const LARGE_TAPS: usize = 256;
/// Bits per sample the cheaper stage-2 filters must save (by estimate) before the 256-tap ones are
/// tried (`FAK_S2_SCREEN`; 0 = always).
const STAGE2_SCREEN: f64 = 0.5;
fn stage2_screen() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_S2_SCREEN").ok().and_then(|v| v.parse().ok()).unwrap_or(STAGE2_SCREEN))
}
fn stage2_target() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_STAGE2_TARGET").ok().and_then(|v| v.parse().ok()).unwrap_or(STAGE2_TARGET))
}

/// Candidate frame lengths (`Normal`/`Max`), each double the one before: a binary tree per
/// `BLOCK_SIZES.last()` span of the chunk. 4096 (`DEFAULT_BLOCK_SIZE`) is one of the levels.
const BLOCK_SIZES: &[usize] = &[1024, 2048, 4096, 8192, 16384];

/// Per-frame overhead charged by the analytic search (header fields, CRC, padding, plus a margin).
/// Calibrated together with `ANALYTIC_COEFF_BITS` (30/68/150/300 tried).
const ANALYTIC_FRAME_BITS: f64 = 150.0;

/// Block-independent frames covering sample-frames `start..stop` of every channel (one chunk).
///
/// Variable block size: candidate frames of every length in `BLOCK_SIZES`, aligned within
/// the chunk, form a binary tree; a bottom-up pass keeps each node's own frame or its two children,
/// whichever costs less. `Max` costs a node by really encoding it -- exact, since a frame's bytes
/// depend only on its own samples and the original samples just before it (history), never
/// on how the rest of the chunk is cut (apart from `refine_precision`'s warm start, which only
/// steers a search). `Normal` costs it with `lpc::analytic_bits` and encodes only what it picks.
fn encode_frames(channels: &[Vec<i64>], start: usize, stop: usize, bits_per_sample: u8, effort: Effort, load: u64) -> Vec<u8> {
    let nch = channels.len();
    let frames_of = |bs: usize| -> Vec<(usize, usize)> { (start..stop).step_by(bs).map(|s| (s, (s + bs).min(stop))).collect() };
    // The frames to encode, per sequence with its own warm-start state, known before any is
    // encoded.
    let levels: Vec<Vec<(usize, usize)>> = match effort {
        Effort::Fast => vec![frames_of(DEFAULT_BLOCK_SIZE)],
        Effort::Normal => {
            let costs: Vec<Vec<f64>> = BLOCK_SIZES.iter()
                .map(|&bs| frames_of(bs).into_iter().map(|(s, e)| frame_analytic_bits(channels, s, e)).collect())
                .collect();
            vec![cheapest_cover(&costs).into_iter().map(|(level, j)| {
                let s = start + j * BLOCK_SIZES[level];
                (s, (s + BLOCK_SIZES[level]).min(stop))
            }).collect()]
        }
        // Each level keeps its own warm-start state, run left to right.
        Effort::Max | Effort::Insane => BLOCK_SIZES.iter().map(|&bs| frames_of(bs)).collect(),
    };
    let active = active_frames(channels, start, stop, effort, &levels);
    let encoded: Vec<Vec<Vec<u8>>> = levels.iter().enumerate().map(|(li, frames)| {
        let mut prec_state = vec![lpc::PRECISION; nch.max(4)];
        let mut gates = vec![LtpGate::default(); nch.max(2)];
        frames.iter().enumerate().map(|(j, &(s, e))| {
            if !active[li][j] { return Vec::new(); }
            //  long-term prediction's decoder-side residual bound (`ltp::LIMIT = 2^20`) was
            // sized so its own dot product stays exact in 32-bit arithmetic at <=24-bit containers;
            // not re-derived for the wider residuals a 32-bit source can legitimately produce, so
            // the search is skipped there rather than risking an unverified margin.
            let lt = if bits_per_sample > 24 { None } else { effort.ltp() };
            encode_frame(channels, start, s, e, bits_per_sample, &mut prec_state, &mut gates, ltp_cap(effort), effort.second_opinions(), effort.cross(), effort.stage2(), lt, load, None)
        }).collect()
    }).collect();
    let (chosen, base): (Vec<(usize, usize)>, Vec<u8>) = match effort {
        Effort::Max | Effort::Insane => {
            let costs: Vec<Vec<f64>> = encoded.iter().map(|l| l.iter().map(|f| if f.is_empty() { f64::INFINITY } else { f.len() as f64 }).collect()).collect();
            let cover = cheapest_cover(&costs);
            let bytes = cover.iter().flat_map(|&(level, j)| encoded[level][j].iter().copied()).collect();
            (cover.into_iter().map(|(level, j)| levels[level][j]).collect(), bytes)
        }
        _ => (levels.iter().flatten().copied().collect(), encoded.into_iter().flatten().flatten().collect()),
    };
    // H140: re-encode the chosen frames in order with a carried 512-tap filter per subframe slot;
    // kept per chunk only if smaller. Research switch `FAK_CARRY=1` (off: config byte 0).
    // `FAK_CARRY` lists the configs to try (`1`, `2`, `12`); the smallest chunk wins.
    let mut best: Option<Vec<u8>> = None;
    // H145: default `12` for Insane only (H144: -0.2..-0.58% on val; ~3x encode time, so not Max);
    // `FAK_CARRY=0` turns it off, `FAK_CARRY=12` forces it on at other levels.
    let carry_default = if matches!(effort, Effort::Insane) { "12" } else { "" };
    for cfg in std::env::var("FAK_CARRY").unwrap_or_else(|_| carry_default.to_string()).bytes().filter_map(|b| b.checked_sub(b'0')) {
        let Some((taps, k)) = stage2::Carried::config(cfg) else { continue };
        let mut prec_state = vec![lpc::PRECISION; nch.max(4)];
        let mut gates = vec![LtpGate::default(); nch.max(2)];
        let mut slots: Vec<stage2::Carried> = (0..nch.max(2)).map(|_| stage2::Carried::new(taps, k)).collect();
        let lt = if bits_per_sample > 24 { None } else { effort.ltp() };
        let mut alt = vec![cfg];
        for &(s, e) in &chosen {
            alt.extend(encode_frame(channels, start, s, e, bits_per_sample, &mut prec_state, &mut gates, ltp_cap(effort), effort.second_opinions(), effort.cross(), effort.stage2(), lt, load, Some(&mut slots)));
        }
        if alt.len() <= best.as_ref().map_or(base.len(), |b| b.len()) { best = Some(alt); }
    }
    // H178: backward-adaptive stereo OLS chunk (config 3) competes with the frame path; Insane only
    // (~17x realtime decode, H177). `FAK_OLS=0` turns it off.
    if effort == Effort::Insane && nch == 2 && bits_per_sample <= 24 && std::env::var("FAK_OLS").map_or(true, |v| v != "0") {
        // H248: config 4 adds the quantized IRLS weight (-0.70% on ech/gh/ap); `FAK_OLS_IRLS=0` writes config 3.
        let irls = std::env::var("FAK_OLS_IRLS").map_or(true, |v| v != "0");
        let alt = encode_ols_chunk(&channels[0][start..stop], &channels[1][start..stop], bits_per_sample, irls);
        if std::env::var_os("FAK_OLS_TRACE").is_some() {
            eprintln!("olschunk frames={} base={} carry={} ols={}", stop - start, base.len() + 1, best.as_ref().map_or(0, |b| b.len()), alt.len());
        }
        if alt.len() <= best.as_ref().map_or(base.len() + 1, |b| b.len()) { return alt; }
    }
    if let Some(b) = best { return b; }
    let mut out = vec![0u8];
    out.extend(base);
    out
}

/// Samples per OLS-chunk block (one carried-LMS shift / Rice partitioning decision per channel).
pub(crate) const OLS_BLOCK: usize = 4096;

/// Config-3 chunk payload (H178): stereo OLS (fresh state at the chunk start) over the whole chunk, then per
/// block of `OLS_BLOCK` frames and per channel `[carried flag][6-bit shift]` and the Rice-coded residual, with
/// a carried 512n:8 LMS per channel over the OLS residual (advanced on every block, as the decoder does).
pub(crate) fn encode_ols_chunk(s0: &[i64], s1: &[i64], bits_per_sample: u8, irls: bool) -> Vec<u8> {
    let mut st = crate::ols::Stereo::new(crate::ols::Params { irls, ..Default::default() });
    let (taps, k) = stage2::Carried::config(1).unwrap();
    let mut carried: [stage2::Carried; 2] = [stage2::Carried::new_ols(taps, k, bits_per_sample), stage2::Carried::new_ols(taps, k, bits_per_sample)];
    let mut w = BitWriter::new();
    w.write_bits(if irls { 4 } else { 3 }, 8);
    for b in (0..s0.len()).step_by(OLS_BLOCK) {
        let e = (b + OLS_BLOCK).min(s0.len());
        let (r0, r1) = st.forward_block(&s0[b..e], &s1[b..e]);
        for (res, c) in [r0, r1].into_iter().zip(carried.iter_mut()) {
            let mut use_s = None;
            let mut out = res.clone();
            if res.iter().all(|e| e.abs() <= stage2::MAX_RESIDUAL) {
                let s = stage2::Params::for_block(&res, c.taps(), c.k(), stage2::Carried::TARGET).s;
                c.forward(&mut out, s);
                if rice::cost_bits(&out) + 5 < rice::cost_bits(&res) { use_s = Some(s); }
            }
            w.write_bits(use_s.is_some() as u64, 1);
            match use_s { Some(s) => { stage2::Carried::write_s(s, &mut w); rice::encode(&mut w, &out); } None => rice::encode(&mut w, &res) }
        }
    }
    w.finish()
}

/// A node off the analytic cover is encoded only if its analytic cost is within this fraction of the
/// cheapest cover of its span (`FAK_PRUNE_TAU`; `inf` keeps every node in the level window). Measured
/// on 18 real files: 0.05 is +0.003..0.010% for -5% (`Max`) and
/// -12% (`Insane`) encode time, 0.02 +0.04% for -26%.
const PRUNE_TAU: f64 = 0.05;
fn prune_tau() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("FAK_PRUNE_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(PRUNE_TAU))
}

/// Which frames of `levels` the exhaustive tree search (`Max`, `Insane`) encodes. Encoding every
/// node at five block sizes is most of `Max`'s time, but the cheap analytic cover (`Normal`'s
/// search) already lands within one or two levels of the winner nearly everywhere, so only nodes
/// whose level is within `down` levels below or `up` above an analytically chosen frame
/// overlapping them are really encoded (the analytic search errs toward short frames, so more
/// room above; measured). The analytic cover itself is always among them, so
/// a cover exists. `FAK_PRUNE="down,up"` overrides (`9` encodes everything).
fn active_frames(channels: &[Vec<i64>], start: usize, stop: usize, effort: Effort, levels: &[Vec<(usize, usize)>]) -> Vec<Vec<bool>> {
    let all = || levels.iter().map(|l| vec![true; l.len()]).collect();
    if !matches!(effort, Effort::Max | Effort::Insane) { return all(); }
    let pr = std::env::var("FAK_PRUNE").unwrap_or_default();
    let mut it = pr.split(',').filter_map(|v| v.parse::<usize>().ok());
    let (down, up) = match (it.next(), it.next()) {
        (Some(d), Some(u)) => (d, u),
        (Some(d), None) => (d, d),
        // `Max` was (1, 2): (1, 1) is +0.005% on two sets for 9-10% less time.
        _ => if effort == Effort::Insane { (0, 2) } else { (0, 1) },
    };
    if down >= BLOCK_SIZES.len() && up >= BLOCK_SIZES.len() { return all(); }
    let costs: Vec<Vec<f64>> = BLOCK_SIZES.iter()
        .map(|&bs| (start..stop).step_by(bs).map(|s| frame_analytic_bits(channels, s, (s + bs).min(stop))).collect())
        .collect();
    let best = cover_costs(&costs);
    let tau = prune_tau();
    let mut active: Vec<Vec<bool>> = levels.iter().map(|l| vec![false; l.len()]).collect();
    for (lc, jc) in cheapest_cover(&costs) {
        let (a, b) = (jc * BLOCK_SIZES[lc], (jc + 1) * BLOCK_SIZES[lc]);
        for l in lc.saturating_sub(down)..=(lc + up).min(BLOCK_SIZES.len() - 1) {
            let bs = BLOCK_SIZES[l];
            for j in a / bs..b.div_ceil(bs).min(active[l].len()) {
                if l == lc || costs[l][j] <= best[l][j] * (1.0 + tau) { active[l][j] = true; }
            }
        }
    }
    active
}

/// A mode-0 chunk payload: the value-map section (`valuemap`), then the frames, coded
/// in the mapped (`k`) domain for every channel that has a map. A fitted map is not always a
/// cheaper one (a sparse 8-bit chunk can fit a gain barely above 1), so unless every map found
/// clearly pays (`valuemap::ChannelMap::clearly_pays`), the chunk is also encoded without maps and
/// the smaller payload kept.
fn encode_block_chunk(channels: &[Vec<i64>], start: usize, stop: usize, bits_per_sample: u8, effort: Effort, load: u64) -> Vec<u8> {
    let _g20 = crate::prof::span(crate::prof::Phase::Chunk);
    let g22 = crate::prof::span(crate::prof::Phase::ValueMap);
    let found: Vec<Option<(valuemap::ChannelMap, Vec<i64>)>> =
        channels.iter().map(|c| valuemap::detect(&c[start..stop], bits_per_sample as u32)).collect();
    drop(g22);
    let plain = || {
        let mut p = valuemap::write_section(&vec![None; channels.len()]);
        p.extend(encode_frames(channels, start, stop, bits_per_sample, effort, load));
        p
    };
    if found.iter().all(Option::is_none) { return plain(); }
    let sure = found.iter().flatten().all(|(m, _)| m.clearly_pays(stop - start));
    let maps: Vec<Option<valuemap::ChannelMap>> = found.iter().map(|f| f.as_ref().map(|(m, _)| m.clone())).collect();
    let mapped: Vec<Vec<i64>> = found.into_iter().zip(channels)
        .map(|(f, c)| f.map_or_else(|| c[start..stop].to_vec(), |(_, k)| k))
        .collect();
    let mut out = valuemap::write_section(&maps);
    out.extend(encode_frames(&mapped, 0, stop - start, bits_per_sample, effort, load));
    if sure { return out; }
    let p = plain();
    if out.len() < p.len() { out } else { p }
}

/// `best[level][j]`: the cheapest cover cost of node (level, j), filled bottom-up.
fn cover_costs(costs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut best: Vec<Vec<f64>> = Vec::with_capacity(costs.len());
    for (level, row) in costs.iter().enumerate() {
        let b: Vec<f64> = row.iter().enumerate().map(|(j, &own)| {
            if level == 0 { return own; }
            let below = &best[level - 1];
            let kids: f64 = below[2 * j..(2 * j + 2).min(below.len())].iter().sum();
            own.min(kids)
        }).collect();
        best.push(b);
    }
    best
}

/// `costs[level][j]` is the cost of the frame covering `j*BLOCK_SIZES[level]..` of the chunk (the
/// last one per level may be shorter). Returns, left to right, the (level, j) frames of the cheapest
/// cover, where each node is either its own frame or the cheapest covers of its (up to) two children.
fn cheapest_cover(costs: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let best = cover_costs(costs);
    fn collect(costs: &[Vec<f64>], best: &[Vec<f64>], level: usize, j: usize, out: &mut Vec<(usize, usize)>) {
        if level > 0 && best[level][j] < costs[level][j] {
            for k in 2 * j..(2 * j + 2).min(costs[level - 1].len()) { collect(costs, best, level - 1, k, out); }
        } else {
            out.push((level, j));
        }
    }
    let top = costs.len() - 1;
    let mut out = Vec::new();
    for j in 0..costs[top].len() { collect(costs, &best, top, j, &mut out); }
    out
}

/// `lpc::analytic_bits` for one frame: the cheapest stereo mode for a pair, else the channels'
/// sum, plus `ANALYTIC_FRAME_BITS` (and the stereo-mode byte for a pair).
fn frame_analytic_bits(channels: &[Vec<i64>], s: usize, e: usize) -> f64 {
    let _g = crate::prof::span(crate::prof::Phase::AnalyticCover);
    let est = |x: &[i64]| lpc::analytic_bits(x, LPC_ORDER_CANDIDATES);
    if channels.len() == 2 {
        let (l, r) = (&channels[0][s..e], &channels[1][s..e]);
        // Mid and side only: the cover decides where frames start and end, which the two signals
        // show as well as all four pairings do (+0.003% on real music against the four-way minimum,
        // half the autocorrelations: 10% of `normal` encode time; L+R alone was +0.02%, M alone +0.04%).
        let (ms, ss) = (stereo::mid(l, r), stereo::side(l, r));
        let (bm, bs) = (est(&ms), est(&ss));
        ANALYTIC_FRAME_BITS + 8.0 + bm + bs
    } else {
        ANALYTIC_FRAME_BITS + channels.iter().map(|c| est(&c[s..e])).sum::<f64>()
    }
}

/// One complete frame (byte-aligned payload) for sample-frames `start..end`, history from
/// `chunk_start..start`.
#[allow(clippy::too_many_arguments)]
fn encode_frame(channels: &[Vec<i64>], chunk_start: usize, start: usize, end: usize, bits_per_sample: u8, prec_state: &mut [u32], gates: &mut [LtpGate], cap: u32, second_opinions: usize, cross: bool, stage2: &[(usize, u32)], lt: Option<ltp::Search>, load: u64, mut carry: Option<&mut [stage2::Carried]>) -> Vec<u8> {
    let nch = channels.len();
    let frame_frames = (end - start) as u32;
    debug_assert!(frame_frames > 0 && frame_frames <= MAX_FRAME_FRAMES);

    let mut fw = BitWriter::new();
    crate::format::write_frame_len(&mut fw, frame_frames);
    let analyses = analyze_frame(channels, chunk_start, start, end, bits_per_sample);
    let analyses = if nch == 2 { prune_stereo(analyses, if cross { -1.0 } else { 0.0 }) } else { analyses };
    let plans = plan_subframes(analyses, prec_state, second_opinions);

    if nch == 2 {
        let base = bits_per_sample as u32;
        // Each of L/R/M/S is planned exactly once; every stereo mode reuses one of these four
        // (S is shared by MidSide/LeftSide/SideRight) instead of re-searching per mode.
        let [mut pl, mut pr, mut pm, mut ps]: [Planned; 4] = plans.try_into().unwrap_or_else(|_| unreachable!("four subframes"));
        for (p, st) in [&mut pl, &mut pr, &mut pm, &mut ps].into_iter().zip(prec_state.iter_mut()) { refine_precision(p, st); }
        // The second subframe of each mode may reference the first (cross-channel).
        let screen = cross_screen();
        let pc = |a: &Planned, b: &Planned| -> Option<Planned> {
            if a.plan.bits >= SKIPPED_BITS || b.plan.bits >= SKIPPED_BITS { return None; }
            if screen > 0.0 && residual_affinity(&residual_signal(a), &residual_signal(b)) < screen { return None; }
            plan_cross(a, b, 0)
        };
        // Every mode gets its cross-channel search: skipping the ones whose plain cost is already
        // above the best mode's cost +0.15% at normal for -15% time (margin 1%), +0.026% for -2% (4%).
        let (xr_l, xs_m, xs_l, xr_s) = if cross {
            (pc(&pl, &pr), pc(&pm, &ps), pc(&pl, &ps), pc(&ps, &pr))
        } else { (None, None, None, None) };
        let second = |x: &Option<Planned>, own: &Planned| -> u64 { x.as_ref().map_or(own.plan.bits, |v| v.plan.bits) };
        let candidates = [
            (StereoMode::LeftRight, pl.plan.bits + second(&xr_l, &pr)),
            (StereoMode::MidSide, pm.plan.bits + second(&xs_m, &ps)),
            (StereoMode::LeftSide, pl.plan.bits + second(&xs_l, &ps)),
            (StereoMode::SideRight, ps.plan.bits + second(&xr_s, &pr)),
        ];
        let best_mode = candidates.iter().min_by_key(|(_, cost)| *cost).unwrap().0;
        fw.write_bits(best_mode as u64, 2);
        let (a, b, x, bits_a, bits_b) = match best_mode {
            StereoMode::LeftRight => (&pl, &pr, &xr_l, base, base),
            StereoMode::MidSide => (&pm, &ps, &xs_m, base, base + 1),
            StereoMode::LeftSide => (&pl, &ps, &xs_l, base, base + 1),
            StereoMode::SideRight => (&ps, &pr, &xr_s, base + 1, base),
        };
        write_subframe(&mut fw, a, bits_a, stage2, lt, load, &mut gates[0], cap, carry.as_deref_mut().map(|c| &mut c[0]));
        write_subframe(&mut fw, x.as_ref().unwrap_or(b), bits_b, stage2, lt, load, &mut gates[1], cap, carry.as_deref_mut().map(|c| &mut c[1]));
    } else {
        // Each channel after the first may reference the earlier channel whose coded residual
        // correlates best with its own (cross-channel).
        let mut done: Vec<(Planned, Vec<i64>)> = Vec::with_capacity(nch);
        for (ci, mut p) in plans.into_iter().enumerate() {
            refine_precision(&mut p, &mut prec_state[ci]);
            if cross && !done.is_empty() {
                let own = residual_signal(&p);
                let best_ref = done.iter().enumerate()
                    .map(|(i, (_, r))| (residual_affinity(r, &own), i))
                    .fold(None, |acc: Option<(f64, usize)>, x| match acc { Some(a) if a.0 >= x.0 => Some(a), _ => Some(x) });
                if let Some((_, i)) = best_ref {
                    if let Some(v) = plan_cross(&done[i].0, &p, i as u8) { p = v; }
                }
            }
            write_subframe(&mut fw, &p, bits_per_sample as u32, stage2, lt, load, &mut gates[ci], cap, carry.as_deref_mut().map(|c| &mut c[ci]));
            if cross && ci + 1 < nch { let r = residual_signal(&p); done.push((p, r)); }
        }
    }

    fw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder;

    fn roundtrip(channels: Vec<Vec<i64>>, sample_rate: u32, bits: u8) {
        let encoded = encode(&channels, sample_rate, bits).unwrap();
        let (dec_header, dec_channels) = decoder::decode(&encoded).unwrap();
        assert_eq!(dec_header.channels as usize, channels.len());
        assert_eq!(dec_header.sample_rate, sample_rate);
        assert_eq!(dec_header.bits_per_sample, bits);
        assert_eq!(dec_channels, channels);
    }

    /// Encodes with FEC explicitly disabled -- for tests that assert a specific compressed
    /// size/ratio, where FEC's real, deliberate size overhead (~1/group_size, much higher still for
    /// a short file whose single chunk forms a group of one) would conflate the codec's own
    /// compression behavior with an unrelated feature. FEC's own overhead/recovery properties are
    /// tested separately.
    fn encode_no_fec(channels: &[Vec<i64>], sample_rate: u32, bits: u8) -> Vec<u8> {
        encode_chunked(channels, sample_rate, bits, MODE_BLOCK_INDEPENDENT, default_chunk_frames(sample_rate), 1, None, &Metadata::default()).unwrap()
    }

    #[test]
    fn mono_roundtrip() {
        roundtrip(vec![(0..10_000i64).map(|i| ((i * 977) % 30001) - 15000).collect()], 44100, 16);
    }

    #[test]
    fn stereo_roundtrip_various_content() {
        let n = 9000;
        let l: Vec<i64> = (0..n).map(|i: i64| ((i * 37) % 2001) - 1000).collect();
        let r: Vec<i64> = (0..n).map(|i: i64| ((i * 91) % 1500) - 700).collect();
        roundtrip(vec![l, r], 44100, 16);
    }

    ///  32-bit int block-mode round trip, including full-range extremes (min/max i32) and an
    /// LPC-friendly tone that exercises real prediction, not just verbatim/constant subframes.
    #[test]
    fn int32_block_mode_roundtrip() {
        let n = 9000i64;
        let (lo, hi) = (i32::MIN as i64, i32::MAX as i64);
        let tone: Vec<i64> = (0..n).map(|i| (1_000_000_000.0 * (i as f64 * 0.01).sin()) as i64).collect();
        roundtrip(vec![tone.clone(), tone.iter().map(|&x| -x).collect()], 44100, 32);
        let extremes: Vec<i64> = (0..n).map(|i| if i % 2 == 0 { hi } else { lo }).collect();
        roundtrip(vec![extremes.clone(), extremes], 48000, 32);
        roundtrip(vec![vec![lo, hi, 0, -1, 1, lo, hi]], 44100, 32);
    }

    /// Stage 2 and long-term prediction are decided together: on a strongly periodic signal LTP alone
    /// is far cheaper than stage 2 followed by LTP, and `Insane` (stage 2 on) once came out ~10x
    /// larger than `Max` on a pure 1 kHz tone because stage 2 was picked by beating the plain
    /// residual without LTP. `Insane` must never be larger than `Max` here, and must round-trip.
    #[test]
    fn insane_is_not_larger_than_max_on_a_periodic_signal() {
        let tone: Vec<i64> = (0..60_000).map(|i| ((i as f64 * std::f64::consts::TAU * 1000.0 / 44100.0).sin() * 12_000.0).round() as i64).collect();
        let chans = vec![tone.clone(), tone.iter().map(|&x| x / 2).collect()];
        let chunk = default_chunk_frames(44100);
        let enc = |e: Effort| encode_chunked_effort(&chans, 44100, 16, MODE_BLOCK_INDEPENDENT, chunk, 2, None, &Metadata::default(), e).unwrap();
        let (max, insane) = (enc(Effort::Max), enc(Effort::Insane));
        assert_eq!(decoder::decode(&insane).unwrap().1, chans);
        assert!(insane.len() <= max.len(), "insane {} bytes vs max {} bytes", insane.len(), max.len());
    }

    #[test]
    fn silence_and_constant_blocks() {
        roundtrip(vec![vec![0i64; 5000], vec![0i64; 5000]], 44100, 16);
        roundtrip(vec![vec![42i64; 5000]], 8000, 8);
    }

    #[test]
    fn palette_wins_on_iid_few_valued_signal() {
        // A fair coin flip between two fixed extreme values, sample-to-sample independent: no
        // predictive structure exists for Fixed/LPC to exploit, but the raw alphabet has size 2.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        let n = 8000;
        let hi = 32767i64;
        let lo = -32768i64;
        let l: Vec<i64> = (0..n).map(|_| if next() % 2 == 0 { hi } else { lo }).collect();
        let r: Vec<i64> = (0..n).map(|_| if next() % 2 == 0 { hi } else { lo }).collect();
        let encoded = encode_no_fec(&[l.clone(), r.clone()], 44100, 16);
        // Verbatim would need 2 * n * 16 bits = 32000 bytes; a palette should land near n/4 bytes
        // (~1 bit/sample/channel + a small table), far below Fixed/LPC's residual-coding cost too.
        assert!(encoded.len() < n as usize, "palette should compress far below 1 byte/sample-pair, got {} bytes for {n} samples", encoded.len());
        roundtrip(vec![l, r], 44100, 16);
    }

    #[test]
    fn palette_rle_wins_on_slow_square_wave() {
        // Mirrors corpus/synthetic's square100 (tools/corpus/gen_synthetic.py g_square): a 2-value
        // square wave with ~220-sample runs. Flat Palette pays 1 bit/sample regardless of run
        // length; PaletteRle should win decisively here.
        let n = 8000i64;
        let period = 220i64;
        let hi = 26214i64; // 0.8 * full-scale, matching g_square's amplitude convention
        let lo = -26214i64;
        let l: Vec<i64> = (0..n).map(|i| if (i / period) % 2 == 0 { hi } else { lo }).collect();
        let encoded = encode_no_fec(&[l.clone()], 44100, 16);
        // Flat Palette would cost ~1 bit/sample (n/8 bytes) plus header; PaletteRle should land
        // far below that -- roughly (n/period) runs * a handful of bytes each.
        assert!(encoded.len() < (n / 8) as usize / 2,
            "PaletteRle should beat flat Palette's ~{} bytes by a wide margin, got {} bytes for {n} samples",
            n / 8, encoded.len());
        roundtrip(vec![l], 44100, 16);
    }

    #[test]
    fn palette_rle_does_not_regress_no_run_structure() {
        // The encoder must never REGRESS to PaletteRle when there's no run structure to exploit --
        // it should fall back to flat Palette (or better) since the real cost comparison decides,
        // not a heuristic. Reuses the i.i.d. two-value signal from the test above this one.
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        let n = 8000;
        let l: Vec<i64> = (0..n).map(|_| if next() % 2 == 0 { 100i64 } else { -100i64 }).collect();
        let flat_only_estimate = n as usize / 8; // ~1 bit/sample, what plain Palette alone would cost
        let encoded = encode_no_fec(&[l.clone()], 44100, 16);
        assert!(encoded.len() < flat_only_estimate * 2,
            "should still compress near flat-Palette cost on i.i.d. data (no RLE regression), got {} bytes", encoded.len());
        roundtrip(vec![l], 44100, 16);
    }

    #[test]
    fn wasted_bits_detected_and_shrinks_output() {
        // Every sample is a multiple of 16 (4 wasted bits), mimicking audio upsampled from a
        // lower bit depth. Without wasted-bits detection every residual carries 4 dead bits.
        let n: usize = 20_000;
        let raw: Vec<i64> = (0..n as i64).map(|i| (((i * 37) % 2001) - 1000) * 16).collect();
        let with_wasted = encode(&[raw.clone(), raw.iter().map(|&s| -s).collect()], 44100, 16).unwrap();
        let shifted: Vec<i64> = raw.iter().map(|&s| s / 16).collect();
        let without_wasted = encode(&[shifted.clone(), shifted.iter().map(|&s| -s).collect()], 44100, 16).unwrap();
        // The wasted-bits path must be close to the cost of encoding the already-shifted signal
        // (same residual entropy, +5 bits/subframe/frame for the wasted-count field) -- and much
        // smaller than if wasted bits weren't detected at all.
        let n_frames = (n + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE;
        assert!(with_wasted.len() <= without_wasted.len() + n_frames * 2,
                "wasted={} shifted={} frames={}", with_wasted.len(), without_wasted.len(), n_frames);
        roundtrip(vec![raw.clone(), raw.iter().map(|&s| -s).collect()], 44100, 16);
    }

    #[test]
    fn extreme_values_24bit() {
        let lo = -(1i64 << 23);
        let hi = (1i64 << 23) - 1;
        let l: Vec<i64> = (0..5000).map(|i| if i % 2 == 0 { hi } else { lo }).collect();
        let r: Vec<i64> = (0..5000).map(|i| if i % 3 == 0 { lo } else { hi }).collect();
        roundtrip(vec![l, r], 96000, 24);
    }

    #[test]
    fn multichannel_and_odd_length_not_multiple_of_block_size() {
        let n = DEFAULT_BLOCK_SIZE * 2 + 137;
        let chans: Vec<Vec<i64>> = (0..6).map(|c| (0..n).map(|i| ((i as i64 * (c as i64 + 3)) % 500) - 250).collect()).collect();
        roundtrip(chans, 48000, 16);
    }

    #[test]
    fn history_warmup_across_frames_with_changing_stereo_modes_and_wasted_bits() {
        // Format v9: a frame's predictor warmup comes from the previous frame of the same
        // chunk, re-derived in whatever channel representation (L/R/M/S) and wasted-bits domain
        // the new frame uses. Consecutive frames here differ in both, so every representation
        // change and wasted-bits change is crossed, in one chunk and across chunk boundaries.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for bits in [16u8, 24, 32] {
            let amp = (1i64 << (bits - 1)) / 4;
            let n = DEFAULT_BLOCK_SIZE * 7 + 999;
            let (mut l, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for i in 0..n {
                let t = i as f64;
                let tone = (amp as f64 * (t * 0.013).sin()) as i64;
                let noise = (next() % 2001) as i64 - 1000;
                let (a, b) = match (i / DEFAULT_BLOCK_SIZE) % 4 {
                    0 => ((tone + noise) & !3, (tone + noise / 2) & !3), // correlated, 2 wasted bits
                    1 => (noise * 7, (next() % 30001) as i64 - 15000),   // independent
                    2 => (tone | 1, tone - noise),                       // odd left, side-friendly
                    _ => (-tone, tone),                                  // anti-correlated
                };
                l.push(a.clamp(-amp * 4, amp * 4 - 1));
                r.push(b.clamp(-amp * 4, amp * 4 - 1));
            }
            roundtrip(vec![l.clone(), r.clone()], 44100, bits);
            // Chunks shorter than a frame, and chunk boundaries mid-way between frames.
            for chunk in [100usize, DEFAULT_BLOCK_SIZE + 17] {
                let enc = encode_chunked(&[l.clone(), r.clone()], 44100, bits, MODE_BLOCK_INDEPENDENT, chunk, 3, None, &Metadata::default()).unwrap();
                assert_eq!(decoder::decode(&enc).unwrap().1, vec![l.clone(), r.clone()]);
            }
            // Mono and 3-channel: per-channel history.
            roundtrip(vec![l.clone()], 44100, bits);
            roundtrip(vec![l.clone(), r.clone(), l.iter().zip(&r).map(|(a, b)| (a + b) / 2).collect()], 44100, bits);
        }
    }

    #[test]
    fn history_warmup_makes_later_frames_cheaper() {
        // The same frames coded as one chunk (warmup from history) vs one chunk per frame (every
        // frame a chunk's first, warmup verbatim, as before v9): the only per-frame differences
        // are the verbatim warmup and the 16-byte chunk header, so the one-chunk encoding must be
        // smaller by more than the headers alone.
        let n = DEFAULT_BLOCK_SIZE * 8;
        let x: Vec<i64> = (0..n).map(|i| (9000.0 * (i as f64 * 0.021).sin() + 3000.0 * (i as f64 * 0.0037).sin()) as i64).collect();
        let one = encode_chunked(&[x.clone()], 44100, 16, MODE_BLOCK_INDEPENDENT, n, 1, None, &Metadata::default()).unwrap();
        let per = encode_chunked(&[x.clone()], 44100, 16, MODE_BLOCK_INDEPENDENT, DEFAULT_BLOCK_SIZE, 1, None, &Metadata::default()).unwrap();
        let headers = 7 * crate::format::CHUNK_HEADER_LEN;
        assert!(one.len() + headers < per.len(), "one chunk {} + headers {headers} vs per-frame chunks {}", one.len(), per.len());
        assert_eq!(decoder::decode(&one).unwrap().1, vec![x]);
    }

    #[test]
    fn cheapest_cover_picks_per_node_minimum_and_handles_a_truncated_tail() {
        // Three levels over a chunk of 5 leaf-sized pieces (last level-1 and level-2 nodes have
        // only one child each).
        let costs = vec![
            vec![1.0, 1.0, 5.0, 5.0, 2.0],   // leaves
            vec![3.0, 8.0, 1.5],             // pairs: (0,1) costs 2 as leaves -> split; (2,3) 8 < 10 -> keep; (4) 1.5 < 2 -> keep
            vec![11.0, 1.0],                 // roots: first 11 > 2+8 -> split; second 1.0 < 1.5 -> keep
        ];
        assert_eq!(cheapest_cover(&costs), vec![(0, 0), (0, 1), (1, 1), (2, 1)]);
        assert_eq!(cheapest_cover(&[vec![4.0, 2.0]]), vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn every_effort_roundtrips_and_max_is_never_larger_than_fast() {
        // Content that changes character mid-chunk (so variable length has something to find),
        // chunk lengths that aren't multiples of any block size, mono/stereo/3 channels.
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        let n = 40_000;
        let x: Vec<i64> = (0..n).map(|i| {
            let t = i as f64;
            if (i / 3000) % 3 == 0 { (8000.0 * (t * 0.03).sin()) as i64 } else { (next() % 4001) as i64 - 2000 + (3000.0 * (t * 0.002).sin()) as i64 }
        }).collect();
        let y: Vec<i64> = x.iter().enumerate().map(|(i, &v)| v / 2 + (i as i64 % 7)).collect();
        let z: Vec<i64> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
        for chans in [vec![x.clone()], vec![x.clone(), y.clone()], vec![x, y, z]] {
            for chunk in [n, 17_000, 999] {
                let enc = |e: Effort| encode_chunked_effort(&chans, 44100, 16, MODE_BLOCK_INDEPENDENT, chunk, 2, None, &Metadata::default(), e).unwrap();
                let (fast, normal, max, insane) = (enc(Effort::Fast), enc(Effort::Normal), enc(Effort::Max), enc(Effort::Insane));
                for data in [&fast, &normal, &max, &insane] { assert_eq!(decoder::decode(data).unwrap().1, chans); }
                assert!(max.len() <= fast.len(), "max {} > fast {} (chunk {chunk})", max.len(), fast.len());
            }
        }
    }

    #[test]
    fn empty_stream_roundtrip() {
        roundtrip(vec![vec![], vec![]], 44100, 16);
    }

    #[test]
    fn single_sample_stream() {
        roundtrip(vec![vec![123i64], vec![-45i64]], 44100, 16);
    }

    #[test]
    fn value_map_roundtrips_and_recovers_the_source_cost() {
        // A 16-bit stereo source exported to 24 bits at -1 dB (x = round(k * 256 * 0.891)) and a
        // 16-bit +3 dB normalisation that clips: both must round-trip exactly and cost about what
        // the underlying source does, across several chunks and every effort.
        let n = 3 * 45056 + 1234;
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut acc = [0f64; 2];
        let src: Vec<Vec<i64>> = (0..2).map(|c| (0..n).map(|i| {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            acc[c] = 0.95 * acc[c] + (st % 601) as f64 - 300.0;
            ((i as f64 * 0.003 * (c + 1) as f64).sin() * 6000.0 + acc[c]).round() as i64
        }).collect()).collect();
        let chunk = default_chunk_frames(44100);
        let enc = |ch: &[Vec<i64>], bits: u8, e: Effort| encode_chunked_effort(ch, 44100, bits, MODE_BLOCK_INDEPENDENT, chunk, 2, None, &Metadata::default(), e).unwrap();
        let g24 = 256.0 * 10f64.powf(-1.0 / 20.0);
        let up24: Vec<Vec<i64>> = src.iter().map(|c| c.iter().map(|&k| (k as f64 * g24).round() as i64).collect()).collect();
        let g16 = 10f64.powf(3.0 / 20.0);
        let norm16: Vec<Vec<i64>> = src.iter().map(|c| c.iter().map(|&k| (k as f64 * g16 * 3.0).round().clamp(-32768.0, 32767.0) as i64).collect()).collect();
        assert!(norm16[0].contains(&32767));
        for e in [Effort::Fast, Effort::Normal] {
            let base = enc(&src, 16, e).len() as f64;
            for (x, bits) in [(&up24, 24u8), (&norm16, 16)] {
                let b = enc(x, bits, e);
                assert_eq!(&decoder::decode(&b).unwrap().1, x);
                assert!((b.len() as f64) < base * 1.02, "{bits}-bit: {} vs source {base}", b.len());
            }
        }
    }

    #[test]
    fn cross_channel_prediction_roundtrips_and_pays() {
        // Channel 0: an AR(2) process; the others: delayed, filtered copies of it plus a little
        // noise -- structure the fixed stereo transforms cannot express, cross-channel prediction
        // can.
        let n = 40_000usize;
        let mut seed = 99u64;
        let mut noise = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 33) % 201) as i64 - 100 };
        let mut x = vec![0i64; n];
        for t in 2..n { x[t] = ((1.6 * x[t - 1] as f64 - 0.8 * x[t - 2] as f64) as i64 + 40 * noise()).clamp(-30000, 30000); }
        let copy = |d: usize, noise: &mut dyn FnMut() -> i64| -> Vec<i64> {
            (0..n).map(|t| ((3 * x[t.saturating_sub(d)] + 2 * x[t.saturating_sub(d + 1)]) / 5 + noise() / 20).clamp(-32768, 32767)).collect()
        };
        for nch in [2usize, 3] {
            let mut chans = vec![x.clone()];
            for c in 1..nch { chans.push(copy(c + 1, &mut noise)); }
            let enc = |effort: Effort| encode_chunked_effort(&chans, 44100, 16, MODE_BLOCK_INDEPENDENT, default_chunk_frames(44100), 1, None, &Metadata::default(), effort).unwrap();
            let (normal, fast) = (enc(Effort::Normal), enc(Effort::Fast));
            let (_, back) = decoder::decode(&normal).unwrap();
            assert_eq!(back, chans, "{nch} channels");
            assert!((normal.len() as f64) < 0.9 * fast.len() as f64, "{nch} channels: {} vs {}", normal.len(), fast.len());
        }
    }

    #[test]
    fn near_map_with_corrections_roundtrips_and_pays() {
        // cellar_44-like: material scaled by -0.02 dB into 24 bits (G ~ 3.99), with a few samples
        // off the lattice (more in the rare loud tails); quiet enough that every lattice point near the
        // densest value is used, as detection assumes. Must round-trip and beat the plain
        // encoding by about the 2 bits/sample the map removes.
        let n = 2 * 45056 + 777;
        let mut st = 0xD1B54A32D192ED03u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let g = 4.0 * 10f64.powf(-0.02 / 20.0);
        let mut acc = [0f64; 2];
        let x: Vec<Vec<i64>> = (0..2).map(|c| (0..n).map(|i| {
            acc[c] = 0.95 * acc[c] + (next() % 601) as f64 - 300.0;
            let k = ((i as f64 * 0.002 * (c + 1) as f64).sin() * 4000.0 + acc[c]).round() as i64;
            let v = (k as f64 * g).round() as i64;
            let rate = if k.abs() > 5500 { 8 } else { 400 };
            if next() % rate == 0 { v + 1 } else { v }
        }).collect()).collect();
        let chunk = default_chunk_frames(44100);
        let enc = encode_chunked_effort(&x, 44100, 24, MODE_BLOCK_INDEPENDENT, chunk, 2, None, &Metadata::default(), Effort::Fast).unwrap();
        assert_eq!(decoder::decode(&enc).unwrap().1, x);
        // The same content without any lattice (every sample +-1 dithered) as the reference.
        let dith: Vec<Vec<i64>> = x.iter().map(|c| c.iter().map(|&v| v + (next() % 3) as i64 - 1).collect()).collect();
        let plain = encode_chunked_effort(&dith, 44100, 24, MODE_BLOCK_INDEPENDENT, chunk, 2, None, &Metadata::default(), Effort::Fast).unwrap();
        assert!((enc.len() as f64) < plain.len() as f64 * 0.95, "mapped {} vs unmapped {}", enc.len(), plain.len());
    }

    #[test]
    fn white_noise_full_range_16bit() {
        // Deterministic pseudo-random full-range residual stress test (no external RNG dependency).
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        let n = 20_000;
        let l: Vec<i64> = (0..n).map(|_| (next() % 65536) as i64 - 32768).collect();
        let r: Vec<i64> = (0..n).map(|_| (next() % 65536) as i64 - 32768).collect();
        roundtrip(vec![l, r], 44100, 16);
    }

    #[test]
    fn file_encoder_matches_whole_file_encode() {
        // Same audio, same chunk length: the streaming encoder (header patched at the end) must
        // write exactly the bytes the whole-file encoder does, with and without FEC and threads,
        // and a short last chunk.
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let chunk = 4096 * 3;
        let n = chunk * 5 + 777;
        let x: Vec<Vec<i64>> = (0..2).map(|c| (0..n).map(|i| (((i as f64 * 0.01 * (c + 1) as f64).sin() * 9000.0) as i64) + (next() % 61) as i64 - 30).collect()).collect();
        for (threads, fec) in [(1, None), (3, None), (2, Some(2)), (1, Some(3))] {
            let whole = encode_chunked_effort(&x, 44100, 24, MODE_BLOCK_INDEPENDENT, chunk, threads, fec, &Metadata::default(), Effort::Fast).unwrap();
            let mut fe = FileEncoder::new(std::io::Cursor::new(Vec::new()), 2, 44100, 24, MODE_BLOCK_INDEPENDENT, threads, fec, &Metadata::default(), Effort::Fast).unwrap();
            for s in (0..n).step_by(chunk) {
                let e = (s + chunk).min(n);
                fe.push_chunk(x.iter().map(|c| c[s..e].to_vec()).collect()).unwrap();
            }
            let streamed = fe.finish().unwrap().into_inner();
            assert!(streamed == whole, "threads {threads} fec {fec:?}: streamed encode differs");
            assert_eq!(decoder::decode(&streamed).unwrap().1, x);
        }
    }

    #[test]
    fn every_path_and_thread_count_writes_the_same_bytes() {
        // Whole-file (`ordered_pipeline`, concurrent hash) and streaming (persistent workers, hash
        // thread) at 1, 2, 5 and 24 threads, with and without FEC, on a stream whose last chunk is
        // short and on one shorter than a chunk: one set of bytes.
        let mut st = 0xD1B54A32D192ED03u64;
        let mut next = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
        let chunk = 2048 * 3;
        for n in [chunk * 7 + 1234, 900, 1] {
            let x: Vec<Vec<i64>> = (0..2).map(|c| (0..n).map(|i| (((i as f64 * 0.013 * (c + 1) as f64).sin() * 7000.0) as i64) + (next() % 41) as i64 - 20).collect()).collect();
            for fec in [None, Some(3)] {
                let reference = encode_chunked_effort(&x, 48000, 16, MODE_BLOCK_INDEPENDENT, chunk, 1, fec, &Metadata::default(), Effort::Normal).unwrap();
                for threads in [1, 2, 5, 24] {
                    let whole = encode_chunked_effort(&x, 48000, 16, MODE_BLOCK_INDEPENDENT, chunk, threads, fec, &Metadata::default(), Effort::Normal).unwrap();
                    assert!(whole == reference, "n {n} fec {fec:?} threads {threads}: whole-file bytes differ");
                    let mut fe = FileEncoder::new(std::io::Cursor::new(Vec::new()), 2, 48000, 16, MODE_BLOCK_INDEPENDENT, threads, fec, &Metadata::default(), Effort::Normal).unwrap();
                    for s in (0..n).step_by(chunk) {
                        let e = (s + chunk).min(n);
                        fe.push_chunk(x.iter().map(|c| c[s..e].to_vec()).collect()).unwrap();
                    }
                    assert!(fe.finish().unwrap().into_inner() == reference, "n {n} fec {fec:?} threads {threads}: streamed bytes differ");
                }
                assert_eq!(decoder::decode(&reference).unwrap().1, x);
            }
        }
    }

    #[test]
    fn a_file_encoder_dropped_mid_stream_stops_its_workers() {
        // An abandoned encode (a cancelled conversion, an error) must not leave threads running or hang the drop.
        for threads in [2, 8] {
            let mut fe = FileEncoder::new(std::io::Cursor::new(Vec::new()), 1, 44100, 16, MODE_BLOCK_INDEPENDENT, threads, None, &Metadata::default(), Effort::Fast).unwrap();
            for i in 0..3 { fe.push_chunk(vec![(0..4096).map(|j| ((i * 4096 + j) % 200) as i64).collect()]).unwrap(); }
            drop(fe);
        }
    }

    #[test]
    fn file_encoder_rejects_uneven_chunks() {
        let mut fe = FileEncoder::new(std::io::Cursor::new(Vec::new()), 1, 44100, 16, MODE_BLOCK_INDEPENDENT, 1, None, &Metadata::default(), Effort::Fast).unwrap();
        fe.push_chunk(vec![vec![0; 100]]).unwrap();
        fe.push_chunk(vec![vec![0; 50]]).unwrap();
        assert!(fe.push_chunk(vec![vec![0; 50]]).is_err());
        assert!(fe.push_chunk(vec![vec![0; 100], vec![0; 100]]).is_err());
    }
}
