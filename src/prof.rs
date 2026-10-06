//! Phase timers for the block encoder (`FAK_PROF=1` prints a table to stderr when the CLI exits).
//!
//! A development aid, not a feature: it says where encode time goes ("Encode-speed
//! pass on x86-64" used it for every decision there). Each phase is a flat span; run with `-t 1` for
//! meaningful shares, since several threads add their times together. When `FAK_PROF` is unset a span
//! costs one relaxed atomic load and a branch.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// What a span times. Spans do not overlap except where a phase is marked as nested in another.
#[derive(Clone, Copy)]
pub enum Phase {
    /// One chunk's whole block encode: the total the shares below are taken against.
    Chunk,
    /// Analytic frame-cost estimates for the block-size search (Tukey window, autocorrelation, Levinson).
    AnalyticCover,
    /// Building a subframe's search inputs: shifted copy, wasted bits, constant check.
    AnalyzePrep,
    MidSide,
    /// Windowed autocorrelation of the LPC candidates.
    Autocorr,
    /// Levinson-Durbin, coefficient quantization and analytic ranking of the LPC candidates.
    Levinson,
    CandidateRanking,
    FixedOrders,
    /// Costing LPC candidates on their residuals.
    LpcCosting,
    /// Coefficient precision: the excess-power model (or the greedy walk under `FAK_PREC=walk`).
    Precision,
    CrossSearch,
    /// Inside `CrossSearch`.
    CrossPrep,
    CrossNormalEq,
    CrossSolve,
    CrossResidual,
    /// The winning subframe's final residual.
    FinalResidual,
    Stage2,
    /// Long-term prediction: the whole search.
    Ltp,
    /// Inside `Ltp`.
    LtpFft,
    LtpRankFit,
    LtpPricing,
    RiceCoding,
    ValueMap,
    Palette,
    Sha,
    /// Decoder: one chunk's whole decode (the total the decode shares are taken against).
    DecChunk,
    /// Inside `DecChunk`: Rice residual decoding.
    DecRice,
    /// Inside `DecChunk`: the carried LMS filter (inverse or advance).
    DecCarried,
    /// Inside `DecChunk`: the stereo OLS predictor (cfg 3/4 chunks).
    DecOls,
    /// Inside `DecChunk`: long-term prediction.
    DecLtp,
    /// Inside `DecChunk`: the per-block stage-2 filter.
    DecStage2,
    /// Inside `DecChunk`: cross-channel prediction.
    DecCross,
    /// Inside `DecChunk`: fixed/LPC sample reconstruction.
    DecPredictor,
}

const NAMES: [&str; 33] = [
    "chunk (total)", "analytic cover", "analyze prep", "mid/side", "autocorrelation", "levinson+quantize", "candidate ranking",
    "fixed orders", "lpc costing", "precision", "cross search", "  cross prep", "  cross normal eq", "  cross solve", "  cross residual",
    "final residual", "stage 2", "ltp (total)", "  ltp fft", "  ltp rank+fit", "  ltp pricing", "rice coding", "value map", "palette", "sha-256",
    "decode chunk (total)", "  dec rice", "  dec carried lms", "  dec ols stereo", "  dec ltp", "  dec stage 2", "  dec cross", "  dec predictor",
];
static TIMES: [AtomicU64; 33] = [const { AtomicU64::new(0) }; 33];
static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

fn enabled() -> bool { *ON.get_or_init(|| std::env::var_os("FAK_PROF").is_some()) }

/// Times from its creation until it is dropped (a no-op unless `FAK_PROF` is set).
pub struct Guard(usize, Option<Instant>);

pub fn span(phase: Phase) -> Guard { Guard(phase as usize, if enabled() { Some(Instant::now()) } else { None }) }

impl Drop for Guard {
    fn drop(&mut self) { if let Some(t) = self.1 { TIMES[self.0].fetch_add(t.elapsed().as_nanos() as u64, Relaxed); } }
}

/// Prints the table: seconds and share of the chunk total (indented phases are inside the one above).
pub fn report() {
    if !enabled() { return; }
    let (enc, dec) = (TIMES[Phase::Chunk as usize].load(Relaxed), TIMES[Phase::DecChunk as usize].load(Relaxed));
    let chunk = if enc > 0 { enc } else { dec };
    eprintln!("--- FAK_PROF: {} phases, seconds and share of chunk time", if enc > 0 { "encode" } else { "decode" });
    for (name, t) in NAMES.iter().zip(&TIMES) {
        let v = t.load(Relaxed);
        if v > 0 { eprintln!("{name:>22} {:8.3}s {:5.1}%", v as f64 / 1e9, 100.0 * v as f64 / chunk.max(1) as f64); }
    }
}
