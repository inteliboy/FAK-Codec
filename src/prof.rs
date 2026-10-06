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
}

const NAMES: [&str; 25] = [
    "chunk (total)", "analytic cover", "analyze prep", "mid/side", "autocorrelation", "levinson+quantize", "candidate ranking",
    "fixed orders", "lpc costing", "precision", "cross search", "  cross prep", "  cross normal eq", "  cross solve", "  cross residual",
    "final residual", "stage 2", "ltp (total)", "  ltp fft", "  ltp rank+fit", "  ltp pricing", "rice coding", "value map", "palette", "sha-256",
];
static TIMES: [AtomicU64; 25] = [const { AtomicU64::new(0) }; 25];
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
    let chunk = TIMES[Phase::Chunk as usize].load(Relaxed);
    eprintln!("--- FAK_PROF: encode phases, seconds and share of chunk time");
    for (name, t) in NAMES.iter().zip(&TIMES) {
        let v = t.load(Relaxed);
        if v > 0 { eprintln!("{name:>22} {:8.3}s {:5.1}%", v as f64 / 1e9, 100.0 * v as f64 / chunk.max(1) as f64); }
    }
}
