//! Opt-in encoder acceleration. Only Apple's Accelerate framework is supported (`fak encode --accel apple`,
//! macOS); every other platform runs the bit-exact CPU kernels in `simd.rs`.

/// `--accel apple` (macOS; the CLI's default there since, `--accel cpu` turns it off): the encoder's LPC autocorrelation runs on Accelerate's
/// `vDSP_convD` (`simd::autocorr_vdsp`) instead of the bit-exact kernel. The sums are the same
/// products in a different order, so they differ in the last few bits (~1e-14 relative) and a
/// near-tie in the LPC search can then break differently, so **output bytes depend on the machine
/// and macOS version** (fine under: only the decoded PCM must be identical). Files stay
/// lossless and decode anywhere. The library default is still off; only the CLI turns it on.
static APPLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enables the Accelerate autocorrelation for the rest of the process.
pub fn enable_apple() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    { APPLE.store(true, std::sync::atomic::Ordering::Relaxed); Ok(()) }
    #[cfg(not(target_os = "macos"))]
    Err("`--accel apple` needs macOS (Accelerate framework)".into())
}

/// Back to the bit-exact scalar/NEON analysis (`--accel cpu`).
pub fn disable_apple() { APPLE.store(false, std::sync::atomic::Ordering::Relaxed); }

/// Whether [`enable_apple`] was called (and not undone by [`disable_apple`]).
#[inline]
pub fn apple_enabled() -> bool { APPLE.load(std::sync::atomic::Ordering::Relaxed) }
