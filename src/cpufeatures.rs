//! Runtime CPU instruction-set detection (target multiple ISAs, never assume a
//! feature is present just because the build machine has it). Exists to replace the blanket
//! `target-cpu=native` compile-time approach: that flag bakes the *build* machine's exact
//! instruction set into the binary, so the program would require AVX2/BMI2/FMA/SHA and crash with an
//! illegal-instruction fault, not a clean error, on any CPU that lacks them. A portable build must instead
//! detect what the *running* machine supports and dispatch accordingly, falling back to the
//! scalar reference path wherever a feature isn't available. This module is that detection layer,
//! also useful on its own as a diagnostic (`fak cpuinfo`). The SIMD kernels (`src/simd.rs`) keep their own cached
//! `is_x86_feature_detected!` checks rather than calling `detect()` here -- `detect()` re-queries every call (fine for a one-shot diagnostic,
//! too slow for a per-sample decode hot path); `simd.rs`'s `OnceLock`-cached check is the pattern
//! any future kernel added here should follow.
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    pub avx2: bool,
    pub avx512f: bool,
    pub sse4_2: bool,
    pub bmi2: bool,
    pub fma: bool,
    pub neon: bool,
}

/// Detects what the *running* CPU actually supports, not what the build was compiled for.
/// `is_x86_feature_detected!`/`is_aarch64_feature_detected!` are std macros backed by CPUID/HWCAP
/// at runtime (no external crate, no unsafe on the caller's part) -- exactly the mechanism 
///  asks for ("do not assume AVX-512 is automatically faster... benchmark real implementations")
/// applied one level earlier: don't assume a feature exists at all.
pub fn detect() -> Features {
    #[cfg(target_arch = "x86_64")]
    {
        Features {
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            sse4_2: is_x86_feature_detected!("sse4.2"),
            bmi2: is_x86_feature_detected!("bmi2"),
            fma: is_x86_feature_detected!("fma"),
            neon: false,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Features {
            avx2: false,
            avx512f: false,
            sse4_2: false,
            bmi2: false,
            fma: false,
            neon: std::arch::is_aarch64_feature_detected!("neon"),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Features { avx2: false, avx512f: false, sse4_2: false, bmi2: false, fma: false, neon: false }
    }
}

/// What this exact binary was *compiled* to require (via `-C target-feature`/`target-cpu`), as
/// opposed to `detect()`'s report of what the running machine actually has. The two can differ --
/// that gap is the portability hazard. `cfg!(target_feature = "...")` is
/// evaluated at compile time by rustc, so this reflects the build, not the host.
pub struct CompiledRequirements {
    pub avx2: bool,
    pub avx512f: bool,
    pub bmi2: bool,
    pub fma: bool,
}

pub fn compiled_requirements() -> CompiledRequirements {
    CompiledRequirements {
        avx2: cfg!(target_feature = "avx2"),
        avx512f: cfg!(target_feature = "avx512f"),
        bmi2: cfg!(target_feature = "bmi2"),
        fma: cfg!(target_feature = "fma"),
    }
}

/// One instruction-set extension the running CPU was probed for: its name, whether it is present,
/// and what FAK uses it for.
pub struct Extension { pub name: &'static str, pub present: bool, pub used_for: &'static str }

/// Every extension FAK can use, probed on the running CPU.
pub fn extensions() -> Vec<Extension> {
    let mut v = Vec::new();
    macro_rules! ext { ($name:expr, $present:expr, $used:expr) => { v.push(Extension { name: $name, present: $present, used_for: $used }) }; }
    #[cfg(target_arch = "x86_64")]
    {
        ext!("sse4.1", is_x86_feature_detected!("sse4.1"), "CRC-32 and SHA-256 helpers");
        ext!("sse4.2", is_x86_feature_detected!("sse4.2"), "baseline vector code");
        ext!("pclmulqdq", is_x86_feature_detected!("pclmulqdq"), "CRC-32 (carry-less multiply)");
        ext!("sha", is_x86_feature_detected!("sha"), "SHA-256 (SHA-NI)");
        ext!("lzcnt", is_x86_feature_detected!("lzcnt"), "Rice decoding");
        ext!("bmi2", is_x86_feature_detected!("bmi2"), "Rice decoding");
        ext!("avx2", is_x86_feature_detected!("avx2"), "LPC residuals, autocorrelation, cross-channel sums, reconstruction");
        ext!("fma", is_x86_feature_detected!("fma"), "not used (results must not depend on it)");
        ext!("avx512f", is_x86_feature_detected!("avx512f"), "LPC residuals and autocorrelation (when faster than AVX2)");
        ext!("avx512bw", is_x86_feature_detected!("avx512bw"), "candidate ranking (with VNNI)");
        ext!("avx512vnni", is_x86_feature_detected!("avx512vnni"), "candidate ranking, 16-bit audio");
    }
    #[cfg(target_arch = "aarch64")]
    {
        ext!("neon", std::arch::is_aarch64_feature_detected!("neon"), "LPC residuals, ranking, reconstruction");
        ext!("crc", std::arch::is_aarch64_feature_detected!("crc"), "CRC-32");
        ext!("sha2", std::arch::is_aarch64_feature_detected!("sha2"), "SHA-256");
        ext!("aes", std::arch::is_aarch64_feature_detected!("aes"), "not used");
        ext!("dotprod", std::arch::is_aarch64_feature_detected!("dotprod"), "not used");
    }
    v
}

impl fmt::Display for Features {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut on = Vec::new();
        if self.avx2 { on.push("avx2"); }
        if self.avx512f { on.push("avx512f"); }
        if self.sse4_2 { on.push("sse4.2"); }
        if self.bmi2 { on.push("bmi2"); }
        if self.fma { on.push("fma"); }
        if self.neon { on.push("neon"); }
        if on.is_empty() { write!(f, "(none detected -- scalar baseline only)") }
        else { write!(f, "{}", on.join(", ")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic_and_is_deterministic_within_a_process() {
        let a = detect();
        let b = detect();
        assert_eq!(a, b, "CPU features must not change mid-process");
    }

    #[test]
    fn avx512_implies_nothing_about_avx2_but_avx2_absence_with_avx512_present_would_be_incoherent() {
        // Real-hardware sanity check, not a logical requirement of the ISA itself: every shipping
        // x86-64 chip with AVX-512 also has AVX2 (AVX-512 was layered on top of the AVX2-era
        // feature set in every real CPU family to date). If this ever fails on real hardware it
        // means a new/unusual CPU exists, not that the detection code is wrong -- worth a second
        // look before assuming a bug.
        let f = detect();
        if f.avx512f { assert!(f.avx2, "unexpected: AVX-512F without AVX2 on real hardware"); }
    }

    #[test]
    fn display_lists_detected_features_or_says_none() {
        let f = Features { avx2: true, avx512f: false, sse4_2: true, bmi2: false, fma: false, neon: false };
        assert_eq!(f.to_string(), "avx2, sse4.2");
        let none = Features { avx2: false, avx512f: false, sse4_2: false, bmi2: false, fma: false, neon: false };
        assert_eq!(none.to_string(), "(none detected -- scalar baseline only)");
    }
}
