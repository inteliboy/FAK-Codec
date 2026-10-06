//! The `fak cpuinfo` report: what the running CPU supports, which kernel FAK uses for each
//! operation on this machine and why, and which further acceleration options exist.
#[cfg(target_os = "macos")]
use crate::accel;
use crate::{cpufeatures, crc, rice, sha256, simd};
use std::fmt::Write;

fn on_off(v: bool) -> &'static str { if v { "yes" } else { "no" } }

fn env_set(name: &str) -> bool { std::env::var_os(name).is_some() }

/// The CPU's marketing name where the architecture exposes it.
fn cpu_name() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        // Safety: CPUID is available on every x86-64 CPU.
        #[allow(unused_unsafe)]
        let max = unsafe { __cpuid(0x8000_0000) }.eax;
        if max < 0x8000_0004 { return None; }
        let mut bytes = Vec::with_capacity(48);
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            #[allow(unused_unsafe)]
            let r = unsafe { __cpuid(leaf) };
            for w in [r.eax, r.ebx, r.ecx, r.edx] { bytes.extend_from_slice(&w.to_le_bytes()); }
        }
        let s = String::from_utf8_lossy(&bytes).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string();
        return if s.is_empty() { None } else { Some(s.split_whitespace().collect::<Vec<_>>().join(" ")) };
    }
    #[allow(unreachable_code)]
    None
}

/// Builds the report. Makes the timed kernel choices first, so the kernels listed are the ones the
/// encoder will use.
pub fn report() -> String {
    simd::choose_kernels();
    let mut s = String::new();
    let w = &mut s;

    let _ = writeln!(w, "FAK {} (format v{}) on {} / {}", env!("CARGO_PKG_VERSION"), crate::format::VERSION, std::env::consts::OS, std::env::consts::ARCH);
    if let Some(n) = cpu_name() { let _ = writeln!(w, "CPU:      {n}"); }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let _ = writeln!(w, "Threads:  {threads} logical CPUs (encode and decode use all of them unless -t N is given)");
    let _ = writeln!(w, "Build:    allocator {}", if cfg!(feature = "fast-alloc") { "mimalloc" } else { "system" });
    let req = cpufeatures::compiled_requirements();
    let _ = writeln!(w, "          compiled to require: avx2={} avx512f={} bmi2={} fma={} (a portable build requires none of them)", req.avx2, req.avx512f, req.bmi2, req.fma);

    let _ = writeln!(w, "\nInstruction sets (detected at run time):");
    for e in cpufeatures::extensions() {
        let _ = writeln!(w, "  {:<11} {:<3}  {}", e.name, on_off(e.present), e.used_for);
    }

    let _ = writeln!(w, "\nKernels in use on this machine:");
    let row = |w: &mut String, what: &str, used: &str| { let _ = writeln!(w, "  {:<38} {}", what, used); };
    row(w, "LPC residuals (encoder search)", simd::lpc_residuals_kernel());
    row(w, "LPC candidate ranking, 16-bit", simd::lpc_estimate_kernel());
    row(w, "autocorrelation", simd::autocorr_kernel());
    row(w, "analysis loops (cross-channel, FFT)", simd::analysis_loops_kernel());
    row(w, "LPC reconstruction (decoder)", simd::reconstruct_kernel());
    row(w, "Rice decoding (decoder)", rice::decode_kernel());
    row(w, "CRC-32 (decoder, encoder)", crc::crc32_kernel());
    row(w, "SHA-256 (verify, encoder)", sha256::sha256_kernel());
    row(w, "AVX-512 selection", simd::avx512_policy());
    let _ = writeln!(w, "  Every kernel of an operation gives identical output, so the choice changes speed only.");

    let _ = writeln!(w, "\nAcceleration options beyond the CPU kernels:");
    #[cfg(target_os = "macos")]
    {
        let _ = writeln!(w, "  Apple Accelerate     available; {} (--accel apple / --accel cpu)",
            if accel::apple_enabled() { "enabled by default for the encoder's autocorrelation" } else { "off" });
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = writeln!(w, "  Apple Accelerate     not available (macOS only)");
    }
    let _ = writeln!(w, "  Threads              -t N sets the worker count (default {threads}); output does not depend on N");
    let _ = writeln!(w, "  Level                -l fast|normal|max|insane|archival trades encode time for size; decoding is unaffected");

    let mut hints: Vec<String> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let f = cpufeatures::detect();
        if !f.avx2 { hints.push("this CPU has no AVX2: encoding and decoding use the baseline code and are noticeably slower".into()); }
        if f.avx512f && (simd::lpc_residuals_kernel() == "avx2" || simd::autocorr_kernel() == "avx2") {
            hints.push("AVX-512F is present but AVX2 measured faster for at least one kernel; FAK_FORCE_AVX512=1 forces AVX-512 (usually slower here)".into());
        }
        if f.avx512f && simd::avx512_policy().starts_with("disabled") {
            hints.push("AVX-512 is disabled by FAK_DISABLE_AVX512; unset it to let FAK time AVX-512 against AVX2".into());
        }
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") && simd::lpc_estimate_kernel() != "avx512vnni" {
            hints.push("AVX-512 VNNI is present but the ranking kernel is not using it (FAK_DISABLE_VNNI set, AVX-512 disabled, or the exact kernel measured faster)".into());
        }
    }
    for (var, what) in [("FAK_DISABLE_AVX512", "the AVX-512 kernels"), ("FAK_FORCE_AVX512", "kernel timing"), ("FAK_DISABLE_VNNI", "the VNNI ranking kernel"),
        ("FAK_DISABLE_AVX2_CLONES", "the AVX2 analysis loops"), ("FAK_DISABLE_HW_CRC", "hardware CRC-32"), ("FAK_DISABLE_SHA_NI", "hardware SHA-256"),
        ("FAK_DISABLE_LZCNT", "lzcnt Rice decoding"), ("FAK_DISABLE_BLOCKED", "blocked reconstruction")] {
        if env_set(var) { hints.push(format!("{var} is set and overrides {what}")); }
    }
    if hints.is_empty() { let _ = writeln!(w, "\nNo CPU kernel is left unused. The options above need other hardware or a different build."); }
    else { let _ = writeln!(w, "\nNotes:"); for h in hints { let _ = writeln!(w, "  - {h}"); } }
    let _ = writeln!(w, "\nOverrides (set to 1, for testing and timing): FAK_DISABLE_AVX512, FAK_FORCE_AVX512, FAK_DISABLE_VNNI, FAK_DISABLE_AVX2_CLONES,\n  FAK_DISABLE_HW_CRC, FAK_DISABLE_SHA_NI, FAK_DISABLE_LZCNT, FAK_DISABLE_BLOCKED");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_names_every_section_and_kernel() {
        let r = report();
        for needle in ["Instruction sets", "Kernels in use", "LPC residuals", "autocorrelation", "CRC-32", "SHA-256", "Rice decoding", "Acceleration options"] {
            assert!(r.contains(needle), "report lacks {needle:?}:\n{r}");
        }
    }
}
