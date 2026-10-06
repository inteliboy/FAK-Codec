//! FAK ("FLAC Audio Killer"): a lossless audio codec. File extension `.fak`.
//! The decoder is deterministic and integer-exact: the same file decodes to the same PCM on every
//! platform and instruction set.
/// This library's version (`fak --version`, the encoder string written into files).
pub const LIB_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod accel;
pub mod bitio;
pub mod cdrip;
pub mod cpufeatures;
pub mod cpuinfo;
pub mod crc;
pub mod crossch;
pub mod decoder;
pub mod detmath;
pub mod encoder;
pub mod floatpcm;
pub mod format;
pub mod lpc;
pub mod ltp;
pub mod ols;
pub mod metadata;
pub mod palette;
pub mod parallel;
pub mod predictors;
pub mod prof;
pub mod rice;
pub mod rs;
pub mod sha256;
pub mod simd;
pub mod stage2;
pub mod stereo;
pub mod valuemap;
pub mod wav;
