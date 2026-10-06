//! The format is frozen (`docs/bitstream-spec.md`): files written once must decode to the same audio
//! forever. These tests decode stored version-21 files from `tests/data/` and compare them with the
//! signals they were made from, so a change to the decoder that alters what an existing file means
//! fails here. Encoder output is not compared (encoders may change); only decoded audio is.
//!
//! The files are made by `regenerate_fixtures` (`cargo test --test format_frozen -- --ignored`). Do
//! not regenerate them to make a failing test pass.
use fak::encoder::{encode_chunked_effort, Effort};
use fak::format::{HEADER_LEN, MODE_BLOCK_INDEPENDENT};
use fak::metadata::Metadata;
use std::path::PathBuf;

struct Fixture { name: &'static str, channels: usize, rate: u32, bits: u8, frames: usize, effort: Effort, fec: Option<usize>, tags: bool }

const FIXTURES: &[Fixture] = &[
    Fixture { name: "stereo16_insane_fec", channels: 2, rate: 44_100, bits: 16, frames: 30_000, effort: Effort::Insane, fec: Some(fak::format::FEC_AUTO), tags: true },
    Fixture { name: "mono24_normal", channels: 1, rate: 48_000, bits: 24, frames: 20_000, effort: Effort::Normal, fec: None, tags: false },
    Fixture { name: "stereo16_fast", channels: 2, rate: 44_100, bits: 16, frames: 12_000, effort: Effort::Fast, fec: None, tags: false },
    Fixture { name: "stereo8_max", channels: 2, rate: 22_050, bits: 8, frames: 10_000, effort: Effort::Max, fec: None, tags: false },
];

/// Deterministic test signal: tones, a repeating pattern (long-term prediction), noise, and a second
/// channel that follows the first through a short filter (cross-channel prediction).
fn signal(f: &Fixture) -> Vec<Vec<i64>> {
    let mut st = 0x9E37_79B9_7F4A_7C15u64 ^ (f.frames as u64);
    let mut next = move || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; st };
    let peak = ((1i64 << (f.bits - 1)) - 1) as f64 * 0.6;
    let mut first = Vec::with_capacity(f.frames);
    for i in 0..f.frames {
        let t = i as f64;
        let tone = (t * 0.031).sin() * 0.5 + (t * 0.0071).sin() * 0.3 + ((i % 441) as f64 / 441.0 - 0.5) * 0.15;
        let noise = ((next() % 2001) as f64 - 1000.0) / 1000.0 * 0.02;
        first.push(((tone + noise) * peak).round() as i64);
    }
    let mut out = vec![first];
    for c in 1..f.channels {
        let src = &out[c - 1];
        let ch: Vec<i64> = (0..f.frames).map(|i| {
            let a = src[i] as f64 * 0.7 + if i > 0 { src[i - 1] as f64 * 0.2 } else { 0.0 };
            a.round() as i64 + (next() % 7) as i64 - 3
        }).collect();
        out.push(ch);
    }
    out
}

fn metadata(f: &Fixture) -> Metadata {
    let mut m = Metadata::default();
    if f.tags { m.tags = vec!["TITLE=Frozen".to_string(), "ARTIST=FAK".to_string()]; }
    m
}

fn path(name: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(format!("{name}.fak")) }

#[test]
#[ignore = "writes tests/data; run only to create the fixtures"]
fn regenerate_fixtures() {
    std::fs::create_dir_all(path("x").parent().unwrap()).unwrap();
    for f in FIXTURES {
        let chunk = fak::format::default_chunk_frames(f.rate);
        let bytes = encode_chunked_effort(&signal(f), f.rate, f.bits, MODE_BLOCK_INDEPENDENT, chunk, 1, f.fec, &metadata(f), f.effort).unwrap();
        std::fs::write(path(f.name), bytes).unwrap();
    }
}

#[test]
fn stored_version_21_files_decode_to_their_source_audio() {
    for f in FIXTURES {
        let bytes = std::fs::read(path(f.name)).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert_eq!(&bytes[..4], b"FAK1", "{}", f.name);
        assert_eq!(bytes[4], 21, "{}: header version byte", f.name);
        assert!(bytes.len() > HEADER_LEN);
        let want = signal(f);
        for threads in [1, 4] {
            let (header, meta, got) = fak::decoder::decode_full(&bytes, threads).unwrap_or_else(|e| panic!("{} ({threads} threads): {e}", f.name));
            assert_eq!((header.channels as usize, header.sample_rate, header.bits_per_sample), (f.channels, f.rate, f.bits), "{}", f.name);
            assert_eq!(got, want, "{} ({threads} threads): decoded audio changed", f.name);
            assert_eq!(meta.tags, metadata(f).tags, "{}", f.name);
        }
        fak::decoder::verify(&bytes, 2).unwrap_or_else(|e| panic!("{}: stored SHA-256 no longer matches: {e}", f.name));
    }
}

#[test]
fn only_header_version_21_is_accepted() {
    let bytes = std::fs::read(path(FIXTURES[2].name)).unwrap();
    for v in [0u8, 1, 20, 22, 255] {
        let mut b = bytes.clone();
        b[4] = v;
        assert!(fak::decoder::decode_full(&b, 1).is_err(), "version byte {v} must be rejected");
    }
}

#[test]
fn a_damaged_chunk_in_the_fec_file_is_still_repaired() {
    let f = &FIXTURES[0];
    let mut bytes = std::fs::read(path(f.name)).unwrap();
    let at = HEADER_LEN + 200;
    bytes[at] ^= 0x55;
    let (_, _, got) = fak::decoder::decode_full(&bytes, 1).expect("one damaged chunk is within the parity's reach");
    assert_eq!(got, signal(f));
}
