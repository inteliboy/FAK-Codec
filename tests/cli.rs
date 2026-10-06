//! End-to-end tests of the `fak` command line: every level round-trips, stdin/stdout work, cue
//! sheets are embedded, exported and used to cut tracks, `edit` changes metadata without touching
//! the audio, and mistakes exit with the usage code (2) rather than a failure code (1).
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const RATE: u32 = 44_100;
const FRAMES: usize = RATE as usize * 6;

fn fak(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_fak")).args(args).output().expect("run fak")
}

fn ok(args: &[&str]) -> Output {
    let o = fak(args);
    assert!(o.status.success(), "fak {args:?} failed: {}", String::from_utf8_lossy(&o.stderr));
    o
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fak_cli_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A stereo 16-bit test signal: a chord with slow noise, different per channel.
fn source(d: &Path) -> (PathBuf, Vec<Vec<i64>>) {
    let mut seed = 12345u32;
    let channels: Vec<Vec<i64>> = (0..2).map(|c| (0..FRAMES).map(|i| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let t = i as f64 / RATE as f64;
        let v = 8000.0 * (t * 440.0 * (1.0 + c as f64 * 0.5) * std::f64::consts::TAU).sin() + 3000.0 * (t * 97.0 * std::f64::consts::TAU).sin();
        v as i64 + (seed >> 26) as i64 - 32
    }).collect()).collect();
    let p = d.join("src.wav");
    fak::wav::write_wav(&p, &fak::wav::Wav { channels: channels.clone(), sample_rate: RATE, bits: 16, channel_mask: None, float_info: None }).unwrap();
    (p, channels)
}

fn pcm(p: &Path) -> Vec<Vec<i64>> { fak::wav::read_wav(p).unwrap().channels }
fn s(p: &Path) -> &str { p.to_str().unwrap() }

// 00:02:00 = 88200 frames, 00:04:00 = 176400 (INDEX 00 pre-gap at 00:03:50 = 169050).
const CUE: &str = "REM made by a test\r\nPERFORMER \"Band\"\r\nTITLE \"Album\"\r\nFILE \"src.wav\" WAVE\r\n  TRACK 01 AUDIO\r\n    TITLE \"One\"\r\n    INDEX 01 00:00:00\r\n  TRACK 02 AUDIO\r\n    TITLE \"Two\"\r\n    INDEX 01 00:02:00\r\n  TRACK 03 AUDIO\r\n    TITLE \"Three\"\r\n    ISRC ABCDE1234567\r\n    INDEX 00 00:03:50\r\n    INDEX 01 00:04:00\r\n";

#[test]
fn every_level_round_trips() {
    let d = dir("levels");
    let (src, want) = source(&d);
    let mut sizes = Vec::new();
    for level in ["fast", "normal", "max", "insane"] {
        let out = d.join(format!("{level}.fak"));
        ok(&["encode", s(&src), s(&out), "--level", level, "-q"]);
        let back = d.join(format!("{level}.wav"));
        ok(&["decode", s(&out), s(&back), "-q"]);
        assert_eq!(pcm(&back), want, "{level}");
        ok(&["verify", s(&out)]);
        sizes.push(std::fs::metadata(&out).unwrap().len());
    }
    // insane is max plus stage 2 only where stage 2 pays, so it is never larger than max.
    assert!(sizes[3] <= sizes[2], "{sizes:?}");
    // Shorthand flags are the same levels; options may come before the file names.
    let a = d.join("a.fak");
    ok(&["encode", "--max", "-q", s(&src), s(&a)]);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(d.join("max.fak")).unwrap());
}

/// `--archival` (= `insane` + FEC): same audio and effort as `insane`, plus parity blocks that
/// `fak info` reports; an explicit `--fec=N` decides the group size; and a damaged chunk is
/// repaired by `decode` (the point of the level).
#[test]
fn archival_is_insane_plus_fec_and_repairs_a_damaged_chunk() {
    let d = dir("archival");
    let (src, want) = source(&d);
    let (ins, arc, arc4) = (d.join("insane.fak"), d.join("archival.fak"), d.join("archival4.fak"));
    ok(&["encode", s(&src), s(&ins), "--insane", "-q"]);
    ok(&["encode", s(&src), s(&arc), "--archival", "-q"]);
    ok(&["encode", s(&src), s(&arc4), "-l", "archival", "--fec=4", "-q"]);
    let (i, a, a4) = (std::fs::metadata(&ins).unwrap().len(), std::fs::metadata(&arc).unwrap().len(), std::fs::metadata(&arc4).unwrap().len());
    assert!(a > i && a4 > a, "parity adds bytes, and a smaller group adds more: {i} {a} {a4}");
    let info = |f: &std::path::Path| String::from_utf8_lossy(&fak(&["info", s(f)]).stdout).to_string();
    assert!(!info(&ins).contains("FEC"), "insane has no parity");
    assert!(info(&arc).contains("FEC"), "archival has parity");
    // Same audio either way.
    let back = d.join("a.wav");
    ok(&["decode", s(&arc), s(&back), "-q"]);
    assert_eq!(pcm(&back), want);
    // Damage a byte in the middle of the payload: decode still returns the exact PCM (FEC repair).
    let mut bytes = std::fs::read(&arc4).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    let bad = d.join("damaged.fak");
    std::fs::write(&bad, &bytes).unwrap();
    let fixed = d.join("fixed.wav");
    ok(&["decode", s(&bad), s(&fixed), "-q"]);
    assert_eq!(pcm(&fixed), want, "a damaged chunk in an archival file must be rebuilt from parity");
}

///  32-bit sources round-trip through the real CLI at every level.
#[test]
fn int32_source_round_trips_every_level() {
    let d = dir("int32");
    let mut seed = 987654u32;
    let channels: Vec<Vec<i64>> = (0..2).map(|c| (0..FRAMES).map(|i| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let t = i as f64 / RATE as f64;
        let v = 900_000_000.0 * (t * 440.0 * (1.0 + c as f64 * 0.5) * std::f64::consts::TAU).sin();
        (v as i64 + (seed as i64) - (1i64 << 31)).clamp(i32::MIN as i64, i32::MAX as i64)
    }).collect()).collect();
    let src = d.join("src32.wav");
    fak::wav::write_wav(&src, &fak::wav::Wav { channels: channels.clone(), sample_rate: RATE, bits: 32, channel_mask: None, float_info: None }).unwrap();
    for level in ["fast", "normal", "max", "insane"] {
        let out = d.join(format!("{level}.fak"));
        ok(&["encode", s(&src), s(&out), "--level", level, "-q"]);
        let back = d.join(format!("{level}.wav"));
        ok(&["decode", s(&out), s(&back), "-q"]);
        assert_eq!(pcm(&back), channels, "{level}");
        ok(&["verify", s(&out)]);
    }
}

/// (b): a real 32-bit float WAV (grid-aligned content, matching typical DAW exports, plus a few
/// genuinely pathological samples -- `NaN`, `Inf`, `-0.0` -- that must fall back to the exception
/// path) round-trips bit-exactly through the real `fak` binary at every level. Verified two ways:
/// the decoded WAV's own grid mapping matches the source's exactly (`channels` + `float_info`,
/// which -- since `unmap`.`map` is the identity for every input, `floatpcm.rs` -- is equivalent to
/// the raw float32 bytes matching exactly), and directly via `floatpcm::unmap_from_pcm` back to the
/// exact bit patterns this test started with.
#[test]
fn float32_source_round_trips_every_level() {
    let d = dir("float32");
    let mut seed = 424242u32;
    let raw: Vec<Vec<i32>> = (0..2).map(|c| (0..FRAMES).map(|i| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let t = i as f64 / RATE as f64;
        let v = 9000.0 * (t * 440.0 * (1.0 + c as f64 * 0.5) * std::f64::consts::TAU).sin();
        (v as i32).wrapping_add(((seed >> 28) as i32) - 8)
    }).collect()).collect();
    let mut bits: Vec<Vec<u32>> = raw.iter().map(|c| c.iter().map(|&x| (x as f32 / 32768.0).to_bits()).collect()).collect();
    bits[0][10] = f32::NAN.to_bits();
    bits[0][20] = f32::INFINITY.to_bits();
    bits[0][30] = f32::NEG_INFINITY.to_bits();
    bits[1][40] = 0x8000_0000; // -0.0, distinct from +0.0
    let (mapped, container_bits, info) = fak::floatpcm::map_to_pcm(&bits);
    let src = d.join("srcf32.wav");
    fak::wav::write_wav(&src, &fak::wav::Wav { channels: mapped, sample_rate: RATE, bits: container_bits, channel_mask: None, float_info: Some(info) }).unwrap();
    let src_read = fak::wav::read_wav(&src).unwrap();
    for level in ["fast", "normal", "max", "insane"] {
        let out = d.join(format!("{level}.fak"));
        ok(&["encode", s(&src), s(&out), "--level", level, "-q"]);
        let back = d.join(format!("{level}.wav"));
        ok(&["decode", s(&out), s(&back), "-q"]);
        let back_read = fak::wav::read_wav(&back).unwrap();
        assert_eq!(back_read.channels, src_read.channels, "{level}");
        assert_eq!(back_read.float_info, src_read.float_info, "{level}");
        let reconstructed = fak::floatpcm::unmap_from_pcm(&back_read.channels, back_read.float_info.as_ref().unwrap());
        assert_eq!(reconstructed, bits, "{level}");
        ok(&["verify", s(&out)]);
    }
}

/// `seek` on a float source: exception indices are absolute over the whole stream, so `cmd_seek`
/// must rebase them (`floatpcm::slice_info`) to the returned sub-range, not apply them raw. `seek`
/// returns only the chunk containing the target frame (from the target onward), not the rest of the
/// stream -- confirmed exactly with `format::default_chunk_frames`, and exceptions are placed both
/// inside and outside that window to exercise both rebasing and correct exclusion.
#[test]
fn seek_on_float_source_rebases_exceptions() {
    let d = dir("float32_seek");
    // Grid-aligned (int16-normalized) base signal, so the only exceptions are the ones placed below
    // -- a raw floating-point signal would also (correctly, but unhelpfully for this test) turn
    // every near-zero-crossing sample into its own exception, since values that close to zero need
    // more fractional precision than `floatpcm::MAX_SCALE_EXP` provides.
    let sample = |c: usize, i: usize| -> u32 {
        let t = i as f64 / RATE as f64;
        let v = (6000.0 * (t * 220.0 * (1.0 + c as f64) * std::f64::consts::TAU).sin()) as i32;
        (v as f32 / 32768.0).to_bits()
    };
    let mut bits: Vec<Vec<u32>> = (0..2).map(|c| (0..FRAMES).map(|i| sample(c, i)).collect()).collect();
    let target: u64 = 150_000;
    let chunk_frames = fak::format::default_chunk_frames(RATE) as u64;
    let chunk_start = target / chunk_frames * chunk_frames;
    assert!(chunk_start < target && target < chunk_start + chunk_frames, "test assumes target isn't on a chunk boundary");
    // Before the target's chunk, and before the target within its own chunk: must both be dropped.
    bits[0][1000] = f32::NAN.to_bits();
    bits[1][(chunk_start + 10) as usize] = f32::INFINITY.to_bits();
    // After the target, still inside the returned chunk: must survive, rebased to be relative to it.
    bits[0][target as usize + 500] = f32::NEG_INFINITY.to_bits();
    bits[1][target as usize + 20_000] = 0x8000_0000;
    let (mapped, container_bits, info) = fak::floatpcm::map_to_pcm(&bits);
    let src = d.join("srcf32.wav");
    fak::wav::write_wav(&src, &fak::wav::Wav { channels: mapped, sample_rate: RATE, bits: container_bits, channel_mask: None, float_info: Some(info) }).unwrap();
    let out = d.join("a.fak");
    ok(&["encode", s(&src), s(&out), "-q"]);
    let back = d.join("seek.wav");
    ok(&["seek", s(&out), &target.to_string(), s(&back)]);
    let back_read = fak::wav::read_wav(&back).unwrap();
    let got_exceptions = &back_read.float_info.as_ref().unwrap().exceptions;
    assert_eq!(got_exceptions.len(), 2, "only the two post-target exceptions should survive: {got_exceptions:?}");
    let reconstructed = fak::floatpcm::unmap_from_pcm(&back_read.channels, back_read.float_info.as_ref().unwrap());
    let len = reconstructed[0].len();
    assert!(len > 20_000 && (target as usize + len) < FRAMES, "test assumes the returned chunk covers both planted post-target exceptions");
    let want: Vec<Vec<u32>> = bits.iter().map(|c| c[target as usize..target as usize + len].to_vec()).collect();
    assert_eq!(reconstructed, want);
}

#[test]
fn stdin_and_stdout() {
    let d = dir("pipes");
    let (src, want) = source(&d);
    let out = d.join("p.fak");
    let enc = Command::new(env!("CARGO_BIN_EXE_fak")).args(["encode", "-", s(&out), "-q"])
        .stdin(std::fs::File::open(&src).unwrap()).output().unwrap();
    assert!(enc.status.success(), "{}", String::from_utf8_lossy(&enc.stderr));
    let dec = Command::new(env!("CARGO_BIN_EXE_fak")).args(["decode", "-", "-", "-q"])
        .stdin(std::fs::File::open(&out).unwrap()).stdout(Stdio::piped()).output().unwrap();
    assert!(dec.status.success());
    let wav = fak::wav::read_wav_from(&dec.stdout[..], None).unwrap();
    assert_eq!(wav.channels, want);
}

#[test]
fn cue_sheet_embed_export_and_tracks() {
    let d = dir("cue");
    let (src, want) = source(&d);
    let cue = d.join("album.cue");
    std::fs::write(&cue, format!("\u{feff}{CUE}")).unwrap(); // with a byte-order mark, as Windows tools write
    let out = d.join("album.fak");
    ok(&["encode", s(&src), s(&out), "--cuesheet", s(&cue), "-T", "ALBUM=Album", "-q"]);

    // Exported verbatim (titles included), BOM stripped.
    let exported = ok(&["cue", s(&out)]).stdout;
    assert_eq!(String::from_utf8(exported).unwrap(), CUE);
    let info = String::from_utf8(ok(&["info", s(&out)]).stdout).unwrap();
    assert!(info.contains("cue sheet:   3 tracks") && info.contains("ISRC ABCDE1234567") && info.contains("ALBUM=Album"), "{info}");

    // Tracks run from INDEX 01 to the next track's INDEX 01 (track 3's pre-gap ends track 2).
    for (n, range) in [(1, 0..88_200), (2, 88_200..176_400), (3, 176_400..FRAMES)] {
        let t = d.join(format!("t{n}.wav"));
        ok(&["decode", s(&out), s(&t), "--track", &n.to_string(), "-q"]);
        let got = pcm(&t);
        let exp: Vec<Vec<i64>> = want.iter().map(|c| c[range.clone()].to_vec()).collect();
        assert_eq!(got, exp, "track {n}");
    }
    assert_eq!(fak(&["decode", s(&out), s(&d.join("x.wav")), "--track", "4"]).status.code(), Some(1));

    // A cue sheet for several files, or one pointing past the end, is refused.
    let bad = d.join("bad.cue");
    std::fs::write(&bad, "FILE \"a.wav\" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00\nFILE \"b.wav\" WAVE\nTRACK 02 AUDIO\nINDEX 01 00:00:00\n").unwrap();
    assert_eq!(fak(&["encode", s(&src), s(&d.join("b.fak")), "--cuesheet", s(&bad)]).status.code(), Some(1));
    std::fs::write(&bad, "FILE \"a.wav\" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00\nTRACK 02 AUDIO\nINDEX 01 00:09:00\n").unwrap();
    assert_eq!(fak(&["encode", s(&src), s(&d.join("b.fak")), "--cuesheet", s(&bad)]).status.code(), Some(1));
}

///  with `--cd-tags`, a 16/44.1 image whose cue sheet and length are whole CD sectors gets its
/// disc IDs and AccurateRip/CTDB checksums as tags (`fak::cdrip::disc_tags`); without it, none; a
/// tag the user set is kept, and an image that is not whole sectors gets none.
#[test]
fn cd_image_gets_disc_tags() {
    let d = dir("cdtags");
    let (src, want) = source(&d); // 6 s = 450 sectors; CUE's tracks start at sectors 0, 150, 300
    let cue = d.join("album.cue");
    std::fs::write(&cue, CUE).unwrap();
    let out = d.join("album.fak");
    // Opt-in: without --cd-tags an encode writes none.
    let plain = d.join("plain.fak");
    ok(&["encode", s(&src), s(&plain), "--cuesheet", s(&cue), "-q"]);
    let pinfo = String::from_utf8(ok(&["info", s(&plain)]).stdout).unwrap();
    assert!(!pinfo.contains("DISCID") && !pinfo.contains("FAK_"), "{pinfo}");

    ok(&["encode", s(&src), s(&out), "--cuesheet", s(&cue), "--cd-tags", "-T", "DISCID=mine", "-q"]);
    let info = String::from_utf8(ok(&["info", s(&out)]).stdout).unwrap();
    let tag = |k: &str| info.lines().find_map(|l| l.trim().strip_prefix(&format!("{k}="))).map(str::to_string);

    let toc = fak::cdrip::Toc::audio(&[0, 150, 300], 450);
    let frames = fak::cdrip::pack16(&want[0], &want[1]).unwrap();
    let v2: Vec<String> = (1..=3).map(|k| { let r = toc.track_frames(k); format!("{:08x}", fak::cdrip::accuraterip(&frames[r.start as usize..r.end as usize], k, 3).1) }).collect();
    assert_eq!(tag("DISCID").as_deref(), Some("mine"), "the user's own tag wins: {info}");
    assert_eq!(tag("MUSICBRAINZ_DISCID"), Some(toc.musicbrainz_id()));
    assert_eq!(tag("FAK_CD_TOC").as_deref(), Some("0:150:300:450"));
    assert_eq!(tag("FAK_ACCURATERIP_V2"), Some(v2.join(" ")));
    assert_eq!(tag("FAK_CTDB_CRC"), Some(format!("{:08x}", fak::cdrip::ctdb_crc(&frames))));
    assert_eq!(pcm_of(&out, &d), want, "the audio is untouched");

    // One sample short of whole sectors: not a CD image, no CD tags.
    let short = d.join("short.wav");
    fak::wav::write_wav(&short, &fak::wav::Wav { channels: want.iter().map(|c| c[..FRAMES - 1].to_vec()).collect(), sample_rate: RATE, bits: 16, channel_mask: None, float_info: None }).unwrap();
    let out2 = d.join("short.fak");
    ok(&["encode", s(&short), s(&out2), "--cuesheet", s(&cue), "--cd-tags", "-q"]);
    let info2 = String::from_utf8(ok(&["info", s(&out2)]).stdout).unwrap();
    assert!(!info2.contains("ACCURATERIP") && !info2.contains("FAK_CD_TOC"), "{info2}");
}

///  through `fak edit` (opt-in): `--cuesheet` alone adds no CD tags; `--cd-tags` writes them;
/// once a file has them, a moved cue sheet recomputes them; `--remove-cuesheet` drops the FAK_ ones
/// (the common keys may come from a ripper and stay), after which a new sheet adds none again; a
/// key the same command sets wins.
#[test]
fn edit_manages_cd_tags() {
    let d = dir("cdedit");
    let (src, want) = source(&d);
    let cue = d.join("album.cue");
    std::fs::write(&cue, CUE).unwrap();
    let moved = d.join("moved.cue");
    std::fs::write(&moved, CUE.replace("INDEX 01 00:02:00", "INDEX 01 00:02:01")).unwrap();
    let f = d.join("a.fak");
    ok(&["encode", s(&src), s(&f), "-q"]);
    let info = |p: &Path| String::from_utf8(ok(&["info", s(p)]).stdout).unwrap();
    let tag = |i: &str, k: &str| i.lines().find_map(|l| l.trim().strip_prefix(&format!("{k}="))).map(str::to_string);

    ok(&["edit", s(&f), "--cuesheet", s(&cue)]);
    assert!(tag(&info(&f), "FAK_CD_TOC").is_none() && tag(&info(&f), "DISCID").is_none(), "opt-in: a cue sheet alone adds none");

    ok(&["edit", s(&f), "--cd-tags", "--set-tag", "ACCURATERIPID=kept"]);
    let i = info(&f);
    let toc = fak::cdrip::Toc::audio(&[0, 150, 300], 450);
    let frames = fak::cdrip::pack16(&want[0], &want[1]).unwrap();
    assert_eq!(tag(&i, "FAK_CD_TOC").as_deref(), Some("0:150:300:450"), "{i}");
    assert_eq!(tag(&i, "FAK_CTDB_CRC"), Some(format!("{:08x}", fak::cdrip::ctdb_crc(&frames))));
    assert_eq!(tag(&i, "MUSICBRAINZ_DISCID"), Some(toc.musicbrainz_id()));
    assert_eq!(tag(&i, "ACCURATERIPID").as_deref(), Some("kept"));
    assert_eq!(i.matches("FAK_ACCURATERIP_V2=").count(), 1);

    ok(&["edit", s(&f), "--cuesheet", s(&moved)]);
    let i = info(&f);
    assert_eq!(tag(&i, "FAK_CD_TOC").as_deref(), Some("0:151:300:450"), "opted in: kept current: {i}");
    assert_eq!(tag(&i, "MUSICBRAINZ_DISCID"), Some(fak::cdrip::Toc::audio(&[0, 151, 300], 450).musicbrainz_id()));

    ok(&["edit", s(&f), "--remove-cuesheet"]);
    let i = info(&f);
    assert!(tag(&i, "FAK_CD_TOC").is_none() && tag(&i, "FAK_ACCURATERIP_V2").is_none(), "{i}");
    assert!(tag(&i, "MUSICBRAINZ_DISCID").is_some(), "the common keys stay");

    ok(&["edit", s(&f), "--cuesheet", s(&cue)]);
    assert!(tag(&info(&f), "FAK_CD_TOC").is_none(), "no FAK_ tags left, so no longer opted in");
    ok(&["edit", s(&f), "--cd-tags"]);
    let i = info(&f);
    assert_eq!(tag(&i, "FAK_CD_TOC").as_deref(), Some("0:150:300:450"), "{i}");
    assert!(tag(&i, "FAK_ACCURATERIP_V1").is_some());
    assert_eq!(pcm_of(&f, &d), want, "the audio is untouched");
}

fn pcm_of(fak_file: &Path, d: &Path) -> Vec<Vec<i64>> {
    let w = d.join("decoded.wav");
    ok(&["decode", s(fak_file), s(&w), "-q"]);
    pcm(&w)
}

#[test]
fn edit_changes_metadata_only() {
    let d = dir("edit");
    let (src, want) = source(&d);
    let out = d.join("e.fak");
    ok(&["encode", s(&src), s(&out), "-T", "TITLE=Old", "-T", "ARTIST=A", "-q"]);
    let cue = d.join("c.cue");
    std::fs::write(&cue, CUE).unwrap();

    ok(&["edit", s(&out), "--set-tag", "TITLE=New", "--tag", "ARTIST=B", "--cuesheet", s(&cue)]);
    let info = String::from_utf8(ok(&["info", s(&out)]).stdout).unwrap();
    assert!(info.contains("TITLE=New") && !info.contains("TITLE=Old") && info.contains("ARTIST=A") && info.contains("ARTIST=B"), "{info}");
    assert!(info.contains("cue sheet:   3 tracks"), "{info}");
    ok(&["verify", s(&out)]);
    let t2 = d.join("t2.wav");
    ok(&["decode", s(&out), s(&t2), "--track", "2", "-q"]);
    assert_eq!(pcm(&t2)[0], want[0][88_200..176_400].to_vec());

    // -o leaves the input alone; removing the cue sheet removes its tag and the tracks.
    let copy = d.join("copy.fak");
    ok(&["edit", s(&out), "--remove-cuesheet", "--remove-tag", "artist", "-o", s(&copy)]);
    let info = String::from_utf8(ok(&["info", s(&copy)]).stdout).unwrap();
    assert!(!info.contains("cue sheet") && !info.contains("CUESHEET") && !info.contains("ARTIST"), "{info}");
    assert!(String::from_utf8(ok(&["info", s(&out)]).stdout).unwrap().contains("cue sheet:   3 tracks"));
    let back = d.join("back.wav");
    ok(&["decode", s(&copy), s(&back), "-q"]);
    assert_eq!(pcm(&back), want);
    assert_eq!(fak(&["cue", s(&copy)]).status.code(), Some(1));
}

#[test]
fn usage_mistakes_exit_2() {
    let d = dir("usage");
    let (src, _) = source(&d);
    let out = d.join("u.fak");
    for args in [
        vec!["encode", s(&src)],
        vec!["encode", s(&src), s(&out), "--level", "ludicrous"],
        vec!["encode", s(&src), s(&out), "--bogus"],
        vec!["encode", s(&src), s(&out), "--threads", "0"],
        vec!["encode", s(&src), s(&out), "-T", "novalue"],
        vec!["decode", s(&out)],
        vec!["edit", s(&out)],
        vec!["frobnicate"],
    ] {
        let o = fak(&args);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
    assert!(ok(&["help"]).stdout.starts_with(b"FAK lossless audio codec"));
    assert!(ok(&["--version"]).stdout.starts_with(b"fak "));
    // A missing file is a failure, not a usage mistake.
    assert_eq!(fak(&["decode", s(&d.join("missing.fak")), s(&d.join("x.wav"))]).status.code(), Some(1));
}

