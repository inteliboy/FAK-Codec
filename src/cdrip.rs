//!: the footprint of a CD rip -- disc identifiers from the table of
//! contents, AccurateRip and CUETools DB (CTDB) checksums, and CD-origin evidence -- computed
//! offline from the audio. Nothing here touches the network; looking the values up in the
//! databases is a separate, explicit step (`tools/cd/cd_lookup.py`).
//!
//! Definitions, from public descriptions and checked against their reference implementations'
//! published test values (the unit tests below), not copied from any implementation:
//!
//! * **TOC.** Track starts (INDEX 01) and the lead-out, in 1/75-s sectors of 588 stereo frames,
//!   counted from the first sector after the 2-s lead-in (logical block addresses).
//! * **AccurateRip disc IDs** (as whipper and EAC form the database path): `id1` = sum of the
//!   audio tracks' starts plus the lead-out; `id2` = sum of `max(start, 1) * track number` plus
//!   `lead-out * (audio tracks + 1)`, both mod 2^32. Data tracks are skipped but still count for
//!   the lead-out (the disc's real one).
//! * **freedb (CDDB) disc ID**: `(n mod 255) << 24 | t << 8 | tracks`, `n` the sum of the decimal
//!   digit sums of every track's start in whole seconds (+2 s lead-in), `t` = lead-out seconds
//!   minus first-track seconds, every track (data too).
//! * **MusicBrainz disc ID**: SHA-1 of the uppercase hex string of first track (2 digits), last
//!   audio track (2), lead-out + 150 (8), then 99 track offsets + 150 (8 each, zero-filled), in
//!   base64 with `+/=` replaced by `._-`. With a trailing data track, the "lead-out" is the data
//!   track's start - 11400 (the enhanced-CD session gap).
//! * **AccurateRip checksums v1/v2** of one track: every stereo frame as a u32 (left in the low
//!   16 bits) times its 1-based position within the track, summed mod 2^32 (v1); v2 also adds
//!   the high 32 bits of each 64-bit product. The first track skips positions below 5 * 588, the
//!   last track positions above `len - 5 * 588` (drive offsets make the disc's edges unreadable).
//! * **CTDB checksum** of a disc: standard CRC-32 (`crc::crc32`) of the audio's bytes (16-bit
//!   little-endian, interleaved) from the start of track 1 to the end of the last audio track,
//!   leaving out the first 10 sectors (5880 frames) and the last `5880 + len mod 5880` frames
//!   (CUETools' `CTDBCRC` with its 10-sector repair stride).
use crate::crc;

/// Stereo sample-frames per CD sector (44100 / 75).
pub const SECTOR_FRAMES: u64 = 588;
/// Frames AccurateRip skips at the start of the first track and the end of the last.
const AR_SKIP: u64 = 5 * SECTOR_FRAMES;
/// CTDB's edge exclusion (half its 10-sector stride, in frames).
const CTDB_EDGE: u64 = 10 * SECTOR_FRAMES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TocTrack { pub start: u32, pub audio: bool }

/// A CD table of contents: tracks in order (numbered from 1), and the lead-out, in sectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toc { pub tracks: Vec<TocTrack>, pub leadout: u32 }

impl Toc {
    /// An all-audio disc from its track starts and lead-out.
    pub fn audio(starts: &[u32], leadout: u32) -> Toc {
        Toc { tracks: starts.iter().map(|&start| TocTrack { start, audio: true }).collect(), leadout }
    }

    pub fn audio_tracks(&self) -> usize { self.tracks.iter().filter(|t| t.audio).count() }

    /// End (exclusive) of the last audio track: the lead-out, or before a trailing data track's
    /// session gap on an enhanced CD.
    pub fn audio_end(&self) -> u32 {
        match self.tracks.last() {
            Some(t) if !t.audio => t.start - 11400,
            _ => self.leadout,
        }
    }

    pub fn accuraterip_ids(&self) -> (u32, u32) {
        let (mut id1, mut id2) = (0u32, 0u32);
        for (i, t) in self.tracks.iter().enumerate().filter(|(_, t)| t.audio) {
            id1 = id1.wrapping_add(t.start);
            id2 = id2.wrapping_add(t.start.max(1).wrapping_mul(i as u32 + 1));
        }
        id1 = id1.wrapping_add(self.leadout);
        id2 = id2.wrapping_add(self.leadout.wrapping_mul(self.audio_tracks() as u32 + 1));
        (id1, id2)
    }

    pub fn freedb_id(&self) -> u32 {
        let digit_sum = |mut v: u32| { let mut s = 0; while v > 0 { s += v % 10; v /= 10; } s };
        let n: u32 = self.tracks.iter().map(|t| digit_sum((t.start + 150) / 75)).sum();
        let t = self.leadout / 75 - self.tracks.first().map_or(0, |t| t.start / 75);
        ((n % 0xff) << 24) | (t << 8) | self.tracks.len() as u32
    }

    /// Path of this disc's entry under `http://www.accuraterip.com/accuraterip/`.
    pub fn accuraterip_path(&self) -> String {
        let (id1, id2) = self.accuraterip_ids();
        let h = format!("{id1:08x}");
        let c: Vec<char> = h.chars().collect();
        format!("{}/{}/{}/dBAR-{:03}-{id1:08x}-{id2:08x}-{:08x}.bin", c[7], c[6], c[5], self.audio_tracks(), self.freedb_id())
    }

    pub fn musicbrainz_id(&self) -> String {
        let audio: Vec<u32> = self.tracks.iter().filter(|t| t.audio).map(|t| t.start).collect();
        let lead = match self.tracks.last() { Some(t) if !t.audio => t.start - 11400, _ => self.leadout };
        let mut s = format!("{:02X}{:02X}{:08X}", 1, audio.len(), lead + 150);
        for i in 0..99 { s += &format!("{:08X}", audio.get(i).map_or(0, |&o| o + 150)); }
        let b64 = base64(&sha1(s.as_bytes()));
        b64.chars().map(|c| match c { '+' => '.', '/' => '_', '=' => '-', c => c }).collect()
    }

    /// The `toc` parameter of a CTDB lookup: every track's start (data tracks prefixed `-`), then
    /// the lead-out, separated by `:`.
    pub fn ctdb_toc(&self) -> String {
        let mut s: String = self.tracks.iter().map(|t| format!("{}{}:", if t.audio { "" } else { "-" }, t.start)).collect();
        s += &self.leadout.to_string();
        s
    }

    /// Frame range of audio track `n` (1-based) in a disc image whose first frame is sector 0
    /// (so audio before track 1's INDEX 01 -- a hidden track -- is in the image but in no track).
    pub fn track_frames(&self, n: usize) -> std::ops::Range<u64> {
        let audio: Vec<&TocTrack> = self.tracks.iter().filter(|t| t.audio).collect();
        let end = audio.get(n).map_or(self.audio_end() as u64, |t| t.start as u64);
        audio[n - 1].start as u64 * SECTOR_FRAMES..end * SECTOR_FRAMES
    }

    /// Frame range from track 1's INDEX 01 to the end of the last audio track: what CTDB checks.
    pub fn disc_frames(&self) -> std::ops::Range<u64> {
        self.track_frames(1).start..self.track_frames(self.audio_tracks()).end
    }

    /// The TOC of a single-file image from its cue sheet, if every position is a whole number of
    /// sectors at `sample_rate` (true for a CD image at 44.1 kHz; `None` otherwise -- itself
    /// evidence against CD origin). The image is assumed to start at track 1's INDEX 01 or its
    /// INDEX 00 (pre-gap included); all tracks are audio.
    pub fn from_cue(cue: &crate::metadata::CueSheet, sample_rate: u32, total_frames: u64) -> Option<Toc> {
        let to_sector = |f: u64| (f * 75 % sample_rate as u64 == 0).then(|| (f * 75 / sample_rate as u64) as u32);
        let mut starts = Vec::new();
        for t in &cue.tracks {
            let i1 = t.indices.iter().find(|i| i.number == 1)?;
            starts.push(to_sector(i1.sample_offset)?);
        }
        let leadout = to_sector(total_frames)?;
        (!starts.is_empty()).then(|| Toc::audio(&starts, leadout))
    }
}

/// Tag keys [`disc_tags`] writes. The first three follow existing practice (freedb's `DISCID` as
/// CUETools and foobar2000 write it, MusicBrainz Picard's `MUSICBRAINZ_DISCID`, CUETools'
/// `ACCURATERIPID`); the rest have no common convention and carry the `FAK_` prefix.
pub const TAG_FREEDB: &str = "DISCID";
pub const TAG_MUSICBRAINZ: &str = "MUSICBRAINZ_DISCID";
pub const TAG_ACCURATERIP_ID: &str = "ACCURATERIPID";
/// The TOC in CTDB's lookup syntax, so the disc can be looked up without the cue sheet.
pub const TAG_TOC: &str = "FAK_CD_TOC";
/// Per-track AccurateRip checksums, 8 hex digits each, space-separated in track order.
pub const TAG_AR_V1: &str = "FAK_ACCURATERIP_V1";
pub const TAG_AR_V2: &str = "FAK_ACCURATERIP_V2";
/// The CTDB whole-disc CRC (8 hex digits). Computed at read offset 0;
///  for why this is not yet validated against the database.
pub const TAG_CTDB_CRC: &str = "FAK_CTDB_CRC";
pub const DISC_TAGS: [&str; 7] = [TAG_FREEDB, TAG_MUSICBRAINZ, TAG_ACCURATERIP_ID, TAG_TOC, TAG_AR_V1, TAG_AR_V2, TAG_CTDB_CRC];

/// The CD identifiers and checksums of a disc image, as `(key, value)` tags -- only when they can be
/// exact: 16-bit 44.1 kHz stereo audio and a cue sheet whose every position (and the image's
/// length) is a whole number of sectors, the image starting at sector 0. `None` otherwise, which
/// is also what a hi-res or resampled "CD image" gets.
pub fn disc_tags(channels: &[Vec<i64>], sample_rate: u32, bits: u8, cue: &crate::metadata::CueSheet) -> Option<Vec<(&'static str, String)>> {
    if channels.len() != 2 || sample_rate != 44100 || bits != 16 { return None; }
    let toc = Toc::from_cue(cue, sample_rate, channels[0].len() as u64)?;
    let frames = pack16(&channels[0], &channels[1])?;
    let n = toc.audio_tracks();
    let sums: Vec<(u32, u32)> = (1..=n).map(|k| { let r = toc.track_frames(k); accuraterip(&frames[r.start as usize..r.end as usize], k, n) }).collect();
    let hex = |v: &mut dyn Iterator<Item = u32>| v.map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ");
    let (id1, id2) = toc.accuraterip_ids();
    let d = toc.disc_frames();
    Some(vec![
        (TAG_FREEDB, format!("{:08x}", toc.freedb_id())),
        (TAG_MUSICBRAINZ, toc.musicbrainz_id()),
        (TAG_ACCURATERIP_ID, format!("{n:03}-{id1:08x}-{id2:08x}-{:08x}", toc.freedb_id())),
        (TAG_TOC, toc.ctdb_toc()),
        (TAG_AR_V1, hex(&mut sums.iter().map(|s| s.0))),
        (TAG_AR_V2, hex(&mut sums.iter().map(|s| s.1))),
        (TAG_CTDB_CRC, format!("{:08x}", ctdb_crc(&frames[d.start as usize..d.end as usize]))),
    ])
}

fn tag_key_is(t: &str, key: &str) -> bool { t.split_once('=').map_or(t, |(k, _)| k).eq_ignore_ascii_case(key) }

/// The four tags that only describe a TOC (removed with the cue sheet); the other three may come
/// from a ripper and are left alone then.
pub const FAK_DISC_TAGS: [&str; 4] = [TAG_TOC, TAG_AR_V1, TAG_AR_V2, TAG_CTDB_CRC];

/// Sets `m`'s CD tags for this audio and `m`'s cue sheet: every CD tag not named in `keep` is
/// removed, then the computed ones are added -- none when the audio and cue sheet are not an exact
/// CD image, so stale values from an earlier cue sheet never survive. Shared by `fak encode
/// --cd-tags`, `fak edit` and the foobar2000 component, so they follow one rule.
pub fn apply_disc_tags(m: &mut crate::metadata::Metadata, channels: &[Vec<i64>], sample_rate: u32, bits: u8, keep: &[String]) {
    let kept = |k: &str| keep.iter().any(|u| u.eq_ignore_ascii_case(k));
    m.tags.retain(|t| !DISC_TAGS.iter().any(|k| !kept(k) && tag_key_is(t, k)));
    let Some(cue) = &m.cue_sheet else { return };
    for (k, v) in disc_tags(channels, sample_rate, bits, cue).unwrap_or_default() {
        if !kept(k) { m.tags.push(format!("{k}={v}")); }
    }
}

/// Whether `m` has any of the [`FAK_DISC_TAGS`]: CD tags are opt-in, and these (never written by
/// rippers) mark a file whose owner asked for them, so later cue sheet changes keep them current.
pub fn has_fak_disc_tags(m: &crate::metadata::Metadata) -> bool {
    m.tags.iter().any(|t| FAK_DISC_TAGS.iter().any(|k| tag_key_is(t, k)))
}

/// Removes the [`FAK_DISC_TAGS`] not named in `keep` (the cue sheet is gone).
pub fn drop_fak_disc_tags(m: &mut crate::metadata::Metadata, keep: &[String]) {
    m.tags.retain(|t| !FAK_DISC_TAGS.iter().any(|k| tag_key_is(t, k) && !keep.iter().any(|u| u.eq_ignore_ascii_case(k))));
}

/// Whether a cue sheet change moves any track or index point (and so the TOC): titles, ISRCs and
/// other text do not. Hosts that rewrite the sheet on every tag edit use this to recompute the CD
/// tags (which needs decoding) only when it matters.
pub fn cue_points_changed(old: Option<&crate::metadata::CueSheet>, new: Option<&crate::metadata::CueSheet>) -> bool {
    let points = |c: Option<&crate::metadata::CueSheet>| c.map(|c| c.tracks.iter().map(|t| (t.number, t.indices.clone())).collect::<Vec<_>>());
    points(old) != points(new)
}

/// Whether a length in stereo frames is a whole number of CD sectors.
pub fn sector_aligned(frames: u64) -> bool { frames % SECTOR_FRAMES == 0 }

/// 16-bit stereo frames packed as AccurateRip reads them (left in the low half). `None` unless
/// both channels hold 16-bit samples.
pub fn pack16(left: &[i64], right: &[i64]) -> Option<Vec<u32>> {
    let ok = |v: i64| (-32768..=32767).contains(&v);
    left.iter().zip(right).map(|(&l, &r)| (ok(l) && ok(r)).then(|| (l as u16 as u32) | ((r as u16 as u32) << 16))).collect()
}

/// AccurateRip v1 and v2 checksums of audio track `track` (1-based) of `tracks` audio tracks.
pub fn accuraterip(frames: &[u32], track: usize, tracks: usize) -> (u32, u32) {
    let n = frames.len() as u64;
    let from = if track == 1 { AR_SKIP } else { 0 };
    let to = if track == tracks { n.saturating_sub(AR_SKIP) } else { n };
    let (mut lo, mut hi) = (0u32, 0u32);
    for (i, &f) in frames.iter().enumerate() {
        let pos = i as u64 + 1;
        if pos >= from && pos <= to {
            let p = f as u64 * pos;
            lo = lo.wrapping_add(p as u32);
            hi = hi.wrapping_add((p >> 32) as u32);
        }
    }
    (lo, lo.wrapping_add(hi))
}

/// CTDB checksum of a whole disc's audio (track 1's start to the end of the last audio track).
pub fn ctdb_crc(disc: &[u32]) -> u32 {
    let n = disc.len() as u64;
    let tail = CTDB_EDGE + n % CTDB_EDGE;
    if n < CTDB_EDGE + tail { return 0; }
    let bytes: Vec<u8> = disc[CTDB_EDGE as usize..(n - tail) as usize].iter().flat_map(|f| f.to_le_bytes()).collect();
    crc::crc32(&bytes)
}

/// SHA-1 (FIPS 180-4), for the MusicBrainz disc ID only.
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 { w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap()); }
        for i in 16..80 { w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1); }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i / 20 {
                0 => ((b & c) | (!b & d), 0x5A827999),
                1 => (b ^ c ^ d, 0x6ED9EBA1),
                2 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d; d = c; c = b.rotate_left(30); b = a; a = t;
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e]) { *x = x.wrapping_add(v); }
    }
    let mut out = [0u8; 20];
    for (i, v) in h.iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes()); }
    out
}

fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let v = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { A[(v >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ladyhawke (0602517818866): 12 audio tracks and a data track. Expected values from whipper's
    /// test suite (verified there against freedb, EAC's AccurateRip request and mb-submit-disc).
    fn ladyhawke() -> Toc {
        let starts = [0, 15537, 31691, 50866, 66466, 81202, 99409, 115920, 133093, 149847, 161560, 177682, 207106];
        Toc { tracks: starts.iter().enumerate().map(|(i, &start)| TocTrack { start, audio: i < 12 }).collect(), leadout: 210385 }
    }

    #[test]
    fn disc_ids_enhanced_cd() {
        let t = ladyhawke();
        assert_eq!(t.accuraterip_ids(), (0x0013bd5a, 0x00b8d489));
        assert_eq!(t.freedb_id(), 0xc60af50d);
        assert_eq!(t.accuraterip_path(), "a/5/d/dBAR-012-0013bd5a-00b8d489-c60af50d.bin");
        assert_eq!(t.musicbrainz_id(), "KnpGsLhvH.lPrNc1PBL21lb9Bg4-");
        assert_eq!(t.ctdb_toc(), "0:15537:31691:50866:66466:81202:99409:115920:133093:149847:161560:177682:-207106:210385");
    }

    #[test]
    fn musicbrainz_id_audio_cd() {
        // Ettella Diamant, the example in MusicBrainz's "Disc ID Calculation" documentation.
        let t = Toc::audio(&[0, 15213, 32164, 46442, 63264, 80339], 95312);
        assert_eq!(t.musicbrainz_id(), "49HHV7Eb8UKF3aQiNmu1GR8vKTY-");
    }

    #[test]
    fn sha1_and_base64() {
        let hex: String = sha1(b"abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn accuraterip_skips_disc_edges() {
        let frames: Vec<u32> = (0..10_000u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let direct = |from: u64, to: u64| -> (u32, u32) {
            let (mut lo, mut hi) = (0u64, 0u64);
            for (i, &f) in frames.iter().enumerate() {
                let p = i as u64 + 1;
                if p >= from && p <= to { let m = f as u64 * p; lo += m & 0xffff_ffff; hi += m >> 32; }
            }
            (lo as u32, (lo + hi) as u32)
        };
        assert_eq!(accuraterip(&frames, 2, 3), direct(0, 10_000));
        assert_eq!(accuraterip(&frames, 1, 3), direct(2940, 10_000));
        assert_eq!(accuraterip(&frames, 3, 3), direct(0, 10_000 - 2940));
        assert_eq!(accuraterip(&frames, 1, 1), direct(2940, 10_000 - 2940));
    }

    #[test]
    fn disc_tags_only_for_exact_cd_images() {
        use crate::metadata::{CueIndex, CueSheet, CueTrack};
        let track = |number, sector: u64| CueTrack { number, isrc: String::new(), indices: vec![CueIndex { number: 1, sample_offset: sector * 588 }] };
        let cue = CueSheet { catalog: String::new(), tracks: vec![track(1, 0), track(2, 30), track(3, 70)] };
        let n: i64 = 100 * 588;
        let l: Vec<i64> = (0..n).map(|i| (i * 7919) % 65536 - 32768).collect();
        let r: Vec<i64> = (0..n).map(|i| (i * 104729) % 65536 - 32768).collect();
        let tags = disc_tags(&[l.clone(), r.clone()], 44100, 16, &cue).expect("a CD image");
        let get = |k: &str| tags.iter().find(|t| t.0 == k).unwrap().1.clone();
        let toc = Toc::audio(&[0, 30, 70], 100);
        assert_eq!(get(TAG_TOC), "0:30:70:100");
        assert_eq!(get(TAG_ACCURATERIP_ID), format!("003-{}", &toc.accuraterip_path()[15..41]));
        let f = pack16(&l, &r).unwrap();
        let v2: Vec<String> = [(0, 30, 1), (30, 70, 2), (70, 100, 3)].iter()
            .map(|&(a, b, k)| format!("{:08x}", accuraterip(&f[a * 588..b * 588], k, 3).1)).collect();
        assert_eq!(get(TAG_AR_V2), v2.join(" "));
        assert_eq!(get(TAG_CTDB_CRC), format!("{:08x}", ctdb_crc(&f)));
        assert!(disc_tags(&[l.clone(), r.clone()], 48000, 16, &cue).is_none());
        assert!(disc_tags(&[l.clone(), r.clone()], 44100, 24, &cue).is_none());
        assert!(disc_tags(&[l[1..].to_vec(), r[1..].to_vec()], 44100, 16, &cue).is_none(), "length not whole sectors");
    }

    #[test]
    fn toc_from_cue_needs_whole_sectors() {
        use crate::metadata::{CueIndex, CueSheet, CueTrack};
        let track = |number, off| CueTrack { number, isrc: String::new(), indices: vec![CueIndex { number: 1, sample_offset: off }] };
        let cue = CueSheet { catalog: String::new(), tracks: vec![track(1, 0), track(2, 15537 * 588)] };
        assert_eq!(Toc::from_cue(&cue, 44100, 31691 * 588), Some(Toc::audio(&[0, 15537], 31691)));
        assert_eq!(Toc::from_cue(&cue, 44100, 31691 * 588 + 1), None);
        assert_eq!(Toc::from_cue(&cue, 96000, 31691 * 1280), None, "15537 * 588 frames is not a sector at 96 kHz");
    }
}
