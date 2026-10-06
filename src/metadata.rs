//! Optional tag/artwork metadata block (format
//! version 3). Field layout deliberately mirrors two existing,
//! well-understood conventions rather than inventing a bespoke scheme: Vorbis
//! Comment (vendor string + freeform `KEY=VALUE` UTF-8 pairs -- the same tagging model FLAC, Opus
//! and Vorbis all already use) for text tags, and FLAC's own `METADATA_BLOCK_PICTURE` layout
//! field-for-field for embedded artwork. This buys a parser that's easy to write correctly and
//! easy to port to a real library later -- it does *not* mean existing tools read `.fak` files
//! without adding dedicated support, since the surrounding container (chunk table, CRCs) is this
//! project's own, not FLAC's or Ogg's.
//!
//! Placement in the stream: an optional block right after the fixed stream header and before the
//! first self-delimited chunk, so a reader that only wants tags
//! never touches audio data and one that only wants audio can skip it in a single length-prefixed
//! jump.
use crate::crc::crc32;
use crate::floatpcm::{FloatException, FloatInfo};

/// Defensive bounds: a hostile file's declared counts/lengths are capped before
/// they're trusted for allocation, the same discipline `format.rs`'s chunk framing already applies.
pub const MAX_TAGS: usize = 4096;
pub const MAX_TAG_LEN: usize = 1 << 16; // 64 KiB per "KEY=VALUE" string
pub const MAX_PICTURES: usize = 64;
pub const MAX_PICTURE_BYTES: usize = 64 << 20; // 64 MiB per picture
pub const MAX_TRACKS: usize = 999; // Red Book's own track-count ceiling; generous for any real use
pub const MAX_INDICES_PER_TRACK: usize = 100;
pub const MAX_CATALOG_LEN: usize = 64; // real UPC/EAN catalog numbers are 13 ASCII digits
pub const MAX_ISRC_LEN: usize = 64; // real ISRCs are 12 ASCII characters
/// (b): a hostile declared exception count is capped well below what would make
/// `MAX_METADATA_BLOCK_LEN` the binding limit instead (16 bytes/entry: channel u32 + index u64 +
/// bits u32) -- real grid-aligned float material needs at most a handful of these per file.
pub const MAX_FLOAT_EXCEPTIONS: usize = 1 << 22;
/// Overall cap on the metadata block's declared body length. The in-memory `read_block` doesn't
/// strictly need this (its declared length is already bounded by the real `data.len()` a hostile
/// *file* can only make so large), but `decoder::decode_stream`'s streaming reader has no such
/// bound -- it must allocate a buffer for the declared length before it can read and validate
/// anything inside it, so an unbounded declared length is a real allocation-bomb vector against a
/// live peer in a way it isn't for a file already fully read into memory. 128 MiB
/// is generous for any real use (`MAX_PICTURES` alone could in principle reach several GiB, but no
/// legitimate file needs anywhere close to that many maximum-sized pictures).
pub const MAX_METADATA_BLOCK_LEN: usize = 128 << 20;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Metadata {
    pub vendor: String,
    /// Each entry is a whole `KEY=VALUE` string, matching Vorbis Comment's own convention -- keys
    /// are freeform by convention (`TITLE`, `ARTIST`, `ALBUM`, `DATE`, `TRACKNUMBER`, `GENRE`, ...),
    /// not a fixed enum, and repeated keys (e.g. multiple `ARTIST=`) are allowed, again matching
    /// Vorbis Comment.
    pub tags: Vec<String>,
    pub pictures: Vec<Picture>,
    /// Track/index points within this file's one continuous stream -- `None` when there's no cue sheet, distinct from
    /// `Some(CueSheet { tracks: vec![], .. })` (present but empty), since a cue sheet with zero
    /// tracks is meaningless and not worth distinguishing from absent on disk (both serialize the
    /// same way, see `write`/`read`).
    pub cue_sheet: Option<CueSheet>,
    /// `dwChannelMask` from the source WAV's `WAVE_FORMAT_EXTENSIBLE` fmt chunk, if it had one
    /// -- which physical channel is front-left,
    /// front-right, LFE, etc. `None` (not `Some(0)`) means the source had no mask at all (a plain
    /// non-extensible fmt chunk), distinct from a genuinely-declared `Some(0)` ("no defined
    /// layout"). This is metadata only: the codec itself still has no cross-channel prediction for
    /// channel counts other than 2 -- storing the mask preserves
    /// which channel is which so a player can route them correctly, it doesn't compress them
    /// better.
    pub channel_mask: Option<u32>,
    /// (b), format version 18: present iff the source was 32-bit float PCM, reduced to the
    /// integer PCM domain by `floatpcm::map_to_pcm` -- the scale and any non-grid sample exceptions
    /// needed to invert that mapping bit-exactly on decode (`floatpcm::unmap_from_pcm`). `None` for
    /// every integer-PCM source, which is unaffected by this field's existence.
    pub float_info: Option<FloatInfo>,
}

/// Track/index points for one continuous audio stream, the same model a standard `.cue` sheet
/// file describes (`FILE ... TRACK ... INDEX ...`), stored as sample-frame offsets (this format's
/// native unit, `format.rs`) rather than CD-style 1/75s frames -- converting once at import time
/// (`fak::metadata::parse_cue_text`) avoids a lossy unit conversion at every seek.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CueSheet {
    pub catalog: String, // whole-disc UPC/EAN, empty if none
    pub tracks: Vec<CueTrack>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueTrack {
    pub number: u8,
    pub isrc: String, // empty if none
    pub indices: Vec<CueIndex>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueIndex {
    pub number: u8, // 0 = pre-gap, 1 = track start, 2+ = sub-indices, matching CUE sheet convention
    pub sample_offset: u64, // sample-frames from the start of the stream
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PictureType {
    Other = 0,
    FrontCover = 3,
    BackCover = 4,
    Artist = 8,
}

impl PictureType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            3 => Self::FrontCover,
            4 => Self::BackCover,
            8 => Self::Artist,
            _ => Self::Other, // any other FLAC picture-type value is preserved as-is, not rejected
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub kind: PictureType,
    pub kind_raw: u8, // the exact on-disk type byte, so an unrecognized-but-valid FLAC type round-trips
    pub mime: String,
    pub description: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub colors: u32, // 0 for non-palette images, matching METADATA_BLOCK_PICTURE
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct MetadataError(pub String);
impl std::fmt::Display for MetadataError { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "{}", self.0) } }
impl std::error::Error for MetadataError {}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl Metadata {
    pub fn is_empty(&self) -> bool {
        self.vendor.is_empty() && self.tags.is_empty() && self.pictures.is_empty() && self.cue_sheet.is_none()
            && self.channel_mask.is_none() && self.float_info.is_none()
    }

    /// Serializes to exactly what `read` parses: not including the outer block-length prefix
    /// (the caller wraps this, matching the chunk table's own length-then-CRC convention).
    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_str(&mut out, &self.vendor);
        out.extend_from_slice(&(self.tags.len() as u32).to_le_bytes());
        for t in &self.tags { write_str(&mut out, t); }
        out.extend_from_slice(&(self.pictures.len() as u32).to_le_bytes());
        for p in &self.pictures {
            out.push(p.kind_raw);
            write_str(&mut out, &p.mime);
            write_str(&mut out, &p.description);
            for v in [p.width, p.height, p.depth, p.colors] { out.extend_from_slice(&v.to_le_bytes()); }
            out.extend_from_slice(&(p.data.len() as u32).to_le_bytes());
            out.extend_from_slice(&p.data);
        }
        // Cue sheet: a track count of 0 means "absent", so `None` and `Some(CueSheet{tracks:
        // vec![], ..})` serialize identically -- a cue sheet with no tracks carries no information
        // either way, so there's no need for a separate presence flag.
        let tracks: &[CueTrack] = self.cue_sheet.as_ref().map_or(&[], |c| &c.tracks);
        write_str(&mut out, self.cue_sheet.as_ref().map_or("", |c| &c.catalog));
        out.extend_from_slice(&(tracks.len() as u32).to_le_bytes());
        for t in tracks {
            out.push(t.number);
            write_str(&mut out, &t.isrc);
            out.extend_from_slice(&(t.indices.len() as u32).to_le_bytes());
            for idx in &t.indices {
                out.push(idx.number);
                out.extend_from_slice(&idx.sample_offset.to_le_bytes());
            }
        }
        // Presence byte + value, not a sentinel: 0 is a real, legal declared channel mask
        // ("no defined layout"), distinct from no mask field being present at all.
        out.push(if self.channel_mask.is_some() { 1 } else { 0 });
        out.extend_from_slice(&self.channel_mask.unwrap_or(0).to_le_bytes());
        out.push(if self.float_info.is_some() { 1 } else { 0 });
        if let Some(fi) = &self.float_info {
            out.push(fi.scale_exp);
            out.extend_from_slice(&(fi.exceptions.len() as u32).to_le_bytes());
            for e in &fi.exceptions {
                out.extend_from_slice(&e.channel.to_le_bytes());
                out.extend_from_slice(&e.index.to_le_bytes());
                out.extend_from_slice(&e.bits.to_le_bytes());
            }
        }
        out
    }

    /// Parses the body written by `write` (i.e. `data[..]` with no outer length prefix left).
    /// Every declared count/length is bounds-checked against what's actually present and against
    /// the `MAX_*` limits above before being trusted for allocation or a string conversion
    /// -- a corrupted or hostile metadata block must error, never panic or OOM.
    pub fn read(data: &[u8]) -> Result<Self, MetadataError> {
        let err = |m: &str| MetadataError(format!("{m} (corrupted metadata?)"));
        let mut pos = 0usize;
        let take = |pos: &mut usize, n: usize, data: &[u8]| -> Result<Vec<u8>, MetadataError> {
            if *pos + n > data.len() { return Err(err("metadata block truncated")); }
            let s = data[*pos..*pos + n].to_vec();
            *pos += n;
            Ok(s)
        };
        let take_u32 = |pos: &mut usize, data: &[u8]| -> Result<u32, MetadataError> {
            Ok(u32::from_le_bytes(take(pos, 4, data)?.try_into().unwrap()))
        };
        let take_str = |pos: &mut usize, data: &[u8], max_len: usize| -> Result<String, MetadataError> {
            let len = take_u32(pos, data)? as usize;
            if len > max_len { return Err(err("metadata string exceeds sanity bound")); }
            String::from_utf8(take(pos, len, data)?).map_err(|_| err("metadata string is not valid UTF-8"))
        };

        let vendor = take_str(&mut pos, data, MAX_TAG_LEN)?;
        let tag_count = take_u32(&mut pos, data)? as usize;
        if tag_count > MAX_TAGS { return Err(err("tag count exceeds sanity bound")); }
        let mut tags = Vec::with_capacity(tag_count);
        for _ in 0..tag_count { tags.push(take_str(&mut pos, data, MAX_TAG_LEN)?); }

        let pic_count = take_u32(&mut pos, data)? as usize;
        if pic_count > MAX_PICTURES { return Err(err("picture count exceeds sanity bound")); }
        let mut pictures = Vec::with_capacity(pic_count);
        for _ in 0..pic_count {
            let kind_raw = take(&mut pos, 1, data)?[0];
            let mime = take_str(&mut pos, data, MAX_TAG_LEN)?;
            let description = take_str(&mut pos, data, MAX_TAG_LEN)?;
            let width = take_u32(&mut pos, data)?;
            let height = take_u32(&mut pos, data)?;
            let depth = take_u32(&mut pos, data)?;
            let colors = take_u32(&mut pos, data)?;
            let data_len = take_u32(&mut pos, data)? as usize;
            if data_len > MAX_PICTURE_BYTES { return Err(err("picture data exceeds sanity bound")); }
            let pic_data = take(&mut pos, data_len, data)?;
            pictures.push(Picture { kind: PictureType::from_u8(kind_raw), kind_raw, mime, description, width, height, depth, colors, data: pic_data });
        }

        let catalog = take_str(&mut pos, data, MAX_CATALOG_LEN)?;
        let track_count = take_u32(&mut pos, data)? as usize;
        if track_count > MAX_TRACKS { return Err(err("cue sheet track count exceeds sanity bound")); }
        let mut tracks = Vec::with_capacity(track_count);
        for _ in 0..track_count {
            let number = take(&mut pos, 1, data)?[0];
            let isrc = take_str(&mut pos, data, MAX_ISRC_LEN)?;
            let index_count = take_u32(&mut pos, data)? as usize;
            if index_count > MAX_INDICES_PER_TRACK { return Err(err("cue sheet index count exceeds sanity bound")); }
            let mut indices = Vec::with_capacity(index_count);
            for _ in 0..index_count {
                let idx_number = take(&mut pos, 1, data)?[0];
                let sample_offset = u64::from_le_bytes(take(&mut pos, 8, data)?.try_into().unwrap());
                indices.push(CueIndex { number: idx_number, sample_offset });
            }
            tracks.push(CueTrack { number, isrc, indices });
        }
        let cue_sheet = if tracks.is_empty() { None } else { Some(CueSheet { catalog, tracks }) };

        let mask_present = take(&mut pos, 1, data)?[0] != 0;
        let mask_value = take_u32(&mut pos, data)?;
        let channel_mask = if mask_present { Some(mask_value) } else { None };

        let float_present = take(&mut pos, 1, data)?[0] != 0;
        let float_info = if float_present {
            let scale_exp = take(&mut pos, 1, data)?[0];
            let exc_count = take_u32(&mut pos, data)? as usize;
            if exc_count > MAX_FLOAT_EXCEPTIONS { return Err(err("float exception count exceeds sanity bound")); }
            let mut exceptions = Vec::with_capacity(exc_count);
            for _ in 0..exc_count {
                let channel = take_u32(&mut pos, data)?;
                let index = u64::from_le_bytes(take(&mut pos, 8, data)?.try_into().unwrap());
                let bits = take_u32(&mut pos, data)?;
                exceptions.push(FloatException { channel, index, bits });
            }
            Some(FloatInfo { scale_exp, exceptions })
        } else { None };

        if pos != data.len() { return Err(err("trailing bytes after metadata block")); }
        Ok(Metadata { vendor, tags, pictures, cue_sheet, channel_mask, float_info })
    }
}

/// Parses a real, standard `.cue` sheet file's text (`CATALOG`/`TRACK NN <type>`/`INDEX NN
/// MM:SS:FF`/`ISRC`; `TITLE`/`PERFORMER`/`REM`/anything else is recognized and ignored, not
/// an error, since real `.cue` files are full of them) into a [`CueSheet`], converting CD-standard
/// `MM:SS:FF` timestamps (`FF` = 1/75s frames, *not* this format's sample-frames) to this file's
/// actual sample-frame offsets at `sample_rate` -- the one lossy-unit-conversion point, done once
/// at import rather than at every seek. Accepts a single multi-track `.cue` describing one
/// continuous audio file (a disc image, this format's own model); a `.cue` referencing several
/// `FILE`s restarts its timestamps per file, cannot describe one stream, and is rejected, as are
/// sheets a player could not turn into tracks (no `INDEX 01`, tracks out of order, index points
/// going backwards).
pub fn parse_cue_text(text: &str, sample_rate: u32) -> Result<CueSheet, MetadataError> {
    let err = |m: &str| MetadataError(format!("cue sheet: {m}"));
    let mut catalog = String::new();
    let mut tracks: Vec<CueTrack> = Vec::new();
    let mut files = 0usize;

    for line in text.lines() {
        let line = line.trim();
        let Some((keyword, rest)) = line.split_once(char::is_whitespace) else { continue };
        let rest = rest.trim();
        match keyword.to_ascii_uppercase().as_str() {
            "CATALOG" => catalog = rest.trim_matches('"').to_string(),
            "FILE" => files += 1,
            "TRACK" => {
                let number: u8 = rest.split_whitespace().next().and_then(|n| n.parse().ok())
                    .ok_or_else(|| err(&format!("invalid TRACK line: {line}")))?;
                tracks.push(CueTrack { number, isrc: String::new(), indices: Vec::new() });
            }
            "ISRC" => {
                tracks.last_mut().ok_or_else(|| err("ISRC before any TRACK"))?.isrc = rest.trim_matches('"').to_string();
            }
            "INDEX" => {
                let mut parts = rest.split_whitespace();
                let number: u8 = parts.next().and_then(|n| n.parse().ok()).ok_or_else(|| err(&format!("invalid INDEX line: {line}")))?;
                let ts = parts.next().ok_or_else(|| err(&format!("invalid INDEX line: {line}")))?;
                let (mm, ss, ff) = {
                    let mut f = ts.splitn(3, ':');
                    let mm: u64 = f.next().and_then(|s| s.parse().ok()).ok_or_else(|| err(&format!("invalid timestamp: {ts}")))?;
                    let ss: u64 = f.next().and_then(|s| s.parse().ok()).ok_or_else(|| err(&format!("invalid timestamp: {ts}")))?;
                    let ff: u64 = f.next().and_then(|s| s.parse().ok()).ok_or_else(|| err(&format!("invalid timestamp: {ts}")))?;
                    (mm, ss, ff)
                };
                // Checked, not plain `*`/`+` (found by coverage-guided fuzzing): an absurd timestamp like `99999999999999999:00:00` overflowed u64 --
                // a panic in debug builds, a silently wrapped garbage offset in release builds.
                let sample_offset = mm.checked_mul(60).and_then(|v| v.checked_add(ss))
                    .and_then(|v| v.checked_mul(75)).and_then(|v| v.checked_add(ff)) // CD standard: 75 frames/second
                    .and_then(|cd_frames| cd_frames.checked_mul(sample_rate as u64))
                    .and_then(|v| v.checked_add(37)) // round to nearest
                    .map(|v| v / 75)
                    .ok_or_else(|| err(&format!("timestamp out of range: {ts}")))?;
                tracks.last_mut().ok_or_else(|| err("INDEX before any TRACK"))?.indices.push(CueIndex { number, sample_offset });
            }
            _ => {} // TITLE, PERFORMER, REM, and anything else: recognized, not acted on
        }
    }
    if files > 1 { return Err(err("it references more than one audio FILE; only a single-file (disc image) cue sheet describes one stream")); }
    if tracks.len() > MAX_TRACKS { return Err(err("too many tracks")); }
    // What a player needs to turn the sheet into tracks (the same rules FLAC's CUESHEET block
    // imposes): every track starts somewhere (INDEX 01), track numbers are unique and ascending,
    // and index points never go backwards in the stream.
    let mut last_offset = 0u64;
    let mut last_number = 0u8;
    for t in &tracks {
        if t.indices.len() > MAX_INDICES_PER_TRACK { return Err(err(&format!("track {} has too many index points", t.number))); }
        if t.number <= last_number { return Err(err(&format!("track {} is out of order or repeated", t.number))); }
        last_number = t.number;
        if !t.indices.iter().any(|i| i.number == 1) { return Err(err(&format!("track {} has no INDEX 01", t.number))); }
        for i in &t.indices {
            if i.sample_offset < last_offset { return Err(err(&format!("track {} index {:02} goes backwards", t.number, i.number))); }
            last_offset = i.sample_offset;
        }
    }
    Ok(CueSheet { catalog, tracks })
}

/// Tag key holding a whole cue sheet's text -- the convention foobar2000 and EAC-made FLAC files
/// use to embed a disc image's cue sheet, including what the binary cue sheet has no room for
/// (per-track `TITLE`/`PERFORMER`). Matched case-insensitively, like every Vorbis Comment key.
pub const CUESHEET_TAG: &str = "CUESHEET";

impl Metadata {
    /// The value of the `CUESHEET` tag, if there is one (the first, if repeated).
    pub fn cuesheet_tag(&self) -> Option<&str> {
        self.tags.iter().find_map(|t| {
            let (k, v) = t.split_once('=')?;
            k.eq_ignore_ascii_case(CUESHEET_TAG).then_some(v)
        })
    }

    /// Replaces any `CUESHEET` tags with `text` (or removes them for `None`).
    pub fn set_cuesheet_tag(&mut self, text: Option<&str>) {
        self.tags.retain(|t| !t.split_once('=').is_some_and(|(k, _)| k.eq_ignore_ascii_case(CUESHEET_TAG)));
        if let Some(text) = text { self.tags.push(format!("{CUESHEET_TAG}={text}")); }
    }

    /// Makes the binary cue sheet agree with the `CUESHEET` tag, which is the source of truth
    /// whenever it is present: the tag is what players and taggers show and edit, the binary copy
    /// is what `fak decode --track` and seeking use (sample-exact, no text parsing). Without the
    /// tag, `keep_untagged` decides whether an existing binary cue sheet stays (a file written by
    /// a tool that only stores the binary form) or goes (the tag was deliberately removed).
    /// Errors on a malformed sheet or one pointing past `total_frames`.
    pub fn sync_cue_sheet(&mut self, sample_rate: u32, total_frames: u64, keep_untagged: bool) -> Result<(), MetadataError> {
        match self.cuesheet_tag() {
            Some(text) => self.cue_sheet = Some(parse_cue_text(text, sample_rate)?),
            None if !keep_untagged => self.cue_sheet = None,
            None => {}
        }
        if let Some(cue) = &self.cue_sheet {
            if cue.tracks.iter().flat_map(|t| &t.indices).any(|i| i.sample_offset > total_frames) {
                return Err(MetadataError("cue sheet: an index point is past the end of the audio".into()));
            }
        }
        Ok(())
    }

    /// Cue sheet text for this file: the `CUESHEET` tag verbatim if there is one, otherwise one
    /// generated from the binary cue sheet (`FILE` named `file_name`; offsets that are not whole
    /// CD frames are rounded down to one, the unit `.cue` files use). `None` without either.
    pub fn cue_sheet_text(&self, sample_rate: u32, file_name: &str) -> Option<String> {
        if let Some(text) = self.cuesheet_tag() { return Some(text.to_string()); }
        let cue = self.cue_sheet.as_ref().filter(|c| !c.tracks.is_empty())?;
        let mut s = String::new();
        if !cue.catalog.is_empty() { s += &format!("CATALOG {}\r\n", cue.catalog); }
        s += &format!("FILE \"{file_name}\" WAVE\r\n");
        for t in &cue.tracks {
            s += &format!("  TRACK {:02} AUDIO\r\n", t.number);
            if !t.isrc.is_empty() { s += &format!("    ISRC {}\r\n", t.isrc); }
            for i in &t.indices {
                let cd = (i.sample_offset as u128 * 75 / sample_rate.max(1) as u128) as u64;
                s += &format!("    INDEX {:02} {:02}:{:02}:{:02}\r\n", i.number, cd / 75 / 60, cd / 75 % 60, cd % 75);
            }
        }
        Some(s)
    }
}

impl CueSheet {
    /// Sample-frame range of track `number`: from its INDEX 01 to the next track's INDEX 01 (or
    /// the end of the stream), the convention CD rippers and players use -- a track's pre-gap
    /// (INDEX 00) plays as the end of the previous track.
    pub fn track_range(&self, number: u8, total_frames: u64) -> Option<std::ops::Range<u64>> {
        let start_of = |t: &CueTrack| t.indices.iter().find(|i| i.number == 1).map(|i| i.sample_offset);
        let pos = self.tracks.iter().position(|t| t.number == number)?;
        let start = start_of(&self.tracks[pos])?;
        let end = self.tracks.get(pos + 1).and_then(start_of).unwrap_or(total_frames);
        (start <= end && end <= total_frames).then_some(start..end)
    }
}

/// Wraps `Metadata::write`'s bytes with a `u32` length prefix and a `u32` CRC-32, matching the
/// chunk table's own framing (`format.rs`) -- an empty `Metadata` still writes a valid, minimal
/// block (8 bytes of header/vendor-length/counts + the 8-byte outer prefix/CRC), so the block is
/// unconditionally present rather than needing its own presence flag.
pub fn write_block(m: &Metadata) -> Vec<u8> {
    let body = m.write();
    let mut out = Vec::with_capacity(8 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_le_bytes());
    out
}

/// Reads a metadata block from `data[offset..]`, returning the parsed `Metadata` and the offset of
/// the first byte after the block (where chunk payloads begin).
pub fn read_block(data: &[u8], offset: usize) -> Result<(Metadata, usize), MetadataError> {
    let err = |m: &str| MetadataError(format!("{m} (corrupted metadata?)"));
    if offset + 4 > data.len() { return Err(err("truncated metadata block length")); }
    let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    let body_end = offset.checked_add(4).and_then(|p| p.checked_add(len)).ok_or_else(|| err("metadata length overflow"))?;
    let crc_end = body_end.checked_add(4).ok_or_else(|| err("metadata length overflow"))?;
    if crc_end > data.len() { return Err(err("metadata block runs past end of file")); }
    let body = &data[offset + 4..body_end];
    let stored_crc = u32::from_le_bytes(data[body_end..crc_end].try_into().unwrap());
    if crc32(body) != stored_crc { return Err(err("metadata block CRC mismatch")); }
    Ok((Metadata::read(body)?, crc_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Metadata {
        Metadata {
            vendor: "fak-1.0.0".to_string(),
            tags: vec!["TITLE=Test Track".to_string(), "ARTIST=Someone".to_string(), "ARTIST=Someone Else".to_string()],
            pictures: vec![Picture {
                kind: PictureType::FrontCover, kind_raw: 3, mime: "image/png".to_string(),
                description: "cover".to_string(), width: 600, height: 600, depth: 24, colors: 0,
                data: vec![0x89, b'P', b'N', b'G', 1, 2, 3, 4],
            }],
            cue_sheet: Some(CueSheet {
                catalog: "0123456789012".to_string(),
                tracks: vec![
                    CueTrack { number: 1, isrc: "USRC17607839".to_string(), indices: vec![
                        CueIndex { number: 0, sample_offset: 0 },
                        CueIndex { number: 1, sample_offset: 88200 },
                    ] },
                    CueTrack { number: 2, isrc: String::new(), indices: vec![
                        CueIndex { number: 1, sample_offset: 12_345_678 },
                    ] },
                ],
            }),
            channel_mask: Some(0x3F), // 5.1: FL|FR|FC|LFE|BL|BR
            float_info: Some(FloatInfo {
                scale_exp: 23,
                exceptions: vec![
                    FloatException { channel: 0, index: 12, bits: f32::NAN.to_bits() },
                    FloatException { channel: 1, index: 999_999, bits: 0x8000_0000 },
                ],
            }),
        }
    }

    #[test]
    fn roundtrip() {
        let m = sample();
        let (parsed, end) = read_block(&write_block(&m), 0).unwrap();
        assert_eq!(parsed, m);
        assert_eq!(end, write_block(&m).len());
    }

    #[test]
    fn empty_roundtrips() {
        let m = Metadata::default();
        assert!(m.is_empty());
        let (parsed, _) = read_block(&write_block(&m), 0).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn block_can_be_preceded_by_other_bytes() {
        let m = sample();
        let mut data = vec![0xAA; 20];
        data.extend_from_slice(&write_block(&m));
        let (parsed, end) = read_block(&data, 20).unwrap();
        assert_eq!(parsed, m);
        assert_eq!(end, data.len());
    }

    #[test]
    fn rejects_corrupted_and_hostile_blocks() {
        let good = write_block(&sample());

        let mut bad_crc = good.clone();
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xFF;
        assert!(read_block(&bad_crc, 0).is_err());

        assert!(read_block(&good[..good.len() - 1], 0).is_err(), "truncated");
        assert!(read_block(&good[..2], 0).is_err(), "truncated length prefix");

        // declared body length runs past end of file
        let mut huge_len = good.clone();
        huge_len[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_block(&huge_len, 0).is_err());

        // tag count far exceeding MAX_TAGS, inside an otherwise-valid-length, CRC-sealed block
        let mut body = Vec::new();
        write_str(&mut body, ""); // vendor
        body.extend_from_slice(&(MAX_TAGS as u32 + 1).to_le_bytes());
        let mut hostile = Vec::new();
        hostile.extend_from_slice(&(body.len() as u32).to_le_bytes());
        hostile.extend_from_slice(&body);
        hostile.extend_from_slice(&crc32(&body).to_le_bytes());
        assert!(read_block(&hostile, 0).is_err(), "oversized tag count");

        // picture data_len declared larger than MAX_PICTURE_BYTES
        let mut body2 = Vec::new();
        write_str(&mut body2, "");
        body2.extend_from_slice(&0u32.to_le_bytes()); // tag_count = 0
        body2.extend_from_slice(&1u32.to_le_bytes()); // pic_count = 1
        body2.push(3); // kind_raw
        write_str(&mut body2, "image/png");
        write_str(&mut body2, "");
        for _ in 0..4 { body2.extend_from_slice(&0u32.to_le_bytes()); } // width/height/depth/colors
        body2.extend_from_slice(&((MAX_PICTURE_BYTES as u32).wrapping_add(1)).to_le_bytes());
        let mut hostile2 = Vec::new();
        hostile2.extend_from_slice(&(body2.len() as u32).to_le_bytes());
        hostile2.extend_from_slice(&body2);
        hostile2.extend_from_slice(&crc32(&body2).to_le_bytes());
        assert!(read_block(&hostile2, 0).is_err(), "oversized picture data_len");

        // invalid UTF-8 in the vendor string
        let mut body3 = Vec::new();
        body3.extend_from_slice(&4u32.to_le_bytes());
        body3.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
        body3.extend_from_slice(&0u32.to_le_bytes());
        body3.extend_from_slice(&0u32.to_le_bytes());
        let mut hostile3 = Vec::new();
        hostile3.extend_from_slice(&(body3.len() as u32).to_le_bytes());
        hostile3.extend_from_slice(&body3);
        hostile3.extend_from_slice(&crc32(&body3).to_le_bytes());
        assert!(read_block(&hostile3, 0).is_err(), "invalid utf-8");

        // float exception count far exceeding MAX_FLOAT_EXCEPTIONS, inside an otherwise-valid,
        // CRC-sealed block ((b))
        let mut body4 = empty_prefix();
        write_str(&mut body4, ""); // cue sheet catalog
        body4.extend_from_slice(&0u32.to_le_bytes()); // track_count
        body4.push(0); // channel_mask presence = absent
        body4.extend_from_slice(&0u32.to_le_bytes());
        body4.push(1); // float_info presence = present
        body4.push(0); // scale_exp
        body4.extend_from_slice(&(MAX_FLOAT_EXCEPTIONS as u32 + 1).to_le_bytes());
        let mut hostile4 = Vec::new();
        hostile4.extend_from_slice(&(body4.len() as u32).to_le_bytes());
        hostile4.extend_from_slice(&body4);
        hostile4.extend_from_slice(&crc32(&body4).to_le_bytes());
        assert!(read_block(&hostile4, 0).is_err(), "oversized float exception count");
    }

    fn wrap(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(&crc32(body).to_le_bytes());
        out
    }

    /// An otherwise-empty metadata body (vendor="", 0 tags, 0 pictures), for tests that only care
    /// about the cue-sheet section coming after it.
    fn empty_prefix() -> Vec<u8> {
        let mut body = Vec::new();
        write_str(&mut body, "");
        body.extend_from_slice(&0u32.to_le_bytes()); // tag_count
        body.extend_from_slice(&0u32.to_le_bytes()); // pic_count
        body
    }

    #[test]
    fn rejects_hostile_cue_sheet_fields() {
        // track count far exceeding MAX_TRACKS
        let mut body = empty_prefix();
        write_str(&mut body, ""); // catalog
        body.extend_from_slice(&(MAX_TRACKS as u32 + 1).to_le_bytes());
        assert!(read_block(&wrap(&body), 0).is_err(), "oversized track count");

        // index count far exceeding MAX_INDICES_PER_TRACK, inside one otherwise-valid track
        let mut body2 = empty_prefix();
        write_str(&mut body2, "");
        body2.extend_from_slice(&1u32.to_le_bytes()); // track_count = 1
        body2.push(1); // track number
        write_str(&mut body2, ""); // isrc
        body2.extend_from_slice(&(MAX_INDICES_PER_TRACK as u32 + 1).to_le_bytes());
        assert!(read_block(&wrap(&body2), 0).is_err(), "oversized index count");

        // catalog string longer than MAX_CATALOG_LEN
        let mut body3 = empty_prefix();
        body3.extend_from_slice(&(MAX_CATALOG_LEN as u32 + 1).to_le_bytes());
        body3.extend(std::iter::repeat(b'0').take(MAX_CATALOG_LEN + 1));
        body3.extend_from_slice(&0u32.to_le_bytes()); // track_count
        assert!(read_block(&wrap(&body3), 0).is_err(), "oversized catalog string");

        // a valid, well-formed cue sheet with tracks decodes correctly (positive control, so the
        // negative cases above are known to be exercising real validation, not an unrelated bug)
        let mut good_body = empty_prefix();
        write_str(&mut good_body, "0123456789012");
        good_body.extend_from_slice(&1u32.to_le_bytes());
        good_body.push(1);
        write_str(&mut good_body, "USRC17607839");
        good_body.extend_from_slice(&1u32.to_le_bytes());
        good_body.push(1);
        good_body.extend_from_slice(&88200u64.to_le_bytes());
        good_body.push(0); // channel_mask presence = absent
        good_body.extend_from_slice(&0u32.to_le_bytes());
        good_body.push(0); // float_info presence = absent
        let (parsed, _) = read_block(&wrap(&good_body), 0).unwrap();
        assert_eq!(parsed.cue_sheet, Some(CueSheet {
            catalog: "0123456789012".to_string(),
            tracks: vec![CueTrack { number: 1, isrc: "USRC17607839".to_string(), indices: vec![CueIndex { number: 1, sample_offset: 88200 }] }],
        }));
    }

    #[test]
    fn parses_a_real_cue_file() {
        let text = r#"
REM GENRE Rock
CATALOG 0123456789012
PERFORMER "Some Artist"
TITLE "Some Album"
FILE "album.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track One"
    PERFORMER "Some Artist"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track Two"
    ISRC USRC17607839
    INDEX 00 03:58:42
    INDEX 01 04:00:17
"#;
        let cue = parse_cue_text(text, 44100).unwrap();
        assert_eq!(cue.catalog, "0123456789012");
        assert_eq!(cue.tracks.len(), 2);
        assert_eq!(cue.tracks[0].number, 1);
        assert_eq!(cue.tracks[0].indices, vec![CueIndex { number: 1, sample_offset: 0 }]);
        assert_eq!(cue.tracks[1].number, 2);
        assert_eq!(cue.tracks[1].isrc, "USRC17607839");
        // 04:00:17 = (4*60+0)*75+17 = 18017 CD frames; *44100/75 = 18017*588 = 10,593,996 sample-frames
        assert_eq!(cue.tracks[1].indices[1], CueIndex { number: 1, sample_offset: 10_593_996 });
        // Round-trips through the real block format too.
        let m = Metadata { cue_sheet: Some(cue.clone()), ..Metadata::default() };
        let (parsed, _) = read_block(&write_block(&m), 0).unwrap();
        assert_eq!(parsed.cue_sheet, Some(cue));
    }

    #[test]
    fn cue_parser_rejects_malformed_lines_not_silently() {
        assert!(parse_cue_text("TRACK 01 AUDIO\nINDEX 01 not-a-timestamp\n", 44100).is_err());
        assert!(parse_cue_text("INDEX 01 00:00:00\n", 44100).is_err(), "INDEX before any TRACK");
        assert!(parse_cue_text("ISRC ABCDEFGHIJKL\n", 44100).is_err(), "ISRC before any TRACK");
    }

    /// Regression for the coverage-guided fuzzing find: each
    /// step of the MM:SS:FF -> sample-offset conversion used to overflow u64 on absurd values.
    #[test]
    fn cue_parser_rejects_overflowing_timestamps() {
        for ts in ["99999999999999999:00:00", "0:99999999999999999999:00", "0:00:18446744073709551615",
                   "307445734561825860:0:0", "0:0:245956587649466880"] {
            assert!(parse_cue_text(&format!("TRACK 01 AUDIO\nINDEX 01 {ts}\n"), 192_000).is_err(), "{ts}");
        }
        // The largest values that still fit convert exactly, not rejected.
        let cue = parse_cue_text("TRACK 01 AUDIO\nINDEX 01 1000000:00:00\n", 192_000).unwrap();
        assert_eq!(cue.tracks[0].indices[0].sample_offset, 1_000_000 * 60 * 192_000);
    }

    #[test]
    fn cue_parser_ignores_unknown_lines() {
        let cue = parse_cue_text("REM this is a comment\nSOMETHING_UNKNOWN here\nTRACK 01 AUDIO\nINDEX 01 00:00:00\n", 44100).unwrap();
        assert_eq!(cue.tracks.len(), 1);
    }

    #[test]
    fn cue_parser_rejects_sheets_that_are_not_one_playable_image() {
        let two_files = "FILE \"a.wav\" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00\nFILE \"b.wav\" WAVE\nTRACK 02 AUDIO\nINDEX 01 00:00:00\n";
        assert!(parse_cue_text(two_files, 44100).is_err(), "multi-file");
        assert!(parse_cue_text("TRACK 01 AUDIO\nINDEX 00 00:00:00\n", 44100).is_err(), "no INDEX 01");
        assert!(parse_cue_text("TRACK 02 AUDIO\nINDEX 01 00:00:00\nTRACK 01 AUDIO\nINDEX 01 00:01:00\n", 44100).is_err(), "order");
        assert!(parse_cue_text("TRACK 01 AUDIO\nINDEX 01 00:00:00\nTRACK 01 AUDIO\nINDEX 01 00:01:00\n", 44100).is_err(), "repeat");
        assert!(parse_cue_text("TRACK 01 AUDIO\nINDEX 01 00:05:00\nTRACK 02 AUDIO\nINDEX 01 00:01:00\n", 44100).is_err(), "backwards");
        // Pre-gap before INDEX 01 and a gap-less start are both fine.
        assert!(parse_cue_text("FILE \"x.wav\" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00\nTRACK 02 AUDIO\nINDEX 00 00:04:00\nINDEX 01 00:05:00\n", 44100).is_ok());
    }

    #[test]
    fn cuesheet_tag_is_the_source_of_truth() {
        let text = "FILE \"x.wav\" WAVE\r\n  TRACK 01 AUDIO\r\n    TITLE \"One\"\r\n    INDEX 01 00:00:00\r\n  TRACK 02 AUDIO\r\n    INDEX 01 00:02:00\r\n";
        let mut m = Metadata { tags: vec!["ARTIST=A".into(), format!("cuesheet={text}")], ..Metadata::default() };
        assert_eq!(m.cuesheet_tag(), Some(text), "key is case-insensitive");
        m.sync_cue_sheet(44100, 1_000_000, false).unwrap();
        let cue = m.cue_sheet.clone().unwrap();
        assert_eq!(cue.tracks.len(), 2);
        assert_eq!(cue.track_range(1, 1_000_000), Some(0..88200));
        assert_eq!(cue.track_range(2, 1_000_000), Some(88200..1_000_000));
        assert_eq!(cue.track_range(3, 1_000_000), None);
        assert_eq!(m.cue_sheet_text(44100, "ignored.wav").as_deref(), Some(text), "the tag is returned verbatim");
        assert!(m.clone().sync_cue_sheet(44100, 88199, false).is_err(), "index past the end");

        // Tag removed: the binary copy goes with it, unless the caller keeps an untagged one.
        m.set_cuesheet_tag(None);
        assert_eq!(m.tags, vec!["ARTIST=A".to_string()]);
        let mut kept = m.clone();
        kept.sync_cue_sheet(44100, 1_000_000, true).unwrap();
        assert_eq!(kept.cue_sheet, Some(cue.clone()));
        m.sync_cue_sheet(44100, 1_000_000, false).unwrap();
        assert_eq!(m.cue_sheet, None);

        // Without a tag the text is generated from the binary cue sheet and parses back to it.
        let bin = Metadata { cue_sheet: Some(cue.clone()), ..Metadata::default() };
        let generated = bin.cue_sheet_text(44100, "image.wav").unwrap();
        assert!(generated.contains("FILE \"image.wav\" WAVE") && generated.contains("INDEX 01 00:02:00"), "{generated}");
        assert_eq!(parse_cue_text(&generated, 44100).unwrap(), cue);
    }

    #[test]
    fn empty_cue_sheet_is_equivalent_to_none() {
        let with_none = Metadata { cue_sheet: None, ..Metadata::default() };
        let with_empty = Metadata { cue_sheet: Some(CueSheet::default()), ..Metadata::default() };
        assert_eq!(write_block(&with_none), write_block(&with_empty));
        let (parsed, _) = read_block(&write_block(&with_empty), 0).unwrap();
        assert_eq!(parsed.cue_sheet, None);
    }

    #[test]
    fn channel_mask_none_and_some_zero_are_distinct() {
        let absent = Metadata { channel_mask: None, ..Metadata::default() };
        let declared_zero = Metadata { channel_mask: Some(0), ..Metadata::default() };
        assert_ne!(write_block(&absent), write_block(&declared_zero), "presence byte must differ even though the value is 0 either way");
        let (parsed_absent, _) = read_block(&write_block(&absent), 0).unwrap();
        let (parsed_zero, _) = read_block(&write_block(&declared_zero), 0).unwrap();
        assert_eq!(parsed_absent.channel_mask, None);
        assert_eq!(parsed_zero.channel_mask, Some(0));
    }

    #[test]
    fn channel_mask_roundtrips_a_real_value() {
        let m = Metadata { channel_mask: Some(0x60F), ..Metadata::default() }; // an arbitrary real-looking mask
        let (parsed, _) = read_block(&write_block(&m), 0).unwrap();
        assert_eq!(parsed.channel_mask, Some(0x60F));
        assert!(!m.is_empty(), "a present channel_mask alone must make the block non-empty");
    }

    #[test]
    fn random_bytes_never_panic() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for _ in 0..2000 {
            let len = (next() % 300) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let _ = read_block(&bytes, 0);
        }
    }
}
