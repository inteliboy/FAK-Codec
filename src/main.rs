// See the `fast-alloc` feature in Cargo.toml.
#[cfg(feature = "fast-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use fak::metadata::{Metadata, Picture, PictureType};
use fak::{cdrip, cpufeatures, decoder, encoder, format, parallel, wav};
use std::io::{Read, Write};
use std::process::ExitCode;
use std::time::Instant;

const HELP: &str = "\
FAK lossless audio codec

usage: fak <command> [options] <files>
  encode <in.wav> <out.fak>     compress a WAV file (\"-\" reads stdin / writes stdout)
  decode <in.fak> <out.wav>     decompress to WAV (\"-\" reads stdin / writes stdout)
  verify <in.fak>...            decode fully and check the stored SHA-256 of the audio
  info <in.fak>...              stream properties, tags, pictures, cue sheet
  edit <in.fak> [options]       change tags, pictures or cue sheet; audio bytes are not touched
  cue <in.fak> [out.cue]        export the embedded cue sheet (stdout if no file is given)
  picture <in.fak> <n> <out>    save embedded picture n (numbered as `fak info` lists them)
  help [advanced]               this text, or the research/diagnostic commands and options
  version                       program and format version

encode options:
  -l, --level LEVEL   compression level (decode speed in parentheses):
                        fast     fixed 4096-sample frames                     (fast decode)
                        normal   frame lengths chosen by estimate [default]   (fast decode)
                        max      frame lengths chosen by trial, ~4x normal    (fast decode)
                        insane   LEVEL_INSANE_HELP
                        archival insane plus --fec (Reed-Solomon parity over the whole file, about 1%
                                 of its chunks rebuildable; an explicit --fec=N wins): for files
                                 you want to keep and be able to repair
      --fast, --normal, --max, --insane, --archival    the same as --level LEVEL
  -t, --threads N     worker threads (default: all). The output does not depend on N.
      --fec[=N]       add Reed-Solomon parity so damaged chunks can be rebuilt: one block over the
                      whole file, or one per N chunks with =N
      --fec-parity M  parity shards per block, i.e. any M damaged chunks per block are rebuilt
                      (default: 1% of the block's chunks, at least 2); implies --fec
  -T, --tag KEY=VALUE add a tag (repeatable), e.g. -T ARTIST=Someone -T \"TITLE=Some Song\"
      --picture [TYPE:]FILE   embed a PNG or JPEG; TYPE is front (default), back or artist
      --cuesheet FILE embed a .cue sheet for a single-file disc image: players show its tracks,
                      `fak decode --track` extracts them (like `flac --cuesheet`, and titles
                      are kept too: the sheet is stored as the CUESHEET tag, as foobar2000 does)
      --cd-tags       with --cuesheet on a 16-bit 44.1 kHz CD image: also store its disc IDs
                      (freedb, MusicBrainz, AccurateRip) and AccurateRip/CTDB checksums as tags
      --accel cpu|apple  apple (macOS, the default there): LPC autocorrelation on Accelerate,
                      ~25% faster encode; near-ties may break differently from cpu, so the
                      bytes can differ across machines, always lossless. cpu: the bit-exact
                      analysis (also the default off macOS).
  -q, --quiet         no summary line

decode options:
  -t, --threads N     worker threads (default: all)
      --track N       decode only track N of the embedded cue sheet (INDEX 01 to next INDEX 01)
  -q, --quiet         no summary line

edit options (applied in the order given; -o writes a new file instead of changing <in.fak>):
  -T, --tag KEY=VALUE        add a tag           --remove-tag KEY      remove every KEY tag
      --set-tag KEY=VALUE    replace KEY's tags  --remove-all-tags     keep no tags (cue sheet too)
      --picture [TYPE:]FILE  add a picture       --remove-pictures     remove every picture
      --cuesheet FILE        embed a cue sheet   --remove-cuesheet     remove the cue sheet
      --cd-tags              write (or recompute) the CD tags -- disc IDs, AccurateRip/CTDB
                             checksums -- from the embedded cue sheet. Once a file has them,
                             `--cuesheet` keeps them current
  -o, --output FILE          write the result to FILE

Examples:
  fak encode album.wav album.fak --cuesheet album.cue -T \"ALBUM=Some Album\"
  fak decode album.fak track3.wav --track 3
  fak edit song.fak --set-tag TITLE=Better --picture front:cover.jpg
";

const HELP_ADVANCED: &str = "\
research and diagnostic commands:
  stream-encode <in.wav> [encode options] > out.fak   encode to a pipe without a seek table
                                                      (total length unknown up front)
  stream-decode <out.wav> < in.fak                    decode a stream as it arrives
  seek <in.fak> <frame> <out.wav>   decode from one sample-frame on, printing the seek latency
  cpuinfo                           CPU features, the kernels in use, and further acceleration options

research encode options:
  --chunk-seconds S   length of the independently decodable chunks (default 1). Longer chunks
                      can be slightly smaller, at coarser seeking and less parallelism.

environment:
  FAK_DISABLE_AVX2_CLONES, FAK_DISABLE_AVX512, FAK_DISABLE_BLOCKED, FAK_DISABLE_HW_CRC
                      set to 1 to switch off a SIMD/hardware path for timing comparisons; the output
                      is identical either way.
  FAK_DISABLE_VNNI    set to 1 to rank LPC candidates without the AVX-512 VNNI kernel (for
                      timing comparisons; the ranking can differ slightly, the output stays lossless).
  FAK_FORCE_AVX512    set to 1 to use the AVX-512 kernels without first timing them against
                      the AVX2 ones (by default each is picked by a short timing at first use).
";

/// Anything that ends a command: a usage mistake (exit 2, with a pointer to the help) or a
/// failure while doing the work (exit 1).
enum CliError { Usage(String), Fail(String) }
type CliResult<T = ()> = Result<T, CliError>;
fn usage<T>(m: impl Into<String>) -> CliResult<T> { Err(CliError::Usage(m.into())) }
fn fail(m: impl std::fmt::Display) -> CliError { CliError::Fail(m.to_string()) }

/// Command-line arguments as options and positionals. Options may come before, between or after
/// the file names; `--name=value` and `--name value` are the same; `--` ends the options; a lone
/// `-` is a positional (stdin/stdout).
struct Args { rest: std::vec::IntoIter<String>, only_positional: bool }
enum Arg { Opt(String, Option<String>), Pos(String) }

impl Args {
    fn new(v: &[String]) -> Self { Args { rest: v.to_vec().into_iter(), only_positional: false } }
    fn next(&mut self) -> Option<Arg> {
        let a = self.rest.next()?;
        if self.only_positional || a == "-" || !a.starts_with('-') { return Some(Arg::Pos(a)); }
        if a == "--" { self.only_positional = true; return self.next(); }
        match a.split_once('=') {
            Some((k, v)) if a.starts_with("--") => Some(Arg::Opt(k.to_string(), Some(v.to_string()))),
            _ => Some(Arg::Opt(a, None)),
        }
    }
    fn value(&mut self, name: &str, inline: Option<String>) -> CliResult<String> {
        match inline.or_else(|| self.rest.next()) { Some(v) => Ok(v), None => usage(format!("{name} needs a value")) }
    }
}

fn parse_number<T: std::str::FromStr + PartialOrd + Default>(name: &str, v: &str) -> CliResult<T> {
    match v.parse::<T>() { Ok(n) if n > T::default() => Ok(n), _ => usage(format!("{name}: expected a positive number, got \"{v}\"")) }
}

fn expect_positionals(pos: &[String], min: usize, max: usize, what: &str) -> CliResult {
    if pos.len() < min || pos.len() > max { return usage(format!("expected {what}")); }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Level { Fast, Normal, Max, Insane, Archival }

impl Level {
    fn parse(s: &str) -> CliResult<Level> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Ok(Level::Fast), "normal" => Ok(Level::Normal), "max" => Ok(Level::Max), "insane" => Ok(Level::Insane),
            "archival" => Ok(Level::Archival),
            _ => usage(format!("unknown level \"{s}\" (fast, normal, max, insane or archival)")),
        }
    }
    fn name(self) -> &'static str {
        match self { Level::Fast => "fast", Level::Normal => "normal", Level::Max => "max", Level::Insane => "insane", Level::Archival => "archival" }
    }
}

struct EncodeOpts {
    level: Level, threads: usize, chunk_secs: Option<usize>, fec_group: Option<usize>,
    metadata: Metadata, cuesheet: Option<String>, quiet: bool,
    /// Write the CD identifiers and AccurateRip/CTDB checksums as tags (off by default).
    cd_tags: bool,
}

fn read_text(path: &str) -> CliResult<String> {
    let bytes = std::fs::read(path).map_err(|e| fail(format!("{path}: {e}")))?;
    // Cue sheets from Windows tools often start with a UTF-8 byte-order mark; some are not UTF-8
    // at all (legacy code pages), which cannot be stored as a text tag without guessing.
    let text = String::from_utf8(bytes).map_err(|_| fail(format!("{path}: not UTF-8 text (convert it to UTF-8 first)")))?;
    Ok(text.strip_prefix('\u{feff}').unwrap_or(&text).to_string())
}

fn read_picture(spec: &str) -> CliResult<Picture> {
    // TYPE:PATH split on the first colon, so a Windows path (C:\...) keeps its own.
    let (path, kind_raw) = match spec.split_once(':') {
        Some(("front", p)) => (p, PictureType::FrontCover as u8),
        Some(("back", p)) => (p, PictureType::BackCover as u8),
        Some(("artist", p)) => (p, PictureType::Artist as u8),
        _ => (spec, PictureType::FrontCover as u8),
    };
    let lower = path.to_ascii_lowercase();
    let mime = if lower.ends_with(".png") { "image/png" } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") { "image/jpeg" } else {
        return usage(format!("--picture {path}: only .png, .jpg and .jpeg files are supported"));
    };
    let data = std::fs::read(path).map_err(|e| fail(format!("{path}: {e}")))?;
    Ok(Picture { kind: PictureType::from_u8(kind_raw), kind_raw, mime: mime.into(), description: String::new(), width: 0, height: 0, depth: 0, colors: 0, data })
}

fn check_tag(kv: &str) -> CliResult<String> {
    match kv.split_once('=') { Some((k, _)) if !k.is_empty() => Ok(kv.to_string()), _ => usage(format!("tag \"{kv}\" is not KEY=VALUE")) }
}

fn parse_encode_opts(args: &[String]) -> CliResult<(EncodeOpts, Vec<String>)> {
    // Accelerate is always present on macOS: on by default there; `--accel cpu` opts out.
    let _ = fak::accel::enable_apple();
    let mut o = EncodeOpts {
        level: Level::Normal, threads: parallel::default_threads(), chunk_secs: None, fec_group: None,
        metadata: Metadata::default(), cuesheet: None, quiet: false, cd_tags: false,
    };
    let mut pos = Vec::new();
    let mut a = Args::new(args);
    while let Some(arg) = a.next() {
        let (name, inline) = match arg { Arg::Pos(p) => { pos.push(p); continue; } Arg::Opt(n, v) => (n, v) };
        match name.as_str() {
            "-l" | "--level" => o.level = Level::parse(&a.value(&name, inline)?)?,
            "--fast" | "--normal" | "--max" | "--insane" | "--archival" => o.level = Level::parse(&name[2..])?,
            "-t" | "--threads" => o.threads = parse_number(&name, &a.value(&name, inline)?)?,
            "--chunk-seconds" => o.chunk_secs = Some(parse_number(&name, &a.value(&name, inline)?)?),
            "--fec" => {
                let g = match inline {
                    Some(v) => {
                        let g = parse_number(&name, &v)?;
                        if g == 0 || g > format::MAX_FEC_GROUP { return usage(format!("--fec: 1 to {}", format::MAX_FEC_GROUP)); }
                        g
                    }
                    None => format::DEFAULT_FEC_GROUP,
                };
                o.fec_group = Some(g);
            }
            "--fec-parity" => {
                let m: usize = parse_number(&name, &a.value(&name, inline)?)?;
                if m == 0 || m > format::MAX_FEC_PARITY { return usage(format!("--fec-parity: 1 to {}", format::MAX_FEC_PARITY)); }
                format::set_fec_parity(Some(m));
                if o.fec_group.is_none() { o.fec_group = Some(format::DEFAULT_FEC_GROUP); }
            }
            "-T" | "--tag" => o.metadata.tags.push(check_tag(&a.value(&name, inline)?)?),
            "--picture" => o.metadata.pictures.push(read_picture(&a.value(&name, inline)?)?),
            "--cuesheet" => o.cuesheet = Some(read_text(&a.value(&name, inline)?)?),
            "-q" | "--quiet" => o.quiet = true,
            "--cd-tags" => o.cd_tags = true,
            "--accel" => match a.value(&name, inline)?.to_ascii_lowercase().as_str() {
                "cpu" => fak::accel::disable_apple(),
                "apple" => if let Err(e) = fak::accel::enable_apple() { eprintln!("fak: --accel apple: {e}; using the default path"); },
                v => return usage(format!("--accel: unknown accelerator \"{v}\" (cpu or apple)")),
            },
            _ => return usage(format!("unknown encode option {name}")),
        }
    }
    // `archival` = `insane` + FEC; an explicit `--fec[=N]` (any position) decides the group size.
    if o.level == Level::Archival && o.fec_group.is_none() { o.fec_group = Some(format::DEFAULT_FEC_GROUP); }
    Ok((o, pos))
}

/// Reads the source and prepares the metadata every encode path shares.
/// How an FEC layout reads in the encoder string and the summary line.
fn fec_label(g: usize) -> String {
    if g == format::FEC_AUTO { "RS whole-file".to_string() } else { format!("RS per {g} chunks") }
}

fn tag_key_is(t: &str, key: &str) -> bool { t.split_once('=').map_or(t, |(k, _)| k).eq_ignore_ascii_case(key) }

fn load_source(path: &str, o: &mut EncodeOpts) -> CliResult<wav::Wav> {
    let w = wav::read_wav(path).map_err(|e| fail(format!("{path}: {e}")))?;
    let frames = w.channels.first().map_or(0, |c| c.len()) as u64;
    prepare_metadata(o, w.sample_rate, frames, w.channel_mask, w.float_info.clone(), Some((&w.channels, w.bits)))?;
    Ok(w)
}

/// The metadata every encode path shares. `cd` is the whole audio, needed only by `--cd-tags`
/// (disc IDs and checksums cover every sample), so the streaming path passes `None` and is not used
/// with that option.
fn prepare_metadata(
    o: &mut EncodeOpts, sample_rate: u32, frames: u64, channel_mask: Option<u32>,
    float_info: Option<fak::floatpcm::FloatInfo>, cd: Option<(&[Vec<i64>], u8)>,
) -> CliResult {
    if let Some(text) = &o.cuesheet { o.metadata.set_cuesheet_tag(Some(text)); }
    o.metadata.sync_cue_sheet(sample_rate, frames, false).map_err(fail)?;
    // `--cd-tags`: a CD image with its cue sheet gets its disc IDs and AccurateRip/CTDB checksums as
    // tags (only when exact). Tags the user set on the command line are kept.
    if o.cd_tags {
        if let Some((channels, bits)) = cd {
            let user_keys: Vec<String> = o.metadata.tags.iter().filter_map(|t| t.split_once('=').map(|(k, _)| k.to_string())).collect();
            cdrip::apply_disc_tags(&mut o.metadata, channels, sample_rate, bits, &user_keys);
        }
    }
    // The source's speaker layout (WAVE_FORMAT_EXTENSIBLE mask) carries through unchanged.
    o.metadata.channel_mask = channel_mask;
    // (b): a 32-bit float source was already reduced to the integer PCM domain by
    // `wav::read_wav`; the scale/exceptions needed to invert that on decode ride along in metadata.
    o.metadata.float_info = float_info;
    // The encoder string names what was asked of the encoder (the level is not recoverable from the
    // stream): "fak 1.1.0 (level=normal; fec=none; chunk=auto)".
    if o.metadata.vendor.is_empty() {
        let level = if o.level == Level::Archival { "insane" } else { o.level.name() };
        o.metadata.vendor = format!(
            "fak {} (level={}; fec={}; chunk={})", env!("CARGO_PKG_VERSION"), level,
            o.fec_group.map_or("none".to_string(), |g| fec_label(g)),
            o.chunk_secs.map_or("auto".to_string(), |s| format!("{s}s")),
        );
    }
    Ok(())
}

fn chunk_frames(o: &EncodeOpts, rate: u32) -> usize {
    match o.chunk_secs {
        Some(s) => (rate as usize).saturating_mul(s).min(format::MAX_CHUNK_FRAMES as usize),
        None => format::default_chunk_frames(rate),
    }
}

/// Block-mode search effort and stream mode for a level.
fn level_plan(o: &EncodeOpts) -> (u8, encoder::Effort) {
    let effort = match o.level { Level::Fast => encoder::Effort::Fast, Level::Normal => encoder::Effort::Normal, Level::Max => encoder::Effort::Max, Level::Insane | Level::Archival => encoder::Effort::Insane };
    let mode = format::MODE_BLOCK_INDEPENDENT;
    (mode, effort)
}

fn human_bytes(n: u64) -> String {
    if n >= 10 << 20 { format!("{:.1} MB", n as f64 / 1e6) } else if n >= 10_000 { format!("{:.1} kB", n as f64 / 1e3) } else { format!("{n} B") }
}

fn cmd_encode(args: &[String]) -> CliResult {
    let (mut o, pos) = parse_encode_opts(args)?;
    expect_positionals(&pos, 2, 2, "encode <in.wav> <out.fak>")?;
    let (input, output) = (&pos[0], &pos[1]);
    let t0 = Instant::now();
    // A WAV file straight to a file: chunk by chunk, so memory does not grow with the file (a 275 MB
    // WAV took 733 MB as 8-byte samples). Float sources, unknown-length WAVs, `--cd-tags` (needs every
    // sample) and stdin/stdout take the whole-file path below; the output bytes are the same.
    if !o.cd_tags && input != "-" && output != "-" {
        if let Some(r) = encode_streaming(input, output, &mut o)? {
            if !o.quiet { report_encode(input, output, &o, r, t0) }
            return Ok(());
        }
    }
    let w = load_source(input, &mut o)?;
    let chunk = chunk_frames(&o, w.sample_rate);
    let (mode, effort) = level_plan(&o);
    let encode = |out: &mut dyn Write, mode: u8| -> CliResult {
        encoder::encode_chunked_to(&mut { out }, &w.channels, w.sample_rate, w.bits, mode, chunk, o.threads, o.fec_group, &o.metadata, effort)
            .map_err(|e| fail(format!("encode error: {e}")))
    };
    let written = write_output(output, |out| encode(out, mode))?;
    if !o.quiet {
        let frames = w.channels.first().map_or(0, |c| c.len());
        let pcm = (frames * w.channels.len() * (w.bits as usize / 8)) as u64;
        report_encode(input, output, &o, EncodeResult { pcm_bytes: pcm, written, frames: frames as u64, sample_rate: w.sample_rate }, t0);
    }
    Ok(())
}

struct EncodeResult { pcm_bytes: u64, written: u64, frames: u64, sample_rate: u32 }

fn report_encode(input: &str, output: &str, o: &EncodeOpts, r: EncodeResult, t0: Instant) {
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("{input} -> {output}: {} -> {} ({:.2}%), level {}{}, {:.1}x realtime",
              human_bytes(r.pcm_bytes), human_bytes(r.written), 100.0 * r.written as f64 / r.pcm_bytes.max(1) as f64, o.level.name(),
              o.fec_group.map_or(String::new(), |g| format!(", FEC {}", fec_label(g))),
              r.frames as f64 / r.sample_rate.max(1) as f64 / secs.max(1e-9));
}

/// Streaming WAV-file-to-file encode (see `cmd_encode`); `None` when the source needs the whole-file
/// path. A failed encode removes the partial output.
fn encode_streaming(input: &str, output: &str, o: &mut EncodeOpts) -> CliResult<Option<EncodeResult>> {
    let f = std::fs::File::open(input).map_err(|e| fail(format!("{input}: {e}")))?;
    let len = f.metadata().map_err(|e| fail(format!("{input}: {e}")))?.len();
    let Some(mut src) = wav::WavStream::open(std::io::BufReader::with_capacity(1 << 20, f), len).map_err(|e| fail(format!("{input}: {e}")))? else {
        return Ok(None);
    };
    prepare_metadata(o, src.sample_rate, src.frames, src.channel_mask, None, None)?;
    let chunk = chunk_frames(o, src.sample_rate);
    let (mode, effort) = level_plan(o);
    let file = std::fs::File::create(output).map_err(|e| fail(format!("{output}: {e}")))?;
    let run = || -> CliResult {
        let mut enc = encoder::FileEncoder::new(
            std::io::BufWriter::with_capacity(1 << 20, file), src.channels, src.sample_rate, src.bits, mode, o.threads, o.fec_group, &o.metadata, effort,
        ).map_err(|e| fail(format!("encode error: {e}")))?;
        enc.set_expected_chunks((src.frames as usize).div_ceil(chunk.max(1)));
        // With several threads the WAV is read on its own thread, a couple of chunks ahead, so reading
        // overlaps encoding (one thread: reading is under 1% of the time and the queue would only
        // cost memory).
        if o.threads <= 1 {
            while let Some(c) = src.next_chunk(chunk).map_err(|e| fail(format!("{input}: {e}")))? {
                enc.push_chunk(c).map_err(|e| fail(format!("encode error: {e}")))?;
            }
            enc.finish().map_err(|e| fail(format!("encode error: {e}")))?;
            return Ok(());
        }
        let read_err = std::thread::scope(|s| -> CliResult {
            let (tx, rx) = std::sync::mpsc::sync_channel::<CliResult<Option<Vec<Vec<i64>>>>>(2);
            let src = &mut src;
            s.spawn(move || loop {
                let next = src.next_chunk(chunk).map_err(|e| fail(format!("{input}: {e}")));
                let done = !matches!(next, Ok(Some(_)));
                if tx.send(next).is_err() || done { break; }
            });
            while let Ok(next) = rx.recv() {
                match next? {
                    Some(c) => enc.push_chunk(c).map_err(|e| fail(format!("encode error: {e}")))?,
                    None => break,
                }
            }
            Ok(())
        });
        read_err?;
        enc.finish().map_err(|e| fail(format!("encode error: {e}")))?;
        Ok(())
    };
    if let Err(e) = run() { let _ = std::fs::remove_file(output); return Err(e); }
    Ok(Some(EncodeResult { pcm_bytes: src.frames * src.channels as u64 * (src.bits as u64 / 8), written: out_len(output), frames: src.frames, sample_rate: src.sample_rate }))
}

/// Runs `produce` against the output file (or stdout for "-") and returns the bytes written. A
/// failed write removes the partial file.
fn write_output(path: &str, produce: impl FnOnce(&mut dyn Write) -> CliResult) -> CliResult<u64> {
    struct Counter<W: Write> { inner: W, n: u64 }
    impl<W: Write> Write for Counter<W> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { let k = self.inner.write(b)?; self.n += k as u64; Ok(k) }
        fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
    }
    if path == "-" {
        let mut out = Counter { inner: std::io::BufWriter::with_capacity(1 << 20, std::io::stdout().lock()), n: 0 };
        produce(&mut out)?;
        out.flush().map_err(|e| fail(format!("stdout: {e}")))?;
        return Ok(out.n);
    }
    let file = std::fs::File::create(path).map_err(|e| fail(format!("{path}: {e}")))?;
    let mut out = Counter { inner: std::io::BufWriter::with_capacity(1 << 20, file), n: 0 };
    let result = produce(&mut out).and_then(|_| out.flush().map_err(|e| fail(format!("{path}: {e}"))));
    if result.is_err() { drop(out); let _ = std::fs::remove_file(path); }
    result.map(|_| out_len(path))
}
fn out_len(path: &str) -> u64 { std::fs::metadata(path).map_or(0, |m| m.len()) }

fn read_input(path: &str) -> CliResult<Vec<u8>> {
    if path == "-" {
        let mut v = Vec::new();
        std::io::stdin().lock().read_to_end(&mut v).map_err(|e| fail(format!("stdin: {e}")))?;
        return Ok(v);
    }
    read_file_parallel(path, parallel::default_threads()).map_err(|e| fail(format!("{path}: {e}")))
}

/// `std::fs::read` in parallel pieces (positioned reads): the copy out of the OS cache and the
/// first-touch page faults of the buffer spread over `threads` cores instead of running serially
/// before decoding starts (~24 ms for a 126 MB file on one core, a fifth of a 24-thread decode). Small files take the plain path.
fn read_file_parallel(path: &str, threads: usize) -> std::io::Result<Vec<u8>> {
    const PIECE: usize = 4 << 20;
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len < 2 * PIECE || threads <= 1 {
        let mut v = Vec::with_capacity(len);
        (&file).read_to_end(&mut v)?;
        return Ok(v);
    }
    let mut data = vec![0u8; len];
    let pieces: Vec<(usize, &mut [u8])> = data.chunks_mut(PIECE).enumerate().map(|(i, c)| (i * PIECE, c)).collect();
    let pieces: Vec<std::sync::Mutex<(usize, &mut [u8])>> = pieces.into_iter().map(std::sync::Mutex::new).collect();
    let results = parallel::par_map(&pieces, threads, |m| -> std::io::Result<()> {
        let mut g = m.lock().unwrap();
        let (mut off, ref mut buf) = *g;
        let mut buf: &mut [u8] = buf;
        while !buf.is_empty() {
            #[cfg(windows)]
            let n = std::os::windows::fs::FileExt::seek_read(&file, buf, off as u64)?;
            #[cfg(unix)]
            let n = std::os::unix::fs::FileExt::read_at(&file, buf, off as u64)?;
            if n == 0 { return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "file shrank while reading")); }
            off += n;
            buf = &mut buf[n..];
        }
        Ok(())
    });
    results.into_iter().collect::<std::io::Result<()>>()?;
    drop(pieces);
    Ok(data)
}

fn open_reader(path: &str, data: Vec<u8>) -> CliResult<decoder::Reader<Vec<u8>>> {
    decoder::Reader::open(data).map_err(|e| fail(format!("{path}: {e}")))
}

/// The CLI's reader: a file (or stdin, held in memory) behind [`decoder::FileReader`], so the
/// compressed stream is never loaded whole.
type CliReader = decoder::FileReader<Box<dyn decoder::ReadSeek>>;

fn open_file_reader(path: &str) -> CliResult<CliReader> {
    let src: Box<dyn decoder::ReadSeek> = if path == "-" {
        Box::new(std::io::Cursor::new(read_input(path)?))
    } else {
        Box::new(std::io::BufReader::with_capacity(64 << 10, std::fs::File::open(path).map_err(|e| fail(format!("{path}: {e}")))?))
    };
    decoder::FileReader::open(src).map_err(|e| fail(format!("{path}: {e}")))
}

/// Writes sample-frames `range` of the stream as a WAV: chunks are read in order, decoded in
/// order-preserving parallel batches and written as they finish, so memory is a batch
/// of chunks (compressed and decoded), not the stream.
fn decode_range_to(out: &mut dyn Write, r: CliReader, range: std::ops::Range<u64>, threads: usize) -> CliResult {
    let h = r.header.clone();
    let werr = |e: std::io::Error| fail(format!("write error: {e}"));
    let frames = range.end - range.start;
    let float_info = r.metadata.float_info.clone();
    let float_info = float_info.as_ref();
    let header = match float_info {
        Some(_) => wav::header_bytes_float(h.channels as u16, h.sample_rate, r.metadata.channel_mask, frames).map_err(fail)?,
        None => wav::header_bytes(h.channels as u16, h.sample_rate, h.bits_per_sample, r.metadata.channel_mask, frames).map_err(fail)?,
    };
    out.write_all(&header).map_err(werr)?;
    if frames > 0 {
        let first = r.chunk_for_frame(range.start).ok_or_else(|| fail("range outside the stream"))?;
        let last = r.chunk_for_frame(range.end - 1).ok_or_else(|| fail("range outside the stream"))?;
        let shared = std::sync::Mutex::new(r);
        // Workers read their chunk (briefly holding the reader), decode and interleave it; this
        // thread only writes, in order. Written buffers go back to `pool`, so steady state
        // allocates (and page-faults) nothing new.
        let bits = h.bits_per_sample;
        let pool = std::sync::Mutex::new(Vec::<Vec<u8>>::new());
        let produce = |k: usize| -> Result<Vec<u8>, String> {
            let i = first + k;
            let mut payload = Vec::new();
            let (start, dec) = {
                let mut g = shared.lock().unwrap();
                g.read_payload(i, &mut payload).map_err(|e| format!("decode error: {e}"))?;
                (g.chunk_start(i), g.payload_decoder(i).map_err(|e| format!("decode error: {e}"))?)
            };
            let chunk = dec.decode(&payload).map_err(|e| format!("decode error: {e}"))?;
            drop(payload);
            let lo = range.start.saturating_sub(start) as usize;
            let hi = ((range.end - start) as usize).min(chunk[0].len());
            let part: Vec<&[i64]> = chunk.iter().map(|c| &c[lo..hi]).collect();
            let mut buf = pool.lock().unwrap().pop().unwrap_or_default();
            buf.clear();
            match float_info {
                Some(fi) => {
                    let sliced = fak::floatpcm::slice_info(fi, start + lo as u64, (hi - lo) as u64);
                    let f = fak::floatpcm::unmap_from_pcm(&part, &sliced);
                    wav::append_interleaved_float(&mut buf, &f);
                }
                None => wav::append_interleaved(&mut buf, &part, bits),
            }
            Ok(buf)
        };
        parallel::ordered_pipeline(last + 1 - first, threads, threads.max(1) * 4, produce, |_, bytes| {
            let bytes = bytes.map_err(fail)?;
            out.write_all(&bytes).map_err(werr)?;
            pool.lock().unwrap().push(bytes);
            Ok(())
        })?;
    }
    let bytes_per_sample = if float_info.is_some() { 4 } else { h.bits_per_sample as usize / 8 };
    if (frames as usize * h.channels as usize * bytes_per_sample) & 1 != 0 { out.write_all(&[0]).map_err(werr)?; }
    Ok(())
}

fn cmd_decode(args: &[String]) -> CliResult {
    let (mut threads, mut track, mut quiet, mut pos) = (parallel::default_threads(), None::<u8>, false, Vec::new());
    let mut a = Args::new(args);
    while let Some(arg) = a.next() {
        match arg {
            Arg::Pos(p) => pos.push(p),
            Arg::Opt(n, v) => match n.as_str() {
                "-t" | "--threads" => threads = parse_number(&n, &a.value(&n, v)?)?,
                "--track" => track = Some(parse_number(&n, &a.value(&n, v)?)?),
                "-q" | "--quiet" => quiet = true,
                _ => return usage(format!("unknown decode option {n}")),
            },
        }
    }
    expect_positionals(&pos, 2, 2, "decode <in.fak> <out.wav>")?;
    let (input, output) = (&pos[0], &pos[1]);
    let t0 = Instant::now();
    let r = open_file_reader(input)?;
    let range = match track {
        None => 0..r.total_frames,
        Some(n) => {
            let cue = r.metadata.cue_sheet.as_ref().ok_or_else(|| fail(format!("{input}: no cue sheet, so no tracks")))?;
            cue.track_range(n, r.total_frames).ok_or_else(|| {
                let have: Vec<String> = cue.tracks.iter().map(|t| t.number.to_string()).collect();
                fail(format!("{input}: no track {n} (tracks: {})", have.join(", ")))
            })?
        }
    };
    let frames = range.end - range.start;
    let rate = r.header.sample_rate;
    write_output(output, |out| decode_range_to(out, r, range, threads))?;
    if !quiet {
        let secs = frames as f64 / rate.max(1) as f64;
        eprintln!("{input} -> {output}{}: {:.1}s of audio, {:.0}x realtime", track.map_or(String::new(), |n| format!(" (track {n})")),
                  secs, secs / t0.elapsed().as_secs_f64().max(1e-9));
    }
    Ok(())
}

fn cmd_verify(args: &[String]) -> CliResult {
    let (mut threads, mut files) = (parallel::default_threads(), Vec::new());
    let mut a = Args::new(args);
    while let Some(arg) = a.next() {
        match arg {
            Arg::Pos(p) => files.push(p),
            Arg::Opt(n, v) if n == "-t" || n == "--threads" => threads = parse_number(&n, &a.value(&n, v)?)?,
            Arg::Opt(n, _) => return usage(format!("unknown verify option {n}")),
        }
    }
    if files.is_empty() { return usage("expected verify <in.fak>..."); }
    let mut failed = 0;
    for f in &files {
        match open_file_reader(f).and_then(|mut r| decoder::verify_file(&mut r, threads).map(|_| r.header.clone()).map_err(fail)) {
            Ok(h) => println!("{f}: OK ({} samples, {} ch, {}-bit, SHA-256 matches)", h.total_frames, h.channels, h.bits_per_sample),
            Err(CliError::Fail(e) | CliError::Usage(e)) => { println!("{f}: FAILED -- {e}"); failed += 1; }
        }
    }
    if failed > 0 { return Err(fail(format!("{failed} of {} files failed", files.len()))); }
    Ok(())
}

fn format_time(frames: u64, rate: u32) -> String {
    let ms = (frames as u128 * 1000 / rate.max(1) as u128) as u64;
    format!("{}:{:02}.{:03}", ms / 60_000, ms / 1000 % 60, ms % 1000)
}

fn cmd_info(args: &[String]) -> CliResult {
    let files: Vec<String> = args.to_vec();
    if files.is_empty() || files.iter().any(|f| f.starts_with('-') && f != "-") { return usage("expected info <in.fak>..."); }
    for (n, f) in files.iter().enumerate() {
        let r = open_file_reader(f)?;
        let size = r.len() as u64;
        let (h, m) = (&r.header, &r.metadata);
        let secs = r.total_frames as f64 / h.sample_rate.max(1) as f64;
        let pcm = r.total_frames * h.channels as u64 * (h.bits_per_sample as u64 / 8);
        if n > 0 { println!(); }
        println!("file:        {f}");
        println!("format:      FAK v{}, block mode", format::VERSION);
        println!("audio:       {} ch, {}-bit, {} Hz, {} ({} samples)", h.channels, h.bits_per_sample, h.sample_rate, format_time(r.total_frames, h.sample_rate), r.total_frames);
        println!("size:        {} ({:.2}% of PCM, {:.0} kbps)", human_bytes(size), 100.0 * size as f64 / pcm.max(1) as f64, size as f64 * 8.0 / secs.max(1e-9) / 1000.0);
        println!("chunks:      {} of up to {:.1}s{}", r.chunk_count(), r.chunk_frames(0) as f64 / h.sample_rate.max(1) as f64,
                 if r.parity_count() > 0 { format!(", FEC: {} parity blocks", r.parity_count()) } else { String::new() });
        if let Some(mask) = m.channel_mask { println!("channel mask: 0x{mask:x}"); }
        println!("audio SHA-256: {}", h.pcm_hash.iter().map(|b| format!("{b:02x}")).collect::<String>());
        if !m.vendor.is_empty() { println!("encoder:     {}", m.vendor); }
        if !m.tags.is_empty() { println!("tags:"); }
        for t in &m.tags {
            match t.split_once('=') {
                Some((k, v)) if k.eq_ignore_ascii_case(fak::metadata::CUESHEET_TAG) => println!("  {k}=<cue sheet, {} lines; `fak cue` prints it>", v.lines().count()),
                _ => println!("  {t}"),
            }
        }
        for (i, p) in m.pictures.iter().enumerate() {
            println!("picture {i}:   type {} ({:?}), {}, {} bytes{}", p.kind_raw, p.kind, p.mime, p.data.len(),
                     if p.description.is_empty() { String::new() } else { format!(", \"{}\"", p.description) });
        }
        if let Some(cue) = &m.cue_sheet {
            println!("cue sheet:   {} tracks{}", cue.tracks.len(), if cue.catalog.is_empty() { String::new() } else { format!(", catalog {}", cue.catalog) });
            for t in &cue.tracks {
                let Some(range) = cue.track_range(t.number, r.total_frames) else { continue };
                println!("  track {:02}  {} +{}{}", t.number, format_time(range.start, h.sample_rate), format_time(range.end - range.start, h.sample_rate),
                         if t.isrc.is_empty() { String::new() } else { format!("  ISRC {}", t.isrc) });
            }
        }
    }
    Ok(())
}

fn cmd_cue(args: &[String]) -> CliResult {
    expect_positionals(args, 1, 2, "cue <in.fak> [out.cue]")?;
    let r = open_reader(&args[0], read_input(&args[0])?)?;
    let stem = std::path::Path::new(&args[0]).file_stem().map_or("audio".into(), |s| s.to_string_lossy().into_owned());
    let text = r.metadata.cue_sheet_text(r.header.sample_rate, &format!("{stem}.wav")).ok_or_else(|| fail(format!("{}: no cue sheet", args[0])))?;
    match args.get(1) {
        Some(out) => std::fs::write(out, text.as_bytes()).map_err(|e| fail(format!("{out}: {e}"))),
        None => std::io::stdout().write_all(text.as_bytes()).map_err(|e| fail(format!("stdout: {e}"))),
    }
}

fn cmd_picture(args: &[String]) -> CliResult {
    expect_positionals(args, 3, 3, "picture <in.fak> <n> <out-file>")?;
    let r = open_reader(&args[0], read_input(&args[0])?)?;
    let i: usize = args[1].parse().map_err(|_| CliError::Usage(format!("picture number \"{}\" is not a number", args[1])))?;
    let p = r.metadata.pictures.get(i).ok_or_else(|| fail(format!("{}: no picture {i} ({} present)", args[0], r.metadata.pictures.len())))?;
    std::fs::write(&args[2], &p.data).map_err(|e| fail(format!("{}: {e}", args[2])))?;
    eprintln!("{} picture {i} -> {} ({} bytes, {})", args[0], args[2], p.data.len(), p.mime);
    Ok(())
}

fn cmd_edit(args: &[String]) -> CliResult {
    let (mut pos, mut output, mut ops) = (Vec::new(), None::<String>, Vec::<(String, Option<String>)>::new());
    let mut a = Args::new(args);
    while let Some(arg) = a.next() {
        let (name, inline) = match arg { Arg::Pos(p) => { pos.push(p); continue; } Arg::Opt(n, v) => (n, v) };
        match name.as_str() {
            "-o" | "--output" => output = Some(a.value(&name, inline)?),
            "-T" | "--tag" | "--set-tag" | "--remove-tag" | "--picture" | "--cuesheet" => { let v = a.value(&name, inline)?; ops.push((name, Some(v))); }
            "--remove-all-tags" | "--remove-pictures" | "--remove-cuesheet" | "--cd-tags" => ops.push((name, None)),
            _ => return usage(format!("unknown edit option {name}")),
        }
    }
    expect_positionals(&pos, 1, 1, "edit <in.fak> [options]")?;
    if ops.is_empty() { return usage("nothing to change (see the edit options in `fak help`)"); }
    let input = &pos[0];
    let r = open_reader(input, read_input(input)?)?;
    let mut m = r.metadata.clone();
    let had_tag = m.cuesheet_tag().is_some();
    let mut drop_cue = false;
    let key_is = tag_key_is;
    // CD tags (opt-in): written or recomputed by `--cd-tags`; kept current when a cue sheet
    // is set on a file that already has them (its FAK_ tags mark the opt-in); the FAK_ ones dropped
    // with the cue sheet. Keys this command sets itself are left to it.
    let opted_in = cdrip::has_fak_disc_tags(&r.metadata);
    let (mut cue_op, mut explicit, mut user_keys) = (None::<bool>, false, Vec::<String>::new());
    for (op, v) in ops {
        let v = v.unwrap_or_default();
        match op.as_str() {
            "-T" | "--tag" => { let kv = check_tag(&v)?; user_keys.push(kv.split_once('=').unwrap().0.to_string()); m.tags.push(kv); }
            "--set-tag" => { let kv = check_tag(&v)?; let key = kv.split_once('=').unwrap().0.to_string(); m.tags.retain(|t| !key_is(t, &key)); m.tags.push(kv); user_keys.push(key); }
            "--remove-tag" => m.tags.retain(|t| !key_is(t, &v)),
            "--remove-all-tags" => { m.tags.clear(); drop_cue = true; cue_op = None; explicit = false; }
            "--picture" => m.pictures.push(read_picture(&v)?),
            "--remove-pictures" => m.pictures.clear(),
            "--cuesheet" => { m.set_cuesheet_tag(Some(&read_text(&v)?)); drop_cue = false; cue_op = Some(true); }
            "--remove-cuesheet" => { m.set_cuesheet_tag(None); drop_cue = true; cue_op = Some(false); }
            "--cd-tags" => explicit = true,
            _ => unreachable!(),
        }
    }
    // The CUESHEET tag decides; a binary-only cue sheet (no tag to edit) is kept unless removed.
    m.sync_cue_sheet(r.header.sample_rate, r.total_frames, !had_tag && !drop_cue).map_err(fail)?;
    if drop_cue { m.cue_sheet = None; }
    let h = &r.header;
    let cd = if explicit || (cue_op == Some(true) && opted_in) { Some(true) } else if cue_op == Some(false) { Some(false) } else { None };
    match cd {
        Some(true) => {
            // Decoding (the checksums need the audio) only when the result can be a CD image.
            let exact = m.cue_sheet.as_ref().is_some_and(|c| h.channels == 2 && h.sample_rate == 44100 && h.bits_per_sample == 16
                && cdrip::Toc::from_cue(c, h.sample_rate, r.total_frames).is_some());
            let channels = if exact { decoder::decode_with_threads(r.data(), parallel::default_threads()).map_err(|e| fail(format!("{input}: {e}")))?.1 } else { Vec::new() };
            cdrip::apply_disc_tags(&mut m, &channels, h.sample_rate, h.bits_per_sample, &user_keys);
        }
        Some(false) => {
            cdrip::drop_fak_disc_tags(&mut m, &user_keys);
        }
        None => {}
    }
    let bytes = decoder::rewrite_metadata(r.data(), &m).map_err(|e| fail(format!("{input}: {e}")))?;
    let target = output.as_deref().unwrap_or(input);
    // Written beside the target and renamed over it, so a failure never leaves a half-written file.
    let tmp = format!("{target}.fak-edit-tmp");
    std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, target)).map_err(|e| { let _ = std::fs::remove_file(&tmp); fail(format!("{target}: {e}")) })?;
    eprintln!("{input} -> {target}: {} tags, {} pictures, {}", m.tags.len(), m.pictures.len(),
              m.cue_sheet.as_ref().map_or("no cue sheet".into(), |c| format!("cue sheet with {} tracks", c.tracks.len())));
    Ok(())
}

fn cmd_stream_encode(args: &[String]) -> CliResult {
    let (mut o, pos) = parse_encode_opts(args)?;
    expect_positionals(&pos, 1, 1, "stream-encode <in.wav> [options] > out.fak")?;
    let w = load_source(&pos[0], &mut o)?;
    let (mode, effort) = level_plan(&o);
    let chunk = chunk_frames(&o, w.sample_rate);
    let n = w.channels[0].len();
    // Never tells the encoder the length: a live source would push chunks the same way without
    // knowing it.
    let stdout = std::io::stdout();
    let mut enc = encoder::StreamEncoder::new(stdout.lock(), w.channels.len(), w.sample_rate, w.bits, mode, &o.metadata)
        .map_err(|e| fail(format!("stream-encode: {e}")))?.with_effort(effort);
    let mut s = 0usize;
    while s < n {
        let e = (s + chunk).min(n);
        let piece: Vec<Vec<i64>> = w.channels.iter().map(|c| c[s..e].to_vec()).collect();
        enc.push_chunk(&piece).map_err(|e| fail(format!("stream-encode: {e}")))?;
        s = e;
    }
    enc.finish().map(drop).map_err(|e| fail(format!("stream-encode: {e}")))?;
    Ok(())
}

fn cmd_stream_decode(args: &[String]) -> CliResult {
    expect_positionals(args, 1, 1, "stream-decode <out.wav> < in.fak")?;
    let mut channels: Vec<Vec<i64>> = Vec::new();
    let (header, meta) = decoder::decode_stream(std::io::stdin().lock(), |chunk| {
        if channels.is_empty() { channels = vec![Vec::new(); chunk.len()]; }
        for (dst, src) in channels.iter_mut().zip(chunk) { dst.extend_from_slice(src); }
        Ok(())
    }).map_err(|e| fail(format!("stream-decode: {e}")))?;
    let frames = channels.first().map_or(0, |c| c.len());
    let w = wav::Wav { channels, sample_rate: header.sample_rate, bits: header.bits_per_sample, channel_mask: meta.channel_mask, float_info: meta.float_info };
    wav::write_wav(&args[0], &w).map_err(|e| fail(format!("{}: {e}", args[0])))?;
    eprintln!("stdin -> {} ({frames} sample-frames)", args[0]);
    Ok(())
}

fn cmd_seek(args: &[String]) -> CliResult {
    expect_positionals(args, 3, 3, "seek <in.fak> <frame> <out.wav>")?;
    let data = read_input(&args[0])?;
    let target: u64 = args[1].parse().map_err(|_| CliError::Usage(format!("invalid frame {}", args[1])))?;
    let t0 = Instant::now();
    let r = decoder::seek(&data, target).map_err(|e| fail(format!("seek: {e}")))?;
    let elapsed = t0.elapsed();
    let offset = (target - r.chunk_start_frame) as usize;
    let channels: Vec<Vec<i64>> = r.channels.iter().map(|c| c[offset..].to_vec()).collect();
    let len = channels.first().map_or(0, |c| c.len()) as u64;
    let float_info = r.metadata.float_info.as_ref().map(|fi| fak::floatpcm::slice_info(fi, target, len));
    let w = wav::Wav { channels, sample_rate: r.header.sample_rate, bits: r.header.bits_per_sample, channel_mask: r.metadata.channel_mask, float_info };
    wav::write_wav(&args[2], &w).map_err(|e| fail(format!("{}: {e}", args[2])))?;
    println!("{} -> {} (from frame {target}, chunk started at {}): decoded in {:.2}ms", args[0], args[2], r.chunk_start_frame, elapsed.as_secs_f64() * 1000.0);
    Ok(())
}

fn cmd_cpuinfo() -> CliResult {
    let req = cpufeatures::compiled_requirements();
    let have = cpufeatures::detect();
    print!("{}", fak::cpuinfo::report());
    if (req.avx2 && !have.avx2) || (req.avx512f && !have.avx512f) || (req.bmi2 && !have.bmi2) || (req.fma && !have.fma) {
        return Err(fail("this binary requires instructions this CPU does not have -- it will crash with an illegal instruction"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() { Some((c, r)) => (c.as_str(), r), None => ("help", &[][..]) };
    let result = match cmd {
        "encode" => cmd_encode(rest),
        "decode" => cmd_decode(rest),
        "verify" | "test" => cmd_verify(rest),
        "info" => cmd_info(rest),
        "edit" => cmd_edit(rest),
        "cue" => cmd_cue(rest),
        "picture" => cmd_picture(rest),
        "stream-encode" => cmd_stream_encode(rest),
        "stream-decode" => cmd_stream_decode(rest),
        "seek" => cmd_seek(rest),
        "cpuinfo" => cmd_cpuinfo(),
        "help" | "-h" | "--help" => {
            print!("{}", HELP.replace("LEVEL_INSANE_HELP", LEVEL_INSANE_HELP));
            if rest.first().is_some_and(|a| a == "advanced") { print!("\n{HELP_ADVANCED}"); }
            Ok(())
        }
        "version" | "-V" | "--version" => {
            println!("fak {} (FAK format v{})", env!("CARGO_PKG_VERSION"), format::VERSION);
            Ok(())
        }
        _ => usage(format!("unknown command \"{cmd}\"")),
    };
    fak::prof::report();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(m)) => { eprintln!("fak: {m}\nrun `fak help` for usage"); ExitCode::from(2) }
        Err(CliError::Fail(m)) => { eprintln!("fak: {m}"); ExitCode::FAILURE }
    }
}

const LEVEL_INSANE_HELP: &str = "max plus an adaptive stage-2 filter per subframe (~0.5-2% smaller,\n                                 ~2x encode time, 3-8x decode time, ~8x on 24-bit: still ~40x realtime or better)";
