#!/usr/bin/env python3
"""
Subtitle Generation — Whisper + stable-ts
Replaces Vosk (poor Hindi accuracy) with OpenAI Whisper small model.

Why Whisper:
- Handles Hindi + transliterated words (टॉक्सिक, मेंटल, डिसीजन) correctly
- Word-level timestamps via stable-ts library
- 244MB model, ~25s processing time for 57s audio on GitHub Actions CPU

Font:
- Runtime fc-query se exact Devanagari font family name detect
- fontsdir parameter FFmpeg ko direct font directory deta hai
- fontconfig bypass hoti hai — Devanagari conjuncts sahi render hote hain

Output: output/subtitles.ass (ASS format, colored, animated)
        output/subtitles.srt (backup)
        output/font_meta.json (font info for edit_video.py)
"""
import os, sys, json, subprocess, time, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

try:
    import stable_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: stable-whisper not available — pip install stable-ts")

# ============================================================================
# CONFIG
# ============================================================================

MODEL_SIZE  = "small"          # 244MB, good Hindi accuracy
LANGUAGE    = "hi"             # Hindi — forces Hindi transcription

OUTPUT_ASS  = Path("output/subtitles.ass")
OUTPUT_SRT  = Path("output/subtitles.srt")
FONT_META   = Path("output/font_meta.json")
TEMP_WAV    = Path("output/temp_16k_mono.wav")

WORDS_PER_BLOCK = 5   # max words per subtitle block
MAX_LINES       = 2   # max lines per block

# Vivid cycling colors (ASS BGR hex)
COLORS = [
    "&H00FFFF00",  # Yellow
    "&H00FFFF00",  # Yellow (weighted)
    "&H0000FFFF",  # Cyan
    "&H0000FF7F",  # Spring green
    "&H00FF8C00",  # Orange
    "&H00FF69B4",  # Hot pink
    "&H0040E0D0",  # Turquoise
    "&H00ADFF2F",  # Green-yellow
]

# ============================================================================
# LOGGING
# ============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================================
# FONT DETECTION
# ============================================================================

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
]
FONT_DIR_CANDIDATES = [
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/noto",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
]
DEFAULT_FONT_NAME = "Noto Sans Devanagari"


def detect_font() -> Dict:
    """
    Detect best available Devanagari font at runtime.
    Returns dict with font_path, font_dir, font_name.
    """
    font_path = ""
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            font_path = fp
            log(f"Font file found: {fp}")
            break

    font_dir = "/usr/share/fonts"
    for fd in FONT_DIR_CANDIDATES:
        if Path(fd).exists():
            font_dir = fd
            break

    font_name = DEFAULT_FONT_NAME
    if font_path:
        try:
            r = subprocess.run(
                ["fc-query", "--format=%{family}", font_path],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                detected = r.stdout.strip().split(",")[0].strip()
                if detected:
                    font_name = detected
                    log(f"Font family detected via fc-query: {font_name}")
        except Exception as e:
            log(f"fc-query failed ({e}), using default: {font_name}")

    log(f"Font config — name: '{font_name}' | dir: {font_dir}")
    return {"font_path": font_path, "font_dir": font_dir, "font_name": font_name}

# ============================================================================
# AUDIO UTILS
# ============================================================================

def find_audio(audio_dir: str) -> Path:
    d = Path(audio_dir)
    for name in ["audio.wav", "final_audio.wav"]:
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            return p
    wavs = sorted(d.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    wavs = [w for w in wavs if w.stat().st_size > 0]
    if wavs:
        return wavs[0]
    raise FileNotFoundError(f"No WAV file in {audio_dir}")


def convert_to_16k_mono(src: Path, dst: Path) -> bool:
    log(f"Converting to 16kHz mono: {src} → {dst}")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True
    )
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    if not ok:
        log(f"Conversion failed: {r.stderr[-300:]}")
    return ok

# ============================================================================
# WHISPER TRANSCRIPTION
# ============================================================================

def transcribe(audio_path: Path) -> List[Dict]:
    """
    Use stable-whisper to get word-level timestamps.
    Returns list of {word, start, end} dicts.
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("stable-whisper not installed — pip install stable-ts")

    log(f"Loading Whisper '{MODEL_SIZE}' model...")
    model = stable_whisper.load_model(MODEL_SIZE)
    log("Transcribing...")

    result = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        word_timestamps=True,
        regroup=False,
    )

    words = []
    for seg in result.segments:
        for w in seg.words:
            word_text = w.word.strip()
            if word_text:
                words.append({
                    "word":  word_text,
                    "start": round(w.start, 3),
                    "end":   round(w.end,   3),
                })

    log(f"Transcribed {len(words)} words")
    return words

# ============================================================================
# SUBTITLE GROUPING
# ============================================================================

def group_words(words: List[Dict]) -> List[Dict]:
    """Group words into subtitle blocks (max WORDS_PER_BLOCK * MAX_LINES words)."""
    max_per_block = WORDS_PER_BLOCK * MAX_LINES
    blocks = []
    for i in range(0, len(words), max_per_block):
        chunk = words[i:i + max_per_block]
        if not chunk:
            continue
        lines = []
        for j in range(0, len(chunk), WORDS_PER_BLOCK):
            line_words = chunk[j:j + WORDS_PER_BLOCK]
            lines.append(" ".join(w["word"] for w in line_words))
        text  = "\\N".join(lines)
        start = chunk[0]["start"]
        end   = chunk[-1]["end"]
        if end - start < 0.4:
            end = start + 0.6
        blocks.append({"text": text, "start": start, "end": end})
    return blocks

# ============================================================================
# ASS WRITER
# ============================================================================

def ass_ts(sec: float) -> str:
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_ts(sec: float) -> str:
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    ms  = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_ass(blocks: List[Dict], path: Path, font: Dict) -> bool:
    try:
        pos_y      = 1056   # ~55% of 1920px height (Shorts 9:16)
        font_name  = font["font_name"]
        font_dir   = font["font_dir"]

        header = (
            "[Script Info]\n"
            "Title: JSR_Auto Subtitles\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "Collisions: Normal\n"
            "WrapStyle: 2\n\n"
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"Style: Default,{font_name},84,&H00FFFF00,&H000000FF,&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,5,0,8,40,40,{pos_y},1\n\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )

        lines = [header]
        for idx, blk in enumerate(blocks):
            color = COLORS[idx % len(COLORS)]
            text  = f"{{\\fad(80,60)\\c{color}}}{blk['text']}"
            lines.append(
                f"Dialogue: 0,{ass_ts(blk['start'])},{ass_ts(blk['end'])},"
                f"Default,,0,0,0,,{text}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        log(f"ASS written: {path} ({len(blocks)} blocks, font: '{font_name}')")
        return True
    except Exception as e:
        log(f"ASS write failed: {e}")
        import traceback; traceback.print_exc()
        return False


def write_srt(blocks: List[Dict], path: Path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            for i, blk in enumerate(blocks, 1):
                text = blk["text"].replace("\\N", "\n")
                f.write(f"{i}\n{srt_ts(blk['start'])} --> {srt_ts(blk['end'])}\n{text}\n\n")
        log(f"SRT written: {path}")
    except Exception as e:
        log(f"SRT write failed: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",     required=True)
    parser.add_argument("--audio-dir",  default="output")
    parser.add_argument("--video-type", default="short", choices=["short", "long"])
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log(f"SUBTITLE GENERATION (Whisper) — {args.run_id}")
    log("=" * 60)

    # 1. Find audio
    try:
        audio_path = find_audio(args.audio_dir)
        log(f"Audio: {audio_path}")
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    # 2. Convert to 16kHz mono for Whisper
    if not convert_to_16k_mono(audio_path, TEMP_WAV):
        sys.exit(1)

    # 3. Detect font
    font = detect_font()

    # 4. Transcribe with Whisper
    try:
        words = transcribe(TEMP_WAV)
    except Exception as e:
        log(f"Transcription failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        if TEMP_WAV.exists():
            TEMP_WAV.unlink()

    if not words:
        log("ERROR: No words transcribed")
        sys.exit(1)

    # 5. Group into subtitle blocks
    blocks = group_words(words)
    log(f"Subtitle blocks: {len(blocks)}")

    # 6. Write ASS + SRT
    ok = write_ass(blocks, OUTPUT_ASS, font)
    if not ok:
        sys.exit(1)
    write_srt(blocks, OUTPUT_SRT)

    # 7. Save font metadata for edit_video.py
    FONT_META.parent.mkdir(parents=True, exist_ok=True)
    FONT_META.write_text(json.dumps(font, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Font meta saved: {FONT_META}")

    log(f"DONE — {OUTPUT_ASS} ({OUTPUT_ASS.stat().st_size // 1024} KB)")
    sys.exit(0)


if __name__ == "__main__":
    main()
