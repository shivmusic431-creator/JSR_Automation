#!/usr/bin/env python3
"""
Subtitle Generation — Forced Alignment with stable-ts

APPROACH: Transcription नहीं, Forced Alignment।
- script.json का exact text use होता है (words 100% correct)
- stable-ts का align() function audio से timing निकालता है
- Result: perfect word-level sync, no mismatch

Font:
- fc-query से runtime पर exact Devanagari font family detect
- fontsdir FFmpeg को direct directory देता है → Devanagari conjuncts सही render

Output:
  output/subtitles.ass   — colored animated ASS subtitles
  output/subtitles.srt   — backup SRT
  output/font_meta.json  — font info for edit_video.py
"""
import os, sys, json, subprocess, argparse, re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    import stable_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: stable-whisper not available — pip install stable-ts")

# ============================================================================
# CONFIG
# ============================================================================

MODEL_SIZE = "small"   # 244MB, accurate Hindi word boundaries
LANGUAGE   = "hi"

OUTPUT_ASS = Path("output/subtitles.ass")
OUTPUT_SRT = Path("output/subtitles.srt")
FONT_META  = Path("output/font_meta.json")
TEMP_WAV   = Path("output/temp_16k_mono.wav")

WORDS_PER_BLOCK = 5
MAX_LINES       = 2

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
]
FONT_DIR_CANDIDATES = [
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/noto",
    "/usr/share/fonts/truetype",
    "/usr/share/fonts",
]

def detect_font() -> Dict:
    font_path, font_dir, font_name = "", "/usr/share/fonts", "Noto Sans Devanagari"
    for fp in FONT_CANDIDATES:
        if Path(fp).exists():
            font_path = fp
            log(f"Font file: {fp}")
            break
    for fd in FONT_DIR_CANDIDATES:
        if Path(fd).exists():
            font_dir = fd
            break
    if font_path:
        try:
            r = subprocess.run(["fc-query", "--format=%{family}", font_path],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                detected = r.stdout.strip().split(",")[0].strip()
                if detected:
                    font_name = detected
        except Exception as e:
            log(f"fc-query failed: {e}")
    log(f"Font: '{font_name}' | dir: {font_dir}")
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
    raise FileNotFoundError(f"No WAV in {audio_dir}")


def convert_to_16k_mono(src: Path, dst: Path) -> bool:
    log(f"Converting to 16kHz mono...")
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
# SCRIPT TEXT EXTRACTION
# ============================================================================

def extract_plain_text(script_obj: Dict) -> str:
    """
    script.json से exact narration text निकालो।
    Emotion/scene markers हटाओ।
    यही text forced alignment में use होगा।
    """
    parts = []

    # Shorts format
    if "hook_line" in script_obj:
        if script_obj.get("hook_line"):
            parts.append(script_obj["hook_line"])
        for pt in script_obj.get("main_points", []):
            n = pt.get("narration") if isinstance(pt, dict) else str(pt)
            if n:
                parts.append(n)
        if script_obj.get("cta_line"):
            parts.append(script_obj["cta_line"])
        text = " ".join(parts)
    else:
        # Long format
        for key in ["hook", "problem_agitation", "promise"]:
            if script_obj.get(key):
                parts.append(script_obj[key])
        for sec in script_obj.get("main_content", []):
            c = sec.get("content") if isinstance(sec, dict) else sec
            if c:
                parts.append(c)
        for tip in script_obj.get("practical_tips", []):
            if isinstance(tip, dict):
                t = ""
                if tip.get("tip_title"):
                    t += tip["tip_title"] + "। "
                if tip.get("explanation"):
                    t += tip["explanation"]
                if t:
                    parts.append(t)
        if script_obj.get("conclusion"):
            parts.append(script_obj["conclusion"])
        text = " ".join(parts)

    # Prefer pre-built full_text
    if script_obj.get("full_text"):
        text = script_obj["full_text"]

    # Clean markers
    text = re.sub(r'\([^)]*\)', '', text)   # (emotion indicators)
    text = re.sub(r'\[[^\]]*\]', '', text)  # [SCENE:...] [PAUSE-N]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================================
# FORCED ALIGNMENT — CORE
# ============================================================================

def forced_align(audio_path: Path, script_text: str) -> List[Dict]:
    """
    stable-ts का align() use करके script text को audio से align करो।

    यह transcription नहीं है — model को exact words पहले से पता हैं।
    Model सिर्फ timing निकालता है।
    Result: words 100% correct, timing 95%+ accurate।
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("stable-whisper not installed — pip install stable-ts")

    log(f"Loading Whisper '{MODEL_SIZE}' model...")
    model = stable_whisper.load_model(MODEL_SIZE)
    log("Running forced alignment...")
    log(f"Script text ({len(script_text.split())} words): {script_text[:80]}...")

    # align() — exact text को audio से match करता है
    result = model.align(
        str(audio_path),
        script_text,
        language=LANGUAGE,
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

    log(f"Aligned {len(words)} words")
    return words

# ============================================================================
# SUBTITLE GROUPING
# ============================================================================

def group_words(words: List[Dict]) -> List[Dict]:
    """Words को subtitle blocks में group करो।"""
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
        if end - start < 0.3:
            end = start + 0.5
        blocks.append({"text": text, "start": start, "end": end})
    return blocks

# ============================================================================
# TIMING VALIDATION — no mismatch
# ============================================================================

def validate_timing(blocks: List[Dict], audio_duration: float) -> List[Dict]:
    """
    Subtitle timing validate और fix करो:
    1. Overlapping blocks fix करो
    2. Audio duration से बाहर जाने वाले blocks trim करो
    3. Negative duration blocks हटाओ
    """
    fixed = []
    for i, blk in enumerate(blocks):
        start = blk["start"]
        end   = blk["end"]

        # Audio boundary
        start = max(0.0, start)
        end   = min(audio_duration, end)

        # Overlap with previous block
        if fixed and start < fixed[-1]["end"]:
            start = fixed[-1]["end"] + 0.02  # 20ms gap

        # Minimum duration
        if end - start < 0.3:
            end = start + 0.5
            end = min(end, audio_duration)

        # Skip if still invalid
        if end <= start:
            log(f"  Skipping block {i} — invalid timing ({start:.2f} → {end:.2f})")
            continue

        fixed.append({"text": blk["text"], "start": round(start, 3), "end": round(end, 3)})

    log(f"Timing validated: {len(blocks)} → {len(fixed)} blocks")
    return fixed


def get_audio_duration(wav_path: Path) -> float:
    """ffprobe से audio duration निकालो।"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 60.0  # fallback

# ============================================================================
# ASS / SRT WRITERS
# ============================================================================

def ass_ts(sec: float) -> str:
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_ts(sec: float) -> str:
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_ass(blocks: List[Dict], path: Path, font: Dict) -> bool:
    try:
        pos_y     = 1056
        font_name = font["font_name"]

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
    parser.add_argument("--script-file", default="output/script.json")
    args = parser.parse_args()

    log("=" * 60)
    log(f"SUBTITLE GENERATION (Forced Alignment) — {args.run_id}")
    log("=" * 60)

    # 1. Find audio
    try:
        audio_path = find_audio(args.audio_dir)
        log(f"Audio: {audio_path}")
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    audio_duration = get_audio_duration(audio_path)
    log(f"Audio duration: {audio_duration:.1f}s")

    # 2. Load script text
    script_text = None
    script_file = Path(args.script_file)
    if script_file.exists():
        try:
            with open(script_file, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            script_obj = script_data.get("script", {})
            script_text = extract_plain_text(script_obj)
            log(f"Script loaded: {len(script_text.split())} words")
        except Exception as e:
            log(f"Script load failed: {e} — will use transcription fallback")
    else:
        log(f"script.json not found at {script_file} — will use transcription fallback")

    # 3. Convert audio to 16kHz mono
    if not convert_to_16k_mono(audio_path, TEMP_WAV):
        sys.exit(1)

    # 4. Detect font
    font = detect_font()

    # 5. Forced alignment (or transcription fallback)
    try:
        if script_text and len(script_text.split()) >= 20:
            log("Using FORCED ALIGNMENT (script text known)")
            words = forced_align(TEMP_WAV, script_text)
        else:
            log("Using TRANSCRIPTION fallback (no script text)")
            words = transcribe_fallback(TEMP_WAV)
    except Exception as e:
        log(f"Alignment failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        if TEMP_WAV.exists():
            TEMP_WAV.unlink()

    if not words:
        log("ERROR: No words aligned")
        sys.exit(1)

    # 6. Group → blocks
    blocks = group_words(words)
    log(f"Subtitle blocks: {len(blocks)}")

    # 7. Validate timing — no mismatch, no overlap
    blocks = validate_timing(blocks, audio_duration)

    # 8. Write ASS + SRT
    ok = write_ass(blocks, OUTPUT_ASS, font)
    if not ok:
        sys.exit(1)
    write_srt(blocks, OUTPUT_SRT)

    # 9. Save font meta for edit_video.py
    FONT_META.parent.mkdir(parents=True, exist_ok=True)
    FONT_META.write_text(json.dumps(font, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Font meta: {FONT_META}")

    log(f"DONE — {OUTPUT_ASS.stat().st_size // 1024} KB")
    sys.exit(0)


def transcribe_fallback(audio_path: Path) -> List[Dict]:
    """Fallback: pure transcription जब script text न हो।"""
    log("Transcribing (fallback mode)...")
    model = stable_whisper.load_model(MODEL_SIZE)
    result = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        word_timestamps=True,
        regroup=False,
    )
    words = []
    for seg in result.segments:
        for w in seg.words:
            wt = w.word.strip()
            if wt:
                words.append({"word": wt, "start": round(w.start, 3), "end": round(w.end, 3)})
    return words


if __name__ == "__main__":
    main()
