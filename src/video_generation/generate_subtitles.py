#!/usr/bin/env python3
"""
Production-Grade Subtitle Generation for YT-AutoPilot
Uses Vosk Hindi Model (vosk-model-hi-0.22)

Features:
- ASS format subtitles (Advanced SubStation Alpha) — supports animations & colors
- REAL-TIME WORD-LEVEL subtitles synced exactly with speech
- Max 2 lines per block, 5 words per line
- Colorful cycling text (yellow/cyan/green/orange...), black outline, NO background
- Fade-in animation per subtitle block
- YouTube Shorts 9:16 (1080×1920) center-positioned
- Accurate Hindi Devanagari output
"""
import os, sys, json, wave, argparse, subprocess, time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict
import gc

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    VOSK_AVAILABLE = True
    SetLogLevel(-1)
except ImportError:
    VOSK_AVAILABLE = False
    print("WARNING: Vosk not available - pip install vosk")

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = Path("models/vosk-model-hi-0.22")
OUTPUT_ASS = Path("output/subtitles.ass")
OUTPUT_SRT = Path("output/subtitles.srt")
TEMP_WAV   = Path("output/temp_audio_16k_mono.wav")
CHUNK_SIZE        = 2000
SAMPLE_RATE       = 16000
HEARTBEAT_INTERVAL = 10
WORDS_PER_LINE = 5
MAX_LINES      = 2

# Vivid colors for cycling subtitles (ASS BGR hex: &H00BBGGRR)
COLORS = [
    "&H00FFFF00",  # Yellow
    "&H00FFFF00",  # Yellow (doubled for more yellow)
    "&H0000FFFF",  # Cyan
    "&H0000FF7F",  # Spring green
    "&H00FF8C00",  # Orange
    "&H00FF69B4",  # Hot pink
    "&H0040E0D0",  # Turquoise
    "&H00ADFF2F",  # Green-yellow
]

# ============================================================================
# AUDIO DETECTION
# ============================================================================
def find_latest_audio_file(output_dir="output") -> Path:
    p = Path(output_dir)
    if not p.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    for name in ["audio.wav", "final_audio.wav"]:
        c = p / name
        if c.exists() and c.stat().st_size > 0:
            print(f"Using audio: {c}"); return c
    wav_files = [f for f in p.glob("*.wav") if f.stat().st_size > 0]
    if not wav_files:
        raise FileNotFoundError(f"No WAV file in '{output_dir}'")
    wav_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    print(f"Using audio: {wav_files[0]}"); return wav_files[0]

# ============================================================================
# UTILS
# ============================================================================
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ass_ts(sec: float) -> str:
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100: cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def srt_ts(sec: float) -> str:
    td = timedelta(seconds=sec)
    h  = td.seconds // 3600
    m  = (td.seconds % 3600) // 60
    s  = td.seconds % 60
    ms = td.microseconds // 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ============================================================================
# WORD GROUPING
# ============================================================================
def group_words(words: List[Dict]) -> List[Dict]:
    """Group into blocks of max 2 lines × 5 words"""
    if not words: return []
    max_per_block = WORDS_PER_LINE * MAX_LINES
    subtitles = []
    for i in range(0, len(words), max_per_block):
        block = words[i:i + max_per_block]
        if not block: continue
        lines = []
        for j in range(0, len(block), WORDS_PER_LINE):
            chunk = block[j:j + WORDS_PER_LINE]
            lines.append(' '.join(w['word'] for w in chunk))
        text  = '\\N'.join(lines)
        start = block[0]['start']
        end   = block[-1]['end']
        if end - start < 0.4:
            end = start + 0.6
        subtitles.append({'text': text, 'start': start, 'end': end})
    return subtitles

# ============================================================================
# ASS WRITER
# ============================================================================
def write_ass(subtitles: List[Dict], path: Path, video_type: str = "short") -> bool:
    try:
        # 1080x1920 coordinate space for Shorts
        # Alignment=8 → top-center; MarginV pushes down from top
        # We place at ~55% height (Y = 1056 from top in 1920px space)
        pos_y = 1056  # ~55% of 1920

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
            # FontSize=20, Bold, Yellow default, Black outline size 4, NO background (BorderStyle=1)
            # Alignment=8 = top-center (we override position with pos_y via MarginV)
            f"Style: Default,Noto Sans Devanagari,84,&H00FFFF00,&H000000FF,&H00000000,"
            f"&H00000000,1,0,0,0,100,100,0,0,1,5,0,8,40,40,{pos_y},1\n\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )

        event_lines = [header]
        for idx, sub in enumerate(subtitles):
            color = COLORS[idx % len(COLORS)]
            # \fad(80,60) = 80ms fade-in, 60ms fade-out
            # \c<color> = text color override for this block
            text = f"{{\\fad(80,60)\\c{color}}}{sub['text']}"
            event_lines.append(
                f"Dialogue: 0,{ass_ts(sub['start'])},{ass_ts(sub['end'])},"
                f"Default,,0,0,0,,{text}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(event_lines))
        log(f"ASS file written: {path} ({len(subtitles)} blocks)")
        return True
    except Exception as e:
        log(f"Failed to write ASS: {e}")
        import traceback; traceback.print_exc()
        return False

def write_srt_backup(subtitles: List[Dict], path: Path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for idx, sub in enumerate(subtitles, 1):
                text = sub['text'].replace('\\N', '\n')
                f.write(f"{idx}\n{srt_ts(sub['start'])} --> {srt_ts(sub['end'])}\n{text}\n\n")
        log(f"SRT backup: {path}")
    except Exception as e:
        log(f"SRT backup failed: {e}")

# ============================================================================
# MODEL CHECK
# ============================================================================
def verify_model() -> bool:
    if not MODEL_PATH.exists():
        log(f"Model not found: {MODEL_PATH}"); return False
    for req in ['am', 'conf/model.conf', 'graph']:
        if not (MODEL_PATH / req).exists():
            log(f"Model missing: {req}"); return False
    log("Vosk model verified"); return True

def convert_audio(src: Path, dst: Path) -> bool:
    log(f"Converting audio to 16kHz mono...")
    r = subprocess.run(
        ['ffmpeg', '-y', '-i', str(src), '-ar', str(SAMPLE_RATE),
         '-ac', '1', '-c:a', 'pcm_s16le', str(dst)],
        capture_output=True, text=True
    )
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    if not ok: log(f"Conversion failed: {r.stderr}")
    else: log("Audio converted")
    return ok

# ============================================================================
# RECOGNITION
# ============================================================================
def generate_subtitles(audio_path: Path, video_type: str = "short") -> bool:
    if not VOSK_AVAILABLE:
        log("Vosk not available"); return False

    log("Starting word-level ASS subtitle generation...")
    try:
        model = Model(str(MODEL_PATH))
        wf    = wave.open(str(audio_path), "rb")

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            log("Audio format mismatch"); return False

        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(True)

        all_words   = []
        last_beat   = time.time()

        while True:
            data = wf.readframes(CHUNK_SIZE)
            if not data: break
            if time.time() - last_beat > HEARTBEAT_INTERVAL:
                log(f"Processing... {wf.tell()/wf.getnframes()*100:.1f}%")
                last_beat = time.time()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if 'result' in res:
                    for w in res['result']:
                        if all(k in w for k in ['word','start','end']):
                            all_words.append(w)

        for w in json.loads(rec.FinalResult()).get('result', []):
            if all(k in w for k in ['word','start','end']):
                all_words.append(w)

        wf.close(); del model; gc.collect()

        log(f"Extracted {len(all_words)} words")
        if not all_words:
            log("No words detected"); return False

        grouped = group_words(all_words)
        log(f"Grouped into {len(grouped)} subtitle blocks")

        ok = write_ass(grouped, OUTPUT_ASS, video_type)
        if ok:
            write_srt_backup(grouped, OUTPUT_SRT)
        return ok

    except Exception as e:
        log(f"Generation failed: {e}")
        import traceback; traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',     required=True)
    parser.add_argument('--force',      action='store_true')
    parser.add_argument('--simple',     action='store_true')  # kept for compatibility
    parser.add_argument('--audio-dir',  default='output')
    parser.add_argument('--video-type', default='short', choices=['short','long'])
    args = parser.parse_args()

    log("=" * 60)
    log(f"SUBTITLE GENERATION — {args.run_id} | {args.video_type}")
    log("=" * 60)

    try:
        audio_path = find_latest_audio_file(args.audio_dir)
    except FileNotFoundError as e:
        log(str(e)); sys.exit(1)

    if not verify_model(): sys.exit(1)
    if not convert_audio(audio_path, TEMP_WAV): sys.exit(1)

    success = generate_subtitles(TEMP_WAV, args.video_type)

    if TEMP_WAV.exists(): TEMP_WAV.unlink()

    if success and OUTPUT_ASS.exists():
        log(f"DONE: {OUTPUT_ASS} ({OUTPUT_ASS.stat().st_size//1024} KB)")
        sys.exit(0)
    else:
        log("FAILED"); sys.exit(1)

if __name__ == '__main__':
    main()
