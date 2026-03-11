#!/usr/bin/env python3
"""
Audio Generation with XTTS v2 - Shorts-Optimized Single-Shot Version

Shorts (~57 seconds, ~185 words) ke liye:
- NO chunking — sentence by sentence XTTS call, natural pauses inserted
- Temperature 0.3 — no hallucination, stays on script
- Voice cloning always on (voices/my_voice.wav)
- Audio smoothing for clean output
"""

import os
import sys
import json
import argparse
import re
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from scipy.io.wavfile import write as write_wav, read as read_wav

try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("WARNING: XTTS not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

XTTS_MODEL_NAME         = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE        = 24000
XTTS_LANGUAGE           = "hi"
VOICE_CLONE_FILE        = "voices/my_voice.wav"

XTTS_SPEED              = 1.2    # slightly fast = energetic delivery
XTTS_TEMPERATURE        = 0.3    # LOW = no hallucination, no extra words
XTTS_REPETITION_PENALTY = 5.0

# Natural pause durations (seconds)
PAUSE_SENTENCE  = 0.45   # end of sentence (। ! ?)
PAUSE_SECTION   = 0.80   # between script sections ([PAUSE-1] marker)

OUTPUT_DIR       = Path("output")
FINAL_AUDIO_FILE = OUTPUT_DIR / "audio.wav"
TEMP_DIR         = OUTPUT_DIR / "audio_temp"
MAX_RETRIES      = 3

# ============================================================================
# UTILITIES
# ============================================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_text(text: str) -> str:
    """Remove emotion indicators, scene markers, pause markers from text."""
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on Hindi/English boundaries (। ! ?)"""
    parts = re.split(r'([।!?])', text)
    out = []
    i = 0
    while i < len(parts) - 1:
        s = (parts[i] + parts[i+1]).strip()
        if s:
            out.append(s)
        i += 2
    # trailing text without delimiter
    if len(parts) % 2 == 1 and parts[-1].strip():
        out.append(parts[-1].strip())
    return [s for s in out if s.strip()]


def make_silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * XTTS_SAMPLE_RATE), dtype=np.int16)


def smooth_audio(audio: np.ndarray) -> np.ndarray:
    """Soft clip + edge fade to prevent pop/crackle."""
    af = audio.astype(np.float32) / 32767.0
    af = np.clip(af, -0.95, 0.95)
    fade = min(400, max(1, len(af) // 100))
    af[:fade]  *= np.linspace(0, 1, fade)
    af[-fade:] *= np.linspace(1, 0, fade)
    return (af * 32767).astype(np.int16)


def assemble_script(script_obj: Dict) -> str:
    """Assemble narration text from script JSON (shorts or long format)."""
    if script_obj.get('full_text'):
        log("Using pre-built full_text")
        return script_obj['full_text']

    parts = []

    if 'hook_line' in script_obj:
        log("Shorts format detected")
        if script_obj.get('hook_line'):
            parts.append(script_obj['hook_line'])
        for pt in script_obj.get('main_points', []):
            narr = pt.get('narration') if isinstance(pt, dict) else str(pt)
            if narr:
                parts.append(narr)
        if script_obj.get('cta_line'):
            parts.append(script_obj['cta_line'])
        return ' [PAUSE-1] '.join(parts)

    log("Long-form format detected")
    for key in ['hook', 'problem_agitation', 'promise']:
        if script_obj.get(key):
            parts.append(script_obj[key])
    for sec in script_obj.get('main_content', []):
        c = sec.get('content') if isinstance(sec, dict) else sec
        if c:
            parts.append(c)
    for tip in script_obj.get('practical_tips', []):
        if isinstance(tip, dict):
            t = ''
            if tip.get('tip_title'):
                t += tip['tip_title'] + '। '
            if tip.get('explanation'):
                t += tip['explanation']
            if t:
                parts.append(t)
    if script_obj.get('conclusion'):
        parts.append(script_obj['conclusion'])
    return ' [PAUSE-2] '.join(parts)

# ============================================================================
# XTTS GENERATOR
# ============================================================================

class ShortsAudioGenerator:
    """
    Sentence-level XTTS generator with natural pauses.
    Each sentence = one XTTS call (short text = no token overflow, consistent voice).
    Silence arrays inserted between sentences and sections.
    """

    def __init__(self):
        self.tts: Optional[object] = None

    def load_model(self):
        if not XTTS_AVAILABLE:
            raise RuntimeError("XTTS not installed — run: pip install TTS")
        os.environ["COQUI_TOS_AGREED"] = "1"
        log("Loading XTTS v2 model...")
        self.tts = TTS(XTTS_MODEL_NAME)
        log("XTTS v2 loaded successfully")

    def generate(self, full_text: str) -> np.ndarray:
        if not self.tts:
            self.load_model()

        if not os.path.exists(VOICE_CLONE_FILE):
            raise FileNotFoundError(
                f"Voice clone file not found: {VOICE_CLONE_FILE}\n"
                "Add voices/my_voice.wav to the repository."
            )

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Split into sections on [PAUSE-N] markers
        sections = re.split(r'\[PAUSE-[123]\]', full_text)
        sections = [clean_text(s) for s in sections]
        sections = [s for s in sections if s]

        log(f"Sections: {len(sections)}")

        pieces: List[np.ndarray] = []
        seg_idx = 0

        for si, section in enumerate(sections):
            sentences = split_sentences(section)
            if not sentences:
                sentences = [section]

            log(f"  Section {si+1}/{len(sections)}: {len(sentences)} sentences")

            for qi, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                log(f"    [{seg_idx+1}] {sentence[:70]}{'...' if len(sentence)>70 else ''}")

                audio = self._synth(sentence, seg_idx)
                if audio is None:
                    log(f"    Segment {seg_idx} failed — using silence")
                    audio = make_silence(0.8)

                pieces.append(smooth_audio(audio))

                # Pause after sentence (not after the last sentence in section)
                if qi < len(sentences) - 1:
                    pieces.append(make_silence(PAUSE_SENTENCE))

                seg_idx += 1

            # Pause between sections
            if si < len(sections) - 1:
                pieces.append(make_silence(PAUSE_SECTION))

        if not pieces:
            raise RuntimeError("No audio generated")

        final = np.concatenate(pieces)
        final = smooth_audio(final)
        dur = len(final) / XTTS_SAMPLE_RATE
        log(f"Total audio duration: {dur:.1f}s")
        return final

    def _synth(self, text: str, idx: int) -> Optional[np.ndarray]:
        """Synthesize one sentence with retry."""
        tmp = TEMP_DIR / f"seg_{idx:04d}.wav"
        for attempt in range(MAX_RETRIES):
            try:
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=VOICE_CLONE_FILE,
                    language=XTTS_LANGUAGE,
                    file_path=str(tmp),
                    speed=XTTS_SPEED,
                    temperature=XTTS_TEMPERATURE,
                    repetition_penalty=XTTS_REPETITION_PENALTY,
                )
                rate, wav = read_wav(str(tmp))
                tmp.unlink(missing_ok=True)
                if rate != XTTS_SAMPLE_RATE:
                    factor = XTTS_SAMPLE_RATE / rate
                    idx2 = np.round(np.arange(0, len(wav), factor)).astype(int)
                    idx2 = idx2[idx2 < len(wav)]
                    wav = wav[idx2]
                return wav.astype(np.int16)
            except Exception as e:
                log(f"    Attempt {attempt+1}/{MAX_RETRIES}: {e}")
                tmp.unlink(missing_ok=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def cleanup(self):
        if TEMP_DIR.exists():
            for f in TEMP_DIR.glob("seg_*.wav"):
                f.unlink(missing_ok=True)
        gc.collect()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Shorts Audio Generator — XTTS v2 Single Shot')
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id',      required=True)
    args = parser.parse_args()

    log("=" * 60)
    log(f"AUDIO GENERATION — {args.run_id}")
    log("=" * 60)

    with open(args.script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)

    script_obj = script_data.get('script', {})
    if not script_obj:
        log("ERROR: No 'script' key in script.json")
        sys.exit(1)

    full_text = assemble_script(script_obj)
    word_count = len(full_text.split())
    log(f"Script: {word_count} words, {len(full_text)} chars")

    if word_count < 30:
        log("ERROR: Script too short")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gen = ShortsAudioGenerator()
    gen.load_model()

    try:
        audio = gen.generate(full_text)
    finally:
        gen.cleanup()

    write_wav(str(FINAL_AUDIO_FILE), XTTS_SAMPLE_RATE, audio)
    duration = len(audio) / XTTS_SAMPLE_RATE
    log(f"Saved: {FINAL_AUDIO_FILE} ({duration:.1f}s)")

    script_data['audio_info'] = {
        'duration_seconds': round(duration, 2),
        'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}",
        'sample_rate': XTTS_SAMPLE_RATE,
        'method': 'xtts_v2_sentence_level_single_shot',
        'temperature': XTTS_TEMPERATURE,
        'voice_clone_used': os.path.exists(VOICE_CLONE_FILE),
        'generated_at': datetime.now().isoformat(),
    }
    with open(args.script_file, 'w', encoding='utf-8') as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    log("Done")

if __name__ == '__main__':
    main()
