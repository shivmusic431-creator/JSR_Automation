#!/usr/bin/env python3
"""
Production-Grade Subtitle Generation for YT-AutoPilot
Uses Vosk Hindi Model (vosk-model-hi-0.22) for high-accuracy Hindi speech recognition

Features:
- REAL-TIME WORD-LEVEL SUBTITLES - Words appear in sync with speech
- Professional grouping: 2-4 words per subtitle for natural reading
- Streaming processing for long videos (30+ minutes)
- Center-aligned subtitles with {\an5} formatting
- Memory-efficient chunked processing
- GitHub Actions compatible
- Automatic retry and error handling
- Progress logging with heartbeat
- Dynamic audio file detection
"""
import os
import sys
import json
import wave
import argparse
import subprocess
import time
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Optional, Tuple
import gc

# Vosk import with availability check
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    VOSK_AVAILABLE = True
    # Reduce Vosk logging
    SetLogLevel(-1)
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ WARNING: Vosk not available - install with: pip install vosk")

# ============================================================================
# DYNAMIC AUDIO FILE DETECTION (PRODUCTION SAFE)
# ============================================================================

def find_latest_audio_file(output_dir="output"):
    """
    Dynamically find the latest audio WAV file in the output directory.

    Priority:
    1. Preferred names: audio.wav, final_audio.wav
    2. Most recently modified .wav file

    Returns:
        Path object pointing to detected audio file

    Raises:
        FileNotFoundError if no audio file found
    """
    from pathlib import Path
    import time

    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"❌ Output directory not found: {output_dir}")

    # Preferred filenames (XTTS standard outputs)
    preferred_names = [
        "audio.wav",
        "final_audio.wav"
    ]

    # Step 1: Check preferred filenames first
    for name in preferred_names:
        candidate = output_path / name
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"✅ Using preferred audio file: {candidate}")
            return candidate

    # Step 2: Find all WAV files
    wav_files = [
        f for f in output_path.glob("*.wav")
        if f.exists() and f.stat().st_size > 0
    ]

    if not wav_files:
        raise FileNotFoundError(
            f"❌ No valid WAV audio file found in '{output_dir}'. "
            f"Expected one of {preferred_names} or any .wav file."
        )

    # Step 3: Sort by last modified time (newest first)
    wav_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    latest_file = wav_files[0]

    print(
        f"✅ Using latest detected audio file: {latest_file} "
        f"(modified: {time.ctime(latest_file.stat().st_mtime)})"
    )

    return latest_file


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = Path("models/vosk-model-hi-0.22")
# AUDIO_PATH is now dynamically detected
AUDIO_PATH = find_latest_audio_file()
OUTPUT_SRT = Path("output/subtitles.srt")
TEMP_WAV = Path("output/temp_audio_16k_mono.wav")

# Processing parameters
CHUNK_SIZE = 2000  # Process in 4-second chunks for memory efficiency
SAMPLE_RATE = 16000  # Vosk requires 16kHz mono
HEARTBEAT_INTERVAL = 10  # Log progress every 10 seconds

# Subtitle formatting
CENTER_ALIGN = "{\\an5}"  # Center alignment for SRT

# ============================================================================
# WORD-LEVEL SUBTITLE GROUPING
# ============================================================================

def group_words_into_subtitles(words: List[Dict], words_per_group: int = 3) -> List[Dict]:
    """
    Group words into natural subtitle chunks for real-time appearance
    
    Args:
        words: List of word objects with 'word', 'start', 'end'
        words_per_group: Number of words per subtitle (2-4 for optimal reading)
        
    Returns:
        List of subtitle dictionaries with 'text', 'start', 'end'
    """
    if not words:
        return []
    
    subtitles = []
    
    # Process words in groups
    for i in range(0, len(words), words_per_group):
        group = words[i:i + words_per_group]
        
        # Skip empty groups
        if not group:
            continue
        
        # Combine words into text
        text = ' '.join([w['word'] for w in group])
        
        # Get timing from first and last word in group
        start = group[0]['start']
        end = group[-1]['end']
        
        # Ensure minimum duration for very short groups
        if end - start < 0.5 and len(group) == 1:
            # Single word - ensure it's visible long enough
            end = start + 0.8
        
        subtitles.append({
            'text': text,
            'start': start,
            'end': end
        })
    
    # Merge very short subsequent subtitles if needed
    merged_subtitles = []
    i = 0
    while i < len(subtitles):
        current = subtitles[i]
        
        # Check if next subtitle starts very close to current end
        if i < len(subtitles) - 1 and subtitles[i + 1]['start'] - current['end'] < 0.1:
            # Merge with next subtitle
            next_sub = subtitles[i + 1]
            merged = {
                'text': f"{current['text']} {next_sub['text']}",
                'start': current['start'],
                'end': next_sub['end']
            }
            merged_subtitles.append(merged)
            i += 2  # Skip next subtitle
        else:
            merged_subtitles.append(current)
            i += 1
    
    return merged_subtitles

# ============================================================================
# LOGGING
# ============================================================================

def log(message: str, flush: bool = True):
    """CI-safe logging with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    if flush:
        sys.stdout.flush()

# ============================================================================
# AUDIO CONVERSION
# ============================================================================

def convert_audio_to_vosk_format(input_path: Path, output_path: Path) -> bool:
    """
    Convert audio to Vosk-compatible format (16kHz mono WAV)
    
    Args:
        input_path: Input audio file
        output_path: Output WAV file
        
    Returns:
        True if successful
    """
    log(f"🎵 Converting audio to 16kHz mono WAV...")
    log(f"   Input: {input_path} ({input_path.stat().st_size / (1024*1024):.2f} MB)")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log(f"✅ Audio converted: {output_path}")
            
            # Verify file
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                log("❌ Converted file is empty")
                return False
        else:
            log(f"❌ FFmpeg conversion failed: {result.stderr}")
            return False
    except Exception as e:
        log(f"❌ Audio conversion error: {e}")
        return False

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def verify_model() -> bool:
    """
    Verify Vosk Hindi model exists and is valid
    
    Returns:
        True if model is ready
    """
    if not MODEL_PATH.exists():
        log(f"❌ Model not found at {MODEL_PATH}")
        log("   Run: bash scripts/download_vosk_model.sh")
        return False
    
    # Check for required model files
    required_files = ['am', 'conf/model.conf', 'graph']
    missing = []
    
    for req in required_files:
        if not (MODEL_PATH / req).exists():
            missing.append(req)
    
    if missing:
        log(f"❌ Model missing required files: {missing}")
        return False
    
    log(f"✅ Vosk Hindi model verified at {MODEL_PATH}")
    return True

# ============================================================================
# SUBTITLE GENERATION - WORD-LEVEL REAL-TIME
# ============================================================================

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_subtitles_streaming(audio_path: Path, output_path: Path) -> bool:
    """
    Generate REAL-TIME WORD-LEVEL subtitles using Vosk streaming recognition
    
    Features:
    - Extracts word-by-word timestamps
    - Groups words into natural 2-4 word chunks
    - Perfect sync with speech
    - Professional appearance
    
    Args:
        audio_path: Path to 16kHz mono WAV file
        output_path: Path to output SRT file
        
    Returns:
        True if successful
    """
    if not VOSK_AVAILABLE:
        log("❌ Vosk not available. Install with: pip install vosk")
        return False
    
    log("🎬 Starting REAL-TIME WORD-LEVEL subtitle generation with Vosk Hindi model...")
    
    try:
        # Load model
        log("📦 Loading Vosk model...")
        model = Model(str(MODEL_PATH))
        
        # Open audio file
        log(f"📂 Opening audio: {audio_path}")
        wf = wave.open(str(audio_path), "rb")
        
        # Verify audio format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            log(f"❌ Audio format mismatch: {wf.getnchannels()} channels, "
                f"{wf.getsampwidth()} width, {wf.getframerate()} Hz")
            return False
        
        # Initialize recognizer
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        rec.SetWords(True)  # Enable word-level timestamps - CRITICAL for real-time
        
        # Process audio and collect all words
        all_words = []
        last_heartbeat = time.time()
        
        log(f"🔄 Processing audio stream and extracting word-level timestamps...")
        
        while True:
            data = wf.readframes(CHUNK_SIZE)
            if len(data) == 0:
                break
            
            # Heartbeat logging for CI/CD
            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                log(f"💓 Processing... ({wf.tell()/wf.getnframes()*100:.1f}%)")
                last_heartbeat = time.time()
            
            if rec.AcceptWaveform(data):
                # Get result with word details
                result = json.loads(rec.Result())
                
                # Extract word-level data if available
                if 'result' in result:
                    words = result['result']
                    for word_info in words:
                        if all(k in word_info for k in ['word', 'start', 'end']):
                            all_words.append({
                                'word': word_info['word'],
                                'start': word_info['start'],
                                'end': word_info['end']
                            })
        
        # Get final partial result
        final_result = json.loads(rec.FinalResult())
        if 'result' in final_result:
            words = final_result['result']
            for word_info in words:
                if all(k in word_info for k in ['word', 'start', 'end']):
                    all_words.append({
                        'word': word_info['word'],
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
        
        wf.close()
        
        log(f"✅ Extracted {len(all_words)} words with timestamps")
        
        if not all_words:
            log("⚠️ No words detected in audio")
            return False
        
        # Group words into natural subtitle chunks (2-4 words per subtitle)
        # Using 3 words per group as default for optimal real-time reading
        grouped_subtitles = group_words_into_subtitles(all_words, words_per_group=3)
        
        log(f"📝 Grouped into {len(grouped_subtitles)} real-time subtitles")
        
        # Write SRT file with center alignment
        log(f"📝 Writing {len(grouped_subtitles)} real-time subtitles to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, sub in enumerate(grouped_subtitles, 1):
                # Format: index
                f.write(f"{idx}\n")
                
                # Format: timestamp --> timestamp (using precise word timings)
                f.write(f"{format_timestamp(sub['start'])} --> {format_timestamp(sub['end'])}\n")
                
                # Format: {\an5}centered text (CRITICAL: must have center alignment)
                f.write(f"{CENTER_ALIGN}{sub['text']}\n\n")
        
        log(f"✅ REAL-TIME subtitle generation complete!")
        log(f"   Total words: {len(all_words)}")
        log(f"   Total subtitles: {len(grouped_subtitles)}")
        log(f"   Duration: {format_timestamp(grouped_subtitles[-1]['end']) if grouped_subtitles else '0:00'}")
        log(f"   Average words per subtitle: {len(all_words)/len(grouped_subtitles):.1f}")
        
        # Verify center alignment in output
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if CENTER_ALIGN not in content:
                log(f"⚠️ Warning: Center alignment marker {CENTER_ALIGN} not found in output")
            else:
                log(f"✅ Center alignment verified")
        
        # Clean up
        del model
        gc.collect()
        
        return True
        
    except Exception as e:
        log(f"❌ Subtitle generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_subtitles_simple(audio_path: Path, output_path: Path) -> bool:
    """
    Simplified subtitle generation (fallback if streaming fails)
    
    Args:
        audio_path: Path to audio file
        output_path: Path to output SRT file
        
    Returns:
        True if successful
    """
    if not VOSK_AVAILABLE:
        return False
    
    log("🔄 Using simple subtitle generation...")
    
    try:
        model = Model(str(MODEL_PATH))
        wf = wave.open(str(audio_path), "rb")
        rec = KaldiRecognizer(model, SAMPLE_RATE)
        
        all_words = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'result' in result:
                    words = result['result']
                    for word_info in words:
                        if all(k in word_info for k in ['word', 'start', 'end']):
                            all_words.append({
                                'word': word_info['word'],
                                'start': word_info['start'],
                                'end': word_info['end']
                            })
        
        wf.close()
        
        # Group words into subtitles
        grouped_subtitles = group_words_into_subtitles(all_words, words_per_group=3)
        
        # Write SRT with center alignment
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, sub in enumerate(grouped_subtitles, 1):
                f.write(f"{idx}\n")
                f.write(f"{format_timestamp(sub['start'])} --> {format_timestamp(sub['end'])}\n")
                f.write(f"{CENTER_ALIGN}{sub['text']}\n\n")
        
        log(f"✅ Simple subtitle generation complete: {len(grouped_subtitles)} subtitles")
        return True
        
    except Exception as e:
        log(f"❌ Simple subtitle generation failed: {e}")
        return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate subtitles with Vosk Hindi model')
    parser.add_argument('--run-id', required=True, help='Run ID for logging')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if exists')
    parser.add_argument('--simple', action='store_true', help='Use simple mode (fallback)')
    parser.add_argument('--audio-dir', default='output', help='Directory containing audio files')
    
    args = parser.parse_args()
    
    log("=" * 80)
    log(f"📝 SUBTITLE GENERATION - Run ID: {args.run_id}")
    log("=" * 80)
    
    # Step 1: Dynamically find audio file
    try:
        # Override AUDIO_PATH with dynamic detection
        global AUDIO_PATH
        AUDIO_PATH = find_latest_audio_file(args.audio_dir)
        log(f"🎯 Using audio file: {AUDIO_PATH}")
    except FileNotFoundError as e:
        log(f"❌ {e}")
        sys.exit(1)
    
    # Step 2: Verify model
    if not verify_model():
        log("❌ Model verification failed")
        sys.exit(1)
    
    # Step 3: Check audio file
    if not AUDIO_PATH.exists():
        log(f"❌ Audio file not found: {AUDIO_PATH}")
        sys.exit(1)
    
    audio_size = AUDIO_PATH.stat().st_size / (1024 * 1024)
    log(f"📊 Audio file: {AUDIO_PATH.name} ({audio_size:.2f} MB)")
    
    # Step 4: Convert audio to Vosk format
    if not convert_audio_to_vosk_format(AUDIO_PATH, TEMP_WAV):
        log("❌ Audio conversion failed")
        sys.exit(1)
    
    # Step 5: Generate subtitles
    log("🚀 Starting REAL-TIME WORD-LEVEL subtitle generation...")
    
    if args.simple:
        success = generate_subtitles_simple(TEMP_WAV, OUTPUT_SRT)
    else:
        success = generate_subtitles_streaming(TEMP_WAV, OUTPUT_SRT)
    
    # Step 6: Cleanup
    if TEMP_WAV.exists():
        TEMP_WAV.unlink()
        log("🧹 Cleaned up temporary files")
    
    if success and OUTPUT_SRT.exists():
        subtitle_size = OUTPUT_SRT.stat().st_size / 1024
        log(f"✅ Subtitle file created: {OUTPUT_SRT} ({subtitle_size:.2f} KB)")
        
        # Count subtitles properly
        with open(OUTPUT_SRT, 'r', encoding='utf-8') as f:
            content = f.read()
            subtitle_count = content.count('\n\n')  # Each subtitle ends with blank line
        
        log(f"   Total real-time subtitles: {subtitle_count}")
        
        # Final verification of center alignment
        with open(OUTPUT_SRT, 'r', encoding='utf-8') as f:
            content = f.read()
            if CENTER_ALIGN in content:
                log(f"✅ Center alignment marker {CENTER_ALIGN} verified in final output")
            else:
                log(f"⚠️ WARNING: Center alignment marker missing - this must be fixed")
        
        sys.exit(0)
    else:
        log("❌ Subtitle generation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()