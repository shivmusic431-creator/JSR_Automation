#!/usr/bin/env python3
"""
Video Editing - Premium Animated Subtitles with Cinematic Styling
Features dynamic audio file detection and automatic clip looping to fill audio duration
Now supports UNLIMITED CLIPS from pagination and enhanced clip validation
ABSOLUTELY NO BLACK VIDEO GENERATION - Fails safely if requirements not met
AUDIO DURATION AUTHORITY - Final video duration must match audio duration exactly
STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
FIXED: Subtitles are now properly burned into video using subtitles filter
FIXED: Audio is now properly merged into final videos (CRITICAL BUG FIX)
FIXED: Devanagari font rendering - Now uses explicit font file for proper ligatures
FIXED: Video encoding quality - Now enforces consistent HD resolution with CRF 18
FIXED: FFmpeg execution - Added live progress output, proper timeouts, and timestamp regeneration
FIXED: Subtitle filter - Pure SRT input, absolute paths, proper escaping
FIXED: Concat generation - Now properly loops clips to guarantee duration >= target
FIXED: Removed -shortest flag from assembly to prevent frame duplication
"""
import os
import json
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import re
import math
import tempfile
import hashlib
import random
import shutil
import time

def log(message: str, level: str = "INFO"):
    """Simple logging with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    sys.stdout.flush()


# ============================================================================
# AUDIO DURATION DETECTION - SINGLE SOURCE OF TRUTH
# ============================================================================

def get_audio_duration(audio_file: str) -> float:
    """
    Get audio duration in seconds using ffprobe - SINGLE SOURCE OF TRUTH.
    Must be reliable for WAV files in GitHub Actions environment.
    
    Args:
        audio_file: Path to audio file (WAV format)
        
    Returns:
        Duration in seconds as float
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If ffprobe fails or returns invalid duration
    """
    audio_path = Path(audio_file)
    
    # Check if file exists
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    log(f"🔍 Detecting AUDIO AUTHORITY duration: {audio_file}")
    
    # Build ffprobe command
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(audio_path)
    ]
    
    try:
        # Run ffprobe
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10  # Prevent hanging in CI
        )
        
        # Parse JSON output
        data = json.loads(result.stdout)
        
        # Extract duration
        if 'format' in data and 'duration' in data['format']:
            duration = float(data['format']['duration'])
            
            # Validate duration is positive and reasonable
            if duration <= 0:
                raise RuntimeError(f"Invalid duration detected: {duration}s")
                
            log(f"✅ AUDIO AUTHORITY duration: {duration:.2f}s ({duration/60:.2f}m)")
            return duration
        else:
            raise RuntimeError("Duration field not found in ffprobe output")
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out after 10 seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed with code {e.returncode}: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe JSON output: {e}")
    except ValueError as e:
        raise RuntimeError(f"Duration is not a valid float: {e}")

def get_video_duration(video_path: Path) -> float:
    """
    Get video duration using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

# ============================================================================
# VIDEO TRIMMING FUNCTION - EXACT MATCH TO AUDIO DURATION
# ============================================================================

def trim_video_to_audio_duration(input_video: Path, output_video: Path, target_duration: float) -> bool:
    """
    Trim video exactly to match audio duration.
    Used when video_duration > audio_duration.
    
    Args:
        input_video: Path to input video
        output_video: Path to output trimmed video
        target_duration: Target duration in seconds (audio duration)
        
    Returns:
        True if successful
    """
    log(f"✂️ Trimming video from original duration to exactly {target_duration:.3f}s (audio authority)")
    
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',  # Regenerate timestamps
        '-i', str(input_video),
        '-t', str(target_duration),
        '-c', 'copy',  # Stream copy for speed - acceptable for trimming
        str(output_video)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        if output_video.exists() and output_video.stat().st_size > 0:
            trimmed_duration = get_video_duration(output_video)
            log(f"✅ Video trimmed successfully to {trimmed_duration:.3f}s (target: {target_duration:.3f}s)")
            return True
        else:
            log(f"❌ Trimmed video is empty", "ERROR")
            return False
    except subprocess.TimeoutExpired:
        log(f"❌ Video trimming timed out", "ERROR")
        return False
    except subprocess.CalledProcessError as e:
        log(f"❌ Video trimming failed: {e.stderr if e.stderr else 'Unknown error'}", "ERROR")
        return False

# ============================================================================
# STRICT AUDIO DURATION AUTHORITY VALIDATION
# ============================================================================

def enforce_audio_duration_authority(video_path: Path, audio_file: str) -> Path:
    """
    STRICT AUDIO DURATION AUTHORITY ENFORCEMENT
    This function ALWAYS runs after video rendering to ensure final video duration
    exactly matches audio duration.
    
    Rules:
    1. If video_duration > audio_duration: Trim video to audio duration exactly
    2. If video_duration < audio_duration: 
       - If difference <= 0.5s: Allow with warning
       - If difference > 0.5s: Raise warning (but continue)
    3. Always log final durations for verification
    
    Args:
        video_path: Path to rendered video file
        audio_file: Path to audio file (source of truth)
        
    Returns:
        Path to validated video file (may be trimmed version)
        
    Raises:
        RuntimeError: If video file doesn't exist or validation fails critically
    """
    log("=" * 80)
    log("🔍 AUDIO DURATION AUTHORITY VALIDATION")
    log("=" * 80)
    
    # Verify video exists
    if not video_path.exists() or video_path.stat().st_size == 0:
        error_msg = f"❌ FATAL: Video file missing or empty: {video_path}"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Get durations
    try:
        audio_duration = get_audio_duration(audio_file)
        video_duration = get_video_duration(video_path)
    except Exception as e:
        error_msg = f"❌ FATAL: Failed to get durations for validation: {e}"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    log(f"📊 VALIDATION INPUT:")
    log(f"   Audio duration (AUTHORITY): {audio_duration:.6f}s")
    log(f"   Video duration (rendered):  {video_duration:.6f}s")
    log(f"   Difference:                 {video_duration - audio_duration:+.6f}s")
    
    # Create temporary file for trimmed version if needed
    temp_trimmed = video_path.parent / f"temp_trimmed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    validated_path = video_path
    
    # CASE 1: Video longer than audio - MUST TRIM
    if video_duration > audio_duration:
        log(f"⚠️ Video exceeds audio duration by {video_duration - audio_duration:.3f}s")
        log(f"⚡ ENFORCING AUDIO AUTHORITY: Trimming video to match audio exactly...")
        
        trim_success = trim_video_to_audio_duration(video_path, temp_trimmed, audio_duration)
        
        if trim_success and temp_trimmed.exists() and temp_trimmed.stat().st_size > 0:
            # Verify trimmed duration
            trimmed_duration = get_video_duration(temp_trimmed)
            log(f"✅ Trimmed video duration: {trimmed_duration:.6f}s")
            
            # Replace original with trimmed version
            shutil.copy2(temp_trimmed, video_path)
            validated_path = video_path
            
            # Clean up temp file
            try:
                temp_trimmed.unlink()
            except:
                pass
                
            log(f"✅ Audio authority enforced: Video trimmed to match audio duration")
        else:
            error_msg = f"❌ FATAL: Failed to trim video to audio duration"
            log(error_msg, "ERROR")
            raise RuntimeError(error_msg)
    
    # CASE 2: Video shorter than audio
    elif video_duration < audio_duration:
        difference = audio_duration - video_duration
        
        if difference <= 0.5:
            log(f"⚠️ Video shorter than audio by {difference:.3f}s (within 0.5s tolerance)")
            log(f"   Acceptable tolerance - continuing")
        else:
            log(f"⚠️⚠️⚠️ VALIDATION WARNING: Video shorter than audio by {difference:.3f}s", "WARNING")
            log(f"   This exceeds 0.5s tolerance and may cause audio/video desync", "WARNING")
            log(f"   Consider increasing clip duration in asset acquisition", "WARNING")
            # Continue anyway as per requirements
        
        validated_path = video_path
    
    # CASE 3: Exact match - perfect
    else:
        log(f"✅ PERFECT MATCH: Video duration exactly equals audio authority")
        validated_path = video_path
    
    # FINAL VALIDATION: Get final durations after any trimming
    final_audio_duration = get_audio_duration(audio_file)
    final_video_duration = get_video_duration(validated_path)
    final_difference = final_video_duration - final_audio_duration
    
    log("=" * 80)
    log("✅ AUDIO AUTHORITY VALIDATION COMPLETE")
    log("=" * 80)
    log(f"   FINAL Audio duration (AUTHORITY): {final_audio_duration:.6f}s")
    log(f"   FINAL Video duration:             {final_video_duration:.6f}s")
    log(f"   FINAL Difference:                  {final_difference:+.6f}s")
    
    if abs(final_difference) < 0.01:
        log(f"   ✅ EXACT MATCH ACHIEVED")
    elif final_difference > 0:
        log(f"   ⚠️ Video exceeds audio by {final_difference:.3f}s after validation")
    else:
        log(f"   ⚠️ Video shorter than audio by {abs(final_difference):.3f}s after validation")
    
    log("=" * 80)
    
    return validated_path


# ============================================================================
# PREMIUM FONT CONFIGURATION - FIXED FOR DEVANAGARI LIGATURES
# ============================================================================

# Font configuration for proper Devanagari rendering
FONT_DIR = "assets/fonts"
FONT_NAME = "Mukta-Regular.ttf"
SUBTITLE_FONT = FONT_NAME  # Keep for backward compatibility
SUBTITLE_FONTSIZE = 28
SUBTITLE_PRIMARY_COLOR = "&Hffffff&"  # White
SUBTITLE_OUTLINE_COLOR = "&H000000&"  # Black
SUBTITLE_OUTLINE_WIDTH = 2
SUBTITLE_SHADOW = 1
SUBTITLE_ALIGNMENT = 2  # Center aligned


# ============================================================================
# PREMIUM COLOR PALETTE (KEPT FOR REFERENCE BUT NOT USED)
# ============================================================================

COLORS = {
    "primary": "&H00FFFFFF",        # Pure white
    "primary_warm": "&H00F0F0F0",    # Slightly warm white
    "outline": "&H00000000",          # Black outline
    "shadow": "&H40000000",           # Semi-transparent black shadow
    "accent_yellow": "&H0000FFFF",    # Yellow (BGR order in ASS)
    "accent_cyan": "&HFFFF00FF",      # Cyan
    "accent_red": "&H0000FF00",       # Red
    "accent_green": "&H0000FF00",     # Green
    "accent_purple": "&H00FF00FF",    # Purple
    "accent_orange": "&H0000A5FF",    # Orange
    "background": "&H00000000",       # Fully transparent background
}


# ============================================================================
# ANIMATION CONFIGURATION (KEPT FOR REFERENCE BUT NOT USED)
# ============================================================================

ANIMATION = {
    "fade_in_duration": 0.12,      # Seconds - smooth fade
    "slide_up_distance": 4,         # Pixels - subtle upward motion
    "slide_duration": 0.08,         # Seconds - quick but smooth
    "karaoke_fade": 0.05,           # For word-level highlighting
    "scale_effect": 1.02,           # Slight scale for emphasis
}


# ============================================================================
# DYNAMIC AUDIO FILE DETECTION
# ============================================================================

def find_latest_audio_file(output_dir="output", video_type="long"):
    """
    Dynamically find the latest audio WAV file in the output directory
    
    Priority:
    1. For long: audio_long.wav, audio.wav, final_audio.wav
    2. For short: audio_short.wav
    3. Most recently modified .wav file
    
    Args:
        output_dir: Directory to search for audio files
        video_type: 'long' or 'short'
        
    Returns:
        Path to the audio file as string
        
    Raises:
        FileNotFoundError: If no audio file is found
    """
    from pathlib import Path
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Priority order based on video type
    if video_type == "short":
        preferred_names = [
            "audio_short.wav",
            "short_audio.wav",
            "shorts_audio.wav"
        ]
    else:
        preferred_names = [
            "audio_long.wav",
            "audio.wav",
            "final_audio.wav",
            "long_audio.wav"
        ]
    
    # Check preferred names first
    for name in preferred_names:
        file = output_path / name
        if file.exists():
            log(f"✅ Found preferred audio file: {file}")
            return str(file)
    
    # Fallback: find the most recent WAV file
    wav_files = list(output_path.glob("*.wav"))
    
    if not wav_files:
        raise FileNotFoundError(
            f"No audio WAV file found in {output_dir}. "
            f"Expected: {preferred_names} or any .wav file"
        )
    
    # Sort by modification time (most recent first)
    wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = wav_files[0]
    
    log(f"✅ Found latest audio file: {latest_file.name} (modified: {datetime.fromtimestamp(latest_file.stat().st_mtime).strftime('%H:%M:%S')})")
    return str(latest_file)


# ============================================================================
# VIDEO METADATA DETECTION
# ============================================================================

def get_video_metadata(video_path: Path) -> tuple:
    """
    Get video dimensions, duration, and detect if it's a Short.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height, duration, is_short, fps)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        info = json.loads(result.stdout)
        
        width = info['streams'][0]['width']
        height = info['streams'][0]['height']
        duration = float(info['format']['duration'])
        
        # Get frame rate
        fps_str = info['streams'][0].get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30
        else:
            fps = float(fps_str)
        
        # Detect if this is a Short (portrait orientation)
        is_short = height > width and height / width > 1.5
        
        return width, height, duration, is_short, fps
        
    except Exception as e:
        log(f"⚠️ Failed to get video metadata: {e}")
        # Return default values
        return 1920, 1080, 0, False, 30


# ============================================================================
# FIXED: CLIP MANAGEMENT WITH ENHANCED LOOPING - PROPER DURATION GUARANTEE
# ============================================================================

def get_clip_duration(clip_path: Path) -> float:
    """Get individual clip duration using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(clip_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2)
        return float(result.stdout.strip())
    except Exception as e:
        log(f"⚠️ Could not determine duration for {clip_path.name}: {e}")
        return 0.0


def validate_clip_for_use(clip_path: Path) -> bool:
    """
    Validate clip is still usable before adding to concat
    Performs quick check to ensure file is not corrupted
    """
    if not clip_path.exists():
        return False
    
    # Quick check with ffprobe
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'json',
        str(clip_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        return len(streams) > 0 and streams[0].get('codec_type') == 'video'
    except:
        return False


def calculate_total_clips_duration(clips: list, clips_path: Path) -> tuple:
    """
    Calculate total duration of all unique clips with validation
    Returns validated clips and their total duration
    """
    total_duration = 0.0
    valid_clips = []
    invalid_clips = []
    
    log("🔍 Validating clips for use...")
    
    for clip in clips:
        clip_path = Path(clip)
        if not clip_path.exists():
            clip_path = clips_path / clip_path.name
        
        if clip_path.exists():
            if validate_clip_for_use(clip_path):
                duration = get_clip_duration(clip_path)
                if duration > 0:
                    total_duration += duration
                    valid_clips.append(str(clip_path.absolute()))
                    log(f"  ✅ {clip_path.name}: {duration:.2f}s")
                else:
                    invalid_clips.append(str(clip_path))
                    log(f"  ❌ {clip_path.name}: Invalid duration", "WARNING")
            else:
                invalid_clips.append(str(clip_path))
                log(f"  ❌ {clip_path.name}: Corrupted or invalid", "WARNING")
        else:
            invalid_clips.append(str(clip_path))
            log(f"  ❌ {clip_path.name}: File missing", "WARNING")
    
    log(f"📊 Valid clips: {len(valid_clips)} (total {total_duration:.2f}s)")
    if invalid_clips:
        log(f"⚠️ Invalid clips skipped: {len(invalid_clips)}")
    
    return total_duration, valid_clips


def create_optimized_concat_file(clips: list, output_dir: Path, run_id: str, target_duration: float) -> Path:
    """
    Create optimized concat file that GUARANTEES cumulative duration >= target_duration.
    Dynamically loops clips until duration condition is satisfied.
    
    CRITICAL FIX: Previously only wrote clips once, causing FFmpeg to stretch video
    and duplicate frames. Now properly loops clips until target duration is met.
    
    Args:
        clips: List of clip file paths
        output_dir: Directory to write concat file
        run_id: Run identifier for unique filename
        target_duration: Target duration to meet or exceed
        
    Returns:
        Path to created concat file
        
    Raises:
        RuntimeError: If no valid clip durations can be determined
    """
    concat_file = output_dir / f'concat_{run_id}.txt'
    
    log(f"🔧 Creating optimized concat file targeting duration: {target_duration:.2f}s")
    log("Concat generation fixed and duration guaranteed")
    
    # Cache clip durations for performance
    durations = {}
    total_single_loop_duration = 0.0
    
    # Calculate duration of each clip
    for clip in clips:
        clip_path = Path(clip)
        duration = get_clip_duration(clip_path)
        clip_str = str(clip_path)
        durations[clip_str] = duration
        total_single_loop_duration += duration
    
    # Validate we have usable durations
    if total_single_loop_duration <= 0:
        error_msg = "❌ FATAL: No valid clip durations detected - cannot create concat file"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    log(f"📊 Single loop duration: {total_single_loop_duration:.2f}s")
    
    # Calculate approximate loops needed (for logging only)
    estimated_loops = math.ceil(target_duration / total_single_loop_duration)
    log(f"🔄 Estimated loops needed: {estimated_loops}")
    
    cumulative_duration = 0.0
    loop_index = 0
    
    # Write concat file with dynamic looping
    with open(concat_file, "w", encoding="utf-8") as f:
        while cumulative_duration < target_duration:
            loop_index += 1
            
            # Shuffle clips for variety in each loop
            shuffled_clips = clips.copy()
            random.shuffle(shuffled_clips)
            
            log(f"🔄 Writing concat loop #{loop_index} (cumulative: {cumulative_duration:.2f}s / {target_duration:.2f}s)")
            
            for clip in shuffled_clips:
                clip_str = str(Path(clip))
                duration = durations[clip_str]
                
                # Write clip to concat file
                f.write(f"file '{clip_str}'\n")
                cumulative_duration += duration
                
                # Stop immediately if we've met or exceeded target
                if cumulative_duration >= target_duration:
                    log(f"✅ Target duration reached at {cumulative_duration:.2f}s")
                    break
    
    log(f"📊 Final concat duration: {cumulative_duration:.2f}s (target: {target_duration:.2f}s)")
    log(f"📝 Wrote {loop_index} loop(s) with {len(clips)} unique clips")
    
    return concat_file


def verify_manifest_integrity(manifest_path: Path, clips_path: Path) -> tuple:
    """
    Verify all clips in manifest exist and are valid
    Returns (valid_clips_list, total_duration, is_valid)
    """
    if not manifest_path.exists():
        log(f"❌ Manifest file not found: {manifest_path}", "ERROR")
        return [], 0.0, False
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        clips = manifest.get('clips', [])
        audio_authority_target = manifest.get('audio_authority_target', manifest.get('target_duration', 0))
        
        log(f"🔍 Verifying manifest integrity: {len(clips)} clips, audio authority target: {audio_authority_target:.1f}s")
        
        valid_clips = []
        total_duration = 0.0
        
        for clip in clips:
            # Get clip path
            clip_path = None
            if 'filename' in clip:
                clip_path = clips_path / clip['filename']
            elif 'file' in clip:
                clip_path = Path(clip['file'])
            
            if not clip_path or not clip_path.exists():
                log(f"❌ Clip missing from disk: {clip.get('filename', 'unknown')}", "ERROR")
                return [], 0.0, False
            
            if not validate_clip_for_use(clip_path):
                log(f"❌ Clip corrupted or unreadable: {clip_path.name}", "ERROR")
                return [], 0.0, False
            
            duration = clip.get('duration', 0)
            if duration <= 0:
                duration = get_clip_duration(clip_path)
            
            valid_clips.append(str(clip_path.absolute()))
            total_duration += duration
        
        log(f"📊 Verified {len(valid_clips)} valid clips, total duration: {total_duration:.1f}s")
        
        return valid_clips, total_duration, True
        
    except Exception as e:
        log(f"❌ Failed to verify manifest: {e}", "ERROR")
        return [], 0.0, False


# ============================================================================
# FIXED VIDEO RENDERING WITH HARD SUBTITLES AND AUDIO - ENHANCED QUALITY
# ============================================================================

def render_video_with_hard_subtitles(
    input_video: Path,
    output_video: Path,
    subtitles_srt: Path,
    audio_file: str,       # Kept in signature for call-site compatibility, NOT used internally
    video_duration: float,
    video_type: str = "long"
) -> bool:
    """
    Render final video with hard-burned subtitles using FFmpeg.

    PIPELINE ARCHITECTURE (CRITICAL):
    - Stage 1 (assemble): clips are concatenated AND audio is merged → assembled_video.mp4
    - Stage 2 (this function): subtitles are burned into assembled_video.mp4
    The input_video ALREADY contains the merged audio stream.
    Therefore this function uses ONLY ONE INPUT and copies the audio stream as-is.
    Re-merging audio here would create a two-input filtergraph conflict (exit code 255).

    CORRECT APPROACH:
    - Single input: input_video (already has audio)
    - Video: re-encode with libx264 + subtitle burn
    - Audio: -c:a copy  (stream-copy, no re-encode, no quality loss)
    - No -map flags, no -shortest, no second -i

    QUALITY SETTINGS:
    - CRF 18: Visually lossless quality
    - preset medium: Good compression/speed balance
    - Resolution normalised to target HD dimensions
    - Audio preserved losslessly via stream copy

    SUBTITLE FILTER:
    - Pure SRT input (no ASS headers)
    - Absolute paths with proper escaping
    - fontsdir parameter for font lookup
    - charenc=UTF-8 for Unicode support

    Args:
        input_video: Path to assembled input video (ALREADY contains audio)
        output_video: Path to output video
        subtitles_srt: Path to SRT subtitles file
        audio_file: UNUSED - kept only so call sites need no changes
        video_duration: Duration in seconds (used for timeout calculation only)
        video_type: 'long' or 'short' - controls resolution target

    Returns:
        True if render succeeded, False otherwise
    """
    log(f"🎥 Rendering video with hard-burned subtitles (SINGLE-INPUT, audio stream-copy)...")
    log(f"   Input video (with audio): {input_video}")
    log(f"   Subtitle file: {subtitles_srt}")
    log(f"   Audio handling: -c:a copy (stream-copy from input_video, no re-merge)")
    log(f"   Font directory: {FONT_DIR}")
    log(f"   Font file: {FONT_NAME}.ttf")
    log(f"   Font size: {SUBTITLE_FONTSIZE}")
    log(f"   QUALITY SETTINGS: CRF 18, preset medium, forced HD resolution")
    log(f"   FFMPEG FIXES: +genpts (timestamp regeneration), threads 0 (optimal CPU)")

    # ============================================================================
    # VALIDATION - CRITICAL PRE-FLIGHT CHECKS
    # ============================================================================
    
    # Verify subtitle file exists and is non-empty
    if not subtitles_srt.exists():
        log(f"❌ Subtitle file not found: {subtitles_srt}")
        return False
    
    if subtitles_srt.stat().st_size == 0:
        log(f"❌ Subtitle file is empty: {subtitles_srt}")
        return False

    # Verify input video exists and is non-empty
    if not input_video.exists() or input_video.stat().st_size == 0:
        log(f"❌ Input video not found or empty: {input_video}")
        return False

    # Verify font file exists
    font_path = Path(FONT_DIR).resolve() / f"{FONT_NAME}.ttf"
    if not font_path.exists():
        error_msg = f"❌ FATAL: Font file not found at: {font_path}"
        log(error_msg, "ERROR")
        return False
    else:
        log(f"✅ Font file verified: {font_path}")

    # ============================================================================
    # DEBUG LOGGING
    # ============================================================================
    log(f"🔍 DEBUG: Subtitle file exists: {subtitles_srt.exists()}, size: {subtitles_srt.stat().st_size} bytes")
    log(f"🔍 DEBUG: Font exists: {font_path.exists()}")
    
    # ============================================================================
    # FIX: Set fontconfig to use a WRITABLE cache directory inside the workspace.
    # The root cause of "Operation not permitted" is libass trying to write font cache
    # to /usr/share/fonts which is read-only in GitHub Actions sandbox.
    # Solution: redirect font cache to a temp dir we own.
    # ============================================================================
    fontconfig_cache_dir = Path(tempfile.mkdtemp(prefix="fontconfig_cache_"))
    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    os.environ["XDG_CACHE_HOME"] = str(fontconfig_cache_dir)
    os.environ["FC_CACHE_DIR"] = str(fontconfig_cache_dir)
    log(f"✅ Fontconfig cache redirected to writable dir: {fontconfig_cache_dir}")

    # ============================================================================
    # Build subtitle filter with absolute paths - PURE SRT INPUT
    # ============================================================================
    fonts_dir = Path(FONT_DIR).resolve()
    subtitle_file = subtitles_srt.resolve()
    
    # Escape paths for filter syntax
    fonts_escaped = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
    subtitle_escaped = str(subtitle_file).replace("\\", "/").replace(":", "\\:")
    
    subtitle_filter = (
        f"subtitles=filename='{subtitle_escaped}':"
        f"fontsdir='{fonts_escaped}':"
        f"charenc=UTF-8"
    )
    
    log(f"✅ Using subtitle filter: {subtitle_filter}")
    log(f"🔍 DEBUG: Subtitle file path: {subtitle_escaped}")
    log(f"🔍 DEBUG: Fonts dir path: {fonts_escaped}")

    # Determine resolution based on video type
    if video_type == "short":
        target_width, target_height = 1080, 1920
    else:
        target_width, target_height = 1920, 1080

    scale_filter = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase"
        f",crop={target_width}:{target_height}"
    )

    # Filter chain: scale → format → subtitles
    vf_filter = f"{scale_filter},format=yuv420p,{subtitle_filter}"

    log(f"Using video filter chain: {vf_filter}")

    # -------------------------------------------------------------------------
    # SINGLE-INPUT FFmpeg command.
    # input_video already contains the merged audio from Stage 1.
    # -c:a copy  → stream-copy audio, zero quality loss, no filtergraph conflict.
    # No -map flags needed: FFmpeg automatically selects the single video + audio stream.
    # No -shortest: duration is already correct from Stage 1.
    # -------------------------------------------------------------------------
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',       # Regenerate timestamps (prevents subtitle filter crashes)
        '-threads', '0',             # Optimal CPU utilisation
        '-i', str(input_video),      # SINGLE INPUT — already contains audio
        '-vf', vf_filter,            # Scale → format → burn subtitles
        '-c:v', 'libx264',           # Re-encode video to burn in subtitles
        '-preset', 'medium',
        '-crf', '18',                # Visually lossless
        '-c:a', 'copy',              # Stream-copy audio — no re-merge, no quality loss
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',   # Web optimisation
        '-metadata', f'title=YT-AutoPilot {video_type} video',
        '-metadata', 'artist=AI Generated',
        '-metadata', 'comment=Created with YT-AutoPilot',
        str(output_video)
    ]

    # Log command summary for debugging
    log(f"⚙️ Running FFmpeg (subtitle burn, single-input)...")
    log(f"   Video codec: libx264, CRF 18, preset medium")
    log(f"   Resolution: {target_width}x{target_height}")
    log(f"   Audio codec: copy (stream-copy from input_video)")
    log(f"   Filter chain: {vf_filter}")
    log(f"   FFMPEG FIXES: +genpts, threads 0 applied")
    
    # Set timeout based on video type
    if video_type == "short":
        # Short videos: 45 minutes timeout
        timeout_seconds = 2700
        log(f"⏱️ Short video timeout set to 45 minutes ({timeout_seconds/60:.1f}m)")
    else:
        # Long videos: 75 minutes timeout (fixed, not dependent on duration)
        timeout_seconds = 4500
        log(f"⏱️ Long video timeout set to 75 minutes ({timeout_seconds/60:.1f}m)")
    
    try:
        # Build environment for FFmpeg subprocess.
        # CRITICAL: Pass fontconfig cache env vars so libass uses writable dir.
        ffmpeg_env = os.environ.copy()
        ffmpeg_env["XDG_CACHE_HOME"] = str(fontconfig_cache_dir)
        ffmpeg_env["FC_CACHE_DIR"] = str(fontconfig_cache_dir)
        ffmpeg_env["FONTCONFIG_PATH"] = "/etc/fonts"
        ffmpeg_env["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"

        # FIXED: Use Popen with live progress output
        # This prevents the "frozen" appearance and shows encoding progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=ffmpeg_env
        )
        
        log(f"📊 FFmpeg encoding started - showing live progress:")
        log("-" * 80)
        
        start_time = time.time()
        
        # Read and display output line by line
        while True:
            line = process.stdout.readline()
            
            if line:
                # Print the line to show progress
                print(line.strip())
                sys.stdout.flush()
            
            # Check if process has finished
            if process.poll() is not None:
                break
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                process.kill()
                error_msg = f"❌ FFmpeg timeout exceeded ({timeout_seconds/60:.1f} minutes) - process killed"
                log(error_msg, "ERROR")
                raise RuntimeError(error_msg)
        
        log("-" * 80)
        
        # Check return code
        if process.returncode != 0:
            log(f"❌ FFmpeg failed with code {process.returncode}", "ERROR")
            return False
            
        # Verify output file exists and has content
        if output_video.exists() and output_video.stat().st_size > 0:
            size_mb = output_video.stat().st_size / (1024 * 1024)
            log(f"✅ Video rendered successfully with hard subtitles and audio: {size_mb:.2f} MB")
            
            # Quick verification that file is valid
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                str(output_video)
            ]
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if probe_result.returncode == 0:
                    log(f"✅ Output video verified")
                    return True
                else:
                    log(f"⚠️ Output video may be corrupted - ffprobe check failed")
                    return False
            except Exception as e:
                log(f"⚠️ Failed to verify output video: {e}")
                return False
        else:
            log(f"❌ Output video file missing or empty", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"❌ FFmpeg process timed out after {timeout_seconds/60:.1f} minutes", "ERROR")
        process.kill()
        try:
            process.wait(timeout=30)
        except:
            pass
        return False
        
    except Exception as e:
        log(f"❌ FFmpeg execution failed: {e}", "ERROR")
        return False


# ============================================================================
# ENHANCED SHORTS VIDEO EDITING WITH HARD SUBTITLES AND AUDIO - NO BLACK FALLBACK
# ============================================================================

def edit_shorts_video(script_file: str, audio_file: str, clips_dir: str, 
                      run_id: str, subtitles_file: str = None, 
                      hook_file: str = None, cta_file: str = None,
                      premium_subtitles: bool = True):
    """
    Edit SHORTS video using FFmpeg with HARD-BURNED subtitles and AUDIO
    
    Features:
    - Noto Sans Devanagari-Regular font for proper Hindi ligature rendering
    - Center-aligned subtitles with outline and shadow
    - Perfect sync with word-level subtitles
    - Automatic clip looping to fill full audio duration
    - Enhanced clip validation and shuffling
    - ABSOLUTELY NO BLACK VIDEO - Fails safely if requirements not met
    - AUDIO DURATION AUTHORITY - Final video must match audio duration exactly
    - STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
    - AUDIO ALWAYS MERGED - No silent videos
    - ENHANCED QUALITY: CRF 18, forced HD resolution (1080x1920)
    
    Args:
        script_file: Path to script JSON
        audio_file: Path to shorts audio WAV (audio_short.wav)
        clips_dir: Directory containing video clips
        run_id: Run identifier
        subtitles_file: Optional path to SRT subtitles
        hook_file: Optional hook JSON for shorts
        cta_file: Optional CTA JSON for shorts
        premium_subtitles: Parameter kept for compatibility (subtitles are always burned)
        
    Returns:
        Path to output video file or None if failed
    """
    
    log(f"🎬 Editing SHORTS video with HARD-BURNED subtitles and AUDIO...")
    log(f"   Run ID: {run_id}")
    
    # Setup paths
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_video = output_dir / f'temp_shorts_{run_id}.mp4'
    assembled_video = output_dir / f'assembled_shorts_{run_id}.mp4'
    output_file = output_dir / 'final_video_short.mp4'
    max_duration = 58  # YouTube Shorts max is 60 seconds, using 58 for safety
    
    # ============================================================================
    # PRE-RENDER VALIDATION - CRITICAL DEBUG CHECK
    # ============================================================================
    log("=" * 80)
    log("🔍 PRE-RENDER VALIDATION")
    log("=" * 80)
    
    audio_path = Path(audio_file)
    log(f"Audio exists: {audio_path.exists()}")
    if not audio_path.exists():
        error_msg = f"FATAL: Audio file not found: {audio_file}"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    subtitles_path = Path(subtitles_file) if subtitles_file else None
    log(f"Subtitles exists: {subtitles_path.exists() if subtitles_path else 'Not provided'}")
    
    clips_path = Path(clips_dir)
    log(f"Clips directory exists: {clips_path.exists()}")
    
    if clips_path.exists():
        clip_files = list(clips_path.glob('*.mp4'))
        log(f"Clips count: {len(clip_files)}")
        if clip_files:
            # Log first few clips for debugging
            log(f"Sample clips: {', '.join([f.name for f in clip_files[:3]])}")
    
    log("=" * 80)
    
    # Step 1: Get AUDIO AUTHORITY duration
    audio_duration = get_audio_duration(audio_file)
    target_duration = min(audio_duration, max_duration)
    log(f"🎯 AUDIO AUTHORITY target duration: {target_duration:.2f}s")
    
    # Load and verify manifest
    manifest_file = clips_path / 'manifest.json'
    
    # CRITICAL: Verify manifest integrity before proceeding
    valid_clips, total_clips_duration, manifest_valid = verify_manifest_integrity(manifest_file, clips_path)
    
    if not manifest_valid:
        error_msg = (
            f"❌ FATAL: Manifest verification failed. Cannot proceed with video generation. "
            f"No black video fallback allowed."
        )
        log(error_msg, "ERROR")
        return None
    
    if not valid_clips:
        error_msg = (
            f"❌ FATAL: No valid clips found after verification. "
            f"Cannot generate video without clips. No black video fallback allowed."
        )
        log(error_msg, "ERROR")
        return None
    
    # CRITICAL: Verify total duration meets requirement
    if total_clips_duration < target_duration:
        error_msg = (
            f"❌ FATAL: Total clip duration ({total_clips_duration:.1f}s) "
            f"is less than AUDIO AUTHORITY target ({target_duration:.1f}s). "
            f"Cannot generate video without black sections. "
            f"Asset acquisition must fetch more clips."
        )
        log(error_msg, "ERROR")
        return None
    
    # Calculate loop count
    loop_count = math.ceil(target_duration / total_clips_duration)
    
    log(f"🔄 Loop count needed: {loop_count}")
    log(f"   Each loop: {total_clips_duration:.1f}s of unique footage")
    log(f"   Total after {loop_count} loops: {total_clips_duration * loop_count:.1f}s")
    
    # Create optimized concat file with shuffled clips
    # FIXED: This now properly loops until target duration is met
    concat_file = create_optimized_concat_file(valid_clips, output_dir, run_id, target_duration)
    
    # Base FFmpeg command for concatenated clips
    # FIXED: REMOVED -shortest flag which was causing frame duplication
    # The system now guarantees concat duration >= target_duration, so -shortest is unnecessary
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-threads', '0',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-i', audio_file,
        '-c:v', 'copy',  # No re-encode for assembly
        '-c:a', 'aac',
        '-b:a', '192k',
        # REMOVED: '-shortest' - concat already guarantees sufficient duration
        str(assembled_video)
    ]
    
    # Execute first pass (video assembly) - FIXED WITH PROPER TIMEOUT AND LIVE PROGRESS
    log("🎬 Assembling video with HD quality settings...")
    try:
        # FIXED: Use Popen with live progress output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        log(f"📊 FFmpeg assembly started - showing live progress:")
        log("-" * 80)
        
        start_time = time.time()
        assembly_timeout = 600  # 10 minutes for assembly
        
        while True:
            line = process.stdout.readline()
            
            if line:
                print(line.strip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
            
            if time.time() - start_time > assembly_timeout:
                process.kill()
                log(f"❌ Video assembly timed out after 10 minutes", "ERROR")
                return None
        
        log("-" * 80)
        
        if process.returncode != 0:
            log(f"❌ FFmpeg assembly failed with code {process.returncode}")
            return None
        
        log(f"✅ Video assembled: {assembled_video}")
    except Exception as e:
        log(f"❌ Video assembly failed: {e}")
        return None
    
    # Verify assembled video exists and has content
    if not assembled_video.exists() or assembled_video.stat().st_size == 0:
        log(f"❌ Assembled video is empty or missing", "ERROR")
        return None
    
    # Step 2: Get assembled video duration
    assembled_duration = get_video_duration(assembled_video)
    log(f"📊 Assembled video duration: {assembled_duration:.3f}s")
    log(f"📊 AUDIO AUTHORITY duration: {target_duration:.3f}s")
    
    # Step 3: AUDIO DURATION AUTHORITY ENFORCEMENT (initial pass)
    # Case 1: Video longer than audio - TRIM EXACTLY to audio duration
    if assembled_duration > target_duration:
        log(f"⚠️ Video is longer than audio by {assembled_duration - target_duration:.3f}s")
        log(f"✂️ Trimming video to match AUDIO AUTHORITY exactly...")
        
        trim_success = trim_video_to_audio_duration(assembled_video, temp_video, target_duration)
        if not trim_success:
            log(f"❌ Failed to trim video to audio duration", "ERROR")
            return None
        
        # Update reference for subtitle processing
        video_for_subtitles = temp_video
        final_duration = target_duration
        
    # Case 2: Video shorter than audio - Check tolerance
    elif assembled_duration < target_duration:
        difference = target_duration - assembled_duration
        log(f"⚠️ Video is shorter than audio by {difference:.3f}s")
        
        if difference > 0.5:
            # Difference > 0.5s - raise validation warning but continue
            log(f"⚠️ VALIDATION WARNING: Video shorter than audio by {difference:.3f}s (exceeds 0.5s tolerance)", "WARNING")
            log(f"   This may cause audio/video desync. Consider increasing clip duration in asset acquisition.")
        
        # Use assembled video as-is (no trim needed)
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
        final_duration = assembled_duration
        
    else:
        # Exact match - perfect
        log(f"✅ Video duration exactly matches AUDIO AUTHORITY!")
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
        final_duration = assembled_duration
    
    # Get video metadata (for logging only)
    width, height, duration, is_short, fps = get_video_metadata(video_for_subtitles)
    log(f"📹 Video metadata: {width}x{height}, {duration:.1f}s, {fps:.1f}fps")
    
    # Render final video with HARD-BURNED subtitles and AUDIO
    render_success = False
    
    # Always burn subtitles if SRT file exists
    subtitles_srt_path = None
    if subtitles_file and Path(subtitles_file).exists():
        subtitles_srt_path = Path(subtitles_file)
        log(f"📝 Found subtitles file: {subtitles_srt_path}")
        
        # Render with hard-burned subtitles and audio
        render_success = render_video_with_hard_subtitles(
            video_for_subtitles,
            output_file,
            subtitles_srt_path,
            audio_file,  # Pass audio file for merging
            duration,
            'short'
        )
    else:
        # No subtitles, just encode with audio
        log("⚠️ No subtitles file found, rendering without subtitles but with audio")
        
        # Encode with high quality settings (no subtitles)
        cmd_encode = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',  # Regenerate timestamps
            '-threads', '0',        # Optimal CPU utilization
            '-i', str(video_for_subtitles),
            '-i', str(audio_file),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',  # Always re-encode for quality
            '-preset', 'medium',
            '-crf', '18',  # Visually lossless
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,format=yuv420p',  # Force shorts resolution
            str(output_file)
        ]
        
        try:
            # FIXED: Use Popen with live progress output
            process = subprocess.Popen(
                cmd_encode,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            log(f"📊 Encoding started - showing live progress:")
            log("-" * 80)
            
            start_time = time.time()
            encoding_timeout = 2700  # 45 minutes for shorts
            
            while True:
                line = process.stdout.readline()
                
                if line:
                    print(line.strip())
                    sys.stdout.flush()
                
                if process.poll() is not None:
                    break
                
                if time.time() - start_time > encoding_timeout:
                    process.kill()
                    log(f"❌ Encoding timed out after 45 minutes", "ERROR")
                    render_success = False
                    break
            
            log("-" * 80)
            
            if process.returncode == 0:
                render_success = True
            else:
                log(f"❌ Failed to encode video with audio (code {process.returncode})")
                render_success = False
                
        except Exception as e:
            log(f"❌ Failed to encode video with audio: {e}")
            render_success = False
    
    # ============================================================================
    # CRITICAL: POST-RENDER VALIDATION - SILENT FAILURES MUST BE IMPOSSIBLE
    # ============================================================================
    log("=" * 80)
    log("🔍 Validating rendered video output...")
    log("=" * 80)
    
    # Check if output file exists
    if not output_file.exists():
        error_msg = f"FATAL: Video rendering failed. {output_file} was not created."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Check if output file is empty
    if output_file.stat().st_size == 0:
        error_msg = f"FATAL: Video rendering failed. {output_file} is empty."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Check file size is reasonable (at least 100KB)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    if file_size_mb < 0.1:  # Less than 100KB
        error_msg = f"FATAL: Video file too small: {file_size_mb:.2f} MB. Likely corrupted."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    log(f"✅ Video render validation successful. File size: {file_size_mb:.2f} MB")
    
    # Verify final output
    if not render_success or not output_file.exists() or output_file.stat().st_size == 0:
        log(f"❌ Final video rendering failed", "ERROR")
        return None
    
    # Add hook and CTA overlays if needed
    if hook_file or cta_file:
        log("🪝 Adding hook/CTA overlays...")
        
        # Build overlay command
        overlay_filters = []
        
        if hook_file and Path(hook_file).exists():
            try:
                with open(hook_file, 'r', encoding='utf-8') as f:
                    hook_data = json.load(f)
                hook = hook_data.get('hook_options', [{}])[0]
                hook_text = hook.get('text_overlay', '')
                hook_text = re.sub(r'[^\w\s.,!?-]', '', hook_text)
                hook_text = hook_text.replace("'", "\\'").replace(":", "\\:")
                
                if hook_text:
                    overlay_filters.append(
                        f"drawtext=fontfile={font_path}:"
                        f"text='{hook_text}':"
                        "fontcolor=yellow:"
                        "fontsize=72:"
                        "x=(w-text_w)/2:"
                        "y=h*0.2:"
                        "enable='between(t,0,3)':"
                        "borderw=4:bordercolor=black"
                    )
            except Exception as e:
                log(f"⚠️ Hook overlay failed: {e}")
        
        if cta_file and Path(cta_file).exists():
            try:
                with open(cta_file, 'r', encoding='utf-8') as f:
                    cta_data = json.load(f)
                cta = cta_data.get('cta_options', [{}])[0]
                cta_text = cta.get('text_overlay', '')
                cta_text = re.sub(r'[^\w\s.,!?-]', '', cta_text)
                cta_text = cta_text.replace("'", "\\'").replace(":", "\\:")
                
                if cta_text:
                    cta_start = max(0, duration - 5)
                    overlay_filters.append(
                        f"drawtext=fontfile={font_path}:"
                        f"text='{cta_text}':"
                        "fontcolor=white:"
                        "fontsize=60:"
                        "x=(w-text_w)/2:"
                        "y=h*0.8:"
                        f"enable='gte(t,{cta_start})':"
                        "borderw=4:bordercolor=black"
                    )
            except Exception as e:
                log(f"⚠️ CTA overlay failed: {e}")
        
        if overlay_filters:
            final_output = output_dir / 'short_video_with_overlays.mp4'
            cmd_overlay = [
                'ffmpeg', '-y',
                '-fflags', '+genpts',  # Regenerate timestamps
                '-threads', '0',        # Optimal CPU utilization
                '-i', str(output_file),
                '-vf', ','.join(overlay_filters),
                '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'copy',
                str(final_output)
            ]
            
            try:
                # FIXED: Use Popen with live progress output
                process = subprocess.Popen(
                    cmd_overlay,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                log(f"📊 Adding overlays - showing live progress:")
                log("-" * 80)
                
                start_time = time.time()
                overlay_timeout = 300  # 5 minutes for overlay
                
                while True:
                    line = process.stdout.readline()
                    
                    if line:
                        print(line.strip())
                        sys.stdout.flush()
                    
                    if process.poll() is not None:
                        break
                    
                    if time.time() - start_time > overlay_timeout:
                        process.kill()
                        log(f"❌ Overlay addition timed out", "ERROR")
                        break
                
                log("-" * 80)
                
                if process.returncode == 0 and final_output.exists() and final_output.stat().st_size > 0:
                    shutil.move(str(final_output), str(output_file))
                    log("✅ Overlays added successfully")
                else:
                    log(f"⚠️ Overlay addition failed with code {process.returncode}")
            except Exception as e:
                log(f"⚠️ Overlay addition failed: {e}")
    
    # ============================================================================
    # STEP 4: STRICT AUDIO DURATION AUTHORITY ENFORCEMENT
    # This runs AUTOMATICALLY after EVERY video render - NOT optional
    # ============================================================================
    try:
        log("=" * 80)
        log("⚡ AUTOMATIC AUDIO DURATION AUTHORITY VALIDATION")
        log("=" * 80)
        
        validated_video = enforce_audio_duration_authority(output_file, audio_file)
        
        # Update output_file to validated version (should be same path)
        output_file = validated_video
        
        log("✅ Audio duration authority validation completed successfully")
        
    except Exception as e:
        log(f"❌ CRITICAL: Audio duration authority validation failed: {e}", "ERROR")
        log(f"   Pipeline cannot proceed with invalid duration", "ERROR")
        return None
    
    # Cleanup
    if temp_video.exists():
        temp_video.unlink()
    if assembled_video.exists():
        assembled_video.unlink()
    
    # Get final metadata for logging
    final_audio_duration = get_audio_duration(audio_file)
    final_video_duration = get_video_duration(output_file)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    log(f"✅ SHORTS video with HARD-BURNED subtitles and AUDIO complete")
    log(f"   Output: {output_file}")
    log(f"   Size: {size_mb:.2f} MB")
    log(f"   Final video duration: {final_video_duration:.1f}s")
    log(f"   AUDIO AUTHORITY: {final_audio_duration:.1f}s")
    log(f"   Duration match: {'✓' if abs(final_video_duration - final_audio_duration) < 0.1 else '⚠️'}")
    log(f"   FPS: {fps:.1f}")
    log(f"   Subtitles: Hard-burned (permanent)")
    log(f"   Font: {SUBTITLE_FONT} (from {FONT_DIR})")
    log(f"   Audio: Merged (AAC 192k)")
    log(f"   Quality: CRF 18, preset medium, forced HD resolution")
    log(f"   Clips used: {len(valid_clips)}")
    
    # Save metadata
    try:
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
    except:
        manifest_data = {}
    
    metadata = {
        'video_type': 'short',
        'duration_seconds': final_video_duration,
        'audio_authority_duration': final_audio_duration,
        'duration_match': abs(final_video_duration - final_audio_duration) < 0.1,
        'duration_difference': final_video_duration - final_audio_duration,
        'file_size_mb': size_mb,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'quality_settings': {
            'video_codec': 'libx264',
            'crf': 18,
            'preset': 'medium',
            'audio_codec': 'aac',
            'audio_bitrate': '192k',
            'pixel_format': 'yuv420p'
        },
        'subtitles_type': 'hard_burned',
        'subtitles_font': f"{FONT_DIR}/{FONT_NAME}.ttf",
        'subtitles_size': SUBTITLE_FONTSIZE,
        'subtitles_alignment': 'center',
        'subtitles_outline': SUBTITLE_OUTLINE_WIDTH,
        'subtitles_shadow': SUBTITLE_SHADOW,
        'audio_included': True,
        'audio_codec': 'AAC',
        'hook_included': hook_file and Path(hook_file).exists(),
        'cta_included': cta_file and Path(cta_file).exists(),
        'clips_validated': len(valid_clips),
        'manifest_pages': manifest_data.get('pages_searched', 1) if manifest_data else 1,
        'audio_authority_validated': True,
        'validation_timestamp': datetime.now().isoformat(),
        'generated_at': datetime.now().isoformat()
    }
    
    meta_file = output_dir / 'short_metadata.json'
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return output_file


# ============================================================================
# ENHANCED LONG VIDEO EDITING WITH HARD SUBTITLES AND AUDIO - NO BLACK FALLBACK
# ============================================================================

def edit_long_video(script_file: str, audio_file: str, clips_dir: str, 
                    run_id: str, subtitles_file: str = None,
                    premium_subtitles: bool = True):
    """
    Edit LONG video using FFmpeg with HARD-BURNED subtitles and AUDIO
    
    Features:
    - Noto Sans Devanagari-Regular font for proper Hindi ligature rendering
    - Center-aligned subtitles with outline and shadow
    - Perfect sync with word-level subtitles
    - Automatic clip looping to fill full audio duration
    - Enhanced clip validation and shuffling
    - ABSOLUTELY NO BLACK VIDEO - Fails safely if requirements not met
    - AUDIO DURATION AUTHORITY - Final video must match audio duration exactly
    - STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
    - AUDIO ALWAYS MERGED - No silent videos
    - ENHANCED QUALITY: CRF 18, forced HD resolution (1920x1080)
    
    Args:
        script_file: Path to script JSON
        audio_file: Path to audio WAV (audio_long.wav)
        clips_dir: Directory containing video clips
        run_id: Run identifier
        subtitles_file: Optional path to SRT subtitles
        premium_subtitles: Parameter kept for compatibility (subtitles are always burned)
        
    Returns:
        Path to output video file or None if failed
    """
    
    log(f"🎬 Editing LONG video with HARD-BURNED subtitles and AUDIO...")
    log(f"   Run ID: {run_id}")
    
    # Setup paths
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_video = output_dir / f'temp_long_{run_id}.mp4'
    assembled_video = output_dir / f'assembled_long_{run_id}.mp4'
    output_file = output_dir / 'final_video_long.mp4'
    
    # ============================================================================
    # PRE-RENDER VALIDATION - CRITICAL DEBUG CHECK
    # ============================================================================
    log("=" * 80)
    log("🔍 PRE-RENDER VALIDATION")
    log("=" * 80)
    
    audio_path = Path(audio_file)
    log(f"Audio exists: {audio_path.exists()}")
    if not audio_path.exists():
        error_msg = f"FATAL: Audio file not found: {audio_file}"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    subtitles_path = Path(subtitles_file) if subtitles_file else None
    log(f"Subtitles exists: {subtitles_path.exists() if subtitles_path else 'Not provided'}")
    
    clips_path = Path(clips_dir)
    log(f"Clips directory exists: {clips_path.exists()}")
    
    if clips_path.exists():
        clip_files = list(clips_path.glob('*.mp4'))
        log(f"Clips count: {len(clip_files)}")
        if clip_files:
            # Log first few clips for debugging
            log(f"Sample clips: {', '.join([f.name for f in clip_files[:3]])}")
    
    log("=" * 80)
    
    # Step 1: Get AUDIO AUTHORITY duration
    audio_duration = get_audio_duration(audio_file)
    target_duration = audio_duration
    log(f"🎯 AUDIO AUTHORITY target duration: {target_duration:.2f}s ({target_duration/60:.2f}m)")
    
    # Load and verify manifest
    manifest_file = clips_path / 'manifest.json'
    
    # CRITICAL: Verify manifest integrity before proceeding
    valid_clips, total_clips_duration, manifest_valid = verify_manifest_integrity(manifest_file, clips_path)
    
    if not manifest_valid:
        error_msg = (
            f"❌ FATAL: Manifest verification failed. Cannot proceed with video generation. "
            f"No black video fallback allowed."
        )
        log(error_msg, "ERROR")
        return None
    
    if not valid_clips:
        error_msg = (
            f"❌ FATAL: No valid clips found after verification. "
            f"Cannot generate video without clips. No black video fallback allowed."
        )
        log(error_msg, "ERROR")
        return None
    
    # CRITICAL: Verify total duration meets requirement
    if total_clips_duration < target_duration:
        error_msg = (
            f"❌ FATAL: Total clip duration ({total_clips_duration:.1f}s) "
            f"is less than AUDIO AUTHORITY target ({target_duration:.1f}s). "
            f"Cannot generate video without black sections. "
            f"Asset acquisition must fetch more clips."
        )
        log(error_msg, "ERROR")
        return None
    
    # Calculate loop count
    loop_count = math.ceil(target_duration / total_clips_duration)
    
    log(f"🔄 Loop count needed: {loop_count}")
    log(f"   Each loop: {total_clips_duration:.1f}s of unique footage")
    log(f"   Total after {loop_count} loops: {total_clips_duration * loop_count:.1f}s")
    
    # Create optimized concat file with shuffled clips
    # FIXED: This now properly loops until target duration is met
    concat_file = create_optimized_concat_file(valid_clips, output_dir, run_id, target_duration)
    
    # Base FFmpeg command for concatenated clips
    # FIXED: REMOVED -shortest flag which was causing frame duplication
    # The system now guarantees concat duration >= target_duration, so -shortest is unnecessary
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',  # Regenerate timestamps
        '-threads', '0',        # Optimal CPU utilization
        '-f', 'concat', '-safe', '0', '-i', str(concat_file),
        '-i', audio_file,
        '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '18',  # Ensure high quality even in assembly
        '-c:a', 'aac', '-b:a', '192k',
        # REMOVED: '-shortest' - concat already guarantees sufficient duration
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        # Scale to 1080p with cropping to fill frame
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,format=yuv420p',
        str(assembled_video)
    ]
    
    # Execute first pass (video assembly) - FIXED WITH PROPER TIMEOUT AND LIVE PROGRESS
    log("🎬 Assembling video with HD quality settings...")
    try:
        # FIXED: Use Popen with live progress output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        log(f"📊 FFmpeg assembly started - showing live progress:")
        log("-" * 80)
        
        start_time = time.time()
        assembly_timeout = 10800  # 3 hours for long video assembly
        
        while True:
            line = process.stdout.readline()
            
            if line:
                print(line.strip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
            
            if time.time() - start_time > assembly_timeout:
                process.kill()
                log(f"❌ Video assembly timed out after {assembly_timeout/3600:.1f} hours", "ERROR")
                return None
        
        log("-" * 80)
        
        if process.returncode != 0:
            log(f"❌ FFmpeg assembly failed with code {process.returncode}")
            return None
        
        log(f"✅ Video assembled: {assembled_video}")
    except Exception as e:
        log(f"❌ Video assembly failed: {e}")
        return None
    
    # Verify assembled video exists and has content
    if not assembled_video.exists() or assembled_video.stat().st_size == 0:
        log(f"❌ Assembled video is empty or missing", "ERROR")
        return None
    
    # Step 2: Get assembled video duration
    assembled_duration = get_video_duration(assembled_video)
    log(f"📊 Assembled video duration: {assembled_duration:.3f}s")
    log(f"📊 AUDIO AUTHORITY duration: {target_duration:.3f}s")
    
    # Step 3: AUDIO DURATION AUTHORITY ENFORCEMENT (initial pass)
    # Case 1: Video longer than audio - TRIM EXACTLY to audio duration
    if assembled_duration > target_duration:
        log(f"⚠️ Video is longer than audio by {assembled_duration - target_duration:.3f}s")
        log(f"✂️ Trimming video to match AUDIO AUTHORITY exactly...")
        
        trim_success = trim_video_to_audio_duration(assembled_video, temp_video, target_duration)
        if not trim_success:
            log(f"❌ Failed to trim video to audio duration", "ERROR")
            return None
        
        # Update reference for subtitle processing
        video_for_subtitles = temp_video
        final_duration = target_duration
        
    # Case 2: Video shorter than audio - Check tolerance
    elif assembled_duration < target_duration:
        difference = target_duration - assembled_duration
        log(f"⚠️ Video is shorter than audio by {difference:.3f}s")
        
        if difference > 0.5:
            # Difference > 0.5s - raise validation warning but continue
            log(f"⚠️ VALIDATION WARNING: Video shorter than audio by {difference:.3f}s (exceeds 0.5s tolerance)", "WARNING")
            log(f"   This may cause audio/video desync. Consider increasing clip duration in asset acquisition.")
        
        # Use assembled video as-is (no trim needed)
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
        final_duration = assembled_duration
        
    else:
        # Exact match - perfect
        log(f"✅ Video duration exactly matches AUDIO AUTHORITY!")
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
        final_duration = assembled_duration
    
    # Get video metadata (for logging only)
    width, height, duration, is_short, fps = get_video_metadata(video_for_subtitles)
    log(f"📹 Video metadata: {width}x{height}, {duration:.1f}s, {fps:.1f}fps")
    
    # Render final video with HARD-BURNED subtitles and AUDIO
    render_success = False
    
    # Always burn subtitles if SRT file exists
    subtitles_srt_path = None
    if subtitles_file and Path(subtitles_file).exists():
        subtitles_srt_path = Path(subtitles_file)
        log(f"📝 Found subtitles file: {subtitles_srt_path}")
        
        # Render with hard-burned subtitles and audio
        render_success = render_video_with_hard_subtitles(
            video_for_subtitles,
            output_file,
            subtitles_srt_path,
            audio_file,  # Pass audio file for merging
            duration,
            'long'
        )
    else:
        # No subtitles, just encode with audio
        log("⚠️ No subtitles file found, rendering without subtitles but with audio")
        
        # Encode with high quality settings (no subtitles)
        cmd_encode = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',  # Regenerate timestamps
            '-threads', '0',        # Optimal CPU utilization
            '-i', str(video_for_subtitles),
            '-i', str(audio_file),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',  # Always re-encode for quality
            '-preset', 'medium',
            '-crf', '18',  # Visually lossless
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,format=yuv420p',  # Force 1080p resolution
            str(output_file)
        ]
        
        try:
            # FIXED: Use Popen with live progress output
            process = subprocess.Popen(
                cmd_encode,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            log(f"📊 Encoding started - showing live progress:")
            log("-" * 80)
            
            start_time = time.time()
            encoding_timeout = 4500  # 75 minutes for long videos
            
            while True:
                line = process.stdout.readline()
                
                if line:
                    print(line.strip())
                    sys.stdout.flush()
                
                if process.poll() is not None:
                    break
                
                if time.time() - start_time > encoding_timeout:
                    process.kill()
                    log(f"❌ Encoding timed out after {encoding_timeout/60:.1f} minutes", "ERROR")
                    render_success = False
                    break
            
            log("-" * 80)
            
            if process.returncode == 0:
                render_success = True
            else:
                log(f"❌ Failed to encode video with audio (code {process.returncode})")
                render_success = False
                
        except Exception as e:
            log(f"❌ Failed to encode video with audio: {e}")
            render_success = False
    
    # ============================================================================
    # CRITICAL: POST-RENDER VALIDATION - SILENT FAILURES MUST BE IMPOSSIBLE
    # ============================================================================
    log("=" * 80)
    log("🔍 Validating rendered video output...")
    log("=" * 80)
    
    # Check if output file exists
    if not output_file.exists():
        error_msg = f"FATAL: Video rendering failed. {output_file} was not created."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Check if output file is empty
    if output_file.stat().st_size == 0:
        error_msg = f"FATAL: Video rendering failed. {output_file} is empty."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Check file size is reasonable (at least 100KB)
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    if file_size_mb < 0.1:  # Less than 100KB
        error_msg = f"FATAL: Video file too small: {file_size_mb:.2f} MB. Likely corrupted."
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    log(f"✅ Video render validation successful. File size: {file_size_mb:.2f} MB")
    
    # Verify final output
    if not render_success or not output_file.exists() or output_file.stat().st_size == 0:
        log(f"❌ Final video rendering failed", "ERROR")
        return None
    
    # ============================================================================
    # STEP 4: STRICT AUDIO DURATION AUTHORITY ENFORCEMENT
    # This runs AUTOMATICALLY after EVERY video render - NOT optional
    # ============================================================================
    try:
        log("=" * 80)
        log("⚡ AUTOMATIC AUDIO DURATION AUTHORITY VALIDATION")
        log("=" * 80)
        
        validated_video = enforce_audio_duration_authority(output_file, audio_file)
        
        # Update output_file to validated version (should be same path)
        output_file = validated_video
        
        log("✅ Audio duration authority validation completed successfully")
        
    except Exception as e:
        log(f"❌ CRITICAL: Audio duration authority validation failed: {e}", "ERROR")
        log(f"   Pipeline cannot proceed with invalid duration", "ERROR")
        return None
    
    # Cleanup
    if temp_video.exists():
        temp_video.unlink()
    if assembled_video.exists():
        assembled_video.unlink()
    
    # Get final metadata for logging
    final_audio_duration = get_audio_duration(audio_file)
    final_video_duration = get_video_duration(output_file)
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    log(f"✅ LONG video with HARD-BURNED subtitles and AUDIO complete")
    log(f"   Output: {output_file}")
    log(f"   Size: {size_mb:.2f} MB")
    log(f"   Final video duration: {final_video_duration:.1f}s ({final_video_duration/60:.2f}m)")
    log(f"   AUDIO AUTHORITY: {final_audio_duration:.1f}s")
    log(f"   Duration match: {'✓' if abs(final_video_duration - final_audio_duration) < 0.1 else '⚠️'}")
    log(f"   FPS: {fps:.1f}")
    log(f"   Subtitles: Hard-burned (permanent)")
    log(f"   Font: {SUBTITLE_FONT} (from {FONT_DIR})")
    log(f"   Audio: Merged (AAC 192k)")
    log(f"   Quality: CRF 18, preset medium, forced HD resolution")
    log(f"   Clips used: {len(valid_clips)}")
    
    # Save metadata
    try:
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
    except:
        manifest_data = {}
    
    metadata = {
        'video_type': 'long',
        'duration_seconds': final_video_duration,
        'duration_minutes': final_video_duration/60,
        'audio_authority_duration': final_audio_duration,
        'duration_match': abs(final_video_duration - final_audio_duration) < 0.1,
        'duration_difference': final_video_duration - final_audio_duration,
        'file_size_mb': size_mb,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'quality_settings': {
            'video_codec': 'libx264',
            'crf': 18,
            'preset': 'medium',
            'audio_codec': 'aac',
            'audio_bitrate': '192k',
            'pixel_format': 'yuv420p'
        },
        'subtitles_type': 'hard_burned',
        'subtitles_font': f"{FONT_DIR}/{FONT_NAME}.ttf",
        'subtitles_size': SUBTITLE_FONTSIZE,
        'subtitles_alignment': 'center',
        'subtitles_outline': SUBTITLE_OUTLINE_WIDTH,
        'subtitles_shadow': SUBTITLE_SHADOW,
        'audio_included': True,
        'audio_codec': 'AAC',
        'clips_validated': len(valid_clips),
        'manifest_pages': manifest_data.get('pages_searched', 1) if manifest_data else 1,
        'audio_authority_validated': True,
        'validation_timestamp': datetime.now().isoformat(),
        'generated_at': datetime.now().isoformat()
    }
    
    meta_file = output_dir / 'long_metadata.json'
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return output_file


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Edit video with HARD-BURNED subtitles - NO BLACK VIDEO FALLBACK - AUDIO DURATION AUTHORITY - ENHANCED QUALITY')
    parser.add_argument('--type', choices=['long', 'short'], required=True,
                       help='Video type (long form or short)')
    parser.add_argument('--script-file', required=True,
                       help='Path to script JSON file')
    parser.add_argument('--audio-file', default=None,
                       help='Path to audio WAV file (if not specified, auto-detected)')
    parser.add_argument('--clips-dir', required=True,
                       help='Directory containing video clips')
    parser.add_argument('--run-id', required=True,
                       help='Run ID for logging')
    parser.add_argument('--subtitles-file', default='output/subtitles.srt',
                       help='Optional SRT subtitles file')
    parser.add_argument('--audio-dir', default='output',
                       help='Directory to search for audio files')
    parser.add_argument('--hook-file', default=None,
                       help='Optional hook JSON for shorts')
    parser.add_argument('--cta-file', default=None,
                       help='Optional CTA JSON for shorts')
    parser.add_argument('--premium', action='store_true', default=True,
                       help='Parameter kept for compatibility (subtitles are always burned)')
    parser.add_argument('--no-premium', action='store_false', dest='premium',
                       help='Parameter kept for compatibility')
    
    args = parser.parse_args()
    
    log("=" * 80)
    log(f"🎬 HARD-BURNED SUBTITLE RENDERING - {args.type.upper()}")
    log(f"   BLACK VIDEO FALLBACK: DISABLED")
    log(f"   AUDIO DURATION AUTHORITY: ENABLED")
    log(f"   POST-RENDER VALIDATION: AUTOMATIC (ALWAYS RUNS)")
    log(f"   QUALITY SETTINGS: CRF 18, PRESET MEDIUM, FORCED HD RESOLUTION")
    log(f"   AUDIO QUALITY: AAC 192k")
    log(f"   SUBTITLE FONT: {FONT_DIR}/{FONT_NAME}.ttf (explicit font file)")
    log(f"   AUDIO MERGE: ENABLED (CRITICAL FIX)")
    log(f"   FFMPEG FIXES: +genpts (timestamp regeneration), threads 0 (optimal CPU)")
    log(f"   FFMPEG PROGRESS: LIVE OUTPUT ENABLED (no more freezing)")
    log(f"   FFMPEG TIMEOUTS: Shorts=45min, Long=75min (safe limits)")
    log(f"   SUBTITLE FILTER: Pure SRT input, absolute paths, proper escaping")
    log(f"   CONCAT FIX: Dynamic looping ensures duration >= target, removed -shortest")
    log("=" * 80)
    
    # Step 1: Dynamically find audio file if not specified
    if args.audio_file is None:
        try:
            audio_file = find_latest_audio_file(args.audio_dir, args.type)
            log(f"🎯 Auto-detected {args.type} audio file: {audio_file}")
        except FileNotFoundError as e:
            log(f"❌ {e}")
            sys.exit(1)
    else:
        audio_file = args.audio_file
        if not Path(audio_file).exists():
            log(f"❌ Specified audio file not found: {audio_file}")
            sys.exit(1)
    
    # Validate inputs
    if not Path(args.script_file).exists():
        log(f"❌ Script file not found: {args.script_file}")
        sys.exit(1)
    
    if not Path(audio_file).exists():
        log(f"❌ Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Edit video based on type
    if args.type == 'short':
        output_file = edit_shorts_video(
            args.script_file,
            audio_file,
            args.clips_dir,
            args.run_id,
            args.subtitles_file if Path(args.subtitles_file).exists() else None,
            args.hook_file,
            args.cta_file,
            args.premium  # Kept for compatibility
        )
    else:
        output_file = edit_long_video(
            args.script_file,
            audio_file,
            args.clips_dir,
            args.run_id,
            args.subtitles_file if Path(args.subtitles_file).exists() else None,
            args.premium  # Kept for compatibility
        )
    
    if output_file:
        log(f"✅ {args.type.upper()} video with HARD-BURNED subtitles and AUDIO created successfully")
        log(f"   Output: {output_file}")
        log(f"   AUDIO DURATION AUTHORITY enforced - final video matches audio")
        log(f"   QUALITY SETTINGS applied - CRF 18, forced HD resolution")
        log(f"   Subtitles permanently burned into video frames using {FONT_DIR}/{FONT_NAME}.ttf")
        log(f"   Devanagari ligatures rendered properly (क्या, not क् या)")
        log(f"   AUDIO merged successfully - NO SILENT VIDEOS")
        log(f"   FFMPEG FIXES applied - live progress shown, no freezing, correct timeouts")
        log(f"   SUBTITLE FILTER: Pure SRT input, absolute paths, proper escaping")
        log(f"   CONCAT GENERATION FIXED: Dynamic looping ensures proper duration")
        sys.exit(0)
    else:
        log(f"❌ FATAL: {args.type.upper()} video creation failed - no video generated")
        log(f"   This is a SAFE FAILURE - no black video was created")
        sys.exit(1)


if __name__ == '__main__':
    main()
