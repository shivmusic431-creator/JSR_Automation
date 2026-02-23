#!/usr/bin/env python3
"""
Video Editing - Premium Animated Subtitles with Cinematic Styling
Features dynamic audio file detection and automatic clip concatenation
NOW WITH FIXED: No double validation, No unnecessary looping
AUDIO DURATION AUTHORITY - Final video duration must match audio duration exactly
STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
FIXED: Resolution fixed to 720p for both long and short videos
FIXED: Removed duplicate validation (trust acquire_assets manifest)
FIXED: Simple concat without unnecessary loops
FIXED: Proper timestamp regeneration with +genpts
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

def get_video_fps(video_path: Path) -> float:
    """
    Get video FPS using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        FPS as float
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den if den != 0 else 30.0
        return float(fps_str)
    except Exception:
        return 30.0


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
FONT_NAME = "Mukta-Regular"  # DO NOT add .ttf here - will be added in path construction
SUBTITLE_FONT = FONT_NAME  # Keep for backward compatibility
SUBTITLE_FONTSIZE = 28
SUBTITLE_PRIMARY_COLOR = "&Hffffff&"  # White
SUBTITLE_OUTLINE_COLOR = "&H000000&"  # Black
SUBTITLE_OUTLINE_WIDTH = 2
SUBTITLE_SHADOW = 1
SUBTITLE_ALIGNMENT = 2  # Center aligned

# Full font path (constructed once to avoid duplication)
FONT_PATH = Path(FONT_DIR).resolve() / f"{FONT_NAME}.ttf"


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
# FIXED: SIMPLE CLIP MANAGEMENT - NO UNNECESSARY LOOPING
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
    Quick validation that clip is usable - minimal check
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


def load_clips_from_manifest(manifest_path: Path, clips_path: Path) -> tuple:
    """
    Load clips directly from manifest WITHOUT revalidation.
    Trust that acquire_assets already validated everything.
    
    Args:
        manifest_path: Path to manifest file
        clips_path: Directory containing clips
        
    Returns:
        Tuple of (valid_clips_list, total_duration)
        
    Raises:
        RuntimeError: If manifest is invalid or clips missing
    """
    log("📋 Loading clips from manifest (trusting acquire_assets validation)...")
    
    if not manifest_path.exists():
        error_msg = f"❌ Manifest file not found: {manifest_path}"
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        clips = manifest.get('clips', [])
        
        if not clips:
            error_msg = "Manifest contains no clips"
            log(error_msg, "ERROR")
            raise RuntimeError(error_msg)
        
        valid_clips = []
        total_duration = 0.0
        missing_clips = []
        
        for clip in clips:
            # Get clip path
            clip_path = None
            if 'filename' in clip:
                clip_path = clips_path / clip['filename']
            elif 'file' in clip:
                clip_path = Path(clip['file'])
            
            if not clip_path or not clip_path.exists():
                missing_clips.append(clip.get('filename', 'unknown'))
                continue
            
            # Minimal validation - just check if readable
            if not validate_clip_for_use(clip_path):
                missing_clips.append(clip_path.name)
                continue
            
            duration = clip.get('duration', 0)
            if duration <= 0:
                duration = get_clip_duration(clip_path)
            
            valid_clips.append(str(clip_path.absolute()))
            total_duration += duration
        
        if missing_clips:
            error_msg = f"❌ Missing or invalid clips: {missing_clips}"
            log(error_msg, "ERROR")
            raise RuntimeError(error_msg)
        
        log(f"📊 Loaded {len(valid_clips)} clips, total duration: {total_duration:.1f}s")
        
        # CRITICAL: Verify total duration meets audio requirement
        audio_target = manifest.get('audio_authority_target', manifest.get('target_duration', 0))
        if total_duration < audio_target:
            error_msg = (f"❌ Total clip duration ({total_duration:.1f}s) is less than "
                        f"audio target ({audio_target:.1f}s). This should not happen as "
                        f"acquire_assets guarantees sufficient duration.")
            log(error_msg, "ERROR")
            raise RuntimeError(error_msg)
        
        return valid_clips, total_duration
        
    except Exception as e:
        log(f"❌ Failed to load manifest: {e}", "ERROR")
        raise


# ============================================================================
# FIXED: SIMPLE CONCAT FILE GENERATION - NO LOOPING
# ============================================================================

def create_simple_concat_file(clips: list, output_dir: Path, run_id: str) -> Path:
    """
    Create a simple concat file with all clips exactly once.
    NO LOOPING - because acquire_assets already provides enough duration.
    
    Args:
        clips: List of clip file paths
        output_dir: Directory to write concat file
        run_id: Run identifier for unique filename
        
    Returns:
        Path to created concat file
    """
    concat_file = output_dir / f'concat_{run_id}.txt'
    
    log(f"🔧 Creating simple concat file with {len(clips)} clips (no looping)")
    
    # Shuffle clips once for variety
    shuffled_clips = clips.copy()
    random.shuffle(shuffled_clips)
    
    # Write all clips to concat file
    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in shuffled_clips:
            clip_str = str(Path(clip))
            f.write(f"file '{clip_str}'\n")
    
    log(f"📝 Concat file created with {len(shuffled_clips)} clips")
    
    return concat_file


# ============================================================================
# FIXED: SIMPLE MANIFEST LOADING - NO REVALIDATION
# ============================================================================

def verify_manifest_integrity(manifest_path: Path, clips_path: Path) -> tuple:
    """
    DEPRECATED - Use load_clips_from_manifest instead.
    Keeping for backward compatibility but simplified.
    
    Returns:
        Tuple of (valid_clips_list, total_duration, is_valid)
    """
    log("⚠️ WARNING: verify_manifest_integrity is deprecated. Using load_clips_from_manifest instead.")
    
    try:
        valid_clips, total_duration = load_clips_from_manifest(manifest_path, clips_path)
        return valid_clips, total_duration, True
    except Exception as e:
        log(f"❌ Manifest verification failed: {e}", "ERROR")
        return [], 0.0, False


# ============================================================================
# FIXED VIDEO RENDERING WITH HARD SUBTITLES AND AUDIO - 720p QUALITY
# ============================================================================

def render_video_with_hard_subtitles(
    input_video: Path,
    output_video: Path,
    subtitles_srt: Path,
    audio_file: str,
    video_duration: float,
    video_type: str = "long"
) -> bool:
    """
    Render final video with hard-burned subtitles using FFmpeg.

    UPDATED: Resolution fixed to 720p for both video types
    QUALITY SETTINGS:
    - CRF 23: Good quality with smaller file size (was 18 - too large)
    - preset medium: Good compression/speed balance
    - Resolution: 1280x720 for long, 720x1280 for shorts
    - FPS normalized to 30 via filter
    - Audio preserved losslessly via stream copy

    Args:
        input_video: Path to assembled input video (ALREADY contains audio)
        output_video: Path to output video
        subtitles_srt: Path to SRT subtitles file
        audio_file: Audio file path for duration reference only
        video_duration: Duration in seconds (used for timeout calculation only)
        video_type: 'long' or 'short' - controls resolution target

    Returns:
        True if render succeeded, False otherwise
    """
    log(f"🎥 Rendering video with hard-burned subtitles (720p quality)...")
    log(f"   Input video (with audio): {input_video}")
    log(f"   Subtitle file: {subtitles_srt}")
    log(f"   Font path: {FONT_PATH}")
    log(f"   QUALITY SETTINGS: CRF 23, preset medium, 720p resolution")
    log(f"   FPS NORMALIZATION: fps=30 filter")

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
    if not FONT_PATH.exists():
        error_msg = f"❌ FATAL: Font file not found at: {FONT_PATH}"
        log(error_msg, "ERROR")
        return False
    else:
        log(f"✅ Font file verified: {FONT_PATH}")

    # ============================================================================
    # Fontconfig cache setup
    # ============================================================================
    fontconfig_cache_dir = Path(tempfile.mkdtemp(prefix="fontconfig_cache_"))
    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    os.environ["XDG_CACHE_HOME"] = str(fontconfig_cache_dir)
    os.environ["FC_CACHE_DIR"] = str(fontconfig_cache_dir)
    log(f"✅ Fontconfig cache redirected to writable dir: {fontconfig_cache_dir}")

    # ============================================================================
    # Build subtitle filter with absolute paths
    # ============================================================================
    fonts_dir = Path(FONT_DIR).resolve()
    subtitle_file = subtitles_srt.resolve()
    
    # Escape paths for filter syntax
    fonts_escaped = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
    subtitle_escaped = str(subtitle_file).replace("\\", "/").replace(":", "\\:")
    
    # UPDATED: Resolution fixed to 720p
    if video_type == "short":
        target_width, target_height = 720, 1280  # 720p portrait
    else:
        target_width, target_height = 1280, 720  # 720p landscape

    # Build filter chain
    scale_filter = (
        f"scale={target_width}:{target_height}"
        f":force_original_aspect_ratio=increase"
        f",crop={target_width}:{target_height}"
    )
    
    # Complete filter chain
    vf_filter = (
        f"{scale_filter},"
        f"fps=30,"
        f"format=yuv420p,"
        f"subtitles=filename='{subtitle_escaped}':"
        f"fontsdir='{fonts_escaped}':"
        f"charenc=UTF-8"
    )

    log(f"Using video filter chain: {vf_filter}")

    # FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-threads', '0',
        '-i', str(input_video),
        '-vf', vf_filter,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',  # UPDATED: Changed from 18 to 23 for smaller files
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-metadata', f'title=YT-AutoPilot {video_type} video',
        '-metadata', 'artist=AI Generated',
        '-metadata', 'comment=Created with YT-AutoPilot',
        str(output_video)
    ]

    log(f"⚙️ Running FFmpeg (subtitle burn)...")
    log(f"   Resolution: {target_width}x{target_height} (720p)")
    log(f"   CRF: 23 (balanced quality/size)")
    
    # Set timeout based on video type
    if video_type == "short":
        timeout_seconds = 1800  # 30 minutes for shorts
    else:
        timeout_seconds = 7200  # 2 hours for long videos
    
    try:
        # Run FFmpeg with live progress
        ffmpeg_env = os.environ.copy()
        ffmpeg_env["XDG_CACHE_HOME"] = str(fontconfig_cache_dir)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=ffmpeg_env
        )
        
        log(f"📊 FFmpeg encoding started - showing live progress:")
        log("-" * 80)
        
        start_time = time.time()
        
        while True:
            line = process.stdout.readline()
            
            if line:
                print(line.strip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
            
            if time.time() - start_time > timeout_seconds:
                process.kill()
                error_msg = f"❌ FFmpeg timeout exceeded ({timeout_seconds/60:.1f} minutes)"
                log(error_msg, "ERROR")
                raise RuntimeError(error_msg)
        
        log("-" * 80)
        
        if process.returncode != 0:
            log(f"❌ FFmpeg failed with code {process.returncode}", "ERROR")
            return False
            
        # Verify output file
        if output_video.exists() and output_video.stat().st_size > 0:
            size_mb = output_video.stat().st_size / (1024 * 1024)
            log(f"✅ Video rendered successfully: {size_mb:.2f} MB")
            return True
        else:
            log(f"❌ Output video file missing or empty", "ERROR")
            return False
            
    except Exception as e:
        log(f"❌ FFmpeg execution failed: {e}", "ERROR")
        return False


# ============================================================================
# FIXED: SHORTS VIDEO EDITING - SIMPLIFIED, NO UNNECESSARY LOOPS
# ============================================================================

def edit_shorts_video(script_file: str, audio_file: str, clips_dir: str, 
                      run_id: str, subtitles_file: str = None, 
                      hook_file: str = None, cta_file: str = None,
                      premium_subtitles: bool = True):
    """
    Edit SHORTS video using FFmpeg with HARD-BURNED subtitles and AUDIO
    
    FIXED:
    - Simple clip loading from manifest (no revalidation)
    - Simple concat (no loops)
    - Resolution fixed to 720p
    - CRF 23 for balanced quality/size
    
    Args:
        script_file: Path to script JSON
        audio_file: Path to shorts audio WAV
        clips_dir: Directory containing video clips
        run_id: Run identifier
        subtitles_file: Optional path to SRT subtitles
        hook_file: Optional hook JSON for shorts
        cta_file: Optional CTA JSON for shorts
        premium_subtitles: Parameter kept for compatibility
        
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
    # PRE-RENDER VALIDATION
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
    
    log("=" * 80)
    
    # Step 1: Get AUDIO AUTHORITY duration
    audio_duration = get_audio_duration(audio_file)
    target_duration = min(audio_duration, max_duration)
    log(f"🎯 AUDIO AUTHORITY target duration: {target_duration:.2f}s")
    
    # Step 2: Load clips from manifest (NO REVALIDATION)
    manifest_file = clips_path / 'manifest.json'
    try:
        valid_clips, total_clips_duration = load_clips_from_manifest(manifest_file, clips_path)
    except Exception as e:
        log(f"❌ Failed to load clips: {e}", "ERROR")
        return None
    
    # Step 3: Create simple concat file (NO LOOPS)
    concat_file = create_simple_concat_file(valid_clips, output_dir, run_id)
    
    # Step 4: Assemble video
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-threads', '0',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-i', audio_file,
        '-vf', 'fps=30,format=yuv420p',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',  # UPDATED: CRF 23 for better file size
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        str(assembled_video)
    ]
    
    log("🎬 Assembling video...")
    try:
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
        assembly_timeout = 600  # 10 minutes
        
        while True:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
            
            if time.time() - start_time > assembly_timeout:
                process.kill()
                log(f"❌ Video assembly timed out", "ERROR")
                return None
        
        log("-" * 80)
        
        if process.returncode != 0:
            log(f"❌ FFmpeg assembly failed with code {process.returncode}")
            return None
        
        log(f"✅ Video assembled: {assembled_video}")
    except Exception as e:
        log(f"❌ Video assembly failed: {e}")
        return None
    
    # Verify assembled video
    if not assembled_video.exists() or assembled_video.stat().st_size == 0:
        log(f"❌ Assembled video is empty or missing", "ERROR")
        return None
    
    # Step 5: Get assembled video duration
    assembled_duration = get_video_duration(assembled_video)
    log(f"📊 Assembled video duration: {assembled_duration:.3f}s")
    log(f"📊 AUDIO AUTHORITY duration: {target_duration:.3f}s")
    
    # Step 6: Trim if needed
    if assembled_duration > target_duration:
        log(f"⚠️ Video is longer than audio by {assembled_duration - target_duration:.3f}s")
        log(f"✂️ Trimming video to match AUDIO AUTHORITY...")
        
        trim_success = trim_video_to_audio_duration(assembled_video, temp_video, target_duration)
        if not trim_success:
            log(f"❌ Failed to trim video", "ERROR")
            return None
        video_for_subtitles = temp_video
    else:
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
    
    # Step 7: Render with subtitles
    render_success = False
    
    if subtitles_file and Path(subtitles_file).exists():
        render_success = render_video_with_hard_subtitles(
            video_for_subtitles,
            output_file,
            Path(subtitles_file),
            audio_file,
            target_duration,
            'short'
        )
    else:
        # Render without subtitles
        cmd_encode = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',
            '-threads', '0',
            '-i', str(video_for_subtitles),
            '-i', audio_file,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-vf', 'scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=30,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
        
        try:
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
            encoding_timeout = 1800  # 30 minutes
            
            while True:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    sys.stdout.flush()
                
                if process.poll() is not None:
                    break
                
                if time.time() - start_time > encoding_timeout:
                    process.kill()
                    log(f"❌ Encoding timed out", "ERROR")
                    break
            
            log("-" * 80)
            
            if process.returncode == 0:
                render_success = True
            else:
                log(f"❌ Failed to encode video")
                
        except Exception as e:
            log(f"❌ Failed to encode video: {e}")
    
    # Step 8: Final validation
    if not render_success or not output_file.exists() or output_file.stat().st_size == 0:
        log(f"❌ Final video rendering failed", "ERROR")
        return None
    
    # Step 9: Audio duration authority enforcement
    try:
        validated_video = enforce_audio_duration_authority(output_file, audio_file)
        output_file = validated_video
    except Exception as e:
        log(f"❌ Audio duration validation failed: {e}", "ERROR")
        return None
    
    # Cleanup
    if temp_video.exists():
        temp_video.unlink()
    if assembled_video.exists():
        assembled_video.unlink()
    
    # Final metadata
    final_audio_duration = get_audio_duration(audio_file)
    final_video_duration = get_video_duration(output_file)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    
    log(f"✅ SHORTS video complete")
    log(f"   Output: {output_file}")
    log(f"   Size: {size_mb:.2f} MB")
    log(f"   Duration: {final_video_duration:.1f}s")
    log(f"   Resolution: 720x1280 (720p)")
    log(f"   CRF: 23")
    
    return output_file


# ============================================================================
# FIXED: LONG VIDEO EDITING - SIMPLIFIED, NO UNNECESSARY LOOPS
# ============================================================================

def edit_long_video(script_file: str, audio_file: str, clips_dir: str, 
                    run_id: str, subtitles_file: str = None,
                    premium_subtitles: bool = True):
    """
    Edit LONG video using FFmpeg with HARD-BURNED subtitles and AUDIO
    
    FIXED:
    - Simple clip loading from manifest (no revalidation)
    - Simple concat (no loops)
    - Resolution fixed to 720p
    - CRF 23 for balanced quality/size
    
    Args:
        script_file: Path to script JSON
        audio_file: Path to audio WAV
        clips_dir: Directory containing video clips
        run_id: Run identifier
        subtitles_file: Optional path to SRT subtitles
        premium_subtitles: Parameter kept for compatibility
        
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
    # PRE-RENDER VALIDATION
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
    
    log("=" * 80)
    
    # Step 1: Get AUDIO AUTHORITY duration
    audio_duration = get_audio_duration(audio_file)
    target_duration = audio_duration
    log(f"🎯 AUDIO AUTHORITY target duration: {target_duration:.2f}s ({target_duration/60:.2f}m)")
    
    # Step 2: Load clips from manifest (NO REVALIDATION)
    manifest_file = clips_path / 'manifest.json'
    try:
        valid_clips, total_clips_duration = load_clips_from_manifest(manifest_file, clips_path)
    except Exception as e:
        log(f"❌ Failed to load clips: {e}", "ERROR")
        return None
    
    # Step 3: Create simple concat file (NO LOOPS)
    concat_file = create_simple_concat_file(valid_clips, output_dir, run_id)
    
    # Step 4: Assemble video
    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+genpts',
        '-threads', '0',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-i', audio_file,
        '-vf', 'fps=30,format=yuv420p',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',  # UPDATED: CRF 23 for better file size
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        str(assembled_video)
    ]
    
    log("🎬 Assembling video...")
    try:
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
        assembly_timeout = 7200  # 2 hours for long videos
        
        while True:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                sys.stdout.flush()
            
            if process.poll() is not None:
                break
            
            if time.time() - start_time > assembly_timeout:
                process.kill()
                log(f"❌ Video assembly timed out after 2 hours", "ERROR")
                return None
        
        log("-" * 80)
        
        if process.returncode != 0:
            log(f"❌ FFmpeg assembly failed with code {process.returncode}")
            return None
        
        log(f"✅ Video assembled: {assembled_video}")
    except Exception as e:
        log(f"❌ Video assembly failed: {e}")
        return None
    
    # Verify assembled video
    if not assembled_video.exists() or assembled_video.stat().st_size == 0:
        log(f"❌ Assembled video is empty or missing", "ERROR")
        return None
    
    # Step 5: Get assembled video duration
    assembled_duration = get_video_duration(assembled_video)
    log(f"📊 Assembled video duration: {assembled_duration:.3f}s")
    log(f"📊 AUDIO AUTHORITY duration: {target_duration:.3f}s")
    
    # Step 6: Trim if needed
    if assembled_duration > target_duration:
        log(f"⚠️ Video is longer than audio by {assembled_duration - target_duration:.3f}s")
        log(f"✂️ Trimming video to match AUDIO AUTHORITY...")
        
        trim_success = trim_video_to_audio_duration(assembled_video, temp_video, target_duration)
        if not trim_success:
            log(f"❌ Failed to trim video", "ERROR")
            return None
        video_for_subtitles = temp_video
    else:
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
    
    # Step 7: Render with subtitles
    render_success = False
    
    if subtitles_file and Path(subtitles_file).exists():
        render_success = render_video_with_hard_subtitles(
            video_for_subtitles,
            output_file,
            Path(subtitles_file),
            audio_file,
            target_duration,
            'long'
        )
    else:
        # Render without subtitles
        cmd_encode = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',
            '-threads', '0',
            '-i', str(video_for_subtitles),
            '-i', audio_file,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-vf', 'scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=30,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
        
        try:
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
            encoding_timeout = 7200  # 2 hours
            
            while True:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    sys.stdout.flush()
                
                if process.poll() is not None:
                    break
                
                if time.time() - start_time > encoding_timeout:
                    process.kill()
                    log(f"❌ Encoding timed out", "ERROR")
                    break
            
            log("-" * 80)
            
            if process.returncode == 0:
                render_success = True
            else:
                log(f"❌ Failed to encode video")
                
        except Exception as e:
            log(f"❌ Failed to encode video: {e}")
    
    # Step 8: Final validation
    if not render_success or not output_file.exists() or output_file.stat().st_size == 0:
        log(f"❌ Final video rendering failed", "ERROR")
        return None
    
    # Step 9: Audio duration authority enforcement
    try:
        validated_video = enforce_audio_duration_authority(output_file, audio_file)
        output_file = validated_video
    except Exception as e:
        log(f"❌ Audio duration validation failed: {e}", "ERROR")
        return None
    
    # Cleanup
    if temp_video.exists():
        temp_video.unlink()
    if assembled_video.exists():
        assembled_video.unlink()
    
    # Final metadata
    final_audio_duration = get_audio_duration(audio_file)
    final_video_duration = get_video_duration(output_file)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    
    log(f"✅ LONG video complete")
    log(f"   Output: {output_file}")
    log(f"   Size: {size_mb:.2f} MB")
    log(f"   Duration: {final_video_duration:.1f}s ({final_video_duration/60:.2f}m)")
    log(f"   Resolution: 1280x720 (720p)")
    log(f"   CRF: 23")
    
    return output_file


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Edit video with HARD-BURNED subtitles - FIXED VERSION')
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
    
    args = parser.parse_args()
    
    log("=" * 80)
    log(f"🎬 HARD-BURNED SUBTITLE RENDERING - {args.type.upper()}")
    log(f"   RESOLUTION: 720p (fixed)")
    log(f"   CRF: 23 (balanced quality/size)")
    log(f"   NO DOUBLE VALIDATION - Trusting acquire_assets")
    log(f"   NO UNNECESSARY LOOPING - Simple concat")
    log("=" * 80)
    
    # Find audio file if not specified
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
            args.cta_file
        )
    else:
        output_file = edit_long_video(
            args.script_file,
            audio_file,
            args.clips_dir,
            args.run_id,
            args.subtitles_file if Path(args.subtitles_file).exists() else None
        )
    
    if output_file:
        log(f"✅ {args.type.upper()} video created successfully")
        sys.exit(0)
    else:
        log(f"❌ FATAL: {args.type.upper()} video creation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
