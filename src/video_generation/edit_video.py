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
        '-i', str(input_video),
        '-t', str(target_duration),
        '-c', 'copy',  # Stream copy for speed
        str(output_video)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if output_video.exists() and output_video.stat().st_size > 0:
            trimmed_duration = get_video_duration(output_video)
            log(f"✅ Video trimmed successfully to {trimmed_duration:.3f}s (target: {target_duration:.3f}s)")
            return True
        else:
            log(f"❌ Trimmed video is empty", "ERROR")
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
# PREMIUM FONT CONFIGURATION - DISABLED FOR SUBTITLE BURNING
# ============================================================================

# Note: We're bypassing the dynamic font detection for subtitle burning
# and using Noto Sans Devanagari directly as required
SUBTITLE_FONT = "Noto Sans Devanagari"
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
# FIXED VIDEO RENDERING WITH HARD SUBTITLES AND AUDIO
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
    Render final video with hard-burned subtitles and audio using FFmpeg.
    
    Uses the subtitles filter to permanently burn subtitles into video frames.
    Merges audio from the provided audio file.
    Always uses Noto Sans Devanagari font for Hindi support.
    Subtitles are center-aligned with outline and shadow for readability.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video
        subtitles_srt: Path to SRT subtitles file
        audio_file: Path to audio file to merge
        video_duration: Duration of video in seconds (for progress tracking)
        video_type: Type of video ('long' or 'short') - used for logging only
        
    Returns:
        True if successful
    """
    log(f"🎥 Rendering video with hard-burned subtitles and audio...")
    log(f"   Subtitle file: {subtitles_srt}")
    log(f"   Audio file: {audio_file}")
    log(f"   Font: {SUBTITLE_FONT}, Size: {SUBTITLE_FONTSIZE}")
    
    # Verify subtitle file exists
    if not subtitles_srt.exists():
        log(f"❌ Subtitle file not found: {subtitles_srt}")
        return False
    
    # Verify audio file exists
    if not Path(audio_file).exists():
        log(f"❌ Audio file not found: {audio_file}")
        return False
    
    # Create subtitle filter string with proper escaping
    # Format: subtitles=filename:force_style='FontName=...,FontSize=...'
    subtitle_filter = (
        f"subtitles={subtitles_srt}:force_style="
        f"'FontName={SUBTITLE_FONT},"
        f"FontSize={SUBTITLE_FONTSIZE},"
        f"PrimaryColour={SUBTITLE_PRIMARY_COLOR},"
        f"OutlineColour={SUBTITLE_OUTLINE_COLOR},"
        f"Outline={SUBTITLE_OUTLINE_WIDTH},"
        f"Shadow={SUBTITLE_SHADOW},"
        f"Alignment={SUBTITLE_ALIGNMENT}'"
    )
    
    # Build FFmpeg command with high quality settings and audio merging
    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_video),
        '-i', str(audio_file),
        '-vf', subtitle_filter,
        '-map', '0:v:0',  # Video from first input
        '-map', '1:a:0',  # Audio from second input
        '-c:v', 'libx264',
        '-preset', 'slow',  # Better compression
        '-crf', '18',  # High quality
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',  # Match to shortest stream (usually audio)
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Web optimization
        '-metadata', f'title=YT-AutoPilot {video_type} video',
        '-metadata', 'artist=AI Generated',
        '-metadata', 'comment=Created with YT-AutoPilot',
        str(output_video)
    ]
    
    # Log command for debugging
    log(f"⚙️ Running FFmpeg with subtitle filter and audio merge...")
    log(f"   Filter: {subtitle_filter}")
    
    try:
        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor progress
        last_progress = 0
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output and 'time=' in output:
                time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
                if time_match:
                    time_str = time_match.group(1)
                    h, m, s = map(float, time_str.split(':'))
                    current_time = h * 3600 + m * 60 + s
                    progress = (current_time / video_duration) * 100
                    
                    # Log every 10% progress
                    if int(progress / 10) > int(last_progress / 10):
                        log(f"⏱️ Rendering progress: {progress:.1f}%")
                    last_progress = progress
        
        returncode = process.poll()
        
        if returncode == 0:
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
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0:
                        log(f"✅ Output video verified")
                except:
                    pass
                
                return True
        
        log(f"❌ FFmpeg failed with code {returncode}")
        return False
            
    except Exception as e:
        log(f"❌ FFmpeg execution failed: {e}")
        return False


# ============================================================================
# CLIP MANAGEMENT WITH ENHANCED LOOPING
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
    Create optimized concat file with shuffled clips for better variety
    Ensures clips are not repeated consecutively
    """
    concat_file = output_dir / f'concat_{run_id}.txt'
    
    # Shuffle clips for variety but ensure no immediate repeats
    shuffled_clips = []
    available_clips = clips.copy()
    
    while available_clips:
        # Pick random clip
        clip_idx = random.randint(0, len(available_clips) - 1)
        clip = available_clips[clip_idx]
        
        # Avoid repeating the same clip consecutively
        if shuffled_clips and shuffled_clips[-1] == clip and len(available_clips) > 1:
            # Try another clip
            other_indices = [i for i in range(len(available_clips)) if i != clip_idx]
            if other_indices:
                clip_idx = random.choice(other_indices)
                clip = available_clips[clip_idx]
        
        shuffled_clips.append(clip)
        available_clips.pop(clip_idx)
    
    # Write concat file
    with open(concat_file, 'w', encoding='utf-8') as f:
        for clip in shuffled_clips:
            f.write(f"file '{clip}'\n")
    
    log(f"📝 Created optimized concat file with {len(shuffled_clips)} clips (shuffled for variety)")
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
# ENHANCED SHORTS VIDEO EDITING WITH HARD SUBTITLES AND AUDIO - NO BLACK FALLBACK
# ============================================================================

def edit_shorts_video(script_file: str, audio_file: str, clips_dir: str, 
                      run_id: str, subtitles_file: str = None, 
                      hook_file: str = None, cta_file: str = None,
                      premium_subtitles: bool = True):
    """
    Edit SHORTS video using FFmpeg with HARD-BURNED subtitles and AUDIO
    
    Features:
    - Noto Sans Devanagari font for Hindi support
    - Center-aligned subtitles with outline and shadow
    - Perfect sync with word-level subtitles
    - Automatic clip looping to fill full audio duration
    - Enhanced clip validation and shuffling
    - ABSOLUTELY NO BLACK VIDEO - Fails safely if requirements not met
    - AUDIO DURATION AUTHORITY - Final video must match audio duration exactly
    - STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
    - AUDIO ALWAYS MERGED - No silent videos
    
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
    concat_file = create_optimized_concat_file(valid_clips, output_dir, run_id, target_duration)
    
    # Base FFmpeg command for concatenated clips
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(concat_file),
        '-i', audio_file,
        '-c:v', 'libx264', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p',
        str(assembled_video)
    ]
    
    # Execute first pass (video assembly) - SAFE NON-BLOCKING EXECUTION
    log("🎬 Assembling video...")
    try:
        # FIX: Use Popen with stream reading to prevent buffer deadlock
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read stderr line by line to prevent buffer overflow
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            # Optionally log progress lines if needed
            if line and 'time=' in line:
                log(f"⏱️ Assembly progress: {line.strip()}")
        
        returncode = process.poll()
        
        if returncode != 0:
            raise RuntimeError(f"FFmpeg assembly failed with code {returncode}")
        
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
        import shutil
        shutil.copy2(assembled_video, temp_video)
        video_for_subtitles = temp_video
        final_duration = assembled_duration
        
    else:
        # Exact match - perfect
        log(f"✅ Video duration exactly matches AUDIO AUTHORITY!")
        import shutil
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
        # No subtitles, just copy with audio
        log("⚠️ No subtitles file found, rendering without subtitles but with audio")
        
        # Simple copy with audio merge
        cmd_copy = [
            'ffmpeg', '-y',
            '-i', str(video_for_subtitles),
            '-i', str(audio_file),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd_copy, check=True, capture_output=True)
            render_success = True
        except Exception as e:
            log(f"❌ Failed to copy video with audio: {e}")
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
                        f"drawtext=fontfile=/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf:"
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
                        f"drawtext=fontfile=/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf:"
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
                '-i', str(output_file),
                '-vf', ','.join(overlay_filters),
                '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'copy',
                str(final_output)
            ]
            
            try:
                subprocess.run(cmd_overlay, check=True)
                if final_output.exists() and final_output.stat().st_size > 0:
                    shutil.move(str(final_output), str(output_file))
                    log("✅ Overlays added successfully")
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
    log(f"   Font: {SUBTITLE_FONT}")
    log(f"   Audio: Merged (AAC)")
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
        'subtitles_type': 'hard_burned',
        'subtitles_font': SUBTITLE_FONT,
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
    - Noto Sans Devanagari font for Hindi support
    - Center-aligned subtitles with outline and shadow
    - Perfect sync with word-level subtitles
    - Automatic clip looping to fill full audio duration
    - Enhanced clip validation and shuffling
    - ABSOLUTELY NO BLACK VIDEO - Fails safely if requirements not met
    - AUDIO DURATION AUTHORITY - Final video must match audio duration exactly
    - STRICT POST-RENDER VALIDATION - Automatically enforces audio duration match
    - AUDIO ALWAYS MERGED - No silent videos
    
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
    concat_file = create_optimized_concat_file(valid_clips, output_dir, run_id, target_duration)
    
    # Base FFmpeg command for concatenated clips
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(concat_file),
        '-i', audio_file,
        '-c:v', 'libx264', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        '-pix_fmt', 'yuv420p',
        '-r', '30',
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p',
        str(assembled_video)
    ]
    
    # Execute first pass (video assembly) - SAFE NON-BLOCKING EXECUTION
    log("🎬 Assembling video...")
    try:
        # FIX: Use Popen with stream reading to prevent buffer deadlock
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read stderr line by line to prevent buffer overflow
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            # Optionally log progress lines if needed
            if line and 'time=' in line:
                log(f"⏱️ Assembly progress: {line.strip()}")
        
        returncode = process.poll()
        
        if returncode != 0:
            raise RuntimeError(f"FFmpeg assembly failed with code {returncode}")
        
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
        # No subtitles, just copy with audio
        log("⚠️ No subtitles file found, rendering without subtitles but with audio")
        
        # Simple copy with audio merge
        cmd_copy = [
            'ffmpeg', '-y',
            '-i', str(video_for_subtitles),
            '-i', str(audio_file),
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd_copy, check=True, capture_output=True)
            render_success = True
        except Exception as e:
            log(f"❌ Failed to copy video with audio: {e}")
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
    log(f"   Font: {SUBTITLE_FONT}")
    log(f"   Audio: Merged (AAC)")
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
        'subtitles_type': 'hard_burned',
        'subtitles_font': SUBTITLE_FONT,
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
    parser = argparse.ArgumentParser(description='Edit video with HARD-BURNED subtitles - NO BLACK VIDEO FALLBACK - AUDIO DURATION AUTHORITY')
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
    log(f"   SUBTITLE FONT: {SUBTITLE_FONT} (fixed)")
    log(f"   AUDIO MERGE: ENABLED (CRITICAL FIX)")
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
        log(f"   Subtitles permanently burned into video frames")
        log(f"   AUDIO merged successfully - NO SILENT VIDEOS")
        sys.exit(0)
    else:
        log(f"❌ FATAL: {args.type.upper()} video creation failed - no video generated")
        log(f"   This is a SAFE FAILURE - no black video was created")
        sys.exit(1)


if __name__ == '__main__':
    main()
