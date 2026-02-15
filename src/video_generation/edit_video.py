#!/usr/bin/env python3
"""
Video Editing - Combines clips, audio, subtitles, and overlays using FFmpeg/MoviePy
Supports center-aligned subtitles with {\an5} formatting
Features dynamic audio file detection
"""
import os
import json
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def log(message: str):
    """Simple logging with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# ============================================================================
# DYNAMIC AUDIO FILE DETECTION
# ============================================================================

def find_latest_audio_file(output_dir="output"):
    """
    Dynamically find the latest audio WAV file in the output directory
    
    Priority:
    1. Preferred names: audio.wav, final_audio.wav
    2. Most recently modified .wav file
    
    Args:
        output_dir: Directory to search for audio files
        
    Returns:
        Path to the audio file as string
        
    Raises:
        FileNotFoundError: If no audio file is found
    """
    from pathlib import Path
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Priority order for preferred filenames
    preferred_names = [
        "audio.wav",
        "final_audio.wav"
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


def format_subtitle_style(video_type: str = "long") -> str:
    """
    Create FFmpeg subtitle filter with center alignment
    
    Args:
        video_type: 'long' or 'short'
        
    Returns:
        FFmpeg subtitle filter string
    """
    # SRT styling parameters
    # Alignment 10 = center ({\an5} in SRT)
    base_style = (
        "FontName=Arial,"
        "FontSize=24,"
        "PrimaryColour=&H00FFFFFF,"  # White
        "OutlineColour=&H00000000,"  # Black outline
        "BorderStyle=3,"              # Outline + shadow
        "Outline=1,"                   # Outline width
        "Shadow=1,"                    # Shadow depth
        "Alignment=10,"                 # Center alignment
        "MarginV=20"                    # Vertical margin from bottom
    )
    
    if video_type == "short":
        # Adjust for vertical video
        base_style = base_style.replace("FontSize=24", "FontSize=28")
        base_style = base_style.replace("MarginV=20", "MarginV=30")
    
    return base_style

def verify_subtitle_center_alignment(subtitles_file: Path) -> bool:
    """Verify subtitle file contains center alignment marker"""
    try:
        with open(subtitles_file, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Read first 1000 chars
            if '{\\an5}' in content:
                log("✅ Subtitle center alignment verified")
                return True
            else:
                log("⚠️ Subtitle file missing center alignment marker {\\an5}")
                return False
    except Exception as e:
        log(f"⚠️ Could not verify subtitle alignment: {e}")
        return False

def get_audio_duration(audio_file: str) -> float:
    """Get audio duration using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        log(f"📊 Audio duration: {duration:.2f}s ({duration/60:.2f}m)")
        return duration
    except Exception as e:
        log(f"⚠️ Could not determine audio duration: {e}")
        return 600.0  # Default 10 minutes if can't determine

def edit_video(video_type: str, script_file: str, audio_file: str, clips_dir: str, 
               run_id: str, subtitles_file: str = None, hook_file: str = None, 
               cta_file: str = None):
    """
    Edit video using FFmpeg with subtitle support
    
    Args:
        video_type: 'long' or 'short'
        script_file: Path to script JSON
        audio_file: Path to audio WAV
        clips_dir: Directory containing video clips
        run_id: Run identifier
        subtitles_file: Optional path to SRT subtitles
        hook_file: Optional hook JSON for shorts
        cta_file: Optional CTA JSON for shorts
    """
    
    log(f"🎬 Editing {video_type} video...")
    log(f"   Run ID: {run_id}")
    
    # Setup paths
    output_dir = Path('output/final')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if video_type == 'long':
        output_file = output_dir / 'long_video.mp4'
        target_duration = None  # Use full audio
    else:
        output_file = output_dir / 'short_video.mp4'
        target_duration = 58  # Shorts max is 60 seconds
    
    # Check audio file
    audio_path = Path(audio_file)
    if not audio_path.exists():
        log(f"❌ Audio file not found: {audio_file}")
        return None
    
    audio_duration = get_audio_duration(audio_file)
    
    # Load clips manifest
    clips_path = Path(clips_dir)
    manifest_file = clips_path / 'manifest.json'
    
    clips = []
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                clips = [c['file'] for c in manifest.get('clips', [])]
            log(f"📋 Loaded manifest with {len(clips)} clips")
        except Exception as e:
            log(f"⚠️ Error loading manifest: {e}")
    
    # If no clips from manifest, find all video files
    if not clips:
        clips = sorted([str(f) for f in clips_path.glob('*.mp4')])
        if clips:
            log(f"📁 Found {len(clips)} video clips in directory")
    
    # Build FFmpeg command
    if not clips:
        log("⚠️ No clips found, creating video with audio only (black background)")
        
        # Create video from audio with black background
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=black:s=1920x1080:d={audio_duration}',
            '-i', audio_file,
            '-shortest',
            '-c:v', 'libx264', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-r', '30'
        ]
        
        # Add subtitles if available
        if subtitles_file and Path(subtitles_file).exists():
            # Verify center alignment
            verify_subtitle_center_alignment(Path(subtitles_file))
            
            subtitle_style = format_subtitle_style(video_type)
            cmd.extend([
                '-vf', f"subtitles={subtitles_file}:force_style='{subtitle_style}'"
            ])
            log("📝 Adding subtitles to video")
        
        cmd.append(str(output_file))
        
    else:
        # Create concat file for FFmpeg
        concat_file = output_dir / 'concat.txt'
        abs_clips = []
        
        with open(concat_file, 'w', encoding='utf-8') as f:
            for clip in clips:
                clip_path = Path(clip)
                if not clip_path.is_absolute():
                    clip_path = clips_path / clip_path
                
                if clip_path.exists():
                    abs_path = str(clip_path.absolute())
                    f.write(f"file '{abs_path}'\n")
                    abs_clips.append(abs_path)
                else:
                    log(f"⚠️ Clip not found: {clip}")
        
        if not abs_clips:
            log("❌ No valid clips found")
            return None
        
        log(f"🔗 Created concat file with {len(abs_clips)} clips")
        
        # Base FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', audio_file,
            '-c:v', 'libx264', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            '-r', '30'
        ]
        
        # Add video filters
        video_filters = []
        
        # Add subtitles if available
        if subtitles_file and Path(subtitles_file).exists():
            # Verify center alignment
            verify_subtitle_center_alignment(Path(subtitles_file))
            
            subtitle_style = format_subtitle_style(video_type)
            video_filters.append(f"subtitles={subtitles_file}:force_style='{subtitle_style}'")
            log("📝 Adding subtitles to video")
        
        # For shorts, add vertical crop
        if video_type == 'short':
            video_filters.append("crop=1080:1920,format=yuv420p")
            
            # Add duration limit
            if target_duration:
                cmd.extend(['-t', str(target_duration)])
        
        # Apply video filters if any
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        cmd.append(str(output_file))
    
    # Log command (truncated for readability)
    log(f"🎬 Running FFmpeg command...")
    cmd_str = ' '.join(str(c) for c in cmd)
    if len(cmd_str) > 500:
        log(f"   Command: {cmd_str[:500]}... [truncated]")
    else:
        log(f"   Command: {cmd_str}")
    
    try:
        # Run FFmpeg with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor progress
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Log only important messages (every 10% progress)
                if 'time=' in output:
                    # Extract time for progress tracking
                    import re
                    time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
                    if time_match:
                        log(f"⏱️ Progress: {time_match.group(1)}")
        
        returncode = process.poll()
        
        if returncode == 0:
            log(f"✅ Video edited successfully: {output_file}")
            
            # Get final video info
            info_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration,size',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_file)
            ]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            
            if info_result.returncode == 0:
                lines = info_result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    duration = float(lines[0])
                    size_bytes = int(lines[1])
                    size_mb = size_bytes / (1024 * 1024)
                    
                    log(f"⏱️ Video duration: {duration:.2f}s ({duration/60:.2f}m)")
                    log(f"💾 File size: {size_mb:.2f} MB")
                    
                    # Save metadata
                    metadata = {
                        'video_type': video_type,
                        'duration_seconds': duration,
                        'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}",
                        'file_size_mb': size_mb,
                        'subtitles_included': subtitles_file and Path(subtitles_file).exists(),
                        'subtitles_center_aligned': True,
                        'audio_file': audio_file,
                        'clips_used': len(clips) if clips else 0,
                        'output_file': str(output_file)
                    }
                    
                    meta_file = output_dir / f'{video_type}_metadata.json'
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
            
            return output_file
        else:
            # Get error output
            _, stderr = process.communicate()
            log(f"❌ FFmpeg failed with code {returncode}")
            if stderr:
                log(f"Error: {stderr[-500:]}")  # Last 500 chars of error
            return None
            
    except Exception as e:
        log(f"❌ Video editing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_subtitles(subtitles_file: Path) -> bool:
    """Verify subtitle file is valid"""
    if not subtitles_file.exists():
        log(f"❌ Subtitle file not found: {subtitles_file}")
        return False
    
    try:
        with open(subtitles_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Basic validation - should have at least one subtitle
        if '-->' not in content:
            log("⚠️ Subtitle file missing timestamp markers")
            return False
        
        # Check for center alignment marker (CRITICAL)
        if '{\\an5}' not in content:
            log("⚠️ Subtitle file missing center alignment marker {\\an5}")
            log("   This will affect subtitle positioning")
            # Not fatal, but log warning
        
        # Count subtitles
        line_count = len(content.strip().split('\n'))
        subtitle_count = line_count // 4
        log(f"📝 Subtitle file contains ~{subtitle_count} subtitles")
        
        return True
        
    except Exception as e:
        log(f"⚠️ Subtitle verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Edit video with clips, audio, and subtitles')
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
    log(f"🎬 VIDEO EDITING PIPELINE - {args.type.upper()}")
    log("=" * 80)
    
    # Step 1: Dynamically find audio file if not specified
    if args.audio_file is None:
        try:
            audio_file = find_latest_audio_file(args.audio_dir)
            log(f"🎯 Auto-detected audio file: {audio_file}")
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
    
    if not Path(args.clips_dir).exists():
        log(f"⚠️ Clips directory not found: {args.clips_dir}")
        # Continue - will create video with audio only
    
    # Verify subtitles if provided
    if args.subtitles_file and Path(args.subtitles_file).exists():
        if verify_subtitles(Path(args.subtitles_file)):
            log("✅ Subtitles verified and ready")
        else:
            log("⚠️ Subtitle file invalid, proceeding without subtitles")
            args.subtitles_file = None
    else:
        log("ℹ️ No subtitles file found, proceeding without subtitles")
        args.subtitles_file = None
    
    # Edit video
    output_file = edit_video(
        args.type,
        args.script_file,
        audio_file,
        args.clips_dir,
        args.run_id,
        args.subtitles_file,
        args.hook_file,
        args.cta_file
    )
    
    if output_file:
        log(f"✅ {args.type.upper()} video creation successful")
        sys.exit(0)
    else:
        log(f"❌ {args.type.upper()} video creation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()