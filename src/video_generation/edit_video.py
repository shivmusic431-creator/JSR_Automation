#!/usr/bin/env python3
"""
Video Editing - Combines clips, audio, and overlays using FFmpeg/MoviePy
"""
import os
import json
import argparse
from pathlib import Path
import subprocess

def edit_video(video_type, script_file, audio_file, clips_dir, run_id, hook_file=None, cta_file=None):
    """Edit video using FFmpeg"""
    
    print(f"🎬 Editing {video_type} video...")
    
    output_dir = Path('output/final')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if video_type == 'long':
        output_file = output_dir / 'long_video.mp4'
        target_duration = None  # Use full audio
    else:
        output_file = output_dir / 'short_video.mp4'
        target_duration = 58  # Shorts max is 60 seconds
    
    # Load clips manifest
    clips_dir = Path(clips_dir)
    manifest_file = clips_dir / 'manifest.json'
    
    clips = []
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            clips = [c['file'] for c in manifest.get('clips', [])]
    
    # If no clips, use a default approach
    if not clips:
        # Find all video files in clips directory
        clips = sorted([str(f) for f in clips_dir.glob('*.mp4')])
    
    if not clips:
        print("⚠️ No clips found, creating video with audio only")
        # Create video from audio with black background
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:d=600',
            '-i', audio_file,
            '-shortest',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
    else:
        # Create concat file for FFmpeg
        concat_file = output_dir / 'concat.txt'
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")
        
        # Build FFmpeg command
        # This is a simplified version - production would have more complex editing
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-i', audio_file,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            '-r', '30'
        ]
        
        # For shorts, add vertical crop and duration limit
        if video_type == 'short':
            cmd.extend([
                '-vf', 'crop=1080:1920,format=yuv420p',
                '-t', str(target_duration)
            ])
        
        cmd.append(str(output_file))
    
    # Run FFmpeg
    print(f"🎬 Running FFmpeg...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video edited: {output_file}")
            
            # Get video info
            info_cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_file)
            ]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            duration = float(info_result.stdout.strip())
            
            print(f"⏱️ Video duration: {duration:.2f}s ({duration/60:.2f}m)")
            
            return output_file
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Video editing failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['long', 'short'], required=True)
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--audio-file', required=True)
    parser.add_argument('--clips-dir', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--hook-file', default=None)
    parser.add_argument('--cta-file', default=None)
    args = parser.parse_args()
    
    edit_video(
        args.type,
        args.script_file,
        args.audio_file,
        args.clips_dir,
        args.run_id,
        args.hook_file,
        args.cta_file
    )

if __name__ == '__main__':
    main()
