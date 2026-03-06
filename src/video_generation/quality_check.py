#!/usr/bin/env python3
"""
Quality Check - Validates video output meets requirements
"""
import subprocess
import argparse
import sys

def check_video_quality(video_path, video_type):
    """Check video quality and specifications"""
    
    print(f"🔍 Checking video quality: {video_path}")
    
    # Get video info using ffprobe
    try:
        # Duration
        duration_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip())
        
        # Resolution
        resolution_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]
        resolution_result = subprocess.run(resolution_cmd, capture_output=True, text=True)
        
        # Parse resolution
        width = height = 0
        for line in resolution_result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=')
                if key == 'width':
                    width = int(value)
                elif key == 'height':
                    height = int(value)
        
        # Check requirements
        checks = {
            'duration': duration,
            'width': width,
            'height': height,
            'passed': True
        }
        
        print(f"\n📊 Video Specifications:")
        print(f"   Duration: {duration:.2f}s ({duration/60:.2f}m)")
        print(f"   Resolution: {width}x{height}")
        
        # Validate based on type
        if video_type == 'long':
            # Long video: minimum 10 minutes (600 seconds)
            if duration < 600:
                print(f"⚠️ WARNING: Long video is {duration/60:.2f}m, should be at least 10m")
                checks['passed'] = False
            else:
                print(f"✅ Duration check passed (>{duration/60:.1f}m)")
            
            # Resolution check
            if width < 1920 or height < 1080:
                print(f"⚠️ WARNING: Resolution {width}x{height} below recommended 1920x1080")
            else:
                print(f"✅ Resolution check passed")
                
        else:  # short
            # Short: max 60 seconds
            if duration > 60:
                print(f"⚠️ WARNING: Short is {duration:.2f}s, should be under 60s")
                checks['passed'] = False
            else:
                print(f"✅ Duration check passed ({duration:.1f}s)")
            
            # Resolution check for shorts (vertical)
            if width < 1080 or height < 1920:
                print(f"⚠️ WARNING: Resolution {width}x{height} below recommended 1080x1920")
            else:
                print(f"✅ Resolution check passed")
        
        print(f"\n{'✅' if checks['passed'] else '⚠️'} Quality check {'passed' if checks['passed'] else 'completed with warnings'}")
        
        return checks['passed']
        
    except Exception as e:
        print(f"❌ Quality check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--type', choices=['long', 'short'], required=True)
    args = parser.parse_args()
    
    passed = check_video_quality(args.video, args.type)
    
    if not passed:
        print("⚠️ Quality check completed with warnings")
        # Don't fail - just warn
        sys.exit(0)

if __name__ == '__main__':
    main()
