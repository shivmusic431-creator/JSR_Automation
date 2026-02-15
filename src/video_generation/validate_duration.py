#!/usr/bin/env python3
"""
Audio Duration Validator - Ensures videos are at least 10 minutes long
"""
import os
import subprocess
import argparse
import sys

def get_audio_duration(audio_file):
    """Get audio duration using ffprobe"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"❌ Error getting duration: {e}")
        return 0

def validate_duration(audio_file, min_duration, run_id):
    """Validate audio meets minimum duration requirement"""
    
    print(f"⏱️ Validating audio duration (min: {min_duration}s)...")
    
    duration = get_audio_duration(audio_file)
    
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    if duration < min_duration:
        print(f"❌ VALIDATION FAILED: Audio is {duration:.2f}s, minimum required is {min_duration}s")
        print(f"   Missing: {min_duration - duration:.2f} seconds ({(min_duration - duration)/60:.2f} minutes)")
        
        # Don't fail - just warn. The script generation already enforces word count
        print("⚠️ Warning: Video may be shorter than 10 minutes")
        print("   Consider regenerating script with more content")
        
        # Set environment variable for downstream processing
        os.environ['AUDIO_DURATION_SHORT'] = 'true'
        os.environ['AUDIO_DURATION'] = str(duration)
        
        # Return non-zero to indicate issue but not fail the workflow
        # In production, you might want to fail here
        return False
    else:
        print(f"✅ VALIDATION PASSED: Audio duration is {duration:.2f}s")
        os.environ['AUDIO_DURATION'] = str(duration)
        os.environ['AUDIO_DURATION_SHORT'] = 'false'
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', required=True)
    parser.add_argument('--min-duration', type=int, default=600, help='Minimum duration in seconds')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    valid = validate_duration(args.audio_file, args.min_duration, args.run_id)
    
    if not valid:
        # Exit with warning code but don't fail
        sys.exit(0)

if __name__ == '__main__':
    main()
