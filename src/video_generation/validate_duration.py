#!/usr/bin/env python3
"""
Audio Duration Validator - Ensures videos meet duration requirements
Supports both minimum and maximum duration validation
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

def validate_duration(audio_file, min_duration, max_duration, run_id):
    """
    Validate audio meets duration requirements
    
    Args:
        audio_file: Path to audio file
        min_duration: Minimum allowed duration in seconds (optional)
        max_duration: Maximum allowed duration in seconds (optional)
        run_id: Run identifier for logging
        
    Returns:
        True if validation passes, False otherwise
    """
    
    print(f"⏱️ Validating audio duration...")
    if min_duration:
        print(f"   Min required: {min_duration}s")
    if max_duration:
        print(f"   Max allowed: {max_duration}s")
    
    duration = get_audio_duration(audio_file)
    
    print(f"📊 Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Check minimum duration if specified
    if min_duration is not None and duration < min_duration:
        print(f"❌ VALIDATION FAILED: Audio is {duration:.2f}s, minimum required is {min_duration}s")
        print(f"   Missing: {min_duration - duration:.2f} seconds ({(min_duration - duration)/60:.2f} minutes)")
        
        # Set environment variable for downstream processing
        os.environ['AUDIO_DURATION_SHORT'] = 'true'
        os.environ['AUDIO_DURATION'] = str(duration)
        
        # Return non-zero to indicate issue but not fail the workflow
        # In production, you might want to fail here
        return False
    
    # Check maximum duration if specified
    if max_duration is not None and duration > max_duration:
        print(f"❌ VALIDATION FAILED: Audio is {duration:.2f}s, maximum allowed is {max_duration}s")
        print(f"   Exceeds by: {duration - max_duration:.2f} seconds")
        
        # Set environment variable for downstream processing
        os.environ['AUDIO_DURATION_EXCEEDS'] = 'true'
        os.environ['AUDIO_DURATION'] = str(duration)
        
        # Return False to indicate failure (will exit with code 1)
        return False
    
    # If we get here, validation passed
    print(f"✅ Audio duration valid: {duration:.1f}s")
    os.environ['AUDIO_DURATION'] = str(duration)
    
    if min_duration and max_duration:
        os.environ['AUDIO_DURATION_VALID'] = 'true'
    elif min_duration:
        os.environ['AUDIO_DURATION_MIN_OK'] = 'true'
    elif max_duration:
        os.environ['AUDIO_DURATION_MAX_OK'] = 'true'
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate audio duration against min/max requirements')
    parser.add_argument('--audio-file', required=True,
                       help='Path to audio WAV file')
    parser.add_argument('--min-duration', type=float, default=None,
                       help='Minimum allowed audio duration in seconds')
    parser.add_argument('--max-duration', type=float, default=None,
                       help='Maximum allowed audio duration in seconds')
    parser.add_argument('--run-id', required=True,
                       help='Run ID for logging')
    
    args = parser.parse_args()
    
    # Validate that at least one duration constraint is provided
    if args.min_duration is None and args.max_duration is None:
        print("❌ ERROR: Must specify at least one of --min-duration or --max-duration")
        sys.exit(1)
    
    # Validate audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ ERROR: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    valid = validate_duration(
        args.audio_file, 
        args.min_duration, 
        args.max_duration, 
        args.run_id
    )
    
    if not valid:
        # Exit with error code for max duration failures
        # For min duration, we keep warning behavior (exit 0)
        if args.max_duration is not None:
            duration = get_audio_duration(args.audio_file)
            if duration > args.max_duration:
                sys.exit(1)
        
        # For min duration only, warn but don't fail
        sys.exit(0)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
