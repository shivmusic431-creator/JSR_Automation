#!/usr/bin/env python3
"""
Quality Check - Validates video output meets requirements
Enhanced validation for unlimited pagination system
Checks for black frames, audio sync, and overall video quality
"""
import subprocess
import argparse
import sys
import json
import os
from pathlib import Path
import re
import tempfile

def log(message: str, level: str = "INFO"):
    """Simple logging with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    sys.stdout.flush()

# ============================================================================
# VIDEO QUALITY CHECKS
# ============================================================================

def check_video_duration(video_path: str, video_type: str) -> tuple:
    """
    Check if video duration meets requirements
    
    Returns:
        tuple: (duration, passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        if video_type == 'long':
            # Long video: minimum 10 minutes (600 seconds)
            min_duration = 600
            if duration < min_duration:
                return duration, False, f"Duration {duration/60:.2f}m is less than minimum {min_duration/60:.2f}m"
            else:
                return duration, True, f"Duration {duration/60:.2f}m meets minimum requirement"
        else:
            # Short: max 60 seconds
            max_duration = 60
            if duration > max_duration:
                return duration, False, f"Duration {duration:.2f}s exceeds maximum {max_duration}s"
            else:
                return duration, True, f"Duration {duration:.2f}s within limit"
                
    except Exception as e:
        return 0, False, f"Failed to get duration: {e}"


def check_resolution(video_path: str, video_type: str) -> tuple:
    """
    Check if video resolution meets requirements
    
    Returns:
        tuple: (width, height, passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return 0, 0, False, "No video stream found"
        
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        
        if video_type == 'long':
            # Long video: minimum 1280x720 (720p)
            min_width, min_height = 1280, 720
            if width < min_width or height < min_height:
                return width, height, False, f"Resolution {width}x{height} below minimum {min_width}x{min_height}"
            else:
                return width, height, True, f"Resolution {width}x{height} meets minimum"
        else:
            # Short: minimum 720x1280 (720p portrait)
            min_width, min_height = 720, 1280
            if width < min_width or height < min_height:
                return width, height, False, f"Resolution {width}x{height} below minimum {min_width}x{min_height}"
            else:
                return width, height, True, f"Resolution {width}x{height} meets minimum"
                
    except Exception as e:
        return 0, 0, False, f"Failed to get resolution: {e}"


def check_frame_rate(video_path: str) -> tuple:
    """
    Check if frame rate is consistent and reasonable
    
    Returns:
        tuple: (fps, passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            return 0, False, "No video stream found"
        
        # Get average frame rate
        fps_str = data['streams'][0].get('avg_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
        
        # Check if frame rate is reasonable (between 23.976 and 60)
        min_fps, max_fps = 23.976, 60
        if fps < min_fps or fps > max_fps:
            return fps, False, f"Frame rate {fps:.2f}fps outside normal range ({min_fps}-{max_fps})"
        else:
            return fps, True, f"Frame rate {fps:.2f}fps is acceptable"
                
    except Exception as e:
        return 0, False, f"Failed to get frame rate: {e}"


def check_bitrate(video_path: str) -> tuple:
    """
    Check video bitrate for quality
    
    Returns:
        tuple: (bitrate_mbps, passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        bitrate = int(result.stdout.strip()) / 1000000  # Convert to Mbps
        
        # Minimum acceptable bitrate: 2 Mbps
        min_bitrate = 2.0
        if bitrate < min_bitrate:
            return bitrate, False, f"Bitrate {bitrate:.2f} Mbps below minimum {min_bitrate} Mbps"
        else:
            return bitrate, True, f"Bitrate {bitrate:.2f} Mbps is acceptable"
                
    except Exception as e:
        return 0, False, f"Failed to get bitrate: {e}"


def check_audio_tracks(video_path: str) -> tuple:
    """
    Check if video has audio tracks and they're valid
    
    Returns:
        tuple: (audio_count, passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        audio_streams = data.get('streams', [])
        audio_count = len(audio_streams)
        
        if audio_count == 0:
            return 0, False, "No audio tracks found"
        else:
            return audio_count, True, f"Found {audio_count} audio track(s)"
                
    except Exception as e:
        return 0, False, f"Failed to check audio: {e}"


def check_for_black_frames(video_path: str, duration: float) -> tuple:
    """
    Check for excessive black frames in video
    Uses ffmpeg blackdetect filter
    
    Returns:
        tuple: (black_duration, passes_check, message)
    """
    try:
        # Create temporary file for blackdetect output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Run blackdetect filter
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'blackdetect=d=1.0:pic_th=0.98',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse blackdetect output from stderr
        black_segments = []
        pattern = r'black_start:(\d+\.?\d*)\s+black_end:(\d+\.?\d*)\s+black_duration:(\d+\.?\d*)'
        
        for line in result.stderr.split('\n'):
            match = re.search(pattern, line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                black_dur = float(match.group(3))
                black_segments.append((start, end, black_dur))
        
        # Calculate total black duration
        total_black = sum(seg[2] for seg in black_segments)
        black_percentage = (total_black / duration) * 100 if duration > 0 else 100
        
        # Allow up to 2% black frames
        max_black_percentage = 2.0
        if black_percentage > max_black_percentage:
            return total_black, False, f"Excessive black frames: {total_black:.2f}s ({black_percentage:.1f}% > {max_black_percentage}%)"
        elif black_segments:
            return total_black, True, f"Acceptable black frames: {total_black:.2f}s ({black_percentage:.1f}%)"
        else:
            return 0, True, "No black frames detected"
            
    except Exception as e:
        return 0, False, f"Failed to check for black frames: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def check_audio_sync(video_path: str) -> tuple:
    """
    Quick check for audio sync issues
    Compares audio and video durations
    
    Returns:
        tuple: (sync_offset, passes_check, message)
    """
    try:
        # Get video duration
        cmd_video = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        video_duration = float(subprocess.run(cmd_video, capture_output=True, text=True, check=True).stdout.strip())
        
        # Get audio duration
        cmd_audio = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd_audio, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            audio_duration = float(result.stdout.strip())
        else:
            # Fallback to format duration
            audio_duration = video_duration
        
        # Check sync (allow 0.1s difference)
        sync_offset = abs(video_duration - audio_duration)
        max_offset = 0.1
        
        if sync_offset > max_offset:
            return sync_offset, False, f"Audio/video duration mismatch: {sync_offset:.3f}s > {max_offset}s"
        else:
            return sync_offset, True, f"Audio in sync (offset: {sync_offset:.3f}s)"
            
    except Exception as e:
        return 0, False, f"Failed to check audio sync: {e}"


def check_file_integrity(video_path: str) -> tuple:
    """
    Check if video file is complete and not corrupted
    
    Returns:
        tuple: (file_size_mb, passes_check, message)
    """
    try:
        # Check if file exists
        if not os.path.exists(video_path):
            return 0, False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Minimum file size: 1MB (very small videos might be corrupted)
        if file_size_mb < 1:
            return file_size_mb, False, f"File too small: {file_size_mb:.2f} MB"
        
        # Try to read the file with ffmpeg
        cmd = [
            'ffmpeg', '-v', 'error', '-i', video_path,
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return file_size_mb, False, f"File may be corrupted: {result.stderr[:200]}"
        
        return file_size_mb, True, f"File integrity OK ({file_size_mb:.2f} MB)"
        
    except Exception as e:
        return 0, False, f"Failed to check file integrity: {e}"


def check_codec_compatibility(video_path: str) -> tuple:
    """
    Check if codecs are YouTube-compatible
    
    Returns:
        tuple: (passes_check, message)
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_name,codec_type',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_codec = None
        audio_codec = None
        
        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video':
                video_codec = stream['codec_name']
            elif stream['codec_type'] == 'audio':
                audio_codec = stream['codec_name']
        
        # Check video codec (should be h264)
        if video_codec not in ['h264', 'libx264']:
            return False, f"Video codec '{video_codec}' may not be YouTube-optimal (h264 recommended)"
        
        # Check audio codec (should be aac or mp3)
        if audio_codec not in ['aac', 'mp3', 'mp4a']:
            return False, f"Audio codec '{audio_codec}' may not be YouTube-optimal (aac recommended)"
        
        return True, f"Codecs compatible: video={video_codec}, audio={audio_codec}"
        
    except Exception as e:
        return False, f"Failed to check codecs: {e}"


def check_manifest_integrity(clips_dir: str) -> tuple:
    """
    Check if manifest file exists and contains valid data
    
    Returns:
        tuple: (passes_check, message, manifest_data)
    """
    manifest_path = Path(clips_dir) / 'manifest.json'
    
    if not manifest_path.exists():
        return False, "Manifest file not found", None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = ['run_id', 'video_type', 'target_duration', 'downloaded_duration', 'clips']
        missing_fields = [field for field in required_fields if field not in manifest]
        
        if missing_fields:
            return False, f"Manifest missing fields: {missing_fields}", manifest
        
        # Check if downloaded duration meets target
        target = manifest.get('target_duration', 0)
        downloaded = manifest.get('downloaded_duration', 0)
        
        if downloaded < target:
            return False, f"Manifest shows insufficient duration: {downloaded:.1f}s < {target:.1f}s", manifest
        
        # Check pages_searched if present (new field)
        pages = manifest.get('pages_searched', 1)
        
        return True, f"Manifest valid: {len(manifest['clips'])} clips from {pages} pages", manifest
        
    except Exception as e:
        return False, f"Failed to read manifest: {e}", None


# ============================================================================
# MAIN QUALITY CHECK FUNCTION
# ============================================================================

def check_video_quality(video_path: str, video_type: str, clips_dir: str = None) -> bool:
    """
    Comprehensive video quality check
    
    Args:
        video_path: Path to video file
        video_type: 'long' or 'short'
        clips_dir: Optional path to clips directory for manifest check
        
    Returns:
        True if all critical checks pass, False otherwise
    """
    
    log("=" * 80)
    log(f"🔍 COMPREHENSIVE VIDEO QUALITY CHECK")
    log("=" * 80)
    log(f"Video: {video_path}")
    log(f"Type: {video_type}")
    
    if not os.path.exists(video_path):
        log(f"❌ Video file not found: {video_path}", "ERROR")
        return False
    
    # Run all checks
    checks = []
    
    # Critical checks (must pass)
    log("\n📊 CRITICAL CHECKS:")
    
    # 1. File integrity
    size_mb, integrity_pass, integrity_msg = check_file_integrity(video_path)
    checks.append(('File Integrity', integrity_pass, integrity_msg))
    log(f"   {'✅' if integrity_pass else '❌'} {integrity_msg}")
    
    if not integrity_pass:
        log("❌ Critical: File integrity check failed", "ERROR")
        return False
    
    # 2. Duration
    duration, duration_pass, duration_msg = check_video_duration(video_path, video_type)
    checks.append(('Duration', duration_pass, duration_msg))
    log(f"   {'✅' if duration_pass else '❌'} {duration_msg}")
    
    # 3. Resolution
    width, height, res_pass, res_msg = check_resolution(video_path, video_type)
    checks.append(('Resolution', res_pass, res_msg))
    log(f"   {'✅' if res_pass else '❌'} {res_msg}")
    
    # 4. Audio tracks
    audio_count, audio_pass, audio_msg = check_audio_tracks(video_path)
    checks.append(('Audio Tracks', audio_pass, audio_msg))
    log(f"   {'✅' if audio_pass else '❌'} {audio_msg}")
    
    # 5. Audio sync
    sync_offset, sync_pass, sync_msg = check_audio_sync(video_path)
    checks.append(('Audio Sync', sync_pass, sync_msg))
    log(f"   {'✅' if sync_pass else '❌'} {sync_msg}")
    
    # 6. Black frames
    black_duration, black_pass, black_msg = check_for_black_frames(video_path, duration)
    checks.append(('Black Frames', black_pass, black_msg))
    log(f"   {'✅' if black_pass else '❌'} {black_msg}")
    
    # Warning checks (informational only)
    log("\n📊 INFORMATIONAL CHECKS:")
    
    # 7. Frame rate
    fps, fps_pass, fps_msg = check_frame_rate(video_path)
    checks.append(('Frame Rate', fps_pass, fps_msg))
    log(f"   {'⚠️' if not fps_pass else '✅'} {fps_msg}")
    
    # 8. Bitrate
    bitrate, bitrate_pass, bitrate_msg = check_bitrate(video_path)
    checks.append(('Bitrate', bitrate_pass, bitrate_msg))
    log(f"   {'⚠️' if not bitrate_pass else '✅'} {bitrate_msg}")
    
    # 9. Codec compatibility
    codec_pass, codec_msg = check_codec_compatibility(video_path)
    checks.append(('Codec', codec_pass, codec_msg))
    log(f"   {'⚠️' if not codec_pass else '✅'} {codec_msg}")
    
    # 10. Manifest integrity (if clips_dir provided)
    if clips_dir:
        manifest_pass, manifest_msg, manifest_data = check_manifest_integrity(clips_dir)
        checks.append(('Manifest', manifest_pass, manifest_msg))
        log(f"   {'✅' if manifest_pass else '⚠️'} {manifest_msg}")
        
        if manifest_data:
            log(f"      - Clips downloaded: {manifest_data.get('clips_downloaded', 0)}")
            log(f"      - Pages searched: {manifest_data.get('pages_searched', 1)}")
            log(f"      - Validation failures: {manifest_data.get('validation_failures', 0)}")
    
    # Summary
    log("\n" + "=" * 80)
    log("📋 QUALITY CHECK SUMMARY")
    log("=" * 80)
    
    critical_passed = all([
        integrity_pass,
        duration_pass,
        res_pass,
        audio_pass,
        sync_pass,
        black_pass
    ])
    
    if critical_passed:
        log("✅ All CRITICAL checks PASSED")
    else:
        log("❌ Some CRITICAL checks FAILED", "ERROR")
        for check_name, check_pass, check_msg in checks[:6]:  # Only show critical fails
            if not check_pass:
                log(f"   - {check_name}: {check_msg}", "ERROR")
    
    # Count warnings
    warning_count = sum(1 for _, passed, _ in checks[6:] if not passed)
    if warning_count > 0:
        log(f"⚠️ {warning_count} warning(s) detected (non-critical)")
    
    # Final recommendation
    log("\n📊 FINAL RECOMMENDATION:")
    if critical_passed:
        log("✅ VIDEO PASSES QUALITY CHECK - Ready for upload")
        
        # Additional recommendations
        if warning_count > 0:
            log("   ⚠️ Address warnings for optimal YouTube performance")
        if black_duration > 0:
            log(f"   ℹ️ Video contains {black_duration:.2f}s of black frames")
    else:
        log("❌ VIDEO FAILS QUALITY CHECK - Do not upload", "ERROR")
        log("   Please regenerate video with proper assets")
    
    # Save detailed report
    report_path = Path(video_path).parent / 'quality_report.json'
    report = {
        'video_path': video_path,
        'video_type': video_type,
        'checks': [
            {
                'name': name,
                'passed': passed,
                'message': msg
            }
            for name, passed, msg in checks
        ],
        'critical_passed': critical_passed,
        'warning_count': warning_count,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    log(f"\n📝 Detailed report saved: {report_path}")
    
    return critical_passed


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive video quality check')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--type', choices=['long', 'short'], required=True,
                       help='Video type (long form or short)')
    parser.add_argument('--clips-dir', default='output/clips',
                       help='Directory containing clips (for manifest check)')
    parser.add_argument('--strict', action='store_true',
                       help='Fail on warnings (not just critical errors)')
    
    args = parser.parse_args()
    
    try:
        passed = check_video_quality(args.video, args.type, args.clips_dir)
        
        if args.strict:
            # In strict mode, fail on any warning
            if not passed:
                sys.exit(1)
        else:
            # Normal mode: only fail on critical errors
            if not passed:
                sys.exit(1)
            else:
                sys.exit(0)
        
    except Exception as e:
        log(f"❌ Quality check failed with error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Import datetime here to avoid circular imports
    from datetime import datetime
    main()
