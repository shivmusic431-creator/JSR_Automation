#!/usr/bin/env python3
"""
Asset Acquisition - Downloads stock footage from Pexels based on target audio duration
Uses open search strategy and prevents clip reuse via Firebase
"""
import os
import json
import argparse
import requests
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("⚠️ Firebase not available - install with: pip install firebase-admin")

# ============================================================================
# CONFIGURATION
# ============================================================================

PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
if not PEXELS_API_KEY:
    print("❌ PEXELS_API_KEY environment variable not set")
    sys.exit(1)

PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_API = "https://api.pexels.com/v1/search"

# Open search queries (prioritized)
SEARCH_QUERIES = [
    "nature",
    "sunset",
    "sunrise",
    "people walking",
    "indian people",
    "city traffic",
    "business meeting",
    "technology",
    "abstract background",
    "office workers",
    "crowd",
    "street market",
    "beach",
    "mountains",
    "urban life"
]

# API rate limiting
REQUEST_DELAY = 0.5  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_PAGES = 10  # Maximum pages to search to prevent infinite loops

# Output directories
OUTPUT_DIR = Path('output')
CLIPS_DIR = OUTPUT_DIR / 'clips'
MANIFEST_FILE = CLIPS_DIR / 'manifest.json'

# ============================================================================
# LOGGING
# ============================================================================

def log(message: str, level: str = "INFO"):
    """Simple logging with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
    sys.stdout.flush()

# ============================================================================
# FIREBASE MANAGEMENT
# ============================================================================

class FirebaseManager:
    """Manages Firebase Firestore operations for clip usage tracking"""
    
    def __init__(self):
        self.db = None
        self.initialized = False
        self._init_firebase()
    
    def _init_firebase(self):
        """Initialize Firebase connection"""
        if not FIREBASE_AVAILABLE:
            log("Firebase not available - skipping initialization", "WARNING")
            return
        
        service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
        
        if not service_account_json:
            log("FIREBASE_SERVICE_ACCOUNT_JSON not set - skipping Firebase", "WARNING")
            return
        
        try:
            # Parse service account JSON
            service_account_info = json.loads(service_account_json)
            cred = credentials.Certificate(service_account_info)
            
            try:
                app = firebase_admin.initialize_app(cred)
            except ValueError:
                # App already initialized
                pass
            
            self.db = firestore.client()
            self.initialized = True
            log("✅ Firebase initialized successfully")
            
        except Exception as e:
            log(f"Firebase initialization failed: {e}", "WARNING")
            self.initialized = False
    
    def is_clip_used(self, clip_id: int) -> bool:
        """Check if clip ID exists in used_clips collection"""
        if not self.initialized or not self.db:
            return False
        
        try:
            doc_ref = self.db.collection('used_clips').document(str(clip_id))
            doc = doc_ref.get()
            return doc.exists
        except Exception as e:
            log(f"Firebase check failed for clip {clip_id}: {e}", "WARNING")
            return False  # Assume not used if check fails
    
    def mark_clip_used(self, clip_id: int, duration: float):
        """Mark clip as used in Firebase"""
        if not self.initialized or not self.db:
            return
        
        try:
            doc_ref = self.db.collection('used_clips').document(str(clip_id))
            doc_ref.set({
                'clip_id': clip_id,
                'duration': duration,
                'used_at': firestore.SERVER_TIMESTAMP
            })
            log(f"✅ Marked clip {clip_id} as used in Firebase")
        except Exception as e:
            log(f"Failed to mark clip {clip_id} as used: {e}", "WARNING")

# ============================================================================
# AUDIO DURATION EXTRACTION
# ============================================================================

def get_audio_duration(audio_file: Path) -> float:
    """
    Get audio duration using ffprobe
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Duration in seconds
        
    Raises:
        RuntimeError: If duration cannot be determined
    """
    if not audio_file.exists():
        raise RuntimeError(f"Audio file not found: {audio_file}")
    
    log(f"🎵 Getting audio duration from: {audio_file}")
    
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        log(f"📊 Audio duration: {duration:.2f}s ({duration/60:.2f}m)")
        return duration
        
    except subprocess.CalledProcessError as e:
        log(f"FFprobe failed: {e}", "ERROR")
        raise RuntimeError(f"Could not determine audio duration: {e}")
    except ValueError as e:
        log(f"Invalid duration value: {e}", "ERROR")
        raise RuntimeError(f"Invalid duration from ffprobe: {result.stdout}")

# ============================================================================
# PEXELS API
# ============================================================================

def search_pexels_videos(query: str, video_type: str, per_page: int = 15, page: int = 1) -> Dict:
    """
    Search Pexels for videos
    
    Args:
        query: Search query
        video_type: 'long' or 'short' for orientation filtering
        per_page: Results per page
        page: Page number for pagination
        
    Returns:
        Pexels API response as dict
    """
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    # Determine orientation based on video_type
    orientation = "landscape" if video_type == "long" else "portrait"
    
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": orientation
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            log(f"🔍 Searching Pexels: '{query}' (page {page}, orientation: {orientation}) (attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(
                PEXELS_VIDEO_API,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            log(f"Pexels search failed: {e}", "WARNING")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                log(f"All retries failed for query: '{query}'", "ERROR")
                return {"videos": []}
    
    return {"videos": []}

def get_best_video_file(video_data: Dict, video_type: str) -> Optional[Tuple[str, float]]:
    """
    Extract best quality video file from Pexels video object
    
    Args:
        video_data: Pexels video object
        video_type: 'long' for landscape, 'short' for portrait
        
    Returns:
        Tuple of (download_url, duration) or None if no suitable file
    """
    video_files = video_data.get('video_files', [])
    duration = video_data.get('duration', 0)
    
    if not video_files:
        return None
    
    # Filter by aspect ratio based on video_type
    suitable_files = []
    for vf in video_files:
        width = vf.get('width', 0)
        height = vf.get('height', 0)
        
        if video_type == 'long' and width > height:
            suitable_files.append(vf)
        elif video_type == 'short' and height > width:
            suitable_files.append(vf)
    
    if not suitable_files:
        return None
    
    # Prefer HD quality
    hd_files = [vf for vf in suitable_files if vf.get('quality') == 'hd']
    if hd_files:
        return (hd_files[0].get('link'), duration)
    
    # Fallback to first suitable
    return (suitable_files[0].get('link'), duration)

def download_video(url: str, output_path: Path) -> bool:
    """
    Download video from URL
    
    Args:
        url: Video download URL
        output_path: Path to save video
        
    Returns:
        True if successful
    """
    try:
        log(f"⬇️ Downloading: {output_path.name}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Get file size for logging
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Log progress for large files
                if total_size > 0 and downloaded % (1024 * 1024) < 8192:
                    percent = (downloaded / total_size) * 100
                    log(f"   Progress: {percent:.1f}%", "DEBUG")
        
        # Verify download
        if output_path.exists() and output_path.stat().st_size > 0:
            log(f"✅ Downloaded: {output_path.name} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        else:
            log(f"❌ Download failed: {output_path.name} is empty", "ERROR")
            return False
            
    except requests.exceptions.RequestException as e:
        log(f"Download failed: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Unexpected download error: {e}", "ERROR")
        return False

# ============================================================================
# MAIN ACQUISITION FUNCTION
# ============================================================================

def acquire_assets(script_file: str, video_type: str, run_id: str) -> Path:
    """
    Acquire stock footage based on audio duration requirement
    
    Args:
        script_file: Path to script JSON (unused but kept for compatibility)
        video_type: 'long' or 'short'
        run_id: Run identifier
        
    Returns:
        Path to manifest file
        
    Raises:
        RuntimeError: If no clips can be downloaded or insufficient duration
    """
    
    log("=" * 80)
    log("🎬 ASSET ACQUISITION")
    log("=" * 80)
    log(f"Run ID: {run_id}")
    log(f"Video type: {video_type}")
    
    # Step 1: Get audio duration
    audio_file = OUTPUT_DIR / 'audio.wav'
    if not audio_file.exists():
        # Try alternative names
        alternatives = [
            OUTPUT_DIR / 'final_audio.wav',
            OUTPUT_DIR / 'output.wav'
        ]
        for alt in alternatives:
            if alt.exists():
                audio_file = alt
                break
    
    target_duration = get_audio_duration(audio_file)
    
    # FIX: Limit Shorts duration to 58 seconds
    if video_type == 'short':
        target_duration = min(target_duration, 58)
    
    log(f"🎯 Target duration: {target_duration:.2f}s ({target_duration/60:.2f}m)")
    
    # Step 2: Initialize Firebase
    firebase = FirebaseManager()
    
    # Step 3: Setup output directory and clear old clips
    if CLIPS_DIR.exists():
        log(f"🧹 Clearing old clips from {CLIPS_DIR}")
        for old_clip in CLIPS_DIR.glob("*.mp4"):
            old_clip.unlink()
            log(f"   Removed: {old_clip.name}", "DEBUG")
    else:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Download clips until target reached
    downloaded_duration = 0.0
    downloaded_clips = []
    used_queries = set()
    skipped_count = 0
    page = 1
    
    log("🚀 Starting clip acquisition...")
    
    # Keep downloading until we have enough duration
    while downloaded_duration < target_duration:
        # Safety limit: prevent infinite loops
        if page > MAX_PAGES:
            raise RuntimeError(
                f"Cannot find enough clips after searching {MAX_PAGES} pages. "
                f"Downloaded {downloaded_duration:.1f}s of required {target_duration:.1f}s."
            )
        
        found_any = False
        
        # Try each search query
        for query in SEARCH_QUERIES:
            if downloaded_duration >= target_duration:
                break
            
            if query in used_queries:
                continue
            
            log(f"\n📋 Searching: '{query}' (page {page})")
            
            # Search Pexels with proper pagination and orientation
            results = search_pexels_videos(query, video_type, per_page=15, page=page)
            videos = results.get('videos', [])
            
            if not videos:
                log(f"No videos found for '{query}' on page {page}")
                continue
            
            log(f"Found {len(videos)} videos for '{query}' on page {page}")
            
            # Process each video
            for video in videos:
                if downloaded_duration >= target_duration:
                    break
                
                video_id = video.get('id')
                if not video_id:
                    continue
                
                # Check if clip was already used
                if firebase.is_clip_used(video_id):
                    skipped_count += 1
                    if skipped_count % 10 == 0:
                        log(f"Skipped {skipped_count} already-used clips", "INFO")
                    continue
                
                # Get best quality video file with correct orientation
                video_info = get_best_video_file(video, video_type)
                if not video_info:
                    continue
                
                url, duration = video_info
                
                # Validate duration
                if duration <= 0:
                    log(f"Invalid duration for video {video_id}: {duration}", "WARNING")
                    continue
                
                # Generate filename
                clip_num = len(downloaded_clips) + 1
                filename = f"clip_{clip_num:03d}.mp4"
                output_path = CLIPS_DIR / filename
                
                # Download video
                if download_video(url, output_path):
                    # Mark as used in Firebase
                    firebase.mark_clip_used(video_id, duration)
                    
                    # Update tracking - FIX: Include both 'filename' and 'file' keys
                    downloaded_clips.append({
                        'clip_id': video_id,
                        'filename': filename,
                        'file': str(output_path.absolute()),  # Add absolute path
                        'duration': duration,
                        'query': query,
                        'url': url
                    })
                    downloaded_duration += duration
                    found_any = True
                    
                    log(f"✅ Added: {filename} ({duration:.1f}s)")
                    log(f"📊 Progress: {downloaded_duration:.1f}s / {target_duration:.1f}s ({downloaded_duration/target_duration*100:.1f}%)")
                    
                    # Rate limiting
                    time.sleep(REQUEST_DELAY)
                else:
                    log(f"Failed to download video {video_id}", "WARNING")
            
            # Mark query as used for this page
            used_queries.add(query)
            time.sleep(REQUEST_DELAY)
        
        # If we couldn't find any new videos on this page, move to next page
        if not found_any:
            page += 1
            used_queries.clear()  # Reset queries for next page
            log(f"Moving to page {page}...")
    
    # Step 5: Final validation - STRICT duration enforcement
    log("\n" + "=" * 80)
    log("📊 ACQUISITION SUMMARY")
    log("=" * 80)
    log(f"Target duration: {target_duration:.2f}s")
    log(f"Downloaded duration: {downloaded_duration:.2f}s")
    log(f"Clips downloaded: {len(downloaded_clips)}")
    log(f"Clips skipped (already used): {skipped_count}")
    
    # Strict duration check - pipeline must never continue with insufficient duration
    if downloaded_duration < target_duration:
        error_msg = (
            f"❌ CRITICAL: Downloaded duration ({downloaded_duration:.1f}s) "
            f"is less than required ({target_duration:.1f}s). "
            f"Cannot proceed with black video sections."
        )
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Step 6: Save manifest
    manifest = {
        'run_id': run_id,
        'video_type': video_type,
        'target_duration': target_duration,
        'downloaded_duration': downloaded_duration,
        'clips_downloaded': len(downloaded_clips),
        'clips_skipped': skipped_count,
        'clips': downloaded_clips,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    log(f"✅ Manifest saved: {MANIFEST_FILE}")
    
    # Step 7: Return manifest path
    return MANIFEST_FILE

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Acquire stock footage from Pexels')
    parser.add_argument('--script-file', required=True,
                       help='Path to script JSON (unused but kept for compatibility)')
    parser.add_argument('--duration', type=float, required=False,
                       help='Target duration (overrides audio detection)')
    parser.add_argument('--run-id', required=True,
                       help='Run identifier')
    parser.add_argument('--video-type', choices=['long', 'short'], default='long',
                       help='Video type for aspect ratio filtering')
    
    args = parser.parse_args()
    
    try:
        # Override duration if provided (for testing)
        if args.duration:
            log(f"Using provided duration: {args.duration}s")
            
            # Mock manifest for testing
            CLIPS_DIR.mkdir(parents=True, exist_ok=True)
            manifest = {
                'run_id': args.run_id,
                'video_type': args.video_type,
                'target_duration': args.duration,
                'downloaded_duration': args.duration,
                'clips_downloaded': 0,
                'clips_skipped': 0,
                'clips': [],
                'generated_at': datetime.now().isoformat(),
                'note': 'Duration override - no actual downloads'
            }
            
            with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            log(f"✅ Test manifest created: {MANIFEST_FILE}")
            return
        
        # Normal acquisition
        manifest_path = acquire_assets(
            args.script_file,
            args.video_type,
            args.run_id
        )
        
        log(f"✅ Asset acquisition complete")
        
    except RuntimeError as e:
        log(f"❌ {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"❌ Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()