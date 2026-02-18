#!/usr/bin/env python3
"""
Asset Acquisition - Downloads stock footage from Pexels based on target audio duration
Uses open search strategy and prevents clip reuse via Firebase
Includes strict video validation to reject static images and low-quality clips
NOW WITH TRUE UNLIMITED PAGINATION - Continues until API returns no results
"""
import os
import json
import argparse
import requests
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
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

# Open search queries (prioritized) - EXPANDED for more variety
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
    "urban life",
    "aerial view",
    "time lapse",
    "slow motion",
    "professional business",
    "corporate",
    "travel",
    "adventure",
    "fitness",
    "sports",
    "meditation",
    "yoga",
    "food",
    "cooking",
    "animals",
    "wildlife",
    "forest",
    "ocean",
    "river",
    "waterfall",
    "clouds",
    "sky",
    "stars",
    "night",
    "city夜景",
    "architecture",
    "modern building",
    "office",
    "workspace",
    "team meeting",
    "presentation",
    "conference",
    "interview",
    "discussion",
    "collaboration",
    "creative",
    "design",
    "art",
    "music",
    "dance",
    "celebration",
    "festival",
    "wedding",
    "family",
    "children",
    "education",
    "learning",
    "science",
    "laboratory",
    "research",
    "medical",
    "healthcare",
    "technology innovation",
    "future",
    "robotics",
    "AI",
    "virtual reality",
    "augmented reality",
    "digital transformation",
    "cyberspace",
    "data",
    "analytics",
    "marketing",
    "social media",
    "mobile phone",
    "laptop",
    "computer",
    "coding",
    "programming",
    "startup",
    "entrepreneur",
    "success",
    "achievement",
    "motivation",
    "inspiration",
    "dream",
    "goal",
    "target",
    "focus",
    "concentration",
    "mindfulness",
    "peace",
    "calm",
    "relaxation",
    "spa",
    "wellness",
    "beauty",
    "fashion",
    "shopping",
    "luxury",
    "car",
    "driving",
    "road trip",
    "journey",
    "exploration",
    "discovery",
    "freedom",
    "happiness",
    "joy",
    "laughter",
    "love",
    "romance",
    "friendship",
    "community",
    "diversity",
    "culture",
    "tradition",
    "heritage",
    "history",
    "ancient",
    "modern",
    "innovation",
    "sustainability",
    "environment",
    "green energy",
    "renewable",
    "solar",
    "wind",
    "nature conservation",
    "earth",
    "planet",
    "space",
    "universe",
    "galaxy",
    "cosmos"
]

# API rate limiting
REQUEST_DELAY = 0.5  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Validation constants - STRICTER VALIDATION
MIN_DURATION = 3.0  # seconds
MIN_FPS = 24.0
MIN_FRAME_COUNT = 72
MIN_FILE_SIZE = 200000  # bytes - INCREASED for better quality
MAX_RETRIES_PER_CLIP = 5  # Maximum attempts to find valid clip
MIN_ACCEPTABLE_RESOLUTION = {
    'long': {'width': 1280, 'height': 720},  # Minimum 720p for long videos
    'short': {'width': 720, 'height': 1280}   # Minimum 720p portrait for shorts
}

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
        self.used_clips_cache: Set[int] = set()  # Cache to reduce Firebase calls
    
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
            
            # Pre-load recently used clips to reduce API calls
            self._cache_recent_clips()
            
        except Exception as e:
            log(f"Firebase initialization failed: {e}", "WARNING")
            self.initialized = False
    
    def _cache_recent_clips(self):
        """Cache recently used clip IDs to reduce Firebase calls"""
        if not self.initialized or not self.db:
            return
        
        try:
            # Get clips used in the last 7 days
            from datetime import datetime, timedelta
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            docs = self.db.collection('used_clips')\
                .where('used_at', '>=', seven_days_ago)\
                .limit(1000)\
                .stream()
            
            for doc in docs:
                try:
                    clip_id = int(doc.id)
                    self.used_clips_cache.add(clip_id)
                except ValueError:
                    pass
            
            log(f"✅ Cached {len(self.used_clips_cache)} recently used clips")
        except Exception as e:
            log(f"Failed to cache recent clips: {e}", "WARNING")
    
    def is_clip_used(self, clip_id: int) -> bool:
        """Check if clip ID exists in used_clips collection"""
        # Check cache first
        if clip_id in self.used_clips_cache:
            return True
        
        if not self.initialized or not self.db:
            return False
        
        try:
            doc_ref = self.db.collection('used_clips').document(str(clip_id))
            doc = doc_ref.get()
            is_used = doc.exists
            
            if is_used:
                self.used_clips_cache.add(clip_id)
            
            return is_used
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
            self.used_clips_cache.add(clip_id)
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
# VIDEO VALIDATION - ENHANCED
# ============================================================================

def validate_video_clip(video_path: Path, video_type: str = "long") -> bool:
    """
    Strict validation of video clip using ffprobe
    Rejects static images, low FPS, short duration, invalid files, and low resolution
    
    Args:
        video_path: Path to video file
        video_type: 'long' or 'short' for resolution validation
        
    Returns:
        True if clip is valid, False otherwise
    """
    if not video_path.exists():
        log(f"Validation failed: File does not exist - {video_path}", "ERROR")
        return False
    
    # Get file size first (cheapest check)
    file_size = video_path.stat().st_size
    if file_size < MIN_FILE_SIZE:
        log(f"Rejected clip: file size too small ({file_size} bytes < {MIN_FILE_SIZE})", "WARNING")
        return False
    
    # Run ffprobe to get video stream info
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_type,avg_frame_rate,nb_frames,width,height',
        '-show_entries', 'format=duration,size,format_name',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        # Time the validation to ensure performance
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5  # Prevent hanging
        )
        
        # Parse JSON output
        data = json.loads(result.stdout)
        
        # Check format duration
        format_info = data.get('format', {})
        duration = float(format_info.get('duration', 0))
        
        if duration < MIN_DURATION:
            log(f"Rejected clip: duration too short ({duration:.2f}s < {MIN_DURATION}s)", "WARNING")
            return False
        
        # Check video stream exists
        streams = data.get('streams', [])
        if not streams:
            log("Rejected clip: no video stream found", "WARNING")
            return False
        
        video_stream = streams[0]
        
        # Verify codec type
        codec_type = video_stream.get('codec_type')
        if codec_type != 'video':
            log(f"Rejected clip: invalid codec type ({codec_type})", "WARNING")
            return False
        
        # Check resolution
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        
        if width == 0 or height == 0:
            log("Rejected clip: invalid resolution", "WARNING")
            return False
        
        # Validate resolution based on video type
        min_res = MIN_ACCEPTABLE_RESOLUTION[video_type]
        
        if video_type == 'long':
            # Landscape validation
            if width < min_res['width'] or height < min_res['height']:
                log(f"Rejected clip: resolution too low ({width}x{height} < {min_res['width']}x{min_res['height']})", "WARNING")
                return False
            # Ensure landscape orientation
            if height > width:
                log(f"Rejected clip: wrong orientation for long video (portrait {width}x{height})", "WARNING")
                return False
        else:
            # Portrait validation for shorts
            if width < min_res['width'] or height < min_res['height']:
                log(f"Rejected clip: resolution too low ({width}x{height} < {min_res['width']}x{min_res['height']})", "WARNING")
                return False
            # Ensure portrait orientation
            if width > height:
                log(f"Rejected clip: wrong orientation for short video (landscape {width}x{height})", "WARNING")
                return False
        
        # Calculate FPS
        fps_str = video_stream.get('avg_frame_rate', '0/1')
        try:
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)
        except (ValueError, ZeroDivisionError):
            fps = 0
        
        if fps < MIN_FPS:
            log(f"Rejected clip: fps too low ({fps:.2f} < {MIN_FPS})", "WARNING")
            return False
        
        # Check frame count
        frame_count = video_stream.get('nb_frames')
        if frame_count is None:
            log("Rejected clip: frame count missing (static image suspected)", "WARNING")
            return False
        
        try:
            frame_count = int(frame_count)
            if frame_count < MIN_FRAME_COUNT:
                log(f"Rejected clip: too few frames ({frame_count} < {MIN_FRAME_COUNT})", "WARNING")
                return False
        except (ValueError, TypeError):
            log("Rejected clip: invalid frame count value", "WARNING")
            return False
        
        # All checks passed
        validation_time = (time.time() - start_time) * 1000
        log(f"Accepted clip: valid video ({duration:.2f}s, {fps:.2f}fps, {frame_count} frames, {width}x{height}) - validation took {validation_time:.1f}ms", "INFO")
        return True
        
    except subprocess.TimeoutExpired:
        log("Rejected clip: ffprobe timeout", "WARNING")
        return False
    except subprocess.CalledProcessError as e:
        log(f"Rejected clip: ffprobe error - {e}", "WARNING")
        return False
    except json.JSONDecodeError as e:
        log(f"Rejected clip: invalid ffprobe output - {e}", "WARNING")
        return False
    except Exception as e:
        log(f"Rejected clip: unexpected validation error - {e}", "WARNING")
        return False

def verify_clip_readable(clip_path: Path) -> bool:
    """
    Quick verification that clip is readable by FFmpeg
    """
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

# ============================================================================
# PEXELS API
# ============================================================================

def search_pexels_videos(query: str, video_type: str, per_page: int = 15, page: int = 1) -> Dict:
    """
    Search Pexels for videos with unlimited pagination
    
    Args:
        query: Search query
        video_type: 'long' or 'short' for orientation filtering
        per_page: Results per page
        page: Page number for pagination (no upper limit)
        
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
                log(f"All retries failed for query: '{query}' page {page}", "WARNING")
                return {"videos": [], "next_page": None}
    
    return {"videos": [], "next_page": None}

def get_best_video_file(video_data: Dict, video_type: str) -> Optional[Tuple[str, float, int, int]]:
    """
    Extract best quality video file from Pexels video object
    
    Args:
        video_data: Pexels video object
        video_type: 'long' for landscape, 'short' for portrait
        
    Returns:
        Tuple of (download_url, duration, width, height) or None if no suitable file
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
        best = hd_files[0]
        return (best.get('link'), duration, best.get('width', 0), best.get('height', 0))
    
    # Fallback to first suitable
    best = suitable_files[0]
    return (best.get('link'), duration, best.get('width', 0), best.get('height', 0))

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
# CLIP VERIFICATION AND REPLACEMENT
# ============================================================================

def verify_and_repair_manifest(manifest_path: Path, clips_dir: Path, video_type: str, firebase: FirebaseManager) -> Tuple[List[Dict], float]:
    """
    Verify all clips in manifest exist and are valid.
    Remove invalid clips and return updated list and total duration.
    
    Returns:
        Tuple of (valid_clips_list, total_duration)
    """
    if not manifest_path.exists():
        return [], 0.0
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        clips = manifest.get('clips', [])
        valid_clips = []
        total_duration = 0.0
        removed_count = 0
        
        log(f"🔍 Verifying {len(clips)} clips from manifest...")
        
        for clip in clips:
            # Get clip path
            clip_path = None
            if 'filename' in clip:
                clip_path = clips_dir / clip['filename']
            elif 'file' in clip:
                clip_path = Path(clip['file'])
            
            if not clip_path or not clip_path.exists():
                log(f"⚠️ Clip missing from disk: {clip.get('filename', 'unknown')}", "WARNING")
                removed_count += 1
                continue
            
            # Validate clip is readable
            if not verify_clip_readable(clip_path):
                log(f"⚠️ Clip corrupted/unreadable: {clip_path.name}", "WARNING")
                # Delete corrupted file
                try:
                    clip_path.unlink()
                except:
                    pass
                removed_count += 1
                continue
            
            # Re-validate clip quality
            if not validate_video_clip(clip_path, video_type):
                log(f"⚠️ Clip failed quality validation: {clip_path.name}", "WARNING")
                # Delete low quality file
                try:
                    clip_path.unlink()
                except:
                    pass
                removed_count += 1
                continue
            
            # Clip is valid
            duration = clip.get('duration', 0)
            if duration <= 0:
                # Try to get duration from file
                duration = get_clip_duration(clip_path)
                clip['duration'] = duration
            
            valid_clips.append(clip)
            total_duration += duration
            log(f"  ✅ {clip_path.name}: {duration:.2f}s")
        
        if removed_count > 0:
            log(f"⚠️ Removed {removed_count} invalid/missing clips from manifest")
            
            # Update manifest file
            manifest['clips'] = valid_clips
            manifest['downloaded_duration'] = total_duration
            manifest['clips_downloaded'] = len(valid_clips)
            manifest['verified_at'] = datetime.now().isoformat()
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        
        return valid_clips, total_duration
        
    except Exception as e:
        log(f"❌ Error verifying manifest: {e}", "ERROR")
        return [], 0.0

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
    except Exception:
        return 0.0

# ============================================================================
# MAIN ACQUISITION FUNCTION - WITH TRUE UNLIMITED PAGINATION
# ============================================================================

def acquire_assets(script_file: str, video_type: str, run_id: str) -> Path:
    """
    Acquire stock footage based on audio duration requirement
    Includes strict validation to reject static images and low-quality clips
    NOW WITH TRUE UNLIMITED PAGINATION - Continues until API returns no results
    
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
    log("🎬 ASSET ACQUISITION - TRUE UNLIMITED PAGINATION MODE")
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
    
    # Add safety margin (5% extra to ensure we have enough)
    required_duration = target_duration * 1.05
    log(f"🎯 Required duration (with 5% margin): {required_duration:.2f}s")
    
    # Step 2: Initialize Firebase
    firebase = FirebaseManager()
    
    # Step 3: Setup output directory
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Check if we have existing valid clips
    valid_clips, existing_duration = verify_and_repair_manifest(MANIFEST_FILE, CLIPS_DIR, video_type, firebase)
    
    if existing_duration >= required_duration:
        log(f"✅ Existing valid clips already meet duration requirement: {existing_duration:.2f}s >= {required_duration:.2f}s")
        return MANIFEST_FILE
    
    if valid_clips:
        log(f"📊 Existing valid clips: {existing_duration:.2f}s / {required_duration:.2f}s needed")
        log(f"🔄 Need additional {required_duration - existing_duration:.2f}s of footage")
    
    # Step 5: Download additional clips until target reached - TRUE UNLIMITED PAGINATION
    downloaded_duration = existing_duration
    downloaded_clips = valid_clips.copy()
    used_queries = set()
    skipped_count = 0
    validation_failures = 0
    page = 1
    total_searched_pages = 0
    query_index = 0
    api_exhausted = False
    
    log("🚀 Starting clip acquisition with TRUE UNLIMITED PAGINATION...")
    log("📊 Will continue searching page by page until either:")
    log("   - Sufficient clips are found, OR")
    log("   - API returns no more results")
    
    # Keep downloading until we have enough duration or API is exhausted
    while downloaded_duration < required_duration and not api_exhausted:
        found_any_on_page = False
        page_results_count = 0
        page_has_more = False
        
        # Try each search query for this page
        for query_offset in range(len(SEARCH_QUERIES)):
            # Rotate through queries
            current_query = SEARCH_QUERIES[(query_index + query_offset) % len(SEARCH_QUERIES)]
            
            if downloaded_duration >= required_duration:
                break
            
            if current_query in used_queries and page == 1:
                # Skip queries we've already used on page 1
                continue
            
            log(f"\n📋 Searching: '{current_query}' (page {page})")
            
            # Search Pexels with proper pagination and orientation
            results = search_pexels_videos(current_query, video_type, per_page=15, page=page)
            videos = results.get('videos', [])
            next_page = results.get('next_page')
            
            if next_page:
                page_has_more = True
            
            if not videos:
                log(f"No videos found for '{current_query}' on page {page}")
                continue
            
            page_results_count += len(videos)
            log(f"Found {len(videos)} videos for '{current_query}' on page {page}")
            
            # Process each video
            for video in videos:
                if downloaded_duration >= required_duration:
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
                
                url, duration, width, height = video_info
                
                # Validate duration
                if duration <= 0:
                    log(f"Invalid duration for video {video_id}: {duration}", "WARNING")
                    continue
                
                # Try to download a valid clip (with retries)
                clip_attempts = 0
                valid_clip_found = False
                
                while clip_attempts < MAX_RETRIES_PER_CLIP and not valid_clip_found and downloaded_duration < required_duration:
                    clip_attempts += 1
                    
                    # Generate temporary filename
                    clip_num = len(downloaded_clips) + 1
                    temp_filename = f"clip_{clip_num:03d}_temp_{clip_attempts}.mp4"
                    temp_path = CLIPS_DIR / temp_filename
                    
                    # Download video
                    if download_video(url, temp_path):
                        # Validate the downloaded clip with video_type specific checks
                        if validate_video_clip(temp_path, video_type):
                            # Valid clip found - rename to final filename
                            final_filename = f"clip_{clip_num:03d}.mp4"
                            final_path = CLIPS_DIR / final_filename
                            
                            # Remove existing final file if it exists
                            if final_path.exists():
                                final_path.unlink()
                            
                            # Rename temp to final
                            temp_path.rename(final_path)
                            
                            # Mark as used in Firebase
                            firebase.mark_clip_used(video_id, duration)
                            
                            # Update tracking
                            downloaded_clips.append({
                                'clip_id': video_id,
                                'filename': final_filename,
                                'file': str(final_path.absolute()),
                                'duration': duration,
                                'query': current_query,
                                'url': url,
                                'width': width,
                                'height': height,
                                'page': page
                            })
                            downloaded_duration += duration
                            found_any_on_page = True
                            valid_clip_found = True
                            
                            log(f"✅ Added: {final_filename} ({duration:.1f}s, {width}x{height})")
                            log(f"📊 Progress: {downloaded_duration:.1f}s / {required_duration:.1f}s ({downloaded_duration/required_duration*100:.1f}%)")
                        else:
                            # Invalid clip - delete and retry
                            validation_failures += 1
                            temp_path.unlink()
                            log(f"🔄 Validation failed for clip {video_id} (attempt {clip_attempts}/{MAX_RETRIES_PER_CLIP})", "WARNING")
                            
                            # Brief pause before retry
                            time.sleep(1)
                    else:
                        # Download failed
                        if temp_path.exists():
                            temp_path.unlink()
                        log(f"🔄 Download failed for clip {video_id} (attempt {clip_attempts}/{MAX_RETRIES_PER_CLIP})", "WARNING")
                        time.sleep(1)
                
                if not valid_clip_found:
                    log(f"❌ Failed to get valid clip after {MAX_RETRIES_PER_CLIP} attempts for video {video_id}", "WARNING")
                
                # Rate limiting
                time.sleep(REQUEST_DELAY)
            
            # Mark query as used for tracking
            if page == 1:
                used_queries.add(current_query)
            time.sleep(REQUEST_DELAY)
        
        # Update query index for next page rotation
        query_index = (query_index + 1) % len(SEARCH_QUERIES)
        
        # Page processing summary
        total_searched_pages += 1
        log(f"\n📊 Page {page} summary: Found {page_results_count} videos, {found_any_on_page and '✅ found valid clips' or '❌ no valid clips'}")
        
        # Check if API is exhausted (no next_page for any query on this page and no videos found)
        if not page_has_more and page_results_count == 0:
            api_exhausted = True
            log(f"⚠️ API returned no more results after {total_searched_pages} pages")
        
        # Always move to next page if there might be more results
        if page_has_more:
            page += 1
            log(f"➡️ Moving to page {page} - Searched {total_searched_pages} pages total")
        
        # Progress report every 10 pages
        if total_searched_pages % 10 == 0:
            log(f"\n{'='*60}")
            log(f"📈 PROGRESS REPORT after {total_searched_pages} pages searched")
            log(f"   Downloaded: {downloaded_duration:.1f}s / {required_duration:.1f}s ({downloaded_duration/required_duration*100:.1f}%)")
            log(f"   Valid clips: {len(downloaded_clips)}")
            log(f"   Skipped (used): {skipped_count}")
            log(f"   Validation failures: {validation_failures}")
            log(f"{'='*60}\n")
    
    # Step 5: Final validation - STRICT duration enforcement
    log("\n" + "=" * 80)
    log("📊 ACQUISITION SUMMARY")
    log("=" * 80)
    log(f"Target duration: {target_duration:.2f}s")
    log(f"Required duration (with margin): {required_duration:.2f}s")
    log(f"Downloaded duration: {downloaded_duration:.2f}s")
    log(f"Clips downloaded: {len(downloaded_clips)}")
    log(f"Clips skipped (already used): {skipped_count}")
    log(f"Validation failures: {validation_failures}")
    log(f"Pages searched: {total_searched_pages}")
    log(f"Final page reached: {page}")
    
    # CRITICAL: Strict duration check - pipeline must never continue with insufficient duration
    if downloaded_duration < required_duration:
        error_msg = (
            f"❌ FATAL: Downloaded duration ({downloaded_duration:.1f}s) "
            f"is less than required ({required_duration:.1f}s) even after exhaustive search "
            f"({total_searched_pages} pages, API exhausted: {api_exhausted}). "
            f"Cannot proceed with black video sections. "
            f"Pipeline must stop immediately."
        )
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    # Step 6: Save manifest
    manifest = {
        'run_id': run_id,
        'video_type': video_type,
        'target_duration': target_duration,
        'required_duration': required_duration,
        'downloaded_duration': downloaded_duration,
        'clips_downloaded': len(downloaded_clips),
        'clips_skipped': skipped_count,
        'validation_failures': validation_failures,
        'pages_searched': total_searched_pages,
        'final_page': page,
        'api_exhausted': api_exhausted,
        'clips': downloaded_clips,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    log(f"✅ Manifest saved: {MANIFEST_FILE}")
    
    # Step 7: Final verification before returning
    final_clips, final_duration = verify_and_repair_manifest(MANIFEST_FILE, CLIPS_DIR, video_type, firebase)
    
    if final_duration < required_duration:
        error_msg = (
            f"❌ FATAL: Post-verification duration ({final_duration:.1f}s) "
            f"is less than required ({required_duration:.1f}s). "
            f"Some clips failed final validation. Cannot proceed."
        )
        log(error_msg, "ERROR")
        raise RuntimeError(error_msg)
    
    return MANIFEST_FILE

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Acquire stock footage from Pexels with TRUE UNLIMITED PAGINATION')
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
                'required_duration': args.duration * 1.05,
                'downloaded_duration': args.duration,
                'clips_downloaded': 0,
                'clips_skipped': 0,
                'validation_failures': 0,
                'pages_searched': 1,
                'final_page': 1,
                'clips': [],
                'generated_at': datetime.now().isoformat(),
                'note': 'Duration override - no actual downloads'
            }
            
            with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            log(f"✅ Test manifest created: {MANIFEST_FILE}")
            return
        
        # Normal acquisition with true unlimited pagination
        manifest_path = acquire_assets(
            args.script_file,
            args.video_type,
            args.run_id
        )
        
        log(f"✅ Asset acquisition complete with TRUE UNLIMITED PAGINATION")
        
    except RuntimeError as e:
        log(f"❌ FATAL: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"❌ FATAL: Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
