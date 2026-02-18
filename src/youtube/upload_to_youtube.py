#!/usr/bin/env python3
"""
YouTube Direct Uploader - Uploads videos directly to YouTube using Firestore OAuth tokens
Supports scheduled publishing and internal video linking
"""
import os
import json
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytz
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
YT_API_SERVICE_NAME = 'youtube'
YT_API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class YouTubeUploader:
    def __init__(self, channel_id: str, files_dir: Path):
        """Initialize YouTube uploader with Firebase and YouTube client"""
        self.channel_id = channel_id
        self.files_dir = Path(files_dir)
        self.youtube = None
        self.long_video_id = None
        self.short_video_id = None
        
        # Initialize Firebase
        self._init_firebase()
        
        # Get credentials and build YouTube client
        self._init_youtube_client()
        
    def _init_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                firebase_creds_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
                if not firebase_creds_json:
                    raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON environment variable not set")
                
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logger.info("✅ Firebase initialized successfully")
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {e}")
            raise
    
    def _get_token_from_firestore(self) -> Dict:
        """Fetch OAuth token from Firestore using document ID (channel ID)"""
        try:
            # Clean the channel ID by stripping whitespace
            clean_channel_id = self.channel_id.strip()
            logger.info(f"🔍 Looking up token with document ID: {clean_channel_id}")
            
            # Direct document reference using channel ID as document ID
            doc_ref = self.db.collection('userTokens').document(clean_channel_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                raise ValueError(f"No token document found for channel ID: {clean_channel_id}")
            
            token_data = doc.to_dict()
            
            if not token_data:
                raise ValueError(f"Token document exists but contains no data for channel: {clean_channel_id}")
            
            # Validate required fields exist
            if not token_data.get('accessToken') or not token_data.get('refreshToken'):
                missing_fields = []
                if not token_data.get('accessToken'): missing_fields.append('accessToken')
                if not token_data.get('refreshToken'): missing_fields.append('refreshToken')
                raise ValueError(f"Token document missing required fields: {', '.join(missing_fields)}")
            
            # Optional: validate channelId field matches if it exists
            stored_channel_id = token_data.get('channelId')
            if stored_channel_id and stored_channel_id.strip() != clean_channel_id:
                logger.warning(f"⚠️ Document ID mismatch: stored channelId '{stored_channel_id}' != lookup ID '{clean_channel_id}'")
            
            logger.info(f"✅ Token found for channel: {token_data.get('name', clean_channel_id)}")
            
            return {
                'token': token_data['accessToken'],
                'refresh_token': token_data['refreshToken'],
                'token_uri': 'https://oauth2.googleapis.com/token',
                'client_id': os.getenv('GOOGLE_CLIENT_ID'),
                'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
                'scopes': SCOPES
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch token from Firestore: {e}")
            raise
    
    def _refresh_token_if_needed(self, credentials: Credentials) -> Credentials:
        """Refresh token if expired"""
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                logger.info("🔄 Token expired, refreshing...")
                credentials.refresh(Request())
                logger.info("✅ Token refreshed successfully")
                
                # Update token in Firestore
                self._update_token_in_firestore(credentials)
            except Exception as e:
                logger.error(f"❌ Token refresh failed: {e}")
                raise
        return credentials
    
    def _update_token_in_firestore(self, credentials: Credentials):
        """Update refreshed token in Firestore using document ID"""
        try:
            # Clean the channel ID by stripping whitespace
            clean_channel_id = self.channel_id.strip()
            
            # Direct document reference using channel ID as document ID
            doc_ref = self.db.collection('userTokens').document(clean_channel_id)
            doc = doc_ref.get()
            
            if doc.exists:
                doc_ref.update({
                    'accessToken': credentials.token,
                    'updatedAt': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"✅ Token updated in Firestore for document: {clean_channel_id}")
            else:
                logger.warning(f"⚠️ Cannot update token: No document found with ID: {clean_channel_id}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to update token in Firestore: {e}")
            # Don't re-raise - token refresh succeeded even if update fails
    
    def _init_youtube_client(self):
        """Initialize YouTube API client with OAuth credentials"""
        for attempt in range(MAX_RETRIES):
            try:
                token_data = self._get_token_from_firestore()
                
                # Create credentials object with explicit full YouTube scopes
                # Using both youtube.upload and youtube scopes to ensure all operations succeed
                credentials = Credentials(
                    token=token_data['token'],
                    refresh_token=token_data['refresh_token'],
                    token_uri=token_data['token_uri'],
                    client_id=token_data['client_id'],
                    client_secret=token_data['client_secret'],
                    scopes=[
                        'https://www.googleapis.com/auth/youtube.upload',
                        'https://www.googleapis.com/auth/youtube'
                    ]
                )
                
                # Refresh if needed
                credentials = self._refresh_token_if_needed(credentials)
                
                # Build YouTube client
                self.youtube = build(
                    YT_API_SERVICE_NAME,
                    YT_API_VERSION,
                    credentials=credentials
                )
                
                logger.info("✅ YouTube client initialized successfully")
                return
                
            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
    
    def _get_scheduled_publish_time(self, time_str: str) -> str:
        """Convert IST time to UTC ISO format for YouTube scheduling"""
        ist = pytz.timezone('Asia/Kolkata')
        utc = pytz.UTC
        
        # Get current date in IST
        now_ist = datetime.now(ist)
        
        # Set to specified time
        hour, minute = map(int, time_str.split(':'))
        scheduled_ist = now_ist.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        
        # If time has passed for today, schedule for tomorrow
        if scheduled_ist <= now_ist:
            scheduled_ist += timedelta(days=1)
        
        # Convert to UTC
        scheduled_utc = scheduled_ist.astimezone(utc)
        
        # Format for YouTube API (RFC 3339)
        return scheduled_utc.isoformat().replace('+00:00', 'Z')
    
    def _load_json_file(self, filename: str) -> Optional[Dict]:
        """Load JSON file if exists"""
        file_path = self.files_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {filename}: {e}")
        return None
    
    def _prepare_video_metadata(self, video_type: str, scheduled_time: str) -> Dict:
        """Prepare metadata for video upload with robust extraction from script.json"""
        # Load script data safely
        script_data = self._load_json_file('script.json') or {}
        
        # Extract metadata from nested structure
        metadata = script_data.get('metadata', {})
        
        # Get title with fallback chain
        title = metadata.get('final_title')
        if not title:
            title = metadata.get('title')
        if not title:
            title = f'{video_type.title()} Video'
            logger.warning(f"⚠️ No title found in script.json, using fallback: {title}")
        
        # Get description with fallback chain
        description = metadata.get('description', '')
        if not description:
            description = metadata.get('description_hook', '')
        if not description:
            description = f'{title}'
            logger.warning(f"⚠️ No description found in script.json, using title as fallback")
        
        # Check for thumbnail
        thumbnail_path = self.files_dir / 'thumbnail.jpg'
        if not thumbnail_path.exists():
            thumbnail_path = next(self.files_dir.glob('thumbnail*.jpg'), None)
        
        # Add long video link for short video
        if video_type == 'short' and self.long_video_id:
            description += f"\n\nWatch Full Video: https://youtube.com/watch?v={self.long_video_id}"
            logger.info(f"🔗 Added long video link to short description")
        
        # Return complete metadata structure for YouTube API
        return {
            'snippet': {
                'title': title,
                'description': description,
                'categoryId': '27',  # Education
                'defaultLanguage': 'hi'  # Hindi
            },
            'status': {
                'privacyStatus': 'private',
                'publishAt': scheduled_time,
                'selfDeclaredMadeForKids': False
            }
        }
    
    def _upload_video_with_retry(self, video_path: Path, metadata: Dict) -> str:
        """Upload video with retry logic and resumable upload"""
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"📤 Upload attempt {attempt + 1}/{MAX_RETRIES}: {video_path.name}")
                
                media = MediaFileUpload(
                    str(video_path),
                    chunksize=1024*1024,  # 1MB chunks
                    resumable=True
                )
                
                request = self.youtube.videos().insert(
                    part=','.join(metadata.keys()),
                    body=metadata,
                    media_body=media
                )
                
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        progress = int(status.progress() * 100)
                        logger.info(f"📊 Upload progress: {progress}%")
                
                video_id = response['id']
                logger.info(f"✅ Upload successful! Video ID: {video_id}")
                return video_id
                
            except HttpError as e:
                logger.error(f"❌ HTTP error: {e}")
                if e.resp.status in [403, 500, 503]:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = RETRY_DELAY * (attempt + 1) * 2
                        logger.info(f"⏳ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                raise
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
        
        raise Exception(f"Failed to upload after {MAX_RETRIES} attempts")
    
    def _update_video_metadata(self, video_id: str, metadata: Dict):
        """Update video metadata for internal linking"""
        try:
            self.youtube.videos().update(
                part='snippet',
                body={
                    'id': video_id,
                    'snippet': metadata['snippet']
                }
            ).execute()
            logger.info(f"✅ Video metadata updated for: {video_id}")
        except Exception as e:
            logger.error(f"❌ Failed to update video metadata: {e}")
            raise
    
    def upload_videos(self) -> Tuple[str, str]:
        """Main upload process for both videos"""
        
        # Check required files
        long_video = self.files_dir / 'long_video.mp4'
        short_video = self.files_dir / 'short_video.mp4'
        
        if not long_video.exists():
            raise FileNotFoundError(f"Long video not found: {long_video}")
        if not short_video.exists():
            raise FileNotFoundError(f"Short video not found: {short_video}")
        
        # Get scheduled times
        long_scheduled_time = self._get_scheduled_publish_time('17:00')  # 5:00 PM IST
        short_scheduled_time = self._get_scheduled_publish_time('17:30')  # 5:30 PM IST
        
        logger.info(f"📅 Long video scheduled for: {long_scheduled_time} UTC (5:00 PM IST)")
        logger.info(f"📅 Short video scheduled for: {short_scheduled_time} UTC (5:30 PM IST)")
        
        # Upload long video first
        logger.info("🎬 Uploading LONG video...")
        long_metadata = self._prepare_video_metadata('long', long_scheduled_time)
        self.long_video_id = self._upload_video_with_retry(long_video, long_metadata)
        
        # Upload short video
        logger.info("🎬 Uploading SHORT video...")
        short_metadata = self._prepare_video_metadata('short', short_scheduled_time)
        self.short_video_id = self._upload_video_with_retry(short_video, short_metadata)
        
        # Update short video metadata with internal YouTube linking
        logger.info("🔗 Updating internal video linking...")
        script_data = self._load_json_file('script.json')
        short_metadata['snippet']['description'] += f"\n\n#LongVideo {self.long_video_id}"
        
        self._update_video_metadata(self.short_video_id, short_metadata)
        
        # Log results
        logger.info("=" * 50)
        logger.info("✅ UPLOAD COMPLETE")
        logger.info(f"📹 Long Video ID: {self.long_video_id}")
        logger.info(f"📱 Short Video ID: {self.short_video_id}")
        logger.info(f"⏰ Long Scheduled: {long_scheduled_time} UTC")
        logger.info(f"⏰ Short Scheduled: {short_scheduled_time} UTC")
        logger.info("=" * 50)
        
        return self.long_video_id, self.short_video_id

def main():
    parser = argparse.ArgumentParser(description='Upload videos directly to YouTube')
    parser.add_argument('--channel-id', required=True, help='YouTube channel ID for token lookup')
    parser.add_argument('--files-dir', required=True, help='Directory containing video files')
    
    args = parser.parse_args()
    
    try:
        uploader = YouTubeUploader(args.channel_id, Path(args.files_dir))
        long_id, short_id = uploader.upload_videos()
        
        # Create manifest
        manifest = {
            'run_id': os.getenv('RUN_ID'),
            'channel_id': args.channel_id,
            'long_video_id': long_id,
            'short_video_id': short_id,
            'uploaded_at': datetime.utcnow().isoformat()
        }
        
        manifest_file = Path(args.files_dir) / 'youtube_manifest.json'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📄 Manifest saved to: {manifest_file}")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

if __name__ == '__main__':
    main()
