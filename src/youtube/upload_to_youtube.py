#!/usr/bin/env python3
"""
YouTube Direct Uploader - Uploads SHORT video to YouTube using Firestore OAuth tokens
FIXED: Long video references removed - only short video upload
"""
import os
import json
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pytz
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

YT_API_SERVICE_NAME = 'youtube'
YT_API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
MAX_RETRIES = 3
RETRY_DELAY = 5

class YouTubeUploader:
    def __init__(self, channel_id: str, files_dir: Path):
        self.channel_id = channel_id
        self.files_dir = Path(files_dir)
        self.youtube = None
        self.short_video_id = None
        self._init_firebase()
        self._init_youtube_client()

    def _init_firebase(self):
        try:
            if not firebase_admin._apps:
                firebase_creds_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
                if not firebase_creds_json:
                    raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON not set")
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("✅ Firebase initialized")
        except Exception as e:
            logger.error(f"❌ Firebase init failed: {e}")
            raise

    def _get_token_from_firestore(self) -> Dict:
        try:
            clean_channel_id = self.channel_id.strip()
            logger.info(f"🔍 Looking up token for channel: {clean_channel_id}")

            # Collection: userTokens (saved by JSR_Auto Node.js OAuth flow)
            doc_ref = self.db.collection('userTokens').document(clean_channel_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.error("=" * 60)
                logger.error("❌ Token not found in Firestore!")
                logger.error(f"   Channel ID: {clean_channel_id}")
                logger.error("   Please connect your YouTube channel via the JSR_Auto dashboard first.")
                logger.error("=" * 60)
                raise ValueError(f"No token found for channel: {clean_channel_id}")

            data = doc.to_dict()

            # Node.js saves: accessToken, refreshToken (camelCase)
            required_fields = ['accessToken', 'refreshToken']
            missing = [f for f in required_fields if not data.get(f)]
            if missing:
                raise ValueError(f"Token document missing fields: {', '.join(missing)}")

            # client_id and client_secret come from env vars (set in GitHub Secrets)
            client_id = os.getenv('YOUTUBE_CLIENT_ID')
            client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
            if not client_id or not client_secret:
                raise ValueError("YOUTUBE_CLIENT_ID or YOUTUBE_CLIENT_SECRET env vars not set")

            return {
                'access_token': data['accessToken'],
                'refresh_token': data['refreshToken'],
                'client_id': client_id,
                'client_secret': client_secret,
            }
        except Exception as e:
            logger.error(f"❌ Firestore token fetch failed: {e}")
            raise

    def _init_youtube_client(self):
        try:
            token_data = self._get_token_from_firestore()
            creds = Credentials(
                token=token_data['access_token'],
                refresh_token=token_data['refresh_token'],
                client_id=token_data['client_id'],
                client_secret=token_data['client_secret'],
                token_uri='https://oauth2.googleapis.com/token',
                scopes=SCOPES
            )
            if creds.expired and creds.refresh_token:
                logger.info("🔄 Refreshing expired token...")
                creds.refresh(Request())
                # Save refreshed token back to userTokens collection (camelCase)
                self.db.collection('userTokens').document(self.channel_id.strip()).update({
                    'accessToken': creds.token,
                    'updatedAt': datetime.now().isoformat()
                })
                logger.info("✅ Token refreshed and saved")
            self.youtube = build(YT_API_SERVICE_NAME, YT_API_VERSION, credentials=creds)
            logger.info("✅ YouTube client initialized")
        except Exception as e:
            logger.error(f"❌ YouTube client init failed: {e}")
            raise

    def _get_scheduled_time(self) -> str:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        publish_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
        if now >= publish_time:
            publish_time += timedelta(days=1)
        return publish_time.strftime('%Y-%m-%dT%H:%M:%S+05:30')

    def _build_video_metadata(self, script_data: dict, video_type: str) -> dict:
        metadata = script_data.get('metadata', {})
        title = metadata.get('final_title', 'YouTube Short')
        description = metadata.get('description', '')
        tags = metadata.get('tags', [])

        if video_type == 'short':
            if not title.endswith('#Shorts'):
                title = f"{title} #Shorts"
            hashtags = metadata.get('hashtags', [])
            description += f"\n\n{' '.join(hashtags)}\n\n#Shorts #YTShorts"

        return {
            'snippet': {
                'title': title[:100],
                'description': description[:5000],
                'tags': tags[:500],
                'categoryId': '22',
                'defaultLanguage': 'hi',
                'defaultAudioLanguage': 'hi'
            },
            'status': {
                'privacyStatus': 'private',
                'publishAt': self._get_scheduled_time(),
                'selfDeclaredMadeForKids': False
            }
        }

    def _upload_video_with_retry(self, video_path: Path, metadata: dict) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"📤 Uploading: {video_path.name} (attempt {attempt + 1})")
                media = MediaFileUpload(
                    str(video_path),
                    mimetype='video/mp4',
                    resumable=True,
                    chunksize=1024*1024*5
                )
                request = self.youtube.videos().insert(
                    part='snippet,status',
                    body=metadata,
                    media_body=media
                )
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        logger.info(f"   Upload progress: {int(status.progress() * 100)}%")
                video_id = response['id']
                logger.info(f"✅ Uploaded: https://youtube.com/watch?v={video_id}")
                return video_id
            except HttpError as e:
                logger.error(f"❌ HTTP error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ Upload error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    def upload_videos(self) -> tuple:
        # Load script data
        script_file = self.files_dir / 'script.json'
        if not script_file.exists():
            raise FileNotFoundError(f"script.json not found: {script_file}")
        with open(script_file, 'r', encoding='utf-8') as f:
            script_data = json.load(f)

        # Check required file
        short_video = self.files_dir / 'short_video.mp4'
        if not short_video.exists():
            raise FileNotFoundError(f"Short video not found: {short_video}")

        logger.info(f"📱 Short video: {short_video} ({short_video.stat().st_size / 1024 / 1024:.1f} MB)")

        # Upload short video
        short_metadata = self._build_video_metadata(script_data, 'short')
        self.short_video_id = self._upload_video_with_retry(short_video, short_metadata)

        logger.info(f"📱 Short Video ID: {self.short_video_id}")
        logger.info(f"📱 Short: https://youtube.com/shorts/{self.short_video_id}")

        # Save manifest
        manifest = {
            'short_video_id': self.short_video_id,
            'upload_time': datetime.now().isoformat()
        }
        manifest_file = self.files_dir / 'youtube_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        return self.short_video_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel-id', required=True)
    parser.add_argument('--files-dir', required=True)
    args = parser.parse_args()

    uploader = YouTubeUploader(args.channel_id, Path(args.files_dir))
    short_id = uploader.upload_videos()

    print(f"::set-output name=short_video_id::{short_id}")

if __name__ == '__main__':
    main()
