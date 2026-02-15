#!/usr/bin/env python3
"""
Backup Check - Determines if backup video generation is needed
"""
import os
import json
from datetime import datetime, timedelta
from firebase_admin import credentials, firestore, initialize_app

def initialize_firebase():
    """Initialize Firebase connection"""
    service_account_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON')
    
    if not service_account_json:
        print("⚠️ FIREBASE_SERVICE_ACCOUNT_JSON not set")
        return None
    
    try:
        # Parse service account JSON
        service_account_info = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_info)
        
        try:
            app = initialize_app(cred)
        except ValueError:
            # App already initialized
            pass
        
        return firestore.client()
    except Exception as e:
        print(f"⚠️ Firebase initialization failed: {e}")
        return None

def check_backup_needed():
    """Check if backup video generation is needed"""
    
    print("🔍 Checking if backup is needed...")
    
    db = initialize_firebase()
    
    if not db:
        print("⚠️ Cannot check backup status - Firebase not available")
        # Default to not needing backup
        os.environ['BACKUP_NEEDED'] = 'false'
        return False
    
    try:
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Check if video was uploaded today
        videos_ref = db.collection('videos')
        today_query = videos_ref.where('uploadDate', '==', today).limit(1)
        today_docs = list(today_query.stream())
        
        if today_docs:
            print(f"✅ Video already uploaded today ({today})")
            os.environ['BACKUP_NEEDED'] = 'false'
            return False
        
        # Check if video was uploaded yesterday (main run)
        yesterday_query = videos_ref.where('uploadDate', '==', yesterday).limit(1)
        yesterday_docs = list(yesterday_query.stream())
        
        if yesterday_docs:
            print(f"✅ Video was uploaded yesterday ({yesterday})")
            os.environ['BACKUP_NEEDED'] = 'false'
            return False
        
        # No video found - backup needed
        print(f"⚠️ No video found for today or yesterday - backup needed!")
        
        # Get next episode info
        episodes_ref = db.collection('episodes').order_by('episode', direction='DESCENDING').limit(1)
        episode_docs = list(episodes_ref.stream())
        
        if episode_docs:
            episode_data = episode_docs[0].to_dict()
            next_episode = episode_data.get('episode', 1) + 1
            category = episode_data.get('category', 'Human Psychology & Behavior')
            sub_category = episode_data.get('sub_category', 'Dark Psychology')
        else:
            # Default values
            next_episode = 1
            category = 'Human Psychology & Behavior'
            sub_category = 'Dark Psychology'
        
        # Set environment variables for workflow
        os.environ['BACKUP_NEEDED'] = 'true'
        os.environ['CATEGORY_OUTPUT'] = category
        os.environ['SUB_CATEGORY_OUTPUT'] = sub_category
        os.environ['EPISODE_OUTPUT'] = str(next_episode)
        
        print(f"📋 Backup will generate:")
        print(f"   Category: {category}")
        print(f"   Sub-category: {sub_category}")
        print(f"   Episode: {next_episode}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Backup check failed: {e}")
        os.environ['BACKUP_NEEDED'] = 'false'
        return False

def main():
    needed = check_backup_needed()
    
    # Output for GitHub Actions
    print(f"::set-output name=backup_needed::{'true' if needed else 'false'}")
    
    if needed:
        print(f"::set-output name=category::{os.getenv('CATEGORY_OUTPUT', '')}")
        print(f"::set-output name=sub_category::{os.getenv('SUB_CATEGORY_OUTPUT', '')}")
        print(f"::set-output name=episode::{os.getenv('EPISODE_OUTPUT', '')}")

if __name__ == '__main__':
    main()
