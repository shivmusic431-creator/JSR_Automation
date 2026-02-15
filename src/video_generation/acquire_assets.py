#!/usr/bin/env python3
"""
Asset Acquisition - Downloads stock footage from Pexels based on script scenes
"""
import os
import json
import argparse
import requests
from pathlib import Path
from urllib.parse import quote
import time

PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
PEXELS_PHOTO_API = "https://api.pexels.com/v1/search"

def search_pexels_videos(query, per_page=5):
    """Search Pexels for videos"""
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape"
    }
    
    try:
        response = requests.get(PEXELS_VIDEO_API, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠️ Pexels search failed: {e}")
        return {"videos": []}

def download_video(url, output_path):
    """Download video from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        return False

def acquire_assets(script_file, duration, run_id):
    """Acquire stock footage based on script scenes"""
    
    print("📹 Acquiring stock footage from Pexels...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    scene_breakdown = script_data.get('script', {}).get('scene_breakdown', [])
    
    # If no scene breakdown, create from script text
    if not scene_breakdown:
        print("⚠️ No scene breakdown found, creating generic scenes...")
        scene_breakdown = [
            {"scene_type": "intro", "stock_search": "indian people thinking"},
            {"scene_type": "main", "stock_search": "office workplace india"},
            {"scene_type": "conclusion", "stock_search": "happy people success"}
        ]
    
    output_dir = Path('output/clips')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_clips = []
    
    for i, scene in enumerate(scene_breakdown):
        search_query = scene.get('stock_search', scene.get('scene_type', 'people'))
        
        print(f"🔍 Searching: {search_query}")
        
        # Search Pexels
        results = search_pexels_videos(search_query)
        videos = results.get('videos', [])
        
        if videos:
            # Download first video
            video = videos[0]
            video_files = video.get('video_files', [])
            
            # Find HD or SD quality
            video_file = None
            for vf in video_files:
                if vf.get('quality') in ['hd', 'sd']:
                    video_file = vf
                    break
            
            if not video_file and video_files:
                video_file = video_files[0]
            
            if video_file:
                url = video_file.get('link')
                ext = 'mp4'
                output_path = output_dir / f"clip_{i:03d}.{ext}"
                
                print(f"⬇️ Downloading clip {i+1}: {output_path.name}")
                
                if download_video(url, output_path):
                    downloaded_clips.append({
                        'scene': scene.get('scene_type', 'unknown'),
                        'file': str(output_path),
                        'query': search_query,
                        'pexels_id': video.get('id')
                    })
                    print(f"✅ Downloaded: {output_path.name}")
                else:
                    print(f"❌ Failed to download: {search_query}")
        else:
            print(f"⚠️ No results for: {search_query}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save clip manifest
    manifest = {
        'run_id': run_id,
        'total_clips': len(downloaded_clips),
        'clips': downloaded_clips
    }
    
    manifest_file = output_dir / 'manifest.json'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Acquired {len(downloaded_clips)} clips")
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    acquire_assets(args.script_file, args.duration, args.run_id)

if __name__ == '__main__':
    main()
