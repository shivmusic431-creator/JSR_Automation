#!/usr/bin/env python3
"""
Webhook Notifier - Sends notifications to Tier 1 server
FIXED: Added --channel-id, --long-video-id, --short-video-id arguments
"""
import os
import json
import argparse
import requests
from datetime import datetime

def notify_webhook(action, run_id, status, video_type=None, backup_needed=None,
                   channel_id=None, long_video_id=None, short_video_id=None):
    """Send notification to Tier 1 webhook"""

    webhook_url = os.getenv('TIER1_WEBHOOK_URL')

    if not webhook_url:
        print("⚠️ TIER1_WEBHOOK_URL not set, skipping notification")
        return

    print(f"📤 Sending notification: {action} - {status}")

    payload = {
        'action': action,
        'run_id': run_id,
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'source': 'github_actions'
    }

    if video_type:
        payload['video_type'] = video_type
    if backup_needed is not None:
        payload['backup_needed'] = backup_needed
    if channel_id:
        payload['channel_id'] = channel_id
    if long_video_id:
        payload['long_video_id'] = long_video_id
    if short_video_id:
        payload['short_video_id'] = short_video_id

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        if response.status_code == 200:
            print(f"✅ Notification sent: {action}")
        else:
            print(f"⚠️ Notification failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Notification error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--video-type', default=None)
    parser.add_argument('--backup-needed', default=None)
    parser.add_argument('--channel-id', default=None)
    parser.add_argument('--long-video-id', default=None)
    parser.add_argument('--short-video-id', default=None)
    args = parser.parse_args()

    notify_webhook(
        args.action,
        args.run_id,
        args.status,
        args.video_type,
        args.backup_needed,
        args.channel_id,
        args.long_video_id,
        args.short_video_id
    )

if __name__ == '__main__':
    main()
