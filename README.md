# JSR_Automation - YouTube Video Production Pipeline

GitHub Actions-based video generation system for automated YouTube content creation.

## 🎯 Overview

This repository contains the video production pipeline that:
- Generates scripts using Gemini AI
- Creates audio using Bark TTS
- Downloads stock footage from Pexels
- Generates thumbnails using Stability AI
- Edits videos using FFmpeg
- Uploads to Cloudinary for Tier 1 server retrieval

## 🏗️ Architecture

```
GitHub Actions Workflow (main.yml)
├── Phase 1: Script Generation (Gemini 2.5 Pro)
│   ├── Generate script (10+ min enforcement)
│   ├── Generate title
│   ├── Generate description
│   └── Generate thumbnail concept
├── Phase 2: Audio Generation (Bark TTS)
│   └── Convert script to Hindi audio
├── Phase 3: Asset Acquisition (Pexels API)
│   └── Download stock footage
├── Phase 4: Thumbnail Generation (Stability AI)
│   └── Generate video thumbnail
├── Phase 5: Video Editing - Long (FFmpeg)
│   └── Combine clips + audio
├── Phase 6: Video Editing - Short (FFmpeg)
│   ├── Generate viral hook
│   ├── Generate CTA
│   └── Create short video
├── Phase 7: Upload to Cloudinary
│   └── Upload all assets
└── Phase 8: Cleanup
    └── Remove artifacts
```

## 📁 Directory Structure

```
.github/workflows/
├── main.yml          # Main production pipeline
└── backup.yml        # Backup check workflow

src/
├── prompts/          # Modular Gemini prompts
│   ├── generate_title.py
│   ├── generate_description.py
│   └── generate_thumbnail_concept.py
├── shorts/           # Shorts optimization
│   ├── hook_generator.py
│   └── cta_generator.py
├── video_generation/ # Core video processing
│   ├── generate_script.py
│   ├── generate_audio.py
│   ├── acquire_assets.py
│   ├── generate_thumbnail.py
│   ├── edit_video.py
│   ├── quality_check.py
│   ├── validate_duration.py
│   ├── notify_webhook.py
│   └── check_backup_needed.py
└── youtube/          # YouTube/Cloud upload
    └── upload_to_cloud.py

config/
└── categories.json   # Content categories
```

## 🔧 Setup

### 1. Repository Secrets

Add these secrets in GitHub Settings → Secrets and Variables → Actions:

| Secret | Description | Get From |
|--------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API | makersuite.google.com |
| `PEXELS_API_KEY` | Stock footage API | pexels.com/api |
| `STABILITY_API_KEY` | Thumbnail generation | platform.stability.ai |
| `CLOUDINARY_CLOUD_NAME` | Cloud storage | cloudinary.com |
| `CLOUDINARY_API_KEY` | Cloud storage | cloudinary.com |
| `CLOUDINARY_API_SECRET` | Cloud storage | cloudinary.com |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Database access | Firebase Console |
| `TIER1_WEBHOOK_URL` | Render server webhook | Your Render URL |
| `TIER1_HEALTH_URL` | Render health check | Your Render URL |

### 2. Copy Environment Template

```bash
cp .env.example .env
# Edit .env with your values
```

## 🚀 Workflows

### Main Pipeline (`main.yml`)

**Trigger:** Repository dispatch from JSR_Auto server

**Timeout:** 360 minutes (6 hours) per job

**Jobs:**
1. `generate-script` - Creates video script with modular prompts
2. `generate-audio` - Bark TTS audio generation
3. `acquire-assets` - Pexels stock footage
4. `generate-thumbnail` - Stability AI thumbnail
5. `edit-video-long` - Long video editing (10+ min)
6. `edit-video-short` - Short video with viral hooks
7. `upload-to-cloud` - Cloudinary upload
8. `cleanup` - Artifact cleanup

### Backup Pipeline (`backup.yml`)

**Schedule:** 4:00 AM IST daily

**Purpose:** Checks if main generation failed and triggers backup

## 🎨 Content Categories

See `config/categories.json` for full category structure:

- Human Psychology & Behavior
- Hidden Historical Truths
- Politics Decoded
- Business Fundamentals
- Education System Exposed
- Society Reality
- Communication Mastery
- Human Life Reality

## 📝 Script Generation

### 10+ Minute Enforcement

Scripts are validated to ensure:
- Minimum 1800 words
- 12-15 minute duration target
- Structured sections (Hook, Problem, Promise, Content, Tips, Conclusion)

### Modular Prompts

Each element has dedicated prompt files:
- `generate_title.py` - 5 viral title options
- `generate_description.py` - SEO-optimized description
- `generate_thumbnail_concept.py` - AI thumbnail prompt

## 🎬 Shorts Optimization

### Viral Hooks

First 3 seconds feature:
- Attention-grabbing text overlay
- Emotional audio cue
- Visual effect (zoom/shake/flash)

### CTA (Call-to-Action)

Last 5 seconds include:
- "Poori Video Description Mein" text
- Animated arrow pointing down
- Link to full video

## 🔌 API Integration

### Gemini 2.5 Pro
- Script generation
- Title/description optimization
- Thumbnail concept creation

### Bark TTS
- Hindi text-to-speech
- Speaker presets
- Marker support (PAUSE, EMPHASIS, etc.)

### Pexels
- Stock video footage
- Landscape orientation
- Multiple clips per video

### Stability AI
- Thumbnail generation
- 16:9 aspect ratio
- High contrast output

### Cloudinary
- Video storage
- CDN delivery
- Metadata tagging

## 📊 Monitoring

Each job sends webhook notifications to JSR_Auto:
- `script_generated`
- `audio_generated`
- `assets_downloaded`
- `thumbnail_generated`
- `video_rendered`
- `upload_ready`

## 🛠️ Development

### Testing Locally

```bash
# Install dependencies
pip install -r src/video_generation/requirements.txt

# Test script generation
python src/video_generation/generate_script.py \
  --category "Human Psychology & Behavior" \
  --sub-category "Dark Psychology" \
  --episode 1 \
  --run-id "test_001"
```

### Debug Mode

Set `DEBUG=true` in environment for verbose logging.

## 📄 License

Private - For JSR Auto use only
