#!/usr/bin/env python3
"""
Shorts CTA Generator - Creates compelling Call-to-Action for YouTube Shorts
Includes link to full video
"""
import os
import json
import argparse
from pathlib import Path
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_cta(script_file, run_id):
    """Generate CTA for YouTube Shorts with long video link"""
    
    print("📢 Generating CTA for Shorts...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    script_text = script_data.get('script', {}).get('full_text', '')[-1000:]  # Last part
    title = script_data.get('metadata', {}).get('final_title', '')
    category = script_data.get('generation_info', {}).get('category', '')
    
    prompt = f"""You are a YouTube Shorts CTA (Call-to-Action) expert. Create CTAs that drive traffic to long videos.

VIDEO TITLE: {title}
CATEGORY: {category}

SCRIPT ENDING: {script_text}

TASK: Create 5 CTA options for YouTube Shorts (Last 5-8 seconds)

CTA PRINCIPLES:
1. CREATE URGENCY:
   - "Poori story ke liye..."
   - "Complete video dekho..."
   - "Link in description..."

2. TEXT OVERLAY (4-6 words):
   - "Poori Video Description Mein"
   - "Link Bio Mein Hai"
   - "Full Story Channel Pe"
   - Arrow pointing down

3. AUDIO CUE:
   - [EXCITED] for enthusiasm
   - Build anticipation
   - Clear direction

4. CTA PATTERNS THAT WORK:
   - "Ye toh bas trailer tha..."
   - "Asli story abhi baaki hai..."
   - "Poora video description mein..."
   - "Channel pe jaake dekho..."

5. MUST INCLUDE:
   - Reference to full video
   - Clear instruction where to find it
   - Reason to watch full video

OUTPUT FORMAT (STRICT JSON):
{{
  "cta_options": [
    {{
      "cta_id": 1,
      "text_overlay": "Text for screen (4-6 words)",
      "audio_cue": "[EXCITED] or [SERIOUS]",
      "spoken_line": "What narrator says (10-15 words)",
      "duration_seconds": 5,
      "visual_effect": "arrow_pointing_down / text_popup / pulse",
      "description_text": "Text to add in description",
      "conversion_score": "1-10 rating"
    }}
  ],
  "recommended_cta": {{
    "cta_id": 1,
    "reason": "Why this CTA is best"
  }},
  "description_template": "Full description with link placeholder",
  "editing_instructions": {{
    "text_style": "Bold, white or yellow, bottom center",
    "animation": "Bounce or pulse",
    "arrow": "Animated arrow pointing down",
    "duration": "5-8 seconds"
  }}
}}

Generate 5 CTA options now."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=1536
            )
        )
        
        # Parse response
        text = response.text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            cta_data = json.loads(text[start_idx:end_idx])
            
            # Add description template with placeholder
            cta_data['description_template'] = """📺 Poori Video Yahan Dekho: {LONG_VIDEO_URL}

⚠️ Ye sirf ek chhota sa hissa tha... Asli story aur bhi zyada shocking hai!

👇 Description mein link hai - Abhi click karo!

#Shorts #YTShorts #YouTubeShorts #{category}"""
            
            # Save CTA data
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cta_file = output_dir / 'cta.json'
            with open(cta_file, 'w', encoding='utf-8') as f:
                json.dump(cta_data, f, ensure_ascii=False, indent=2)
            
            recommended = cta_data.get('recommended_cta', {})
            print(f"✅ CTA generated (CTA #{recommended.get('cta_id', 1)})")
            return cta_data
        
    except Exception as e:
        print(f"⚠️ CTA generation failed: {e}")
        # Create fallback CTA
        fallback = {
            "cta_options": [{
                "cta_id": 1,
                "text_overlay": "👇 POORI VIDEO YAHAN",
                "audio_cue": "[EXCITED]",
                "spoken_line": "Ye toh bas trailer tha! Poori story description mein hai!",
                "duration_seconds": 5,
                "visual_effect": "arrow_pointing_down",
                "description_text": "Poori video description mein - Abhi dekho!",
                "conversion_score": "8"
            }],
            "recommended_cta": {"cta_id": 1, "reason": "Fallback CTA"},
            "description_template": "📺 Poori Video: {LONG_VIDEO_URL}\n\n#Shorts #YTShorts",
            "editing_instructions": {
                "text_style": "Bold, white, bottom center",
                "animation": "Bounce",
                "arrow": "Animated arrow",
                "duration": "5 seconds"
            }
        }
        
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'cta.json', 'w', encoding='utf-8') as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        
        return fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    generate_cta(args.script_file, args.run_id)

if __name__ == '__main__':
    main()
