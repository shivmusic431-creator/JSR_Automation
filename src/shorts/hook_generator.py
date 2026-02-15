#!/usr/bin/env python3
"""
Shorts Viral Hook Generator - Creates attention-grabbing hooks for YouTube Shorts
"""
import os
import json
import argparse
from pathlib import Path
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_hook(script_file, run_id):
    """Generate viral hook for YouTube Shorts"""
    
    print("🪝 Generating viral hook for Shorts...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    script_text = script_data.get('script', {}).get('full_text', '')[:1500]
    title = script_data.get('metadata', {}).get('final_title', '')
    category = script_data.get('generation_info', {}).get('category', '')
    
    prompt = f"""You are a YouTube Shorts viral content expert. Create hooks that stop the scroll.

VIDEO TITLE: {title}
CATEGORY: {category}

SCRIPT CONTEXT: {script_text}

TASK: Create 5 viral hook options for YouTube Shorts (First 3 seconds text overlay + audio cue)

VIRAL HOOK PRINCIPLES:
1. FIRST 1 SECOND - STOP THE SCROLL:
   - Shocking statement or question
   - Contrarian opinion
   - "Tumhe pata hai..." pattern
   - Numbers + surprise element

2. TEXT OVERLAY (3-5 words max):
   - Large, bold, centered
   - ALL CAPS or mixed case
   - Emoji for emotion
   - Color: Yellow/Red/White

3. AUDIO CUE:
   - [EXCITED] for energy
   - [SERIOUS] for shock
   - [QUESTION] for curiosity
   - Sound effect suggestion

4. HOOK PATTERNS THAT WORK:
   - "3 Cheezein Jo Main..."
   - "Kya Tumne Kabhi Socha..."
   - "Ye Sach Hai Ya Jhooth?"
   - "Stop! ⚠️ Ye Mat Karna"
   - "1 Minute Mein Jaan Lo..."

OUTPUT FORMAT (STRICT JSON):
{{
  "hook_options": [
    {{
      "hook_id": 1,
      "text_overlay": "Text for screen (3-5 words)",
      "audio_cue": "[EXCITED] or [SERIOUS] etc",
      "spoken_line": "What narrator says",
      "duration_seconds": 3,
      "visual_effect": "zoom_in / shake / flash / color_pop",
      "why_it_works": "Psychology explanation",
      "viral_score": "1-10 rating"
    }}
  ],
  "recommended_hook": {{
    "hook_id": 1,
    "reason": "Why this hook is best"
  }},
  "editing_instructions": {{
    "text_style": "Bold, yellow, centered",
    "animation": "Typewriter or fade in",
    "background": "Blur or dark overlay",
    "sound_effect": "Suggested SFX"
  }}
}}

Generate 5 hook options now."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=1536
            )
        )
        
        # Parse response
        text = response.text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            hook_data = json.loads(text[start_idx:end_idx])
            
            # Save hook data
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            hook_file = output_dir / 'hook.json'
            with open(hook_file, 'w', encoding='utf-8') as f:
                json.dump(hook_data, f, ensure_ascii=False, indent=2)
            
            recommended = hook_data.get('recommended_hook', {})
            print(f"✅ Viral hook generated (Hook #{recommended.get('hook_id', 1)})")
            return hook_data
        
    except Exception as e:
        print(f"⚠️ Hook generation failed: {e}")
        # Create fallback hook
        fallback = {
            "hook_options": [{
                "hook_id": 1,
                "text_overlay": "⚠️ YE MAT KARNA!",
                "audio_cue": "[SERIOUS]",
                "spoken_line": "Tumhe pata hai ye galti sabse zyada log karte hain?",
                "duration_seconds": 3,
                "visual_effect": "zoom_in",
                "why_it_works": "Creates curiosity and fear of missing out",
                "viral_score": "8"
            }],
            "recommended_hook": {"hook_id": 1, "reason": "Fallback hook"}
        }
        
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'hook.json', 'w', encoding='utf-8') as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        
        return fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    generate_hook(args.script_file, args.run_id)

if __name__ == '__main__':
    main()
