#!/usr/bin/env python3
"""
Modular Title Generation - Generates optimized YouTube titles
"""
import os
import json
import argparse
from pathlib import Path
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_title(script_file, run_id):
    """Generate optimized YouTube title based on script content"""
    
    print("🎯 Generating optimized title...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    script_text = script_data.get('script', {}).get('full_text', '')[:2000]
    current_title = script_data.get('metadata', {}).get('final_title', '')
    
    prompt = f"""You are a YouTube title optimization expert for Hindi content.

CURRENT TITLE: {current_title}

SCRIPT PREVIEW: {script_text}

TASK: Create 5 viral YouTube title options following these rules:

TITLE OPTIMIZATION RULES:
1. Length: 50-70 characters (optimal for visibility)
2. Use numbers when possible (3, 5, 7, 10 work best)
3. Create curiosity gap - don't reveal everything
4. Use power words: Shocking, Secret, Truth, Reality, Exposed
5. Include emotional triggers: Fear, Curiosity, Hope
6. Make it feel personal: "Tum", "Aap", "Kya"
7. Add urgency or surprise element

EXAMPLES OF GREAT TITLES:
- "7 Signs Someone Is Secretly Manipulating You"
- "The Dark Truth About Indian Education System"
- "Kya Tum Bhi Is Trap Mein Fase Ho?"
- "Ye 3 Baatein Tumhe Kabhi Nahi Batayi Gayi"
- "Psychology Ke Ye Secrets 99% Log Nahi Jaante"

OUTPUT FORMAT (STRICT JSON):
{{
  "title_options": [
    {{
      "title": "Option 1",
      "character_count": 55,
      "why_it_works": "Explanation"
    }}
  ],
  "recommended_title": "Best option",
  "title_analysis": "Why this title will perform well"
}}

Generate 5 title options now."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.8,
                max_output_tokens=1024
            )
        )
        
        # Parse response
        text = response.text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            title_data = json.loads(text[start_idx:end_idx])
            
            # Update script with new title
            script_data['metadata']['title_options'] = title_data.get('title_options', [])
            script_data['metadata']['final_title'] = title_data.get('recommended_title', current_title)
            script_data['metadata']['title_analysis'] = title_data.get('title_analysis', '')
            
            # Save updated script
            with open(script_file, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Title generated: {script_data['metadata']['final_title']}")
            return title_data
        
    except Exception as e:
        print(f"⚠️ Title generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    generate_title(args.script_file, args.run_id)

if __name__ == '__main__':
    main()
