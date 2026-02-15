#!/usr/bin/env python3
"""
Modular Description Generation - Generates SEO-optimized YouTube descriptions
"""
import os
import json
import argparse
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_description(script_file, run_id):
    """Generate optimized YouTube description"""
    
    print("📝 Generating optimized description...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    script_text = script_data.get('script', {}).get('full_text', '')[:1500]
    title = script_data.get('metadata', {}).get('final_title', '')
    category = script_data.get('generation_info', {}).get('category', '')
    
    prompt = f"""You are a YouTube SEO expert for Hindi educational content.

VIDEO TITLE: {title}
CATEGORY: {category}

SCRIPT PREVIEW: {script_text}

TASK: Create an SEO-optimized YouTube description following this exact structure:

DESCRIPTION STRUCTURE:
1. HOOK (First 2 lines - visible in search):
   - Compelling summary that makes people click
   - Include main keyword naturally
   - Create curiosity

2. VIDEO SUMMARY (100-150 words):
   - What viewers will learn
   - Key topics covered
   - Value proposition

3. TIMESTAMPS (if applicable):
   - 0:00 - Intro
   - Key section markers

4. KEY TAKEAWAYS:
   - Bullet points of main insights

5. CALL TO ACTION:
   - Subscribe prompt
   - Comment engagement
   - Share request

6. HASHTAGS (10-15 relevant):
   - Mix of popular and niche
   - Include #Shorts for short videos

7. CHANNEL TAGLINE:
   - Brand message

OUTPUT FORMAT (STRICT JSON):
{{
  "description": "Full description text with all sections",
  "hook": "First 2 lines only",
  "keywords_used": ["keyword1", "keyword2"],
  "hashtags": ["#tag1", "#tag2"],
  "seo_score": "Estimated SEO effectiveness"
}}

Generate the description now."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        # Parse response
        text = response.text
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > 0:
            desc_data = json.loads(text[start_idx:end_idx])
            
            # Update script with description
            script_data['metadata']['description'] = desc_data.get('description', '')
            script_data['metadata']['description_hook'] = desc_data.get('hook', '')
            script_data['metadata']['seo_keywords'] = desc_data.get('keywords_used', [])
            
            # Save updated script
            with open(script_file, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Description generated ({len(desc_data.get('description', ''))} chars)")
            return desc_data
        
    except Exception as e:
        print(f"⚠️ Description generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    generate_description(args.script_file, args.run_id)

if __name__ == '__main__':
    main()
