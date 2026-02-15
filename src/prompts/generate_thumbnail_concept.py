#!/usr/bin/env python3
"""
Modular Thumbnail Concept Generation - Generates AI-optimized thumbnail concepts
"""
import os
import json
import argparse
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_thumbnail_concept(script_file, run_id):
    """Generate AI-optimized thumbnail concept"""
    
    print("🎨 Generating thumbnail concept...")
    
    # Load script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_data = json.load(f)
    
    title = script_data.get('metadata', {}).get('final_title', '')
    script_text = script_data.get('script', {}).get('full_text', '')[:1000]
    category = script_data.get('generation_info', {}).get('category', '')
    
    prompt = f"""You are a YouTube thumbnail design expert specializing in high-CTR thumbnails for Hindi content.

VIDEO TITLE: {title}
CATEGORY: {category}

SCRIPT CONTEXT: {script_text}

TASK: Create a detailed thumbnail concept for AI image generation following these rules:

THUMBNAIL DESIGN PRINCIPLES:
1. ATTENTION GRABBING:
   - High contrast colors (red, yellow, orange work best)
   - Clear focal point
   - Emotional facial expression

2. TEXT ELEMENTS:
   - Maximum 3-5 words
   - Large, bold, readable font
   - Hindi or Hinglish text

3. VISUAL COMPOSITION:
   - Rule of thirds
   - Face should take 30-40% of thumbnail
   - Eyes looking at camera or text

4. COLOR PSYCHOLOGY:
   - Red: Urgency, warning, important
   - Yellow: Attention, curiosity
   - Orange: Energy, excitement
   - Blue: Trust, information

5. ELEMENTS TO INCLUDE:
   - Expressive face (shocked, curious, concerned)
   - Arrow or circle highlighting key element
   - Numbers or symbols for curiosity

OUTPUT FORMAT (STRICT JSON):
{{
  "thumbnail_concept": {{
    "main_subject": "Description of main person/character",
    "facial_expression": "Specific expression details",
    "background": "Background description",
    "text_overlay": "Text to display",
    "color_scheme": ["color1", "color2", "color3"],
    "lighting": "Lighting description",
    "additional_elements": ["element1", "element2"],
    "composition_notes": "How elements should be arranged"
  }},
  "stability_ai_prompt": "Detailed prompt for Stability AI image generation",
  "alternative_concepts": [
    {{
      "concept_name": "Alternative 1",
      "description": "Brief description"
    }}
  ],
  "ctr_prediction": "Estimated click-through rate factors"
}}

Generate the thumbnail concept now."""

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
            concept_data = json.loads(text[start_idx:end_idx])
            
            # Update script with thumbnail concept
            script_data['metadata']['thumbnail_concept'] = concept_data.get('thumbnail_concept', {})
            script_data['metadata']['stability_ai_prompt'] = concept_data.get('stability_ai_prompt', '')
            script_data['metadata']['thumbnail_alternatives'] = concept_data.get('alternative_concepts', [])
            
            # Save updated script
            with open(script_file, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Thumbnail concept generated")
            return concept_data
        
    except Exception as e:
        print(f"⚠️ Thumbnail concept generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script-file', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    
    generate_thumbnail_concept(args.script_file, args.run_id)

if __name__ == '__main__':
    main()
