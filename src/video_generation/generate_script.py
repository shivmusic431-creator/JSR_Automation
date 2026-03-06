#!/usr/bin/env python3
"""
YT-AutoPilot Pro - Script Generation with Gemini 2.5 API
Single API Call: script + title + description + thumbnail concept
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import re

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("❌ google.genai module not found! pip install google-genai")
    sys.exit(1)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set in environment variables")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

CATEGORIES_CONFIG = {
    "Human Psychology & Behavior": {
        "hindi_name": "मानव मनोविज्ञान और व्यवहार",
        "sub_categories": {
            "Dark Psychology": "डार्क साइकोलॉजी",
            "Life Hacks Psychology": "लाइफ हैक्स मनोविज्ञान",
            "Behavioral Psychology": "व्यवहार मनोविज्ञान",
            "Body Language Secrets": "बॉडी लैंग्वेज सीक्रेट्स"
        }
    },
    "Hidden Historical Truths": {
        "hindi_name": "इतिहास की छुपी सच्चाई",
        "sub_categories": {
            "Untold School History": "वो सच जो स्कूलों में नहीं पढ़ाए",
            "Historical Conspiracies": "ऐतिहासिक षड्यंत्र",
            "Real Stories of Kings": "राजाओं की असली कहानियां",
            "Unknown Freedom Struggle": "स्वतंत्रता संग्राम के अनसुने पहलू"
        }
    },
    "Politics Decoded": {
        "hindi_name": "राजनीति का खेल",
        "sub_categories": {
            "Vote Bank Psychology": "वोट बैंक की साइकोलॉजी",
            "Real Intent Behind Schemes": "स्कीमों का असली मकसद",
            "Leader Manipulation": "नेताओं की मैनिपुलेशन ट्रिक्स",
            "Election Strategies": "चुनावी रणनीतियां"
        }
    },
    "Business Fundamentals": {
        "hindi_name": "बिजनेस की बुनियाद",
        "sub_categories": {
            "Businessman Mindset": "बिजनेसमैन माइंडसेट",
            "Building Systems": "सिस्टम बनाना सीखो",
            "Money Works For You": "पैसे काम करें आप नहीं",
            "Startup Psychology": "स्टार्टअप साइकोलॉजी"
        }
    },
    "Education System Exposed": {
        "hindi_name": "स्टडी सिस्टम रिव्यू",
        "sub_categories": {
            "Why Old Education Fails": "पुरानी पढ़ाई क्यों फेल है",
            "School vs Real Life": "स्कूल vs रियल लाइफ",
            "Real Education for Success": "सक्सेस की असली पढ़ाई",
            "Daily Routine Mastery": "डेली रूटीन मास्टरी"
        }
    },
    "Society Reality": {
        "hindi_name": "समाज का सच",
        "sub_categories": {
            "Cycle of Poverty": "गरीबी का चक्र",
            "Secrets of Rich Society": "अमीर समाज के रहस्य",
            "Social Class Psychology": "सोशल क्लास साइकोलॉजी",
            "Breaking the System": "ब्रेकिंग द सिस्टम"
        }
    },
    "Communication Mastery": {
        "hindi_name": "कम्युनिकेशन मास्टरी",
        "sub_categories": {
            "Presentation Psychology": "प्रेजेंटेशन साइकोलॉजी",
            "Less Education More Impact": "कम पढ़े लिखे का जादू",
            "Art of Speaking": "बोलने की कला",
            "Impactful Writing": "प्रभावशाली लेखन"
        }
    },
    "Human Life Reality": {
        "hindi_name": "इंसानी जिंदगी की हकीकत",
        "sub_categories": {
            "Lies About Success": "सक्सेस का झूठ",
            "Relations Marketplace": "रिश्तों का बाजार",
            "Emotional Manipulation": "भावनाओं की दुकानदारी",
            "Real Way of Living": "जीने का असली तरीका"
        }
    }
}

EPISODE_IDEAS = {
    ("Human Psychology & Behavior", "Dark Psychology"): [
        "Gaslighting: Kaise Log Tumhari Yaadashth Ko Control Karte Hain",
        "Emotional Blackmail Ke 5 Chehre Jo Tum Pehchan Nahi Paate",
        "Manipulation Ke 7 Signs - Pehchano Aur Bacho",
        "Love Bombing: Pyar Ya Phasaav?",
        "Toxic Log Kaise Tumhari Energy Churate Hain",
        "Narcissist Ki Pehchan Kaise Karein?",
        "Guilt Tripping Se Kaise Bachein?",
        "Passive-Aggressive Behavior Samajhna",
        "Trauma Bonding Kya Hai?",
        "Toxic Workplace Se Kaise Nikle?"
    ],
    ("Human Psychology & Behavior", "Life Hacks Psychology"): [
        "Baat Maanwane Ka Psychology",
        "Interview Mein Select Hone Ke Tarike",
        "First Impression Kaise Banaye?",
        "Logon Ko Apni Taraf Kaise Karein?",
        "Negotiation Ke Psychology Tricks",
        "Memory Improve Karne Ke Tarike",
        "Decision Making Ke Shortcuts",
        "Social Anxiety Kaise Kam Karein?",
        "Confidence Dikhane Ke Tarike",
        "Influence Kaise Badhayein?"
    ]
}

def get_episode_title(category, sub_category, episode):
    key = (category, sub_category)
    if key in EPISODE_IDEAS:
        ideas = EPISODE_IDEAS[key]
        if episode <= len(ideas):
            return ideas[episode - 1]
    return f"{sub_category} - Episode {episode}"

def create_script_prompt(category, sub_category, episode, title):
    hindi_category = CATEGORIES_CONFIG.get(category, {}).get("hindi_name", category)
    hindi_sub = CATEGORIES_CONFIG.get(category, {}).get("sub_categories", {}).get(sub_category, sub_category)

    return f"""You are an elite Hindi content strategist and scriptwriter creating viral YouTube content.

TASK: In ONE response, generate: complete script + title options + SEO description + thumbnail concept.

INPUT:
- Main Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title: {title}
- Target Duration: 10-15 MINUTES (1400-1900 Hindi words)
- Audience: 18-35 years, Hindi-speaking, Indian urban/semi-urban
- Tone: Conversational, conspiratorial, empowering
- Language: PURE HINDI (देवनागरी लिपि only)

ABSOLUTE LANGUAGE RULE:
- ZERO English letters in narration (a-z or A-Z)
- English words in Hindi phonetics: brain→ब्रेन, psychology→साइकोलॉजी, reality→रियलिटी

SCENE MARKERS (separate line only, never spoken):
[SCENE: nature_morning] [SCENE: office_tension] [SCENE: family_dining]
[SCENE: phone_scrolling] [SCENE: thinking_alone] [SCENE: celebration]
[SCENE: dark_alley] [SCENE: books_study] [SCENE: city_traffic]

EMOTION INDICATORS (separate line before narration):
(धीरे से) (गंभीर स्वर में) (रहस्यमय स्वर में) (उत्साह से)
(हल्की मुस्कान के साथ) (फुसफुसाते हुए) (आश्चर्य से) (प्यार से)

CORRECT:
(गंभीर स्वर में)
तुम्हें सच जानना होगा।

WRONG:
(गंभीर स्वर में) तुम्हें सच जानना होगा।

SCRIPT STRUCTURE (1400-1900 words):
1. HOOK (100-130 words): Shocking question, immediate connection
2. PROBLEM AGITATION (200-280 words): Pain point in detail, use "तुम"
3. PROMISE (150-200 words): What they'll learn, build anticipation
4. MAIN CONTENT (1000-1400 words): 4-6 sections, concept+example+psychology+application
5. PRACTICAL TIPS (300-400 words): 5-7 specific actionable steps
6. CONCLUSION (200-250 words): Summary, CTA, next episode teaser

TITLE RULES: 50-70 chars, use numbers, curiosity gap, power words (Secret/Truth/Reality)

SEO DESCRIPTION: Hook (2 lines) + Summary (100-150 words) + Takeaways + CTA + 10-15 hashtags

THUMBNAIL: High contrast, shocked face, 3-5 word overlay, red/yellow/orange colors

Return ONLY raw JSON (no markdown, no explanation):
{{
  "metadata": {{
    "final_title": "Best title string",
    "title_options": [
      {{"title": "Option 1", "character_count": 55, "why_it_works": "explanation"}},
      {{"title": "Option 2", "character_count": 60, "why_it_works": "explanation"}},
      {{"title": "Option 3", "character_count": 58, "why_it_works": "explanation"}},
      {{"title": "Option 4", "character_count": 62, "why_it_works": "explanation"}},
      {{"title": "Option 5", "character_count": 57, "why_it_works": "explanation"}}
    ],
    "title_analysis": "Why recommended title will perform well",
    "description": "Full SEO description with all sections",
    "description_hook": "First 2 lines only",
    "seo_keywords": ["keyword1", "keyword2", "keyword3"],
    "hashtags": ["#tag1", "#tag2", "#tag3"],
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "thumbnail_concept": {{
      "main_subject": "Description of main person",
      "facial_expression": "Specific shocked/curious expression",
      "background": "Background description",
      "text_overlay": "3-5 words for thumbnail",
      "color_scheme": ["red", "yellow", "black"],
      "lighting": "Dramatic cinematic lighting",
      "additional_elements": ["arrow", "circle highlight"],
      "composition_notes": "Rule of thirds, face 40% frame"
    }},
    "stability_ai_prompt": "Detailed Stability AI image generation prompt",
    "thumbnail_alternatives": [
      {{"concept_name": "Alternative 1", "description": "Brief description"}}
    ],
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode}
  }},
  "script": {{
    "hook": "Hook text with emotion indicators on separate lines and scene markers",
    "problem_agitation": "Problem text with indicators and markers",
    "promise": "Promise text with indicators",
    "main_content": [
      {{"section_title": "Hindi section title", "content": "Detailed content with indicators and markers"}}
    ],
    "practical_tips": [
      {{"tip_number": 1, "tip_title": "Hindi tip title", "explanation": "Detailed explanation"}}
    ],
    "conclusion": "Conclusion with indicators",
    "full_text": "COMPLETE script as single string joining all sections in order",
    "word_count": 1600,
    "estimated_duration": "12:30"
  }}
}}

CRITICAL: full_text must be ALL sections joined together as one complete readable script."""

def extract_json_from_response(text: str) -> str:
    print("🔍 Extracting JSON from response...")

    try:
        json.loads(text)
        print("✓ Response is pure valid JSON")
        return text
    except json.JSONDecodeError:
        pass

    code_block_patterns = [
        r'```json\s*\n(.*?)\n```', r'```\s*\n(.*?)\n```',
        r'```json(.*?)```', r'```(.*?)```',
        r'```json\s*\n(.*)', r'```\s*\n(.*)',
    ]
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                repaired = repair_json(match)
                if repaired:
                    return repaired
                salvaged = salvage_truncated_json(match)
                if salvaged:
                    return salvaged

    json_candidate = find_balanced_json(text)
    if json_candidate:
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            repaired = repair_json(json_candidate)
            if repaired:
                return repaired
            salvaged = salvage_truncated_json(json_candidate)
            if salvaged:
                return salvaged

    salvaged = salvage_truncated_json(text)
    if salvaged:
        return salvaged

    raise ValueError("No valid JSON found in response.")

def find_balanced_json(text: str) -> str:
    brace_stack = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                if brace_stack == 0:
                    start = i
                brace_stack += 1
            elif ch == '}':
                brace_stack -= 1
                if brace_stack == 0 and start is not None:
                    return text[start:i+1]
    if start is not None and brace_stack > 0:
        return text[start:]
    return None

def repair_json(json_str: str) -> str:
    original = json_str
    repairs = []
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    if fixed != json_str:
        repairs.append("trailing_commas")
        json_str = fixed
    fixed = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    if fixed != json_str:
        repairs.append("comments")
        json_str = fixed
    if repairs:
        try:
            json.loads(json_str)
            return json_str
        except:
            return aggressive_repair(original)
    return None

def aggressive_repair(json_str: str) -> str:
    try:
        json.loads(json_str)
    except json.JSONDecodeError as e:
        truncated = json_str[:e.pos]
        ob = truncated.count('{') - truncated.count('}')
        ob2 = truncated.count('[') - truncated.count(']')
        repaired = truncated + ']' * ob2 + '}' * ob
        try:
            json.loads(repaired)
            return repaired
        except:
            pass
    return None

def salvage_truncated_json(text: str) -> str:
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    json_text = text[start_idx:]
    in_string = False
    escape_next = False
    last_valid_pos = 0
    open_braces = 0
    open_brackets = 0
    for i, ch in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{': open_braces += 1
            elif ch == '}': open_braces -= 1
            elif ch == '[': open_brackets += 1
            elif ch == ']': open_brackets -= 1
            if open_braces >= 0 and open_brackets >= 0:
                last_valid_pos = i
    candidate = json_text[:last_valid_pos + 1]
    candidate = re.sub(r',\s*$', '', candidate.rstrip())
    in_string = False
    escape_next = False
    ob = ob2 = 0
    for ch in candidate:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{': ob += 1
            elif ch == '}': ob -= 1
            elif ch == '[': ob2 += 1
            elif ch == ']': ob2 -= 1
    repaired = candidate + ']' * ob2 + '}' * ob
    try:
        parsed = json.loads(repaired)
        if 'metadata' in parsed and 'script' in parsed:
            return repaired
    except:
        pass
    for step_back in [50, 100, 200, 500, 1000, 2000]:
        truncate_at = len(json_text) - step_back
        if truncate_at < 100:
            break
        candidate = re.sub(r',\s*$', '', json_text[:truncate_at].rstrip())
        ob = ob2 = 0
        in_string = escape_next = False
        for ch in candidate:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{': ob += 1
                elif ch == '}': ob -= 1
                elif ch == '[': ob2 += 1
                elif ch == ']': ob2 -= 1
        if ob > 15 or ob2 > 15:
            continue
        repaired = candidate + ']' * ob2 + '}' * ob
        try:
            parsed = json.loads(repaired)
            if 'metadata' in parsed and 'script' in parsed:
                return repaired
        except:
            continue
    return None

def build_full_text(script: dict) -> str:
    parts = []
    if script.get('hook'): parts.append(script['hook'])
    if script.get('problem_agitation'): parts.append(script['problem_agitation'])
    if script.get('promise'): parts.append(script['promise'])
    for section in script.get('main_content', []):
        if section.get('content'): parts.append(section['content'])
    for tip in script.get('practical_tips', []):
        if tip.get('explanation'): parts.append(tip['explanation'])
    if script.get('conclusion'): parts.append(script['conclusion'])
    return '\n\n'.join(parts)

def generate_script(category, sub_category, episode, run_id):
    print(f"📝 Generating script: {category} - {sub_category} (Ep {episode})")
    print(f"🔄 Single API call: script + title + description + thumbnail")

    title = get_episode_title(category, sub_category, episode)
    prompt = create_script_prompt(category, sub_category, episode, title)

    models_to_try = [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
    ]

    response = None
    model_used = None

    for model_name in models_to_try:
        try:
            print(f"🔄 Trying model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                    top_p=0.9,
                    top_k=40
                )
            )
            model_used = model_name
            print(f"✅ Model used: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {e}")
            continue

    if not response or not hasattr(response, 'text'):
        raise Exception("All Gemini models failed")

    try:
        response_text = response.text.strip()
        print(f"📏 Response length: {len(response_text)} chars")

        json_str = extract_json_from_response(response_text)
        script_data = json.loads(json_str)

        if 'metadata' not in script_data:
            script_data['metadata'] = {
                'final_title': title,
                'category': category,
                'sub_category': sub_category,
                'episode': episode
            }

        if 'script' not in script_data:
            raise ValueError("JSON missing required field: script")

        # Ensure full_text is populated
        script_section = script_data['script']
        if not script_section.get('full_text'):
            print("⚠️ full_text missing, building from sections...")
            script_section['full_text'] = build_full_text(script_section)

        word_count = script_section.get('word_count', 0)
        if word_count < 1400:
            print(f"⚠️ Word count {word_count} below minimum 1400")

        script_data['generation_info'] = {
            'category': category,
            'sub_category': sub_category,
            'episode': episode,
            'run_id': run_id,
            'generated_at': datetime.now().isoformat(),
            'model_used': model_used,
            'word_count_validated': word_count >= 1400,
            'response_length_chars': len(response_text),
            'api_calls': 1,
            'includes': ['script', 'title', 'description', 'thumbnail_concept']
        }

        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'script.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Title: {script_data['metadata'].get('final_title', 'N/A')}")
        print(f"📝 Words: {word_count} | ⏱️ Duration: {script_section.get('estimated_duration', 'N/A')}")
        print(f"🖼️ Thumbnail: {'✅' if script_data['metadata'].get('thumbnail_concept') else '❌'}")
        print(f"📄 Description: {'✅' if script_data['metadata'].get('description') else '❌'}")
        print(f"💾 Saved: {output_file}")

        return script_data

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate script + title + description + thumbnail in one API call')
    parser.add_argument('--category', required=True)
    parser.add_argument('--sub-category', required=True)
    parser.add_argument('--episode', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    try:
        script_data = generate_script(args.category, args.sub_category, args.episode, args.run_id)
        print(f"::set-output name=script_data::{json.dumps(script_data)}")
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
