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
except ImportError:
    print("❌ google.genai module not found! pip install google-genai")
    sys.exit(1)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set")
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

    # NOTE: full_text is NOT in the prompt — we build it in code from sections
    # This avoids huge JSON strings with unescaped newlines breaking parsing
    return f"""You are an elite Hindi content strategist. Generate a complete YouTube video package in ONE JSON response.

INPUT:
- Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title hint: {title}
- Duration: 10-15 minutes (1400-1900 Hindi words)
- Audience: 18-35 years, Hindi-speaking Indians
- Language: PURE HINDI DEVANAGARI ONLY - zero English letters in narration

LANGUAGE RULE: Write ALL narration in देवनागरी. English words phonetically:
brain→ब्रेन, psychology→साइकोलॉजी, reality→रियलिटी, manipulation→मैनिपुलेशन

EMOTION INDICATORS (always on separate line BEFORE narration):
(धीरे से) (गंभीर स्वर में) (रहस्यमय स्वर में) (उत्साह से)
(हल्की मुस्कान के साथ) (फुसफुसाते हुए) (आश्चर्य से) (प्यार से)

CORRECT format:
(गंभीर स्वर में)
तुम्हें सच जानना होगा।

WRONG format:
(गंभीर स्वर में) तुम्हें सच जानना होगा।

SCENE MARKERS (separate line, never spoken):
[SCENE: nature_morning] [SCENE: office_tension] [SCENE: family_dining]
[SCENE: phone_scrolling] [SCENE: thinking_alone] [SCENE: city_traffic]
[SCENE: dark_alley] [SCENE: books_study] [SCENE: crowd_walking]

SCRIPT STRUCTURE:
1. hook: 100-130 words, shocking opening
2. problem_agitation: 200-280 words, pain point detail
3. promise: 150-200 words, what they will learn
4. main_content: array of 4-6 sections, 1000-1400 words total
5. practical_tips: array of 5-7 tips, 300-400 words total
6. conclusion: 200-250 words, CTA + next episode teaser

TITLE RULES: 50-70 chars, use numbers, curiosity gap, power words

THUMBNAIL: High contrast, shocked face, 3-5 word overlay, red/yellow/orange

Return ONLY a raw JSON object. No markdown. No explanation. No text before or after JSON.

{{
  "metadata": {{
    "final_title": "best title string here",
    "title_options": [
      {{"title": "option 1", "character_count": 55, "why_it_works": "reason"}},
      {{"title": "option 2", "character_count": 60, "why_it_works": "reason"}},
      {{"title": "option 3", "character_count": 58, "why_it_works": "reason"}}
    ],
    "title_analysis": "why this title performs well",
    "description": "full SEO YouTube description",
    "description_hook": "first 2 lines only",
    "seo_keywords": ["keyword1", "keyword2"],
    "hashtags": ["#tag1", "#tag2"],
    "tags": ["tag1", "tag2", "tag3"],
    "thumbnail_concept": {{
      "main_subject": "person description",
      "facial_expression": "shocked expression",
      "background": "dark dramatic",
      "text_overlay": "3-5 words",
      "color_scheme": ["red", "yellow", "black"],
      "lighting": "dramatic cinematic",
      "additional_elements": ["arrow", "circle"],
      "composition_notes": "rule of thirds"
    }},
    "stability_ai_prompt": "detailed image generation prompt",
    "thumbnail_alternatives": [
      {{"concept_name": "alt 1", "description": "brief description"}}
    ],
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode}
  }},
  "script": {{
    "hook": "hook narration text here",
    "problem_agitation": "problem narration text here",
    "promise": "promise narration text here",
    "main_content": [
      {{"section_title": "section name in Hindi", "content": "section narration text"}}
    ],
    "practical_tips": [
      {{"tip_number": 1, "tip_title": "tip name in Hindi", "explanation": "tip explanation"}}
    ],
    "conclusion": "conclusion narration text here",
    "word_count": 1600,
    "estimated_duration": "12:30"
  }}
}}"""


def build_full_text(script: dict) -> str:
    """Build full_text by joining all script sections — done in code, not by Gemini"""
    parts = []
    if script.get('hook'):
        parts.append(script['hook'])
    if script.get('problem_agitation'):
        parts.append(script['problem_agitation'])
    if script.get('promise'):
        parts.append(script['promise'])
    for section in script.get('main_content', []):
        if section.get('content'):
            parts.append(section['content'])
    for tip in script.get('practical_tips', []):
        if tip.get('explanation'):
            parts.append(tip['explanation'])
    if script.get('conclusion'):
        parts.append(script['conclusion'])
    return '\n\n'.join(parts)


def extract_json_robust(text: str) -> dict:
    """
    Multi-strategy JSON extraction that always returns a parsed dict.
    Prints debug info to help diagnose future failures.
    """
    print(f"🔍 Extracting JSON (response: {len(text)} chars)...")

    # Show first 300 chars to understand response format
    print(f"📄 Response start: {repr(text[:300])}")

    strategies_tried = []

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        print("✓ Strategy 1: Direct parse succeeded")
        return result
    except json.JSONDecodeError as e:
        strategies_tried.append(f"direct_parse: {e}")

    # Strategy 2: Strip common wrappers
    stripped = text.strip()
    for prefix in ['```json', '```JSON', '```']:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped.endswith('```'):
        stripped = stripped[:-3]
    stripped = stripped.strip()
    try:
        result = json.loads(stripped)
        print("✓ Strategy 2: Strip markdown succeeded")
        return result
    except json.JSONDecodeError as e:
        strategies_tried.append(f"strip_markdown: {e}")

    # Strategy 3: Find { ... } balanced block
    start = text.find('{')
    if start != -1:
        # Walk forward counting braces
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

        if end != -1:
            candidate = text[start:end+1]
            try:
                result = json.loads(candidate)
                print("✓ Strategy 3: Balanced brace extraction succeeded")
                return result
            except json.JSONDecodeError as e:
                strategies_tried.append(f"balanced_brace: {e}")
                # Try repair on this candidate
                repaired = repair_json_string(candidate)
                if repaired:
                    try:
                        result = json.loads(repaired)
                        print("✓ Strategy 3b: Brace extraction + repair succeeded")
                        return result
                    except json.JSONDecodeError as e2:
                        strategies_tried.append(f"brace_repair: {e2}")

    # Strategy 4: Progressive truncation — find last valid JSON
    if start != -1:
        print("🔧 Strategy 4: Progressive truncation...")
        partial = text[start:]
        for cutback in [0, 10, 50, 100, 200, 500, 1000]:
            candidate = partial[:len(partial)-cutback] if cutback > 0 else partial
            candidate = re.sub(r',\s*$', '', candidate.rstrip())
            # Count open structures and close them
            depth_b = 0
            depth_sq = 0
            in_str2 = False
            esc2 = False
            for c in candidate:
                if esc2:
                    esc2 = False
                    continue
                if c == '\\' and in_str2:
                    esc2 = True
                    continue
                if c == '"':
                    in_str2 = not in_str2
                    continue
                if not in_str2:
                    if c == '{': depth_b += 1
                    elif c == '}': depth_b -= 1
                    elif c == '[': depth_sq += 1
                    elif c == ']': depth_sq -= 1
            closing = ']' * max(0, depth_sq) + '}' * max(0, depth_b)
            attempt = candidate + closing
            try:
                result = json.loads(attempt)
                if isinstance(result, dict) and ('metadata' in result or 'script' in result):
                    print(f"✓ Strategy 4: Truncation -{cutback} + close succeeded")
                    return result
            except json.JSONDecodeError:
                continue

    # All failed — print full response for debugging
    print("❌ All extraction strategies failed")
    print(f"Strategies tried: {strategies_tried}")
    print(f"Full response (first 3000 chars):\n{text[:3000]}")
    print(f"Full response (last 500 chars):\n{text[-500:]}")
    raise ValueError(f"No valid JSON found. Strategies: {strategies_tried}")


def repair_json_string(s: str) -> str:
    """Light repairs: trailing commas, comment removal"""
    try:
        # Remove trailing commas before } or ]
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        # Remove JS comments
        s = re.sub(r'//[^\n]*\n', '\n', s)
        json.loads(s)
        return s
    except Exception:
        return None


def generate_script(category, sub_category, episode, run_id):
    print(f"📝 Generating: {category} - {sub_category} (Ep {episode})")
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
            print(f"✅ Model responded: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {e}")
            continue

    if not response or not hasattr(response, 'text'):
        raise Exception("All Gemini models failed to respond")

    response_text = response.text.strip()
    print(f"📏 Response length: {len(response_text)} chars")

    # Extract JSON with robust multi-strategy parser
    script_data = extract_json_robust(response_text)

    # Validate structure
    if 'metadata' not in script_data:
        print("⚠️ metadata missing, adding defaults")
        script_data['metadata'] = {
            'final_title': title,
            'category': category,
            'sub_category': sub_category,
            'episode': episode
        }

    if 'script' not in script_data:
        raise ValueError("Response JSON missing 'script' field")

    # Build full_text in code (NOT from Gemini — avoids JSON escape issues)
    script_data['script']['full_text'] = build_full_text(script_data['script'])

    word_count = script_data['script'].get('word_count', 0)
    if word_count < 1400:
        print(f"⚠️ Word count {word_count} below 1400")

    script_data['generation_info'] = {
        'category': category,
        'sub_category': sub_category,
        'episode': episode,
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'model_used': model_used,
        'word_count_validated': word_count >= 1400,
        'api_calls': 1,
        'includes': ['script', 'title', 'description', 'thumbnail_concept']
    }

    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'script.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Title: {script_data['metadata'].get('final_title', 'N/A')}")
    print(f"📝 Words: {word_count} | ⏱️ {script_data['script'].get('estimated_duration', 'N/A')}")
    print(f"🖼️ Thumbnail: {'✅' if script_data['metadata'].get('thumbnail_concept') else '❌'}")
    print(f"📄 Description: {'✅' if script_data['metadata'].get('description') else '❌'}")
    print(f"💾 Saved: {output_file}")

    return script_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', required=True)
    parser.add_argument('--sub-category', required=True)
    parser.add_argument('--episode', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()

    try:
        script_data = generate_script(
            args.category,
            args.sub_category,
            args.episode,
            args.run_id
        )
        print(f"::set-output name=script_data::{json.dumps(script_data)}")
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
