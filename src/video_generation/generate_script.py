#!/usr/bin/env python3
"""
YT-AutoPilot Pro - Script Generation with Gemini 2.5 API
Single API Call: script + title + description + thumbnail concept

FIXES:
- Single quotes in Hindi text breaking JSON → replaced with double-quote-safe approach + post-processing
- Truncated responses → retry with fresh call if response < 5000 chars
- max_output_tokens increased to 65536 for complete responses
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
    print("❌ google.genai not found! pip install google-genai")
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
        "Love Bombing: Pyar Ya Phasaav",
        "Toxic Log Kaise Tumhari Energy Churate Hain",
        "Narcissist Ki Pehchan Kaise Karein",
        "Guilt Tripping Se Kaise Bachein",
        "Passive-Aggressive Behavior Samajhna",
        "Trauma Bonding Kya Hai",
        "Toxic Workplace Se Kaise Nikle"
    ],
    ("Human Psychology & Behavior", "Life Hacks Psychology"): [
        "Baat Maanwane Ka Psychology",
        "Interview Mein Select Hone Ke Tarike",
        "First Impression Kaise Banaye",
        "Logon Ko Apni Taraf Kaise Karein",
        "Negotiation Ke Psychology Tricks",
        "Memory Improve Karne Ke Tarike",
        "Decision Making Ke Shortcuts",
        "Social Anxiety Kaise Kam Karein",
        "Confidence Dikhane Ke Tarike",
        "Influence Kaise Badhayein"
    ]
}

def get_episode_title(category, sub_category, episode):
    key = (category, sub_category)
    if key in EPISODE_IDEAS:
        ideas = EPISODE_IDEAS[key]
        if episode <= len(ideas):
            return ideas[episode - 1]
    return f"{sub_category} Episode {episode}"


def create_script_prompt(category, sub_category, episode, title):
    hindi_category = CATEGORIES_CONFIG.get(category, {}).get("hindi_name", category)
    hindi_sub = CATEGORIES_CONFIG.get(category, {}).get("sub_categories", {}).get(sub_category, sub_category)

    return f"""You are an elite Hindi content strategist. Generate a complete YouTube video package.

CRITICAL JSON RULE: All string values must use ONLY double quotes. Never use single quotes (apostrophes) inside JSON string values. Replace any apostrophe with a space or remove it entirely. This is the most important rule.

WRONG: "why_it_works": "यह शीर्षक 'एपिसोड' का उल्लेख करता है"
CORRECT: "why_it_works": "यह शीर्षक एपिसोड का उल्लेख करता है"

INPUT:
- Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title hint: {title}
- Duration: 10-15 minutes (1400-1900 Hindi words total across all script sections)
- Audience: 18-35 years, Hindi-speaking Indians
- Language: PURE HINDI DEVANAGARI ONLY in all narration fields

EMOTION INDICATORS (on separate line BEFORE narration, never inline):
(धीरे से) (गंभीर स्वर में) (रहस्यमय स्वर में) (उत्साह से) (प्यार से)

SCENE MARKERS (on separate line, never spoken):
[SCENE: nature_morning] [SCENE: office_tension] [SCENE: city_traffic]
[SCENE: dark_alley] [SCENE: books_study] [SCENE: crowd_walking]

SCRIPT SECTIONS (all narration in pure Hindi Devanagari):
1. hook: 100-130 words
2. problem_agitation: 200-280 words
3. promise: 150-200 words
4. main_content: 4-6 sections, 1000-1400 words total
5. practical_tips: 5-7 tips, 300-400 words total
6. conclusion: 200-250 words with CTA

Return ONLY raw JSON. No markdown. No text outside JSON. No single quotes inside string values.

{{
  "metadata": {{
    "final_title": "title string without apostrophes",
    "title_options": [
      {{"title": "option 1", "character_count": 55, "why_it_works": "reason without apostrophes"}},
      {{"title": "option 2", "character_count": 60, "why_it_works": "reason without apostrophes"}},
      {{"title": "option 3", "character_count": 58, "why_it_works": "reason without apostrophes"}}
    ],
    "title_analysis": "analysis without apostrophes",
    "description": "full SEO description",
    "description_hook": "first 2 lines",
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
      "additional_elements": ["arrow"],
      "composition_notes": "rule of thirds"
    }},
    "stability_ai_prompt": "image generation prompt",
    "thumbnail_alternatives": [
      {{"concept_name": "alt 1", "description": "brief description"}}
    ],
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode}
  }},
  "script": {{
    "hook": "hook narration in pure Hindi",
    "problem_agitation": "problem narration in pure Hindi",
    "promise": "promise narration in pure Hindi",
    "main_content": [
      {{"section_title": "section name", "content": "section narration in pure Hindi"}}
    ],
    "practical_tips": [
      {{"tip_number": 1, "tip_title": "tip name", "explanation": "tip explanation in pure Hindi"}}
    ],
    "conclusion": "conclusion narration in pure Hindi",
    "word_count": 1600,
    "estimated_duration": "12:30"
  }}
}}"""


def fix_single_quotes_in_json(text: str) -> str:
    """
    Post-process: replace single quotes inside JSON string values with
    the Hindi danda or just remove them — keeps JSON valid.
    Only targets apostrophes INSIDE string values, not JSON structure.
    """
    # Replace ' that appears between Hindi/word characters (apostrophe in text)
    # Pattern: word char ' word char  → replace ' with space
    result = re.sub(r'(?<=[^\s{}\[\]:,"])\'(?=[^\s{}\[\]:,"])', '', text)
    return result


def build_full_text(script: dict) -> str:
    """Build full_text by joining all script sections"""
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
    """Multi-strategy JSON extraction with single-quote fix"""
    print(f"🔍 Extracting JSON (response: {len(text)} chars)...")
    print(f"📄 Response start: {repr(text[:200])}")

    # Check if response is suspiciously short (truncated)
    if len(text) < 3000:
        print(f"⚠️ Response seems truncated ({len(text)} chars) — will attempt extraction anyway")

    def try_parse(s):
        """Try parse → if fails due to single quote, fix and retry"""
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            err_str = str(e)
            # Single quote / apostrophe issue
            if "delimiter" in err_str or "Expecting" in err_str:
                fixed = fix_single_quotes_in_json(s)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            # Trailing comma issue
            cleaned = re.sub(r',(\s*[}\]])', r'\1', s)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
            return None

    # Strategy 1: Direct parse
    result = try_parse(text)
    if result:
        print("✓ Strategy 1: Direct parse succeeded")
        return result

    # Strategy 2: Strip markdown fences
    stripped = text.strip()
    for prefix in ['```json', '```JSON', '```']:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if stripped.endswith('```'):
        stripped = stripped[:-3]
    stripped = stripped.strip()
    result = try_parse(stripped)
    if result:
        print("✓ Strategy 2: Strip markdown succeeded")
        return result

    # Strategy 3: Find balanced { ... } block
    start = text.find('{')
    if start != -1:
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
            result = try_parse(candidate)
            if result:
                print("✓ Strategy 3: Balanced brace extraction succeeded")
                return result

        # Strategy 4: Partial / truncated — close open braces
        print("🔧 Strategy 4: Closing truncated JSON...")
        partial = text[start:]
        # Remove trailing partial string or comma
        partial = re.sub(r',\s*$', '', partial.rstrip())
        if partial.endswith('"'):
            partial = partial[:-1]
        partial = re.sub(r',\s*$', '', partial.rstrip())

        # Count open structures
        depth_b = depth_sq = 0
        in_str2 = esc2 = False
        for c in partial:
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
        attempt = partial + closing
        result = try_parse(attempt)
        if result and isinstance(result, dict):
            print(f"✓ Strategy 4: Closed truncated JSON succeeded")
            return result

        # Strategy 5: Progressive backtrack
        for cutback in [50, 100, 200, 500]:
            if len(partial) - cutback < 100:
                break
            cut = re.sub(r',\s*$', '', partial[:len(partial)-cutback].rstrip())
            depth_b2 = depth_sq2 = 0
            in_s = es = False
            for c in cut:
                if es:
                    es = False
                    continue
                if c == '\\' and in_s:
                    es = True
                    continue
                if c == '"':
                    in_s = not in_s
                    continue
                if not in_s:
                    if c == '{': depth_b2 += 1
                    elif c == '}': depth_b2 -= 1
                    elif c == '[': depth_sq2 += 1
                    elif c == ']': depth_sq2 -= 1
            cl = ']' * max(0, depth_sq2) + '}' * max(0, depth_b2)
            att = cut + cl
            result = try_parse(att)
            if result and isinstance(result, dict) and ('metadata' in result or 'script' in result):
                print(f"✓ Strategy 5: Backtrack -{cutback} succeeded")
                return result

    print("❌ All strategies failed")
    print(f"Full response:\n{text[:5000]}")
    raise ValueError(f"Could not extract valid JSON from response ({len(text)} chars)")


def call_gemini_with_retry(prompt: str, max_attempts: int = 3) -> tuple:
    """
    Call Gemini with retry logic.
    Returns (response_text, model_used)
    Retries if response is truncated (< 3000 chars)
    """
    models_to_try = [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
    ]

    last_error = None

    for model_name in models_to_try:
        for attempt in range(max_attempts):
            try:
                print(f"🔄 Model: {model_name} (attempt {attempt+1}/{max_attempts})...")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=65536,  # Maximum — ensures complete response
                        top_p=0.9,
                        top_k=40
                    )
                )

                if not hasattr(response, 'text') or not response.text:
                    print(f"⚠️ Empty response from {model_name}")
                    continue

                text = response.text.strip()
                print(f"📏 Response: {len(text)} chars")

                # If suspiciously short, retry
                if len(text) < 2000 and attempt < max_attempts - 1:
                    print(f"⚠️ Response too short ({len(text)} chars), retrying...")
                    continue

                print(f"✅ Got response from {model_name}")
                return text, model_name

            except Exception as e:
                err_msg = str(e)
                print(f"⚠️ {model_name} attempt {attempt+1} failed: {err_msg[:200]}")
                last_error = e
                if '429' in err_msg or 'RESOURCE_EXHAUSTED' in err_msg:
                    break  # Skip to next model immediately on quota error
                continue

    raise Exception(f"All models failed. Last error: {last_error}")


def generate_script(category, sub_category, episode, run_id):
    print(f"📝 Generating: {category} - {sub_category} (Ep {episode})")
    print(f"🔄 Single API call: script + title + description + thumbnail")

    title = get_episode_title(category, sub_category, episode)
    prompt = create_script_prompt(category, sub_category, episode, title)

    response_text, model_used = call_gemini_with_retry(prompt)

    script_data = extract_json_robust(response_text)

    # Validate and fill missing metadata
    if 'metadata' not in script_data:
        print("⚠️ metadata missing, adding defaults")
        script_data['metadata'] = {}

    meta = script_data['metadata']
    if not meta.get('final_title'):
        meta['final_title'] = title
    if not meta.get('category'):
        meta['category'] = category
    if not meta.get('sub_category'):
        meta['sub_category'] = sub_category
    if not meta.get('episode'):
        meta['episode'] = episode

    if 'script' not in script_data:
        raise ValueError("Response JSON missing 'script' field")

    # Build full_text in code (avoids JSON escape issues with Hindi text)
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

    print(f"✅ Title: {meta.get('final_title', 'N/A')}")
    print(f"📝 Words: {word_count} | ⏱️ {script_data['script'].get('estimated_duration', 'N/A')}")
    print(f"🖼️ Thumbnail: {'✅' if meta.get('thumbnail_concept') else '❌'}")
    print(f"📄 Description: {'✅' if meta.get('description') else '❌'}")
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
