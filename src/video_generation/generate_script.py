#!/usr/bin/env python3
"""
YT-AutoPilot Pro - Viral Shorts Script Generation with Gemini 2.5 API
Generates complete YouTube Shorts scripts in Pure Hindi (24-58 seconds)

FEATURES:
- Viral content optimization for maximum engagement
- Category-based voice tone selection
- Multi-channel support
- Enhanced JSON extraction with multi-pass validation
- Automatic JSON repair for common LLM errors
- Optimized for Coqui XTTS Hindi voice generation
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import re
import random

# Use the new google.genai package
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("❌ google.genai module not found!")
    print("\n📦 Installing required package...")
    print("   pip install google-genai")
    sys.exit(1)

# Configure Gemini Client
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not set in environment variables")
    sys.exit(1)

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)

# Load category configurations from JSON
def load_categories_config():
    """Load categories configuration from JSON file"""
    config_path = Path('config/categories.json')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Global categories config
CATEGORIES_CONFIG_DATA = load_categories_config()

def get_voice_tone_for_category(category, sub_category):
    """Get voice tone for a specific category/sub-category"""
    if not CATEGORIES_CONFIG_DATA:
        return "confident_clear"
    
    for cat in CATEGORIES_CONFIG_DATA.get('categories', []):
        if cat['name'] == category:
            # Check sub-category first
            for sub in cat.get('sub_categories', []):
                if sub['name'] == sub_category:
                    return sub.get('voice_tone', cat.get('voice_tone', 'confident_clear'))
            # Fallback to category voice tone
            return cat.get('voice_tone', 'confident_clear')
    
    return 'confident_clear'

def get_emotion_indicators(voice_tone):
    """Get emotion indicators for a voice tone"""
    if not CATEGORIES_CONFIG_DATA:
        return ["(आत्मविश्वास से)", "(स्पष्ट स्वर में)"]
    
    voice_tones = CATEGORIES_CONFIG_DATA.get('voice_tones', {})
    tone_data = voice_tones.get(voice_tone, {})
    return tone_data.get('emotion_indicators', ["(आत्मविश्वास से)"])

def get_category_hindi_name(category):
    """Get Hindi name for category"""
    if not CATEGORIES_CONFIG_DATA:
        return category
    
    for cat in CATEGORIES_CONFIG_DATA.get('categories', []):
        if cat['name'] == category:
            return cat.get('hindi_name', category)
    return category

def get_subcategory_hindi_name(category, sub_category):
    """Get Hindi name for sub-category"""
    if not CATEGORIES_CONFIG_DATA:
        return sub_category
    
    for cat in CATEGORIES_CONFIG_DATA.get('categories', []):
        if cat['name'] == category:
            for sub in cat.get('sub_categories', []):
                if sub['name'] == sub_category:
                    return sub.get('hindi_name', sub_category)
    return sub_category

# Episode ideas database - VIRAL TOPICS for Shorts
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
    ],
    ("Hidden Historical Truths", "Untold School History"): [
        "Veer Savarkar: Bhoole Hue Veer",
        "Netaji Subhash: Gumnaam Shaheed",
        "Bhagat Singh Ka Asli Vichar",
        "Sardar Patel: Iron Man Ka Sach",
        "Rani Laxmi Bai: Jhansi Ki Rani",
        "Tatya Tope: Kranti Ka Nayak",
        "Chandra Shekhar Azad: Amar Balidan",
        "Mangal Pandey: Kranti Ki Shuruaat",
        "Udham Singh: Jallianwala Ka Badla",
        "Khudiram Bose: 18 Saal Ka Shaheed"
    ],
    ("Politics Decoded", "Vote Bank Psychology"): [
        "Vote Bank Kaise Banaya Jaata Hai?",
        "Jumlebaazi Ka Science",
        "Neta Kaise Chunaav Jeette Hain?",
        "Rally Mein Bheed Kaise Aati Hai?",
        "Manifesto Ke Jhooth",
        "Religion Card Ka Istemal",
        "Caste Politics Ka Sach",
        "Freebies Ka Psychology",
        "Media Manipulation",
        "Social Engineering in Elections"
    ],
    ("Business Fundamentals", "Businessman Mindset"): [
        "Amir Log Kaise Sochte Hain?",
        "Gareebi Ka Mindset",
        "Paise Se Paisa Kaise Banaye?",
        "Successful Log Ki Aadatein",
        "Failure Se Seekhna",
        "Risk Lene Ka Sahi Tarika",
        "Opportunity Pehchanne Ka Tarika",
        "Network = Net Worth",
        "Time Management Secrets",
        "Decision Making Power"
    ],
    ("Education System Exposed", "Why Old Education Fails"): [
        "School Ne Kya Nahi Sikhai?",
        "Degree Ka Sach",
        "Job vs Business Mindset",
        "Skills Over Marks",
        "Self Learning Ka Power",
        "Internet Se Seekhna",
        "Mentor Ki Jaroorat",
        "Practical Knowledge",
        "Exam Pressure Ka Sach",
        "Real Education Kya Hai?"
    ],
    ("Society Reality", "Cycle of Poverty"): [
        "Gareebi Ka Chakravyuh",
        "Middle Class Trap",
        "Amir vs Gareeb Mindset",
        "Society Ke Rules",
        "System Ko Beat Karna",
        "Financial Literacy",
        "Paise Ka Psychology",
        "Status Symbol Ka Jaal",
        "Rat Race Se Bahar Nikalna",
        "Wealth Building Basics"
    ],
    ("Communication Mastery", "Presentation Psychology"): [
        "Public Speaking Ka Dar",
        "Body Language Secrets",
        "Voice Modulation Tips",
        "Confidence Kaise Dikhe?",
        "Stage Fear Kaise Hataye?",
        "Effective Communication",
        "Storytelling Ka Power",
        "Audience Ko Kaise Pakde?",
        "Impromptu Speaking",
        "Presentation Skills"
    ],
    ("Human Life Reality", "Lies About Success"): [
        "Success Ka Jhootha Definition",
        "Overnight Success Myth",
        "Hard Work vs Smart Work",
        "Luck vs Preparation",
        "Success Ka Real Formula",
        "Failure Ka Matlab",
        "Consistency Ka Power",
        "Small Steps Big Results",
        "Patience Ki Shakti",
        "Real Success Kya Hai?"
    ],
    ("Mythology", "Mahabharat Secrets"): [
        "Karna: Adharmi Ya Traasdi?",
        "Draupadi: Sati Ya Shakti?",
        "Krishna Ka Rajneeti Gyan",
        "Kurukshetra Ka Sach",
        "Bhishma Pitamah Ki Kasam",
        "Eklavya: Nyay Ya Anyay?",
        "Karn Aur Arjun",
        "Duryodhan Ka Paksh",
        "Geeta Gyan Ka Asar",
        "Mahabharat Ke Lessons"
    ],
    ("Health Wellness", "Ayurveda"): [
        "Roj Uthne Ka Sahi Samay",
        "Khane Ka Sahi Tarika",
        "Neend Ka Mahatva",
        "Pani Peene Ka Sahi Tarika",
        "Subah Ki Routine",
        "Immunity Badhane Ke Tarike",
        "Stress Se Mukti",
        "Yoga Ka Power",
        "Desi Nuskhe",
        "Swasth Rehne Ke Rules"
    ],
    ("Finance", "Saving Psychology"): [
        "Paisa Bachane Ka Sahi Tarika",
        "50-30-20 Rule",
        "Emergency Fund Ka Mahatva",
        "Impulse Buying Se Bachna",
        "Budget Banana Seekho",
        "Needs vs Wants",
        "Financial Discipline",
        "Small Savings Big Impact",
        "Paise Ki Value",
        "Rich Dad Poor Dad Lessons"
    ],
    ("Technology", "AI Impact"): [
        "AI Tumhari Naukri Lega?",
        "ChatGPT Se Paise Kaise Kamaye?",
        "AI Tools For Everyone",
        "Future Of Work",
        "Learn AI Or Lose",
        "AI Revolution Ka Sach",
        "Machine Learning Basics",
        "AI In Daily Life",
        "Automation Ka Asar",
        "Tech Skills For Future"
    ]
}

def get_episode_title(category, sub_category, episode):
    """Get episode title from database or generate"""
    key = (category, sub_category)
    if key in EPISODE_IDEAS:
        ideas = EPISODE_IDEAS[key]
        if episode <= len(ideas):
            return ideas[episode - 1]
    return f"{sub_category} - Episode {episode}"

def create_viral_shorts_script_prompt(category, sub_category, episode, title, voice_tone):
    """
    Create prompt for VIRAL YouTube Shorts script (24-58 seconds)
    Optimized for maximum engagement and shareability
    """
    
    hindi_category = get_category_hindi_name(category)
    hindi_sub = get_subcategory_hindi_name(category, sub_category)
    emotion_indicators = get_emotion_indicators(voice_tone)
    
    # Select 3-4 emotion indicators for variety
    selected_emotions = random.sample(emotion_indicators, min(3, len(emotion_indicators)))
    
    prompt = f"""You are an elite Hindi content strategist specializing in VIRAL YOUTUBE SHORTS.

TASK: Create a VIRAL YouTube Shorts script that will get maximum views, shares, and engagement.

INPUT PARAMETERS:
- Main Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title: {title}
- **Target Duration: 24-58 SECONDS SPEAKING TIME (70-160 Hindi words)**
- Target Audience: 18-35 years, Hindi-speaking, Indian urban/semi-urban
- Voice Tone: {voice_tone}
- Language: **PURE HINDI (देवनागरी लिपि में शुद्ध हिंदी)**

**VIRAL CONTENT RULES (CRITICAL):**

1. **HOOK (0-3 seconds / 8-12 words):**
   - PATTERN INTERRUPT - Start with something unexpected
   - Use "Tum", "Aap", "Kya" to create personal connection
   - Create curiosity gap - viewer MUST know what comes next
   - Examples: "Tum ye galti roz karte ho...", "Kya tum jaante ho ki...", "Ye sach hai ya jhooth?"

2. **CONTENT (3-45 seconds / 60-120 words):**
   - ONE powerful insight or fact
   - Make it PERSONAL - use "Tum" extensively
   - Create "Aha!" moment - something they didn't know
   - Add credibility: "Research ke mutabik...", "Scientists ne kaha..."
   - Build emotional connection
   - Hint at deeper knowledge

3. **CTA (45-58 seconds / 15-30 words):**
   - Strong call-to-action for engagement
   - Create FOMO (Fear Of Missing Out)
   - Ask question to drive comments
   - "Comment mein batao...", "Share karo apne doston ke saath..."

**VOICE TONE EMOTION INDICATORS (USE THESE EXACTLY):**
{chr(10).join(selected_emotions)}

**EMOTION PLACEMENT RULE (CRITICAL):**
Emotion indicators must ALWAYS be placed on a separate line BEFORE narration.

CORRECT format:
(आत्मविश्वास से)
तुम्हें सच जानना होगा।

WRONG format:
(आत्मविश्वास से) तुम्हें सच जानना होगा।

**VIRAL ELEMENTS TO INCLUDE:**
- Controversy or surprising fact
- Relatable situation
- Emotional trigger
- Practical value
- Share-worthy insight

**SCENE MARKERS FOR VIDEO EDITING (USE EXACTLY):**
[SCENE: hook_intense] [SCENE: explain_main] [SCENE: reaction_closeup] [SCENE: cta_energy]

**ABSOLUTE LANGUAGE RULE:**
Narration must contain ZERO English letters.
Only Hindi Devanagari script is allowed.
English words must be written using Hindi phonetics:
- Psychology → साइकोलॉजी
- Brain → ब्रेन  
- Reality → रियलिटी

**CRITICAL OUTPUT INSTRUCTION:**
You MUST return ONLY a valid JSON object. Do NOT include any explanation, preamble, or text before or after the JSON.
Do NOT wrap it in markdown code blocks (```json). Return the raw JSON object directly.

The JSON MUST have this EXACT structure:
{{
  "metadata": {{
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode},
    "title": "{title}",
    "voice_tone": "{voice_tone}
  }},
  "script": {{
    "hook": "Hook text with emotion indicator",
    "content": "Main content text with emotion indicators",
    "cta": "Call to action text with emotion indicator",
    "full_text": "Complete script combining all parts",
    "word_count": 120,
    "estimated_duration": "45 seconds"
  }},
  "viral_elements": {{
    "hook_type": "pattern_interrupt",
    "emotion_trigger": "curiosity",
    "share_factor": "surprising_fact"
  }}
}}

**ENSURE THE JSON IS COMPLETE AND VALID.**
**REMEMBER: Pure Hindi (देवनागरी लिपि), NOT Hinglish**
**REMEMBER: Emotional indicators must be on separate lines BEFORE sentences**
**REMEMBER: This is a STANDALONE VIRAL SHORTS script**"""
    
    return prompt


def extract_json_from_response(text: str) -> str:
    """Enhanced JSON extraction with multiple strategies"""
    print("🔍 Extracting JSON from response...")
    
    # Strategy 1: Check if entire text is valid JSON
    try:
        json.loads(text)
        print("✓ Response is pure valid JSON")
        return text
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'```json(.*?)```',
        r'```(.*?)```',
        r'```json\s*\n(.*)',
        r'```\s*\n(.*)',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                print("✓ Extracted valid JSON from code block")
                return match
            except json.JSONDecodeError:
                repaired = repair_json(match)
                if repaired:
                    print("✓ Extracted and repaired JSON from code block")
                    return repaired
    
    # Strategy 3: Find balanced JSON object
    json_candidate = find_balanced_json(text)
    if json_candidate:
        try:
            json.loads(json_candidate)
            print("✓ Extracted valid balanced JSON")
            return json_candidate
        except json.JSONDecodeError:
            repaired = repair_json(json_candidate)
            if repaired:
                print("✓ Extracted and repaired balanced JSON")
                return repaired
    
    # Emergency salvage
    salvaged = salvage_truncated_json(text)
    if salvaged:
        print("✓ Salvaged truncated JSON")
        return salvaged
    
    raise ValueError("No valid JSON found in response")


def find_balanced_json(text: str) -> str:
    """Find the first balanced JSON object in text"""
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
        
        if ch == '"' and not escape_next:
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
    """Attempt to repair common JSON errors"""
    repairs_made = []
    
    # Remove trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    if fixed != json_str:
        repairs_made.append("trailing_commas")
        json_str = fixed
    
    # Remove comments
    fixed = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    if fixed != json_str:
        repairs_made.append("comments")
        json_str = fixed
    
    # Fix unescaped newlines
    fixed = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', json_str)
    if fixed != json_str:
        repairs_made.append("unescaped_newlines")
        json_str = fixed
    
    if repairs_made:
        try:
            json.loads(json_str)
            print(f"✓ JSON repaired: {', '.join(repairs_made)}")
            return json_str
        except json.JSONDecodeError:
            pass
    
    return None


def salvage_truncated_json(text: str) -> str:
    """Emergency salvage for truncated JSON"""
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    json_text = text[start_idx:]
    
    # Intelligent string-aware closing
    in_string = False
    escape_next = False
    brace_count = 0
    bracket_count = 0
    last_valid_pos = 0
    
    for i, ch in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        
        if ch == '\\':
            escape_next = True
            continue
        
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
            elif ch == '[':
                bracket_count += 1
            elif ch == ']':
                bracket_count -= 1
            
            if brace_count >= 0 and bracket_count >= 0:
                last_valid_pos = i
    
    if in_string:
        for i in range(len(json_text) - 1, -1, -1):
            if json_text[i] == '"' and (i == 0 or json_text[i-1] != '\\'):
                last_valid_pos = i - 1
                break
    
    candidate = json_text[:last_valid_pos + 1]
    
    # Recount and close
    in_string = False
    escape_next = False
    open_braces = 0
    open_brackets = 0
    
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
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces -= 1
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                open_brackets -= 1
    
    closing = ']' * open_brackets + '}' * open_braces
    repaired = candidate + closing
    
    try:
        parsed = json.loads(repaired)
        if 'metadata' in parsed and 'script' in parsed:
            return repaired
    except json.JSONDecodeError:
        pass
    
    return None


def validate_script_integrity(script_data: dict) -> bool:
    """Validate script data integrity"""
    print("🔒 Validating script integrity...")
    
    if "script" not in script_data:
        raise RuntimeError("Missing 'script' field")
    
    script = script_data["script"]
    
    # Check required fields
    required_fields = ['hook', 'content', 'cta', 'full_text', 'word_count']
    for field in required_fields:
        if field not in script:
            raise RuntimeError(f"Missing required field: {field}")
    
    # Validate word count
    word_count = script.get('word_count', 0)
    if word_count < 70 or word_count > 160:
        print(f"⚠️ Word count {word_count} outside optimal range (70-160)")
    
    print("✅ Script integrity validated")
    return True


def generate_script(category, sub_category, episode, run_id, channel_id=None):
    """
    Generate viral shorts script using Gemini 2.5 API
    
    Args:
        category: Main category
        sub_category: Sub category
        episode: Episode number
        run_id: Run ID
        channel_id: YouTube Channel ID (for multi-channel support)
    
    Returns:
        Script data dictionary
    """
    
    print(f"📝 Generating VIRAL SHORTS script for: {category} - {sub_category} (Ep {episode})")
    
    # Get voice tone for category
    voice_tone = get_voice_tone_for_category(category, sub_category)
    print(f"🎙️ Voice tone: {voice_tone}")
    
    # Get episode title
    title = get_episode_title(category, sub_category, episode)
    
    # Create prompt
    prompt = create_viral_shorts_script_prompt(category, sub_category, episode, title, voice_tone)
    
    # Models to try
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
                    temperature=0.8,
                    max_output_tokens=8192,
                    top_p=0.9,
                    top_k=40,
                    response_mime_type="application/json"
                )
            )
            
            model_used = model_name
            print(f"✅ Successfully used model: {model_name}")
            break
            
        except Exception as e:
            print(f"⚠️ Model {model_name} failed: {e}")
            continue
    
    if not response:
        print("❌ All models failed")
        raise Exception("All Gemini models failed")
    
    # Parse JSON response
    try:
        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                response_text = candidate.content.parts[0].text.strip()
            else:
                raise RuntimeError("Gemini returned candidates but no text content")
        else:
            raise RuntimeError("Gemini returned empty response")
        
        print(f"📏 Response length: {len(response_text)} chars")
        
        # Extract JSON
        json_str = extract_json_from_response(response_text)
        
        # Parse JSON
        script_data = json.loads(json_str)
        
        # Validate structure
        if 'script' not in script_data:
            raise ValueError("JSON missing required field: script")
        
        # Ensure full_text is present
        if 'full_text' not in script_data['script']:
            script_data['script']['full_text'] = (
                script_data['script'].get('hook', '') + ' ' +
                script_data['script'].get('content', '') + ' ' +
                script_data['script'].get('cta', '')
            )
        
        # Add metadata if missing
        if 'metadata' not in script_data:
            script_data['metadata'] = {
                'category': category,
                'sub_category': sub_category,
                'episode': episode,
                'title': title,
                'voice_tone': voice_tone,
                'channel_id': channel_id
            }
        
        # Validate integrity
        validate_script_integrity(script_data)
        
        # Add generation metadata
        script_data['generation_info'] = {
            'category': category,
            'sub_category': sub_category,
            'episode': episode,
            'run_id': run_id,
            'channel_id': channel_id,
            'generated_at': datetime.now().isoformat(),
            'model_used': model_used,
            'voice_tone': voice_tone,
            'video_type': 'shorts'
        }
        
        # Save to file
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'script_short.json'
        
        # Normalize to chunks format for audio generation
        if "chunks" not in script_data:
            script_text = script_data['script']['full_text'].strip()
            
            script_data = {
                "chunks": [
                    {
                        "chunk_id": 1,
                        "text": script_text
                    }
                ],
                "full_script": script_text,
                "script": script_data['script'],
                "metadata": script_data['metadata'],
                "viral_elements": script_data.get('viral_elements', {}),
                "generation_info": script_data['generation_info']
            }
        
        # Write the final script data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ VIRAL SHORTS script generated")
        print(f"📝 Word count: {script_data['script'].get('word_count', 'N/A')}")
        print(f"⏱️ Estimated duration: {script_data['script'].get('estimated_duration', 'N/A')}")
        print(f"🎙️ Voice tone: {voice_tone}")
        print(f"💾 Saved to: {output_file}")
        
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
    parser = argparse.ArgumentParser(description='Generate viral YouTube Shorts script')
    parser.add_argument('--category', required=True, help='Main category')
    parser.add_argument('--sub-category', required=True, help='Sub category')
    parser.add_argument('--episode', required=True, type=int, help='Episode number')
    parser.add_argument('--run-id', required=True, help='Run ID')
    parser.add_argument('--channel-id', default=None, help='YouTube Channel ID')
    
    args = parser.parse_args()
    
    try:
        script_data = generate_script(
            args.category,
            args.sub_category,
            args.episode,
            args.run_id,
            args.channel_id
        )
        
        # Output for GitHub Actions
        print(f"::set-output name=script_data::{json.dumps(script_data)}")
        
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
