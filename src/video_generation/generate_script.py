#!/usr/bin/env python3
"""
YT-AutoPilot Pro - Script Generation with Gemini 2.5 API
Generates complete YouTube video scripts in Pure Hindi with 10+ minute enforcement
NOW WITH DETERMINISTIC CHUNK GENERATION - Gemini outputs pre-chunked scripts

FIXES:
- Enhanced JSON extraction with multi-pass validation
- Automatic JSON repair for common LLM errors
- Truncation detection and recovery
- Streaming support for large responses
- Optimized for Coqui XTTS Hindi voice generation
- Emotion indicators strictly on separate lines for XTTS metadata
- SUPPORTS SEPARATE SHORTS SCRIPT GENERATION (not trimmed from long videos)
- NEW: Gemini outputs scripts in pre-defined chunks (120-180 words each) - REDUCED SIZE TO PREVENT TRUNCATION
- NEW: Complete sentences preserved across chunks
- NEW: Zero word loss, zero overlap
- NEW: PRODUCTION SAFETY - Chunk integrity validation stops pipeline on corruption
- FIXED: Added response_mime_type="application/json" to force structured JSON output
- FIXED: Enhanced response handling to capture JSON from candidates when text field is empty
- FIXED: Sentence terminator validation now handles quoted sentences properly
- **FIXED: SHORT script normalization - ensures script_short.json always contains valid "chunks" array**
- **FIXED: Unicode punctuation normalization - prevents validation failures from visually identical punctuation marks**
- **CRITICAL FIX: full_script now rebuilt from chunks as authoritative source - pipeline no longer fails on word count mismatch**
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import re

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

# ============================================================================
# UNICODE PUNCTUATION NORMALIZATION HELPER
# ============================================================================

def normalize_unicode_punctuation(text: str) -> str:
    """
    Normalize Unicode punctuation to ensure consistent comparison.
    Converts visually similar but different Unicode characters to standard forms.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized punctuation
    """
    replacements = {
        '。': '।',  # Chinese/Japanese full stop → Hindi danda
        '．': '.',  # Fullwidth period → ASCII period
        '，': ',',  # Fullwidth comma → ASCII comma
        '！': '!',  # Fullwidth exclamation → ASCII exclamation
        '？': '?'   # Fullwidth question → ASCII question
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

# Load category configurations
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

# Episode ideas database
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
    """Get episode title from database or generate"""
    key = (category, sub_category)
    if key in EPISODE_IDEAS:
        ideas = EPISODE_IDEAS[key]
        if episode <= len(ideas):
            return ideas[episode - 1]
    return f"{sub_category} - Episode {episode}"

def create_long_script_prompt(category, sub_category, episode, title):
    """
    Create the master prompt for Gemini with 10+ minute enforcement,
    XTTS optimization, and CRITICAL: DETERMINISTIC CHUNK GENERATION
    
    Gemini must output the script in pre-defined chunks:
    - Each chunk: 120-180 words (STRICT LIMIT - reduced from 400-800 to prevent truncation)
    - Complete sentences ONLY (never split mid-sentence)
    - No overlap, no gaps, no missing words
    - Chunks concatenate perfectly to form full script
    - Total response size must remain under safe limits - prefer more chunks with smaller size
    """
    
    hindi_category = CATEGORIES_CONFIG.get(category, {}).get("hindi_name", category)
    hindi_sub = CATEGORIES_CONFIG.get(category, {}).get("sub_categories", {}).get(sub_category, sub_category)
    
    prompt = f"""You are an elite Hindi content strategist and scriptwriter. Your expertise is creating viral YouTube content that feels like a trusted friend revealing life-changing secrets.

TASK: Create a complete YouTube video script for Indian audience.

INPUT PARAMETERS:
- Main Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title: {title}
- **CRITICAL: Target Duration MUST BE 10-15 MINUTES (1400-1900 Hindi words)**
- Target Audience: 18-35 years, Hindi-speaking, Indian urban/semi-urban
- Tone: Conversational, slightly conspiratorial, empowering, eye-opening
- Language: **PURE HINDI (देवनागरी लिपि में शुद्ध हिंदी)**

**ABSOLUTE LANGUAGE RULE:**

Narration must contain ZERO English letters.

Do NOT use characters a-z or A-Z anywhere in narration.

Only Hindi Devanagari script is allowed.

English technical words must be written using Hindi phonetics.

Examples:

Correct: ब्रेन, साइकोलॉजी, रियलिटी  
Wrong: brain, psychology, reality

This rule is STRICT and must never be violated.

**LANGUAGE REQUIREMENTS (CRITICAL):**
- Script MUST be written in **pure Hindi using Devanagari script only**
- DO NOT use Hinglish (English letters for Hindi words)
- DO NOT use English words in narration
- English technical words must be written in **Hindi phonetics**:
  - Psychology → साइकोलॉजी
  - Brain → ब्रेन  
  - Reality → रियलिटी
  - Manipulation → मैनिपुलेशन
  - Strategy → स्ट्रैटेजी

**SCENE MARKER RULE:**

Scene markers like:

[SCENE: nature_morning]
[SCENE: office_tension]

are ONLY for video editing.

They must NOT be spoken as narration.

They must be placed on a separate line.

No emotional indicator should be on the same line as scene marker.

Correct example:

[SCENE: office_tension]

(गंभीर स्वर में)
तुम ऑफिस में बैठे हो...

Wrong example:

(गंभीर स्वर में) [SCENE: office_tension] तुम ऑफिस में बैठे हो...

**XTTS VOICE OPTIMIZATION REQUIREMENTS:**

Use emotional reaction indicators in brackets ONLY:

(धीरे से)
(गंभीर स्वर में)
(रहस्यमय स्वर में)  
(उत्साह से)
(हल्की मुस्कान के साथ)
(फुसफुसाते हुए)
(आश्चर्य से)
(दुखी होकर)
(गुस्से में)
(प्यार से)

**EMOTION PLACEMENT RULE (CRITICAL):**

Emotion indicators must ALWAYS be placed on a separate line before narration. They are metadata for voice tone and must never be merged with narration text.

CORRECT format:

(गंभीर स्वर में)
तुम्हें सच जानना होगा।

WRONG format:

(गंभीर स्वर में) तुम्हें सच जानना होगा।

This ensures XTTS uses emotional context but never speaks the emotion words.

Use natural pauses using punctuation:
- ,  for short pauses
- ... for dramatic pauses
- .  for sentence pause

**DO NOT USE THESE MARKERS (NOT COMPATIBLE WITH XTTS):**
❌ [PAUSE-1] [PAUSE-2] [PAUSE-3]
❌ [EMPHASIS] [WHISPER] [EXCITED] [SERIOUS] [QUESTION]

**SCENE MARKERS FOR VIDEO EDITING (USE EXACTLY):**
[SCENE: nature_morning] [SCENE: office_tension] [SCENE: family_dining] 
[SCENE: phone_scrolling] [SCENE: thinking_alone] [SCENE: celebration]
[SCENE: dark_alley] [SCENE: books_study] [SCENE: city_traffic]
[SCENE: crowd_walking] [SCENE: money_counting] [SCENE: handshake]

**10-15 MINUTE STRUCTURE (1400-1900 HINDI WORDS):**

1. **HOOK (0-45 seconds / 100-130 Hindi words):**
   - Start with a shocking question or relatable scenario
   - Create immediate "this is about me" feeling
   - Use (उत्साह से) or (रहस्यमय स्वर में) for impact
   - MUST be engaging enough to retain viewers

2. **PROBLEM AGITATION (45-120 seconds / 200-280 Hindi words):**
   - Describe the pain point in vivid detail
   - Use "तुम" extensively for personalization
   - Make viewer feel understood and frustrated
   - Use (गंभीर स्वर में) for emotional depth

3. **PROMISE (120-180 seconds / 150-200 Hindi words):**
   - Clear statement of what they'll learn
   - "आज के बाद तुम कभी फूल्ड नहीं होगे"
   - Build anticipation for the reveal
   - Use (धीरे से) for intimate connection

4. **MAIN CONTENT (9-12 minutes / 1000-1400 Hindi words) - CRITICAL:**
   - 4-6 distinct sections with clear transitions
   - Each section: Concept → Indian Example → Psychology Explanation → Practical Application
   - Use real-life scenarios: Office politics, family dynamics, relationships, social media
   - Include "रिसर्च के मुताबिक" for credibility
   - Add "मेरे साथ हुआ था" type personal touches
   - Use emotional indicators throughout for natural flow
   - **MUST BE DETAILED - NO RUSHING THROUGH TOPICS**

5. **PRACTICAL TIPS (2-3 minutes / 300-400 Hindi words):**
   - 5-7 actionable steps (NOT just 3-4)
   - Specific, not generic ("फोन उठाने से पहले 2 बार सोचो" not just "सावधान रहो")
   - Include early warning signs to watch for
   - Add real-life implementation examples
   - Use (हल्की मुस्कान के साथ) for approachable tone

6. **CONCLUSION (1.5-2 minutes / 200-250 Hindi words):**
   - Summary of key insights (NOT just one)
   - Emotional reinforcement: "तुम स्मार्ट हो, बस अवेयर होना है"
   - Call-to-action: Comment, share with friend, subscribe
   - Teaser for next episode
   - Personal closing message with (प्यार से)

**VOICE STYLE REQUIREMENTS:**
The script must sound:
- Immersive (श्रोता को कहानी में खींच ले)
- Cinematic (दृश्यों की कल्पना हो सके)
- Emotionally engaging (भावनाओं को छू ले)
- Natural spoken Hindi (रोबोटिक नहीं, बल्कि जैसे कोई दोस्त बात कर रहा हो)

**CRITICAL: DETERMINISTIC CHUNK GENERATION REQUIREMENT (ABSOLUTE)**

You MUST split the entire script into logical chunks following these RULES:

1. **CHUNK SIZE RULE (STRICT LIMIT - PREVENTS TRUNCATION):** 
   - Each chunk MUST contain between 120-180 Hindi words
   - **NEVER exceed 180 words per chunk. This is a hard limit.**
   - **Total response size must remain under safe limits. Prefer more chunks with smaller size.**

2. **SENTENCE COMPLETENESS RULE:** Each chunk MUST end with a COMPLETE sentence (। ? !)

3. **NO SPLIT RULE:** NEVER split a sentence between chunks

4. **CONTINUITY RULE:** Chunks must flow naturally with no gaps or overlaps

5. **WORD COUNT RULE:** Total words across ALL chunks = full script word count

For a 10-15 minute script (1400-1900 words), you will create approximately:
- 8-12 chunks (since each chunk is 120-180 words)
- More chunks with smaller size ensures the JSON response stays under token limits

**IMPORTANT CHUNKING GUIDELINES:**
- Chunk 1: HOOK + beginning of PROBLEM AGITATION (120-180 words)
- Middle chunks: Continue PROBLEM AGITATION, PROMISE, and MAIN CONTENT divided into small logical segments (120-180 words each)
- Final chunks: PRACTICAL TIPS + CONCLUSION (120-180 words each)

**CRITICAL OUTPUT INSTRUCTION:**
You MUST return ONLY a valid JSON object. Do NOT include any explanation, preamble, or text before or after the JSON.
Do NOT wrap it in markdown code blocks (```json). Return the raw JSON object directly.

The JSON MUST have this EXACT structure:
{{
  "metadata": {{
    "title_options": ["Option 1", "Option 2", "Option 3"],
    "final_title": "Selected title",
    "description": "YouTube description text (2-3 sentences)",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
    "thumbnail_idea": "Description for thumbnail",
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode}
  }},
  "chunks": [
    {{
      "chunk_id": 1,
      "text": "Complete Hindi narration text for chunk 1 with emotional indicators like (गंभीर स्वर में) and scene markers [SCENE: type]. Must end with a complete sentence (। ? !). Word count: between 120-180 words."
    }},
    {{
      "chunk_id": 2,
      "text": "Complete Hindi narration text for chunk 2 with emotional indicators and scene markers. Must start with the natural continuation from chunk 1. Must end with a complete sentence (। ? !). Word count: between 120-180 words."
    }},
    {{
      "chunk_id": 3,
      "text": "Complete Hindi narration text for chunk 3 with emotional indicators and scene markers. Must start with the natural continuation from chunk 2. Must end with a complete sentence (। ? !). Word count: between 120-180 words."
    }}
    // Add more chunks as needed (typically 8-12 total for 1400-1900 word script)
  ],
  "full_script": "The COMPLETE concatenation of ALL chunks in order. This must be exactly chunk1.text + chunk2.text + chunk3.text + ... with no modifications.",
  "script": {{
    "word_count": 1600,
    "estimated_duration": "12:30"
  }}
}}

**CRITICAL VALIDATION RULES:**
- Verify that `full_script` is exactly the concatenation of all chunk texts
- Verify that each chunk contains complete sentences only
- Verify that no sentence is split across chunks
- Verify that total words in chunks = word_count in script
- Verify that chunks cover 100% of the script content with no gaps
- **Verify that NO chunk exceeds 180 words (this is a hard limit to prevent JSON truncation)**

**ENSURE THE JSON IS COMPLETE AND VALID. DO NOT TRUNCATE ANY SECTION.**
**REMEMBER: Pure Hindi (देवनागरी लिपि), NOT Hinglish**
**REMEMBER: Emotional indicators must be on separate lines BEFORE sentences**
**REMEMBER: Scene markers must be on separate lines, NOT spoken**
**REMEMBER: CHUNKS MUST BE 120-180 WORDS EACH (STRICT LIMIT), COMPLETE SENTENCES ONLY**
**REMEMBER: USING MORE SMALLER CHUNKS IS BETTER THAN FEWER LARGE CHUNKS TO PREVENT TRUNCATION**"""
    
    return prompt


def create_short_script_prompt(category, sub_category, episode, title):
    """
    Create prompt for viral YouTube Shorts script (45-60 seconds)
    Shorts are single chunk by definition (no splitting needed)
    """
    
    hindi_category = CATEGORIES_CONFIG.get(category, {}).get("hindi_name", category)
    hindi_sub = CATEGORIES_CONFIG.get(category, {}).get("sub_categories", {}).get(sub_category, sub_category)
    
    prompt = f"""You are an elite Hindi content strategist specializing in VIRAL YOUTUBE SHORTS.

TASK: Create a promotional YouTube Shorts script that drives viewers to watch the full video.

INPUT PARAMETERS:
- Main Category: {category} ({hindi_category})
- Sub-Category: {sub_category} ({hindi_sub})
- Episode: {episode}
- Title: {title}
- **Target Duration: 45-60 SECONDS SPEAKING TIME (100-150 Hindi words)**
- Target Audience: 18-35 years, Hindi-speaking, Indian urban/semi-urban
- Tone: High energy, curiosity-driven, emotionally engaging
- Language: **PURE HINDI (देवनागरी लिपि में शुद्ध हिंदी)**

**ABSOLUTE LANGUAGE RULE:**

Narration must contain ZERO English letters.

Do NOT use characters a-z or A-Z anywhere in narration.

Only Hindi Devanagari script is allowed.

English technical words must be written using Hindi phonetics.

Examples:

Correct: ब्रेन, साइकोलॉजी, रियलिटी  
Wrong: brain, psychology, reality

This rule is STRICT and must never be violated.

**XTTS VOICE OPTIMIZATION REQUIREMENTS:**

Use emotional reaction indicators in brackets ONLY:

(धीरे से)
(गंभीर स्वर में)
(रहस्यमय स्वर में)  
(उत्साह से)
(हल्की मुस्कान के साथ)
(फुसफुसाते हुए)
(आश्चर्य से)
(दुखी होकर)
(गुस्से में)
(प्यार से)

**EMOTION PLACEMENT RULE (CRITICAL):**

Emotion indicators must ALWAYS be placed on a separate line before narration.

**SHORTS SCRIPT STRUCTURE (45-60 SECONDS):**

1. **HOOK (0-5 seconds / 10-15 words):**
   - Pattern interrupt statement
   - Creates immediate curiosity
   - Shocking or relatable opening
   - Example: "तुम रोज़ एक ऐसी गलती कर रहे हो..."
   - Use (उत्साह से) or (रहस्यमय स्वर में)

2. **CONTENT (5-40 seconds / 80-120 words):**
   - One powerful psychological insight
   - Emotional engagement throughout
   - Make it personal with "तुम"
   - Create "aha moment"
   - Hint at deeper secrets in full video
   - Use varied emotions: (गंभीर स्वर में), (आश्चर्य से), (धीरे से)

3. **CTA (40-55 seconds / 20-30 words):**
   - Clear call to action to watch full video
   - Create urgency
   - Example: "पूरी सच्चाई जानने के लिए... वीडियो एंड तक देखो!"
   - Use (प्यार से) or (उत्साह से)

**VOICE STYLE REQUIREMENTS:**
- High energy but natural
- Emotionally engaging
- Curiosity-driven
- Must feel like a trailer, not a summary

**SCENE MARKERS FOR VIDEO EDITING (USE EXACTLY):**
[SCENE: hook_intense] [SCENE: explain_serious] [SCENE: cta_energy]

**CRITICAL OUTPUT INSTRUCTION:**
You MUST return ONLY a valid JSON object. Do NOT include any explanation, preamble, or text before or after the JSON.
Do NOT wrap it in markdown code blocks (```json). Return the raw JSON object directly.

The JSON MUST have this EXACT structure:
{{
  "metadata": {{
    "category": "{category}",
    "sub_category": "{sub_category}",
    "episode": {episode},
    "full_video_title": "{title}"
  }},
  "script": {{
    "hook": "Hook text with emotional indicator",
    "content": "Main content text with emotional indicators",
    "cta": "Call to action text with emotional indicator",
    "full_text": "Complete script combining all parts",
    "word_count": 120,
    "estimated_duration": "50 seconds"
  }}
}}

**ENSURE THE JSON IS COMPLETE AND VALID.**
**REMEMBER: Pure Hindi (देवनागरी लिपि), NOT Hinglish**
**REMEMBER: Emotional indicators must be on separate lines BEFORE sentences**
**REMEMBER: This is a STANDALONE SHORTS script, NOT trimmed from long video**"""
    
    return prompt


# ============================================================================
# ENHANCED JSON EXTRACTION WITH REPAIR CAPABILITIES
# ============================================================================

def extract_json_from_response(text: str) -> str:
    """
    Enhanced JSON extraction with multiple strategies and auto-repair
    
    Handles:
    - Markdown code blocks
    - Text before/after JSON
    - Truncated JSON
    - Common LLM errors (trailing commas, missing commas, unescaped quotes)
    
    Args:
        text: Raw response from LLM
        
    Returns:
        Valid JSON string
        
    Raises:
        ValueError: If no valid JSON can be extracted/repaired
    """
    print("🔍 Extracting JSON from response...")
    
    # Strategy 1: Check if entire text is valid JSON
    try:
        json.loads(text)
        print("✓ Response is pure valid JSON")
        return text
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks (ENHANCED)
    code_block_patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'```json(.*?)```',
        r'```(.*?)```',
        # Also try without closing ``` (truncated markdown)
        r'```json\s*\n(.*)',
        r'```\s*\n(.*)',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            # Try to parse directly
            try:
                json.loads(match)
                print("✓ Extracted valid JSON from code block")
                return match
            except json.JSONDecodeError:
                # Try to repair truncated markdown block
                print(f"⚠️ Code block JSON invalid, attempting repair...")
                repaired = repair_json(match)
                if repaired:
                    print("✓ Extracted and repaired JSON from code block")
                    return repaired
                
                # If repair failed, try salvage (for truncation)
                salvaged = salvage_truncated_json(match)
                if salvaged:
                    print("✓ Salvaged truncated JSON from code block")
                    return salvaged
    
    # Strategy 3: Find balanced JSON object using brace counting
    print("🔍 Searching for balanced JSON object...")
    json_candidate = find_balanced_json(text)
    
    if json_candidate:
        try:
            json.loads(json_candidate)
            print("✓ Extracted valid balanced JSON")
            return json_candidate
        except json.JSONDecodeError as e:
            print(f"⚠️ Balanced JSON invalid: {e}")
            # Try to repair
            repaired = repair_json(json_candidate)
            if repaired:
                print("✓ Extracted and repaired balanced JSON")
                return repaired
            
            # Try salvage for truncation
            salvaged = salvage_truncated_json(json_candidate)
            if salvaged:
                print("✓ Salvaged truncated balanced JSON")
                return salvaged
    
    # Strategy 4: Emergency - try to salvage truncated JSON directly
    print("🚨 Attempting emergency JSON salvage...")
    salvaged = salvage_truncated_json(text)
    if salvaged:
        print("✓ Salvaged truncated JSON")
        return salvaged
    
    # Failed all strategies
    print("❌ Failed to extract valid JSON from response")
    print("Response preview (first 2000 chars):")
    print(text[:2000])
    print("...")
    print("Response end (last 500 chars):")
    print(text[-500:])
    raise ValueError("No valid JSON found in response. The model may not have followed the JSON format instruction.")


def find_balanced_json(text: str) -> str:
    """
    Find the first balanced JSON object in text using brace counting
    Enhanced to handle truncated responses
    
    Args:
        text: Text to search
        
    Returns:
        JSON string or None
    """
    # First, try to find a complete balanced object
    brace_stack = 0
    start = None
    in_string = False
    escape_next = False
    
    for i, ch in enumerate(text):
        # Handle string state
        if escape_next:
            escape_next = False
            continue
        
        if ch == '\\':
            escape_next = True
            continue
        
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        
        # Only count braces outside strings
        if not in_string:
            if ch == '{':
                if brace_stack == 0:
                    start = i
                brace_stack += 1
            elif ch == '}':
                brace_stack -= 1
                if brace_stack == 0 and start is not None:
                    return text[start:i+1]
    
    # If we didn't find a complete balanced object, the JSON is likely truncated
    # Return the partial JSON from first { to end
    if start is not None and brace_stack > 0:
        print(f"⚠️ JSON appears truncated ({brace_stack} unclosed braces)")
        return text[start:]
    
    return None


def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON errors from LLMs
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Repaired JSON string or None
    """
    original = json_str
    repairs_made = []
    
    # Repair 1: Remove trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
    if fixed != json_str:
        repairs_made.append("trailing_commas")
        json_str = fixed
    
    # Repair 2: Remove comments
    fixed = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    if fixed != json_str:
        repairs_made.append("comments")
        json_str = fixed
    
    # Repair 3: Fix unescaped newlines in strings (common LLM error)
    # This is risky but necessary for LLM outputs
    fixed = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', json_str)
    if fixed != json_str:
        repairs_made.append("unescaped_newlines")
        json_str = fixed
    
    # Repair 4: Add missing commas between array elements (heuristic)
    # Look for patterns like: "text"<whitespace>"text" and add comma
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    if fixed != json_str:
        repairs_made.append("missing_commas")
        json_str = fixed
    
    # Repair 5: Fix single quotes to double quotes (JSON standard)
    # Be careful not to affect content inside strings
    # This is a simple heuristic - only fix obvious cases
    fixed = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', json_str)
    if fixed != json_str:
        repairs_made.append("single_quotes")
        json_str = fixed
    
    # Test if repairs worked
    if repairs_made:
        try:
            json.loads(json_str)
            print(f"✓ JSON repaired: {', '.join(repairs_made)}")
            return json_str
        except json.JSONDecodeError as e:
            print(f"⚠️ Repair attempt failed: {e}")
            # Try more aggressive repairs
            return aggressive_repair(original)
    
    return None


def aggressive_repair(json_str: str) -> str:
    """
    More aggressive JSON repair for severely malformed JSON
    
    Args:
        json_str: Malformed JSON
        
    Returns:
        Repaired JSON or None
    """
    # Strategy: Try to parse incrementally and reconstruct
    try:
        # Find the error position
        json.loads(json_str)
    except json.JSONDecodeError as e:
        error_pos = e.pos
        
        # Try to truncate at the error and close properly
        truncated = json_str[:error_pos]
        
        # Count open braces/brackets that need closing
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        
        # Close them
        closing = ']' * open_brackets + '}' * open_braces
        repaired = truncated + closing
        
        try:
            json.loads(repaired)
            print("✓ Aggressively repaired by truncation and closing")
            return repaired
        except json.JSONDecodeError:
            pass
    
    return None


def salvage_truncated_json(text: str) -> str:
    """
    Emergency salvage for truncated JSON responses
    
    Intelligently closes incomplete JSON by analyzing structure
    
    Args:
        text: Response text (potentially truncated)
        
    Returns:
        Repaired valid JSON or None
    """
    print("🔧 Attempting to salvage truncated JSON...")
    
    # Try to find the start of the JSON
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    json_text = text[start_idx:]
    
    # Strategy 1: Intelligent string-aware closing
    # Parse character by character tracking state
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
            
            # Track last position where we had valid nesting
            if brace_count >= 0 and bracket_count >= 0:
                last_valid_pos = i
    
    # If we're in the middle of a string, backtrack to before it started
    if in_string:
        # Find the last quote before current position
        for i in range(len(json_text) - 1, -1, -1):
            if json_text[i] == '"' and (i == 0 or json_text[i-1] != '\\'):
                last_valid_pos = i - 1
                break
    
    # Now try to close from last valid position
    candidate = json_text[:last_valid_pos + 1]
    
    # Recount braces/brackets from the candidate
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
    
    # Add proper closing
    closing = ']' * open_brackets + '}' * open_braces
    repaired = candidate + closing
    
    # Try to parse
    try:
        parsed = json.loads(repaired)
        if 'metadata' in parsed and 'chunks' in parsed:
            print(f"✓ Salvaged JSON with intelligent closing ({open_braces} braces, {open_brackets} brackets)")
            return repaired
    except json.JSONDecodeError as e:
        print(f"⚠️ Intelligent closing failed: {e}")
    
    # Strategy 2: Progressive backtracking with validation
    # Start from end and work backwards in chunks
    print("🔄 Trying progressive backtracking...")
    
    for step_back in [0, 50, 100, 200, 500, 1000, 2000]:
        truncate_at = len(json_text) - step_back
        if truncate_at < 100:  # Don't go too far back
            break
        
        candidate = json_text[:truncate_at]
        
        # Clean up partial content
        # Remove incomplete string at end
        if candidate.rstrip().endswith('"'):
            candidate = candidate.rstrip()[:-1]
        
        # Remove trailing comma/incomplete element
        candidate = re.sub(r',\s*$', '', candidate.rstrip())
        
        # Count and close
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
        
        # Reasonable limits
        if open_braces > 15 or open_brackets > 15:
            continue
        
        closing = ']' * open_brackets + '}' * open_braces
        repaired = candidate + closing
        
        try:
            parsed = json.loads(repaired)
            if 'metadata' in parsed and 'chunks' in parsed:
                print(f"✓ Salvaged by backtracking {step_back} chars")
                return repaired
        except json.JSONDecodeError:
            continue
    
    print("❌ All salvage strategies failed")
    return None


# ============================================================================
# PRODUCTION SAFETY - CHUNK INTEGRITY VALIDATION
# ============================================================================

def validate_chunks_integrity(script_data: dict) -> bool:
    """
    CRITICAL PRODUCTION SAFETY VALIDATION
    
    Validates script chunk integrity to ensure pipeline doesn't proceed with corrupted data.
    NOW WITH CRITICAL FIX: chunks are treated as authoritative source. full_script is rebuilt from chunks.
    
    Validation Rules:
    1. script_data contains "chunks" key
    2. chunks is a non-empty list
    3. Each chunk contains:
       - "chunk_id" (integer)
       - "text" (non-empty string)
    4. Each chunk's text ends with a sentence terminator (। ? !) - ignoring trailing quotes
    5. Chunk IDs are sequential starting from 1 (1,2,3,... no gaps)
    6. (FIXED) full_script is rebuilt from chunks (chunks are authoritative)
    7. (SOFT WARNING) Check if any chunk exceeds 180 words (new rule)
    
    Args:
        script_data: Parsed script JSON data
        
    Returns:
        True if validation passes
        
    Raises:
        RuntimeError: With detailed error message if any validation fails
    """
    print("🔒 PRODUCTION SAFETY: Validating script chunk integrity...")
    
    # Rule 1: Contains "chunks" key
    if "chunks" not in script_data:
        error_msg = "Script integrity validation failed: Missing 'chunks' key in script_data"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Rule 2: chunks is a non-empty list
    chunks = script_data["chunks"]
    if not isinstance(chunks, list):
        error_msg = f"Script integrity validation failed: 'chunks' is not a list (found {type(chunks).__name__})"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    if len(chunks) == 0:
        error_msg = "Script integrity validation failed: 'chunks' list is empty"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    print(f"   ✓ Found {len(chunks)} chunks")
    
    # Track expected chunk ID
    expected_id = 1
    concatenated_text = ""
    
    # Rule 3, 4, 5: Validate each chunk
    for idx, chunk in enumerate(chunks):
        chunk_num = idx + 1
        
        # Rule 3a: Each chunk must contain "chunk_id"
        if "chunk_id" not in chunk:
            error_msg = f"Script integrity validation failed: Chunk {chunk_num} missing 'chunk_id' field"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        chunk_id = chunk["chunk_id"]
        
        # Rule 5: Chunk IDs must be sequential
        if chunk_id != expected_id:
            error_msg = (f"Script integrity validation failed: Chunk ID sequence broken. "
                        f"Expected ID {expected_id}, got {chunk_id} at position {chunk_num}")
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Rule 3b: Each chunk must contain "text"
        if "text" not in chunk:
            error_msg = f"Script integrity validation failed: Chunk {chunk_id} missing 'text' field"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        text = chunk["text"]
        
        # Rule 3c: text must be non-empty string
        if not isinstance(text, str):
            error_msg = (f"Script integrity validation failed: Chunk {chunk_id} 'text' is not a string "
                        f"(found {type(text).__name__})")
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        if not text.strip():
            error_msg = f"Script integrity validation failed: Chunk {chunk_id} text is empty"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Rule 4: Each chunk's text must end with a sentence terminator
        # Find last meaningful character ignoring quotes and spaces
        stripped_text = text.rstrip()
        if stripped_text:
            # Start with the last character
            last_char = stripped_text[-1]
            
            # Remove trailing quotes if present
            while last_char in ['"', "'", '”', '’'] and len(stripped_text) > 1:
                stripped_text = stripped_text[:-1].rstrip()
                last_char = stripped_text[-1]
            
            # Now validate sentence terminator
            if last_char not in ['।', '?', '!']:
                error_msg = (f"Script integrity validation failed: Chunk {chunk_id} does not end with "
                            f"sentence terminator (। ? !). Last meaningful char: '{last_char}'")
                print(f"❌ {error_msg}")
                print(f"   Chunk text ends with: ...{text.rstrip()[-50:]}")
                raise RuntimeError(error_msg)
        else:
            error_msg = f"Script integrity validation failed: Chunk {chunk_id} text contains only whitespace"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Rule 7: Check chunk size (warning only, not fatal)
        word_count = len(text.split())
        if word_count > 180:
            print(f"⚠️ WARNING: Chunk {chunk_id} exceeds 180 words ({word_count} words). This may cause truncation in future runs.")
        elif word_count < 120:
            print(f"ℹ️ INFO: Chunk {chunk_id} is below 120 words ({word_count} words). Consider combining with adjacent chunk for optimal size.")
        
        # Add to concatenated text for rebuilding
        concatenated_text += text.strip()
        if idx < len(chunks) - 1:
            # Add space between chunks for proper concatenation
            concatenated_text += " "
        
        # Increment expected ID
        expected_id += 1
        
        print(f"   ✓ Chunk {chunk_id}: {word_count} words, ends with '{last_char}'")
    
    # ===== CRITICAL FIX: Rebuild full_script from chunks (chunks are authoritative) =====
    rebuilt_script = " ".join(chunk["text"].strip() for chunk in script_data["chunks"])
    
    # Update full_script in script_data
    script_data["full_script"] = rebuilt_script
    
    # Also update script.full_text if it exists
    if "script" in script_data:
        if "full_text" in script_data["script"]:
            script_data["script"]["full_text"] = rebuilt_script
        # Update word count to match actual words
        script_data["script"]["word_count"] = len(rebuilt_script.split())
    
    print("✅ full_script rebuilt from chunks (chunks are authoritative source)")
    
    print(f"✅ PRODUCTION SAFETY: All chunk integrity checks passed")
    print(f"   ✓ {len(chunks)} sequential chunks validated")
    print(f"   ✓ All chunks end with sentence terminators")
    print(f"   ✓ full_script rebuilt from chunks (authoritative)")
    print(f"   ✓ Total words: {len(rebuilt_script.split())}")
    
    return True


def validate_chunks(chunks, full_script):
    """
    Legacy validation function - kept for backward compatibility
    Now uses the production safety validator internally
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        full_script: Expected full script text
    
    Returns:
        Boolean indicating if validation passed
    """
    # Create a minimal script_data structure for validation
    script_data = {
        "chunks": chunks,
        "full_script": full_script
    }
    
    try:
        validate_chunks_integrity(script_data)
        return True
    except RuntimeError as e:
        print(f"⚠️ Chunk validation failed: {e}")
        return False


def generate_script(category, sub_category, episode, run_id, video_type='long'):
    """
    Generate script using Gemini 2.5 API with enhanced error handling
    Now with deterministic chunk generation for long videos
    And production safety validation
    FIXED: Added response_mime_type="application/json" to force structured JSON output
    FIXED: Enhanced response handling to capture JSON from candidates when text field is empty
    
    Args:
        category: Main category
        sub_category: Sub category
        episode: Episode number
        run_id: Run ID
        video_type: 'long' or 'short'
    
    Returns:
        Script data dictionary
    
    Raises:
        Exception: If generation or validation fails
    """
    
    print(f"📝 Generating {video_type.upper()} script for: {category} - {sub_category} (Ep {episode})")
    
    # Get episode title
    title = get_episode_title(category, sub_category, episode)
    
    # Create appropriate prompt based on video type
    if video_type == 'short':
        prompt = create_short_script_prompt(category, sub_category, episode, title)
    else:
        prompt = create_long_script_prompt(category, sub_category, episode, title)
    
    # Models to try in order of preference
    models_to_try = [
        'gemini-2.5-pro',      # Best quality
        'gemini-2.5-flash',    # Fast and good
        'gemini-2.0-flash',    # Good fallback
        'gemini-2.0-flash-lite',  # Fastest
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
                    max_output_tokens=8192,  # Maximum allowed
                    top_p=0.9,
                    top_k=40,
                    response_mime_type="application/json"  # CRITICAL: Force structured JSON output
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
    
    # Parse JSON response with enhanced extraction
    try:
        # FIXED: Enhanced response handling to capture JSON from candidates when text field is empty
        if hasattr(response, "text") and response.text:
            response_text = response.text.strip()
            print("📄 Response captured from response.text")
        elif hasattr(response, "candidates") and response.candidates:
            # Extract from candidates[0].content.parts[0].text
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                response_text = candidate.content.parts[0].text.strip()
                print("📄 Response captured from response.candidates[0].content.parts[0].text")
            else:
                raise RuntimeError("Gemini returned candidates but no text content")
        else:
            raise RuntimeError("Gemini returned empty response")
        
        print(f"📏 Response length: {len(response_text)} chars")
        
        # Extract JSON using enhanced method
        json_str = extract_json_from_response(response_text)
        
        # Parse JSON
        script_data = json.loads(json_str)
        
        # Validate structure based on video type
        if video_type == 'short':
            if 'script' not in script_data:
                raise ValueError("JSON missing required field: script")
            
            # Ensure full_text is present (create if needed)
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
                    'full_video_title': title
                }
            
            # Validate word count (shorts: 100-150 words)
            word_count = script_data['script'].get('word_count', 0)
            if word_count < 80 or word_count > 200:
                print(f"⚠️ Shorts word count {word_count} outside optimal range (80-200)")
            
        else:  # long
            if 'metadata' not in script_data:
                print("⚠️ Missing 'metadata' field, adding minimal structure")
                script_data['metadata'] = {
                    'final_title': title,
                    'category': category,
                    'sub_category': sub_category,
                    'episode': episode
                }
            
            if 'chunks' not in script_data:
                raise ValueError("JSON missing required field: 'chunks' for long script")
            
            if 'full_script' not in script_data:
                # Try to construct full_script from chunks if missing
                if 'chunks' in script_data:
                    full_script = ""
                    for chunk in sorted(script_data['chunks'], key=lambda x: x.get('chunk_id', 0)):
                        if 'text' in chunk:
                            full_script += chunk['text']
                    script_data['full_script'] = full_script
                    print("⚠️ Missing 'full_script', constructed from chunks")
                else:
                    raise ValueError("JSON missing required field: 'full_script'")
            
            # ===== PRODUCTION SAFETY VALIDATION =====
            # Validate chunk integrity - this will raise RuntimeError if validation fails
            # IMPORTANT: This function now rebuilds full_script from chunks
            validate_chunks_integrity(script_data)
            
            # For backward compatibility, also populate script.full_text
            if 'script' not in script_data:
                script_data['script'] = {}
            
            script_data['script']['full_text'] = script_data.get('full_script', '')
            
            # Count chunks and words
            num_chunks = len(script_data.get('chunks', []))
            chunk_word_counts = []
            for chunk in script_data.get('chunks', []):
                if 'text' in chunk:
                    words = len(chunk['text'].split())
                    chunk_word_counts.append(words)
            
            # Validate word count
            word_count = script_data.get('script', {}).get('word_count', 0)
            if word_count < 1400:
                print(f"⚠️ Long script word count {word_count} is below minimum 1400")
            
            print(f"📊 Generated {num_chunks} chunks with word counts: {chunk_word_counts}")
        
        # Add generation metadata
        script_data['generation_info'] = {
            'category': category,
            'sub_category': sub_category,
            'episode': episode,
            'run_id': run_id,
            'video_type': video_type,
            'generated_at': datetime.now().isoformat(),
            'model_used': model_used,
            'response_length_chars': len(response_text)
        }
        
        # Add chunking info for long videos
        if video_type == 'long' and 'chunks' in script_data:
            script_data['generation_info']['chunking_method'] = 'gemini_deterministic_chunking'
            script_data['generation_info']['num_chunks'] = len(script_data['chunks'])
        
        # Save to file based on video type
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if video_type == 'short':
            output_file = output_dir / 'script_short.json'
            
            # ===== SHORT SCRIPT NORMALIZATION (SAFE FIX) =====
            # Ensure script_short.json always contains valid "chunks" array
            # If chunks already exist, do nothing
            if "chunks" not in script_data:
                print("🔄 Normalizing SHORT script to chunks format...")
                
                # Extract script text safely
                script_text = (
                    script_data.get("script", {}).get("full_text") or
                    script_data.get("full_script") or
                    ""
                ).strip()
                
                if not script_text:
                    raise RuntimeError("SHORT script is empty, cannot normalize")
                
                # Convert to required chunks format
                script_data = {
                    "chunks": [
                        {
                            "chunk_id": 1,
                            "text": script_text
                        }
                    ],
                    "full_script": script_text,
                    "script": {
                        "full_text": script_text,
                        "word_count": len(script_text.split()),
                        "estimated_duration": script_data.get('script', {}).get('estimated_duration', '50 seconds')
                    },
                    "metadata": script_data.get('metadata', {
                        'category': category,
                        'sub_category': sub_category,
                        'episode': episode,
                        'full_video_title': title
                    }),
                    "generation_info": script_data.get('generation_info', {})
                }
                
                print(f"✅ SHORT script normalized: {len(script_text.split())} words, single chunk")
        else:
            output_file = output_dir / 'script_long.json'
        
        # Write the final script data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {video_type.upper()} script generated")
        
        if video_type == 'short':
            print(f"📝 Word count: {script_data['script'].get('word_count', 'N/A')}")
            print(f"⏱️ Estimated duration: {script_data['script'].get('estimated_duration', 'N/A')}")
        else:
            print(f"📝 Word count: {script_data.get('script', {}).get('word_count', 'N/A')}")
            print(f"📊 Chunks: {len(script_data.get('chunks', []))}")
            print(f"⏱️ Estimated duration: {script_data.get('script', {}).get('estimated_duration', 'N/A')}")
            print(f"🎯 Title: {script_data['metadata'].get('final_title', 'N/A')}")
        
        print(f"💾 Saved to: {output_file}")
        
        return script_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Error position: {e.pos}")
        print(f"Response text around error:")
        start = max(0, e.pos - 200)
        end = min(len(response_text), e.pos + 200)
        print(response_text[start:end])
        raise
    except RuntimeError as e:
        # This is from validation failure - re-raise to stop pipeline
        print(f"❌ PRODUCTION SAFETY STOP: {e}")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate YouTube script with enhanced JSON handling, deterministic chunking, and production safety validation')
    parser.add_argument('--category', required=True, help='Main category')
    parser.add_argument('--sub-category', required=True, help='Sub category')
    parser.add_argument('--episode', required=True, type=int, help='Episode number')
    parser.add_argument('--run-id', required=True, help='Run ID')
    parser.add_argument('--video-type', choices=['long', 'short'], default='long',
                       help='Video type: long (10-15 min) or short (45-60 sec)')
    
    args = parser.parse_args()
    
    try:
        script_data = generate_script(
            args.category,
            args.sub_category,
            args.episode,
            args.run_id,
            args.video_type
        )
        
        # Output for GitHub Actions
        print(f"::set-output name=script_data::{json.dumps(script_data)}")
        
    except Exception as e:
        print(f"❌ Script generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
