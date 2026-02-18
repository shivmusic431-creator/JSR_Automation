#!/usr/bin/env python3
"""
Enhanced Audio Generation with Coqui XTTS v2 - CI/CD Safe Version
Production-grade pause-aware, resumable audio generation system

Features:
- Intelligent pause-aware chunking (~2 min chunks)
- XTTS v2-safe micro-segmentation (FIXED: 600 chars max per segment)
- Resume-safe persistent session state
- Automatic retry logic (3 attempts per chunk)
- CLI modes: generate, resume, stitch, status, chunk-id
- CI/CD safe execution with heartbeat logging
- GitHub Actions watchdog protection
- Backwards compatible with existing pipeline
- Production-safe voice cloning with my_voice.wav
- Studio-quality audio output with smoothing
- SUPPORTS BOTH LONG AND SHORT SCRIPTS with separate output files
- PROFESSIONAL YOUTUBE PACING: Confident, energetic narration with optimized speed
- ENHANCED CONFIDENCE: Optimized voice characteristics for authoritative narration

ARCHITECTURE:
- Single model load per process (XTTS v2 optimization)
- Reusable TTS instance across all chunks
- Chunk-level inference with immediate WAV writes
- Memory-safe iteration with cleanup
- No Bark dependencies
"""

import os
import sys
import json
import argparse
import re
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from threading import Thread, Event

# Core dependencies (always needed)
import numpy as np
from scipy.io.wavfile import write as write_wav, read as read_wav

# XTTS imports with availability check
try:
    from TTS.api import TTS
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("⚠️ WARNING: XTTS not available - running in compatibility mode")


# ============================================================================
# CONFIGURATION
# ============================================================================

class ChunkStatus(Enum):
    """Chunk generation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChunkMetadata:
    """Metadata for a single audio chunk"""
    chunk_id: int
    text: str
    estimated_duration: float
    status: str
    retries: int
    error: Optional[str]
    wav_path: Optional[str]
    timestamp: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SessionState:
    """Persistent session state for resume capability"""
    run_id: str
    script_file: str
    script_type: str  # 'long' or 'short'
    total_chunks: int
    chunks_completed: int
    chunks_failed: int
    voice_preset: str
    sample_rate: int
    created_at: str
    updated_at: str
    chunks: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# CONSTANTS - OPTIMIZED FOR PROFESSIONAL YOUTUBE PACING
# ============================================================================

# Audio generation parameters - CI-SAFE LIMITS
TARGET_CHUNK_DURATION = 60   # seconds (target duration)
MIN_CHUNK_DURATION = 45       # minimum chunk duration
MAX_CHUNK_DURATION = 75       # HARD CAP - NEVER EXCEED

# FIX 1: INCREASED WORDS_PER_MINUTE for YouTube-optimized pacing
# Professional YouTube narration: 170-190 WPM (was 150 - too slow/audiobook style)
# 180 WPM provides confident, energetic pacing while maintaining clarity
WORDS_PER_MINUTE = 180        # YouTube-optimized speaking rate for Hindi

# FIX 2: OPTIMIZED micro-segment size for better voice continuity
# XTTS v2 has a hard limit of 400 tokens per generation
# 600 characters is a safe upper bound that respects this limit
# Keeping segments as complete sentences improves voice confidence
XTTS_MICRO_SEGMENT_CHARS = 600
MAX_RETRIES = 3

# XTTS Configuration
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000  # XTTS v2 native sample rate
XTTS_LANGUAGE = "hi"  # Hindi language code

# FIX 3: OPTIMIZED SPEED for final audio (applied after stitching)
# 1.20x provides confident, energetic pacing while maintaining:
# - Audio clarity
# - Emotional expressiveness  
# - No robotic distortion
# - Natural speech rhythm
# Applied globally after stitching for consistent pacing
FINAL_SPEED_MULTIPLIER = 1.20  # Optimized for YouTube narration

# Voice cloning configuration
VOICE_CLONE_FILE = "voices/my_voice.wav"
USE_VOICE_CLONE = True  # Always True for professional output

# CI/CD Safety parameters
HEARTBEAT_INTERVAL = 20  # seconds - log activity every 20s
MAX_SILENT_TIME = 30  # seconds - never silent for more than 30s
MEMORY_CHECK_INTERVAL = 5  # Check memory every N micro-segments

# Directories
OUTPUT_DIR = Path('output')
CHUNKS_DIR = OUTPUT_DIR / 'audio_chunks'
SESSION_FILE = CHUNKS_DIR / 'session.json'
FINAL_AUDIO_FILE_LONG = OUTPUT_DIR / 'audio_long.wav'
FINAL_AUDIO_FILE_SHORT = OUTPUT_DIR / 'audio_short.wav'


# ============================================================================
# HEARTBEAT SYSTEM (CI/CD SAFETY)
# ============================================================================

class HeartbeatLogger:
    """
    Continuous heartbeat logger for CI/CD environments
    Prevents GitHub Actions watchdog timeouts
    """
    
    def __init__(self, interval: int = HEARTBEAT_INTERVAL):
        self.interval = interval
        self.running = False
        self.thread = None
        self.stop_event = Event()
        self.last_message = "Starting..."
        self.start_time = None
        
    def start(self, initial_message: str = "Processing..."):
        """Start heartbeat logging"""
        if self.running:
            return
            
        self.running = True
        self.stop_event.clear()
        self.last_message = initial_message
        self.start_time = time.time()
        
        self.thread = Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        
    def update(self, message: str):
        """Update heartbeat message"""
        self.last_message = message
        
    def stop(self):
        """Stop heartbeat logging"""
        if not self.running:
            return
            
        self.running = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=2)
            
    def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while not self.stop_event.wait(self.interval):
            elapsed = time.time() - self.start_time
            log(f"💓 Heartbeat [{elapsed:.0f}s] - {self.last_message}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(message: str, flush: bool = True):
    """
    CI-safe logging with immediate flush
    
    Args:
        message: Log message
        flush: Whether to flush stdout immediately
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    if flush:
        sys.stdout.flush()


def memory_cleanup():
    """Force memory cleanup (CI safety)"""
    gc.collect()


def smooth_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Apply smoothing to audio to prevent popping and cracking
    
    Args:
        audio_array: Input audio array (int16)
        
    Returns:
        Smoothed audio array (int16)
    """
    # Convert to float32 for processing
    if audio_array.dtype == np.int16:
        audio_float = audio_array.astype(np.float32) / 32767.0
    else:
        audio_float = audio_array.astype(np.float32)
    
    # Apply soft clipping to prevent cracking
    audio_float = np.clip(audio_float, -0.99, 0.99)
    
    # Apply small fade at edges to prevent popping
    if len(audio_float) > 1000:
        fade_len = min(500, len(audio_float) // 100)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        audio_float[:fade_len] *= fade_in
        audio_float[-fade_len:] *= fade_out
    
    # Convert back to int16
    return (audio_float * 32767).astype(np.int16)


def estimate_duration_from_text(text: str) -> float:
    """
    Estimate audio duration from text using word count
    
    Args:
        text: Input text
        
    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    # Using updated WORDS_PER_MINUTE (180) for YouTube-optimized pacing
    duration = (words / WORDS_PER_MINUTE) * 60
    return duration


def clean_text_for_synthesis(text: str) -> str:
    """
    Remove ALL non-spoken metadata safely before XTTS synthesis.
    This ensures emotion indicators are NOT spoken while preserving narration.
    
    Removes:
    - Emotion indicators in parentheses: (गंभीर स्वर में)
    - Scene markers in square brackets: [SCENE: office_tension]
    - Pause markers: [PAUSE-1], [PAUSE-2], [PAUSE-3]
    """
    # Remove emotion indicators - anything inside parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove scene markers - anything inside square brackets
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Normalize whitespace (remove multiple spaces, trim)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def trim_silence(audio: np.ndarray, threshold: float = 0.008, min_silence: int = 100) -> np.ndarray:
    """
    Trim excessive silence from beginning and end of audio.
    Preserves natural breathing pauses while removing dead air.
    
    Args:
        audio: Input audio array (int16)
        threshold: Silence threshold (amplitude ratio) - lowered for better sensitivity
        min_silence: Minimum samples to consider as silence
        
    Returns:
        Trimmed audio array with natural pacing
    """
    if len(audio) == 0:
        return audio
    
    # Convert to float for analysis
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32767.0
    else:
        audio_float = audio.astype(np.float32)
    
    # Find where audio exceeds threshold (not silent)
    abs_audio = np.abs(audio_float)
    above_threshold = abs_audio > threshold
    
    if not np.any(above_threshold):
        # Entire audio is silent - return as is
        return audio
    
    # Find first and last non-silent positions
    start_idx = np.argmax(above_threshold)
    end_idx = len(audio) - np.argmax(above_threshold[::-1]) - 1
    
    # Add conservative padding to preserve natural attack/decay
    # 30ms padding (reduced from 50ms) to minimize dead air while preserving naturalness
    padding = int(0.03 * XTTS_SAMPLE_RATE)  # 30ms padding
    start_idx = max(0, start_idx - padding)
    end_idx = min(len(audio), end_idx + padding)
    
    # Trim silence
    trimmed = audio[start_idx:end_idx]
    
    return trimmed


def increase_audio_speed(audio: np.ndarray, speed: float = FINAL_SPEED_MULTIPLIER) -> np.ndarray:
    """
    Apply speed increase for confident YouTube narration pacing.
    Used globally after stitching for consistent pacing.
    
    Args:
        audio: Input audio array (int16)
        speed: Speed multiplier (1.20 optimized for YouTube)
        
    Returns:
        Sped-up audio array with maintained quality
    """
    if speed <= 1.0:
        return audio
    
    # Ensure speed doesn't exceed safe limits (max 1.25x for quality preservation)
    speed = min(speed, 1.25)
    
    indices = np.round(np.arange(0, len(audio), speed))
    indices = indices[indices < len(audio)].astype(int)
    
    sped_up = audio[indices]
    
    # Apply smoothing after speed change
    return smooth_audio(sped_up)


# ============================================================================
# PRODUCTION-SAFE SENTENCE-BASED CHUNKING SYSTEM
# ============================================================================

class ScriptChunker:
    """
    Deterministic sentence-based chunking system for XTTS audio generation
    
    CRITICAL DESIGN:
    - Splits ONLY at Hindi sentence boundaries (। ? !)
    - Never cuts mid-sentence
    - Hard duration caps: 45s min, 75s max
    - No pause marker dependency
    - Fully deterministic iteration
    - Memory-safe operation
    - CI/CD safe execution
    
    PRODUCTION REQUIREMENTS:
    - Zero text overlap between chunks
    - Zero text loss (full narration preserved)
    - Zero mid-sentence cuts
    - Stable processing time
    - Reduced XTTS freeze probability
    """
    
    def __init__(self, script_text: str, script_type: str = 'long'):
        self.script_text = script_text
        self.script_type = script_type
        self.chunks: List[ChunkMetadata] = []
        
    def chunk_script(self) -> List[ChunkMetadata]:
        """
        Split script into sentence-based chunks with strict duration caps
        
        ALGORITHM:
        1. Split full script into sentences using Hindi boundaries (। ? !)
        2. Accumulate sentences until duration approaches MAX_CHUNK_DURATION
        3. Never split mid-sentence
        4. Enforce 45s min / 75s max per chunk
        5. Preserve complete narration continuity
        
        For shorts: Scripts are short enough to be a single chunk
        For long: Split into multiple chunks as needed
        
        Returns:
            List of chunk metadata objects
        """
        log("=" * 80)
        log(f"🔨 PRODUCTION CHUNKING: Sentence-Based Deterministic Split ({self.script_type.upper()})")
        log("=" * 80)
        log(f"📋 Script analysis:")
        log(f"   Total length: {len(self.script_text)} chars")
        
        # For shorts, always treat as single chunk
        if self.script_type == 'short':
            log("📊 Shorts script detected - generating as single chunk")
            estimated_duration = estimate_duration_from_text(self.script_text)
            
            # Validate shorts duration
            if estimated_duration > 75:
                log(f"⚠️ WARNING: Shorts script duration ({estimated_duration:.1f}s) exceeds 75s")
                log(f"   This may exceed YouTube Shorts limits")
            
            chunk = ChunkMetadata(
                chunk_id=0,
                text=self.script_text.strip(),
                estimated_duration=estimated_duration,
                status=ChunkStatus.PENDING.value,
                retries=0,
                error=None,
                wav_path=None,
                timestamp=None
            )
            self.chunks = [chunk]
            
            log(f"\n📊 Chunk Statistics:")
            log(f"   Total chunks: 1")
            log(f"   Duration: {estimated_duration:.1f}s")
            
            return self.chunks
        
        # For long scripts, use sentence-based chunking
        log(f"   Min chunk: {MIN_CHUNK_DURATION}s")
        log(f"   Target: {TARGET_CHUNK_DURATION}s")
        log(f"   Max chunk: {MAX_CHUNK_DURATION}s (HARD CAP)")
        
        # Step 1: Split into sentences using Hindi boundaries
        # Pattern matches । ? ! followed by whitespace
        sentences = re.split(r'(?<=[।?!])\s+|\n+', self.script_text)
        
        # Clean empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        log(f"📊 Found {len(sentences)} sentences")
        
        if not sentences:
            log("❌ No sentences found in script")
            return []
        
        # Log first few sentences for debugging
        log("\n📝 First 5 sentences:")
        for i, s in enumerate(sentences[:5]):
            preview = s[:50] + "..." if len(s) > 50 else s
            log(f"   {i+1}. {preview}")
        
        # Step 2-6: Build chunks by accumulating sentences
        chunks = self._build_chunks_from_sentences(sentences)
        
        # Step 7: Validate all chunks
        chunks = self._validate_chunks(chunks)
        
        log("\n" + "=" * 80)
        log(f"✅ CHUNKING COMPLETE: {len(chunks)} chunks created")
        log("=" * 80)
        
        # Log chunk statistics
        if chunks:
            total_duration = sum(c.estimated_duration for c in chunks)
            max_duration = max(c.estimated_duration for c in chunks)
            min_duration = min(c.estimated_duration for c in chunks)
            
            log(f"\n📊 Chunk Statistics:")
            log(f"   Total chunks: {len(chunks)}")
            log(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
            log(f"   Min chunk: {min_duration:.1f}s")
            log(f"   Max chunk: {max_duration:.1f}s")
            log(f"   Avg chunk: {total_duration/len(chunks):.1f}s")
            
            # Log each chunk's composition
            log("\n📋 Chunk Composition:")
            for i, chunk in enumerate(chunks):
                sentence_count = len(chunk.text.split('।'))  # Rough estimate
                log(f"   Chunk {i:2d}: {chunk.estimated_duration:5.1f}s | "
                    f"{len(chunk.text):5d} chars | ~{sentence_count} sentences")
        
        return chunks
    
    def _build_chunks_from_sentences(self, sentences: List[str]) -> List[ChunkMetadata]:
        """
        Build chunks by accumulating sentences until duration cap
        
        CRITICAL DESIGN:
        - NEVER splits a sentence
        - Single pass through sentences
        - Deterministic accumulation
        - No backtracking or overlap
        
        RULES:
        1. HARD CAP: If candidate_duration > MAX_CHUNK_DURATION (75s)
           - Finalize current chunk immediately
           - Start new chunk with current sentence
           
        2. TARGET REACHED: If candidate_duration >= TARGET_CHUNK_DURATION (60s)
           - Finalize current chunk INCLUDING current sentence
           
        3. ACCUMULATION: If candidate_duration < TARGET_CHUNK_DURATION
           - Keep accumulating sentences
        
        Args:
            sentences: List of sentences from script
            
        Returns:
            List of chunk metadata
        """
        chunks = []
        current_chunk_sentences = []
        current_chunk_text = ""
        
        for i, sentence in enumerate(sentences):
            # Calculate duration if we add this sentence
            candidate_text = (current_chunk_text + " " + sentence).strip() if current_chunk_text else sentence
            candidate_duration = estimate_duration_from_text(candidate_text)
            
            log(f"\n🔍 Processing sentence {i+1}/{len(sentences)}:")
            log(f"   Sentence length: {len(sentence)} chars")
            log(f"   Current chunk: {len(current_chunk_sentences)} sentences")
            log(f"   Current duration: {estimate_duration_from_text(current_chunk_text):.1f}s" if current_chunk_text else "   Current duration: 0.0s")
            log(f"   Candidate duration: {candidate_duration:.1f}s")
            
            # RULE A: HARD CAP - Never exceed MAX_CHUNK_DURATION
            if candidate_duration > MAX_CHUNK_DURATION:
                if current_chunk_sentences:
                    # Case 1: We have existing sentences - finalize current chunk first
                    chunk_id = len(chunks)
                    chunk_duration = estimate_duration_from_text(current_chunk_text)
                    
                    log(f"   ⚠️ Candidate exceeds MAX ({MAX_CHUNK_DURATION}s)")
                    log(f"   ✅ Finalizing chunk {chunk_id} with {len(current_chunk_sentences)} sentences ({chunk_duration:.1f}s)")
                    
                    chunk = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=current_chunk_text.strip(),
                        estimated_duration=chunk_duration,
                        status=ChunkStatus.PENDING.value,
                        retries=0,
                        error=None,
                        wav_path=None,
                        timestamp=None
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with current sentence only
                    current_chunk_sentences = [sentence]
                    current_chunk_text = sentence
                    
                    log(f"   🆕 Started new chunk with current sentence only")
                    
                else:
                    # Case 2: Single sentence exceeds MAX cap - create chunk with this single sentence
                    # This should be extremely rare with proper script writing
                    chunk_id = len(chunks)
                    
                    log(f"   ⚠️ WARNING: Single sentence exceeds {MAX_CHUNK_DURATION}s cap!")
                    log(f"   📝 Sentence: {sentence[:100]}...")
                    
                    chunk = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=sentence.strip(),
                        estimated_duration=candidate_duration,
                        status=ChunkStatus.PENDING.value,
                        retries=0,
                        error=None,
                        wav_path=None,
                        timestamp=None
                    )
                    chunks.append(chunk)
                    
                    # Reset current chunk (sentence already processed)
                    current_chunk_sentences = []
                    current_chunk_text = ""
            
            # RULE B: TARGET REACHED - Candidate meets or exceeds target duration
            elif candidate_duration >= TARGET_CHUNK_DURATION:
                # Add current sentence to chunk
                current_chunk_sentences.append(sentence)
                current_chunk_text = candidate_text
                
                chunk_id = len(chunks)
                final_duration = estimate_duration_from_text(current_chunk_text)
                
                log(f"   ✅ TARGET duration reached: {candidate_duration:.1f}s")
                log(f"   📦 Finalizing chunk {chunk_id} with {len(current_chunk_sentences)} sentences")
                
                chunk = ChunkMetadata(
                    chunk_id=chunk_id,
                    text=current_chunk_text.strip(),
                    estimated_duration=final_duration,
                    status=ChunkStatus.PENDING.value,
                    retries=0,
                    error=None,
                    wav_path=None,
                    timestamp=None
                )
                chunks.append(chunk)
                
                log(f"   ✅ Chunk {chunk_id} finalized: {final_duration:.1f}s, {len(current_chunk_text)} chars")
                
                # Reset for next chunk
                current_chunk_sentences = []
                current_chunk_text = ""
            
            # RULE C: ACCUMULATION - Continue building current chunk
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_text = candidate_text
                log(f"   📚 Accumulating: now {len(current_chunk_sentences)} sentences")
        
        # Handle remaining sentences after loop
        if current_chunk_sentences:
            chunk_id = len(chunks)
            chunk_duration = estimate_duration_from_text(current_chunk_text)
            
            # Check if remaining chunk is too small (optional warning)
            if chunk_duration < MIN_CHUNK_DURATION and chunks:
                log(f"\n⚠️ Final chunk {chunk_id} is smaller than minimum: {chunk_duration:.1f}s < {MIN_CHUNK_DURATION}s")
                log(f"   This is acceptable for the last chunk - continuing...")
            
            log(f"\n📦 Finalizing last chunk {chunk_id} with {len(current_chunk_sentences)} sentences ({chunk_duration:.1f}s)")
            
            chunk = ChunkMetadata(
                chunk_id=chunk_id,
                text=current_chunk_text.strip(),
                estimated_duration=chunk_duration,
                status=ChunkStatus.PENDING.value,
                retries=0,
                error=None,
                wav_path=None,
                timestamp=None
            )
            chunks.append(chunk)
        
        # Verify no text loss
        reconstructed_text = " ".join([chunk.text for chunk in chunks])
        original_words = set(self.script_text.split())
        reconstructed_words = set(reconstructed_text.split())
        
        if not original_words.issubset(reconstructed_words):
            log(f"\n⚠️ WARNING: Possible text loss detected in chunking!")
        
        log(f"\n✅ Chunking complete: {len(chunks)} chunks created")
        
        return chunks
    
    def _validate_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Final validation pass to ensure all chunks meet requirements
        
        Validation checks:
        1. No empty chunks
        2. No chunks exceeding MAX_DURATION
        3. All chunks contain complete sentences
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validated chunks (with warnings if issues found)
        """
        log("\n🔍 Validating chunks...")
        
        validated_chunks = []
        issues_found = 0
        
        for i, chunk in enumerate(chunks):
            # Check 1: Empty chunk
            if not chunk.text.strip():
                log(f"  ❌ Chunk {i} is empty - SKIPPING")
                issues_found += 1
                continue
            
            # Check 2: Duration cap
            if chunk.estimated_duration > MAX_CHUNK_DURATION:
                log(f"  ⚠️ Chunk {i} exceeds MAX cap: {chunk.estimated_duration:.1f}s > {MAX_CHUNK_DURATION}s")
                log(f"     Text preview: {chunk.text[:100]}...")
                issues_found += 1
                # Still include chunk (can't split mid-sentence)
            
            # Check 3: Ends with sentence boundary
            if not chunk.text.rstrip().endswith(('।', '?', '!')):
                log(f"  ⚠️ Chunk {i} does not end with sentence boundary")
                issues_found += 1
            
            # Check 4: Reasonable text length (sanity check)
            if len(chunk.text) > 5000:
                log(f"  ⚠️ Chunk {i} has unusual length: {len(chunk.text)} chars")
            
            validated_chunks.append(chunk)
        
        # Re-index chunks sequentially (ensures clean IDs)
        for i, chunk in enumerate(validated_chunks):
            chunk.chunk_id = i
        
        if issues_found > 0:
            log(f"\n⚠️ Validation complete: {issues_found} issues found")
        else:
            log(f"\n✅ All chunks validated successfully")
        
        return validated_chunks


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """
    Persistent session state management for resume capability
    """
    
    def __init__(self, run_id: str, script_file: str, script_type: str = 'long'):
        self.run_id = run_id
        self.script_file = script_file
        self.script_type = script_type
        self.session: Optional[SessionState] = None
        
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_or_create_session(self, chunks: List[ChunkMetadata], voice_preset: str) -> SessionState:
        """Load existing session or create new one"""
        if SESSION_FILE.exists():
            log("📂 Loading existing session...")
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if run_id AND script_type match
            if data['run_id'] == self.run_id and data.get('script_type') == self.script_type:
                log(f"✓ Session loaded successfully (type: {self.script_type})")
                chunks_data = [ChunkMetadata(**c) for c in data['chunks']]
                
                self.session = SessionState(
                    run_id=data['run_id'],
                    script_file=data['script_file'],
                    script_type=data.get('script_type', self.script_type),
                    total_chunks=data['total_chunks'],
                    chunks_completed=data['chunks_completed'],
                    chunks_failed=data['chunks_failed'],
                    voice_preset=data['voice_preset'],
                    sample_rate=data['sample_rate'],
                    created_at=data['created_at'],
                    updated_at=datetime.now().isoformat(),
                    chunks=[c.to_dict() for c in chunks_data]
                )
                return self.session
            else:
                log(f"⚠️ Different run_id or script_type - creating new session for {self.script_type}")
        
        log(f"📝 Creating new session for {self.script_type}...")
        self.session = SessionState(
            run_id=self.run_id,
            script_file=self.script_file,
            script_type=self.script_type,
            total_chunks=len(chunks),
            chunks_completed=0,
            chunks_failed=0,
            voice_preset=voice_preset,
            sample_rate=XTTS_SAMPLE_RATE,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            chunks=[c.to_dict() for c in chunks]
        )
        
        self._save_session()
        return self.session
    
    def update_chunk_status(self, chunk: ChunkMetadata):
        """Update chunk status in session"""
        if not self.session:
            return
        
        self.session.chunks[chunk.chunk_id] = chunk.to_dict()
        self.session.updated_at = datetime.now().isoformat()
        
        completed = sum(1 for c in self.session.chunks if c['status'] == ChunkStatus.COMPLETED.value)
        failed = sum(1 for c in self.session.chunks if c['status'] == ChunkStatus.FAILED.value)
        
        self.session.chunks_completed = completed
        self.session.chunks_failed = failed
        
        self._save_session()
    
    def _save_session(self):
        """Save session to disk"""
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.session.to_dict(), f, ensure_ascii=False, indent=2)


# ============================================================================
# XTTS v2 AUDIO GENERATION (CI-SAFE) WITH VOICE CLONING
# ============================================================================

class XTTSAudioGenerator:
    """
    CI-safe XTTS v2 audio generator with heartbeat logging
    
    CRITICAL DESIGN:
    - Single model load per process instance
    - Reuses TTS object for all chunks
    - Prevents repeated model loading
    - Chunk-level inference with immediate WAV writes
    - Memory cleanup after each chunk
    - Always uses voice cloning with my_voice.wav
    - Applies audio smoothing to prevent cracking
    - Micro-segments capped at 600 chars to prevent XTTS token limit crash
    - FIX 4: Generates clean, silence-trimmed chunks WITHOUT speed increase
    - Speed increase applied globally after stitching for consistent pacing
    - Optimized segment boundaries for confident voice delivery
    """
    
    def __init__(self, voice_preset: str = 'hi'):
        self.voice_preset = voice_preset
        self.language = XTTS_LANGUAGE
        self.tts: Optional[TTS] = None
        self.model_loaded = False
        self.heartbeat = HeartbeatLogger()
        
        # Always use voice cloning
        if not os.path.exists(VOICE_CLONE_FILE):
            log(f"⚠️ WARNING: Voice clone file not found at {VOICE_CLONE_FILE}")
            log("   Please ensure voices/my_voice.wav exists for studio-quality output")
    
    def load_model(self):
        """Load XTTS v2 model ONCE per process"""
        if not XTTS_AVAILABLE:
            log("⚠️ XTTS not available - skipping model load")
            return
        
        if self.model_loaded:
            log("✓ XTTS model already loaded - reusing instance")
            return
        
        # Set Coqui TOS agreement for CI/CD environments
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        log("🔄 Loading XTTS v2 model...")
        log(f"   Model: {XTTS_MODEL_NAME}")
        log(f"   Language: {self.language}")
        log(f"   Using voice clone: {VOICE_CLONE_FILE if os.path.exists(VOICE_CLONE_FILE) else 'NOT FOUND'}")
        
        self.heartbeat.start("Loading XTTS v2 model...")
        
        try:
            self.tts = TTS(XTTS_MODEL_NAME)
            self.model_loaded = True
            log("✅ XTTS v2 model loaded successfully")
            
        except Exception as e:
            log(f"❌ Failed to load XTTS model: {e}")
            raise
        finally:
            self.heartbeat.stop()
    
    def generate_chunk_audio(self, chunk: ChunkMetadata, retries: int = 0) -> Optional[np.ndarray]:
        """
        Generate audio for a chunk with CI-safe retry logic
        
        FIX 5: Generates clean audio with silence trimming only.
        NO SPEED INCREASE at chunk level - applied globally after stitching.
        
        Args:
            chunk: Chunk metadata
            retries: Current retry count
            
        Returns:
            Audio array or None if failed
        """
        if not XTTS_AVAILABLE:
            log("⚠️ XTTS not available - cannot generate audio")
            return None
        
        if not self.model_loaded:
            log("⚠️ Model not loaded - loading now")
            self.load_model()
        
        # Strictly clean text before synthesis
        clean_text = clean_text_for_synthesis(chunk.text)
        
        for attempt in range(MAX_RETRIES):
            try:
                log(f"🎙️ Attempt {attempt + 1}/{MAX_RETRIES} for chunk {chunk.chunk_id}")
                
                audio_array = self._generate_with_heartbeat(clean_text, chunk.chunk_id)
                
                if audio_array is not None:
                    log(f"✅ Generation successful on attempt {attempt + 1}")
                    
                    # FIX 6: Professional audio processing chain for chunks
                    # 1. Trim excessive silence only (preserves natural pacing)
                    # 2. NO speed increase at chunk level
                    # 3. Light smoothing for quality
                    
                    # Trim silence to remove dead air while preserving natural pauses
                    trimmed_audio = trim_silence(audio_array)
                    log(f"   ✂️ Silence trimmed: {len(audio_array)/XTTS_SAMPLE_RATE:.2f}s → {len(trimmed_audio)/XTTS_SAMPLE_RATE:.2f}s")
                    
                    # Light smoothing for quality (NO speed increase)
                    final_audio = smooth_audio(trimmed_audio)
                    
                    memory_cleanup()
                    return final_audio
                
            except Exception as e:
                log(f"❌ Attempt {attempt + 1} failed: {e}")
                memory_cleanup()
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    log(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        log(f"❌ All {MAX_RETRIES} attempts failed for chunk {chunk.chunk_id}")
        return None
    
    def _generate_with_heartbeat(self, text: str, chunk_id: int) -> Optional[np.ndarray]:
        """
        Generate audio with continuous heartbeat logging
        
        Split into larger micro-segments (600 chars max) while preserving sentence boundaries.
        Larger segments improve voice continuity and confidence.
        
        Args:
            text: Clean text to synthesize
            chunk_id: Chunk identifier
            
        Returns:
            Audio array or None
        """
        # Split into micro-segments (600 chars max) preserving sentences
        segments = self._split_into_micro_segments(text)
        log(f"📝 Split into {len(segments)} micro-segments (max {XTTS_MICRO_SEGMENT_CHARS} chars each)")
        
        self.heartbeat.start(f"Generating chunk {chunk_id} (0/{len(segments)} segments)")
        
        try:
            audio_arrays = []
            
            for i, segment in enumerate(segments):
                segment_start = time.time()
                
                self.heartbeat.update(
                    f"Chunk {chunk_id}: segment {i+1}/{len(segments)} "
                    f"({len(segment)} chars)"
                )
                
                log(f"  🔊 Segment {i+1}/{len(segments)}: {len(segment)} chars")
                
                # Always use voice cloning with speaker_wav parameter
                try:
                    # Generate audio to file first (more stable than direct array)
                    temp_file = CHUNKS_DIR / f"temp_seg_{chunk_id}_{i}.wav"
                    
                    self.tts.tts_to_file(
                        text=segment,
                        speaker_wav=VOICE_CLONE_FILE,
                        language=self.language,
                        file_path=str(temp_file)
                    )
                    
                    # Read the generated audio
                    rate, wav = read_wav(temp_file)
                    
                    # Clean up temp file
                    temp_file.unlink(missing_ok=True)
                    
                except Exception as e:
                    log(f"⚠️ Generation failed: {e}")
                    
                    # Fallback to direct array generation if file method fails
                    try:
                        wav_list = self.tts.tts(
                            text=segment,
                            speaker_wav=VOICE_CLONE_FILE,
                            language=self.language
                        )
                        
                        if isinstance(wav_list, list):
                            wav = np.array(wav_list, dtype=np.float32)
                        else:
                            wav = wav_list
                            
                    except Exception as e2:
                        log(f"❌ Fallback also failed: {e2}")
                        raise
                
                # Convert to numpy array if needed
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)
                
                # Apply audio smoothing
                if wav.dtype == np.float32:
                    # Already in float32 [-1, 1]
                    wav_float = wav
                else:
                    # Convert to float32 for processing
                    wav_float = wav.astype(np.float32) / 32767.0
                
                # Apply soft clipping to prevent cracking
                wav_float = np.clip(wav_float, -0.99, 0.99)
                
                # Convert to int16 for storage
                audio_int16 = (wav_float * 32767).astype(np.int16)
                
                # Apply smoothing to prevent popping
                audio_int16 = smooth_audio(audio_int16)
                
                audio_arrays.append(audio_int16)
                
                segment_time = time.time() - segment_start
                log(f"  ✓ Segment {i+1} completed in {segment_time:.1f}s")
                
                if (i + 1) % MEMORY_CHECK_INTERVAL == 0:
                    log(f"  🧹 Memory cleanup after {i+1} segments")
                    memory_cleanup()
            
            # Concatenate audio chunks
            log(f"🔗 Concatenating {len(audio_arrays)} segments...")
            final_audio = np.concatenate(audio_arrays, axis=0)
            
            memory_cleanup()
            return final_audio
            
        except Exception as e:
            log(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.heartbeat.stop()
    
    def _split_into_micro_segments(self, text: str) -> List[str]:
        """
        Split text into XTTS-safe micro-segments
        
        Optimized for voice confidence:
        - Prefers complete sentences
        - Avoids fragmentation
        - Maintains natural speech flow
        - 600 chars max per segment (XTTS v2 safe limit)
        
        Args:
            text: Text to split
            
        Returns:
            List of segments (each <= 600 chars)
        """
        # Split on Hindi/Devanagari sentence boundaries
        # । - Hindi danda (full stop)
        # ? - question mark
        # ! - exclamation mark
        sentences = re.split(r'([।?!]+)', text)
        segments = []
        current_segment = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            full_sentence = sentence + delimiter
            
            # If adding this sentence would exceed the limit, finalize current segment
            if len(current_segment) + len(full_sentence) > XTTS_MICRO_SEGMENT_CHARS:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = full_sentence
            else:
                current_segment += full_sentence
        
        # Add any remaining text
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        # Safety check: if any segment still exceeds limit, do a hard character split as last resort
        # This should rarely happen, but protects against extremely long sentences without punctuation
        final_segments = []
        for segment in segments:
            if len(segment) <= XTTS_MICRO_SEGMENT_CHARS:
                final_segments.append(segment)
            else:
                # Hard split by characters as absolute last resort
                log(f"  ⚠️ Segment exceeds {XTTS_MICRO_SEGMENT_CHARS} chars, performing hard split")
                for i in range(0, len(segment), XTTS_MICRO_SEGMENT_CHARS):
                    final_segments.append(segment[i:i + XTTS_MICRO_SEGMENT_CHARS])
        
        return final_segments


# ============================================================================
# AUDIO STITCHING WITH GLOBAL SPEED OPTIMIZATION
# ============================================================================

class AudioStitcher:
    """Stitch multiple WAV chunks into final audio with global speed optimization"""
    
    @staticmethod
    def stitch_chunks(chunks: List[ChunkMetadata], output_file: Path, script_type: str = 'long') -> bool:
        """
        Stitch chunk WAV files into single output with global speed optimization
        
        FIX 7: CORRECT PIPELINE ORDER:
        1. Load all chunks (already silence-trimmed)
        2. Concatenate with no gaps
        3. Apply global speed increase once for consistent pacing
        4. Final smoothing
        5. Save
        
        Args:
            chunks: List of chunk metadata
            output_file: Output file path
            script_type: 'long' or 'short'
            
        Returns:
            True if successful
        """
        if not XTTS_AVAILABLE:
            log("⚠️ XTTS not available - cannot stitch audio")
            return False
        
        log(f"🔗 Stitching {script_type} audio chunks...")
        
        completed_chunks = [c for c in chunks if c.status == ChunkStatus.COMPLETED.value and c.wav_path]
        
        if not completed_chunks:
            log("❌ No completed chunks to stitch")
            return False
        
        log(f"   Found {len(completed_chunks)} chunks to stitch")
        
        audio_segments = []
        expected_sample_rate = None
        total_original_duration = 0.0
        
        for chunk in sorted(completed_chunks, key=lambda x: x.chunk_id):
            wav_path = Path(chunk.wav_path)
            
            if not wav_path.exists():
                log(f"⚠️ Chunk {chunk.chunk_id} WAV not found: {wav_path}")
                continue
            
            rate, audio = read_wav(wav_path)
            chunk_duration = len(audio) / rate
            total_original_duration += chunk_duration
            
            if expected_sample_rate is None:
                expected_sample_rate = rate
            elif rate != expected_sample_rate:
                log(f"⚠️ Sample rate mismatch: {rate} vs {expected_sample_rate}")
            
            # Chunks are already silence-trimmed, just load them
            audio_segments.append(audio)
            log(f"  ✓ Loaded chunk {chunk.chunk_id}: {chunk_duration:.1f}s")
        
        if not audio_segments:
            log("❌ No audio segments loaded")
            return False
        
        log(f"\n📊 Original total duration: {total_original_duration:.1f}s")
        
        # STEP 1: Concatenate with no gaps or overlaps
        log("🔗 Concatenating segments...")
        concatenated_audio = np.concatenate(audio_segments, axis=0)
        
        # STEP 2: Apply global speed increase once for consistent pacing
        log(f"⚡ Applying global speed increase: {FINAL_SPEED_MULTIPLIER:.2f}x")
        sped_audio = increase_audio_speed(concatenated_audio, speed=FINAL_SPEED_MULTIPLIER)
        
        # STEP 3: Final smoothing pass
        log("✨ Applying final smoothing...")
        final_audio = smooth_audio(sped_audio)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Enforce XTTS sample rate
        write_wav(str(output_file), XTTS_SAMPLE_RATE, final_audio)
        
        final_duration = len(final_audio) / XTTS_SAMPLE_RATE
        log(f"✅ Final {script_type} audio: {final_duration:.1f}s ({final_duration/60:.2f} minutes)")
        log(f"   Speed improvement: {total_original_duration/final_duration:.2f}x")
        log(f"   Saved to: {output_file}")
        
        return True


# ============================================================================
# ORCHESTRATION
# ============================================================================

class AudioGenerationOrchestrator:
    """
    Main orchestration layer for audio generation workflow
    Supports both long and short scripts
    """
    
    def __init__(self, script_file: str, run_id: str, voice_preset: str = 'hi', script_type: str = 'long'):
        self.script_file = script_file
        self.run_id = run_id
        self.voice_preset = voice_preset
        self.script_type = script_type
        
        self.script_data: Optional[Dict] = None
        self.script_text: Optional[str] = None
        self.chunks: List[ChunkMetadata] = []
        
        self.session_manager: Optional[SessionManager] = None
        self.xtts_generator: Optional[XTTSAudioGenerator] = None
    
    def load_script(self):
        """Load script from JSON file based on script type"""
        log(f"📖 Loading {self.script_type} script: {self.script_file}")
        
        script_path = Path(self.script_file)
        if not script_path.exists():
            # Try default locations based on script type
            if self.script_type == 'short':
                default_path = OUTPUT_DIR / 'script_short.json'
            else:
                default_path = OUTPUT_DIR / 'script_long.json'
            
            if default_path.exists():
                self.script_file = str(default_path)
                log(f"📖 Using default location: {self.script_file}")
            else:
                raise FileNotFoundError(f"Script file not found: {self.script_file}")
        
        with open(self.script_file, 'r', encoding='utf-8') as f:
            self.script_data = json.load(f)
        
        # Extract script text based on structure
        if self.script_type == 'short':
            # Shorts script structure
            if 'script' in self.script_data:
                script_obj = self.script_data['script']
                # Use full_text if available, otherwise combine parts
                if 'full_text' in script_obj:
                    self.script_text = script_obj['full_text']
                else:
                    parts = []
                    for key in ['hook', 'content', 'cta']:
                        if key in script_obj:
                            parts.append(script_obj[key])
                    self.script_text = ' '.join(parts)
            else:
                raise ValueError("Short script missing 'script' field")
            
            log(f"✓ Loaded short script: {len(self.script_text)} characters")
            return
        
        # Long script structure
        if 'hindi_script' in self.script_data:
            self.script_text = self.script_data['hindi_script']
            log(f"✓ Loaded script (old format): {len(self.script_text)} characters")
            return
        
        if 'script' in self.script_data:
            log("📋 Detected new structured script format")
            self.script_text = self._assemble_script_from_sections(self.script_data['script'])
            log(f"✓ Assembled script from sections: {len(self.script_text)} characters")
            return
        
        raise ValueError("No valid script format found in JSON")
    
    def _assemble_script_from_sections(self, script_obj: Dict) -> str:
        """
        Assemble complete script text from structured sections
        
        Args:
            script_obj: Script object with sections
            
        Returns:
            Complete script text with all markers
        """
        sections = []
        
        # 1. Hook
        if 'hook' in script_obj:
            sections.append(script_obj['hook'])
        
        # 2. Problem Agitation
        if 'problem_agitation' in script_obj:
            sections.append(script_obj['problem_agitation'])
        
        # 3. Promise
        if 'promise' in script_obj:
            sections.append(script_obj['promise'])
        
        # 4. Main Content (array of sections)
        if 'main_content' in script_obj and isinstance(script_obj['main_content'], list):
            for section in script_obj['main_content']:
                if isinstance(section, dict) and 'content' in section:
                    sections.append(section['content'])
                elif isinstance(section, str):
                    sections.append(section)
        
        # 5. Practical Tips (array of tips)
        if 'practical_tips' in script_obj and isinstance(script_obj['practical_tips'], list):
            for tip in script_obj['practical_tips']:
                if isinstance(tip, dict):
                    # Assemble tip text
                    tip_text = ""
                    if 'tip_title' in tip:
                        tip_text += f"[PAUSE-1] {tip['tip_title']}. [PAUSE-1] "
                    if 'explanation' in tip:
                        tip_text += tip['explanation']
                    if tip_text:
                        sections.append(tip_text)
                elif isinstance(tip, str):
                    sections.append(tip)
        
        # 6. Conclusion
        if 'conclusion' in script_obj:
            sections.append(script_obj['conclusion'])
        
        # Join all sections with pause markers
        complete_script = ' [PAUSE-2] '.join(sections)
        
        log(f"  ✓ Assembled {len(sections)} sections into complete script")
        
        return complete_script
    
    def prepare_chunks(self) -> bool:
        """
        JOB A: Prepare chunks and save session (FAST - no audio generation)
        
        Returns:
            True if successful
        """
        log("=" * 80)
        log(f"JOB A: CHUNK PREPARATION ({self.script_type.upper()})")
        log("=" * 80)
        
        self.load_script()
        
        chunker = ScriptChunker(self.script_text, self.script_type)
        self.chunks = chunker.chunk_script()
        
        self.session_manager = SessionManager(self.run_id, self.script_file, self.script_type)
        self.session_manager.load_or_create_session(self.chunks, self.voice_preset)
        
        log(f"\n✅ Chunk preparation complete for {self.script_type}!")
        log(f"   Total chunks: {len(self.chunks)}")
        log(f"   Session saved to: {SESSION_FILE}")
        
        return True
    
    def generate_all_chunks(self, resume: bool = False) -> bool:
        """
        Legacy mode: Generate all chunks sequentially (NOT CI-SAFE FOR LONG VIDEOS)
        
        Args:
            resume: Whether to resume from existing session
            
        Returns:
            True if all chunks generated successfully
        """
        log("=" * 80)
        log(f"LEGACY MODE: SEQUENTIAL CHUNK GENERATION ({self.script_type.upper()})")
        if self.script_type == 'long':
            log("⚠️ WARNING: Not recommended for videos >10 minutes")
        log("=" * 80)
        
        self.load_script()
        
        if resume and SESSION_FILE.exists():
            log("📂 Resuming from existing session...")
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Verify script type matches
            if session_data.get('script_type') != self.script_type:
                log(f"⚠️ Session type mismatch: expected {self.script_type}, found {session_data.get('script_type')}")
                log("   Creating new session...")
                chunker = ScriptChunker(self.script_text, self.script_type)
                self.chunks = chunker.chunk_script()
            else:
                self.chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        else:
            chunker = ScriptChunker(self.script_text, self.script_type)
            self.chunks = chunker.chunk_script()
        
        self.session_manager = SessionManager(self.run_id, self.script_file, self.script_type)
        self.session_manager.load_or_create_session(self.chunks, self.voice_preset)
        
        self.xtts_generator = XTTSAudioGenerator(self.voice_preset)
        self.xtts_generator.load_model()
        
        log(f"\n🎬 Starting generation of {len(self.chunks)} chunks for {self.script_type}...")
        
        for chunk in self.chunks:
            if chunk.status == ChunkStatus.COMPLETED.value:
                log(f"\n⏭️ Skipping chunk {chunk.chunk_id} (already completed)")
                continue
            
            success = self._generate_single_chunk(chunk)
            
            if not success:
                log(f"\n❌ Generation failed for chunk {chunk.chunk_id}")
                return False
        
        log("\n" + "=" * 80)
        log(f"🔗 STITCHING FINAL {self.script_type.upper()} AUDIO")
        log("=" * 80)
        
        # Select appropriate output file
        if self.script_type == 'short':
            output_file = FINAL_AUDIO_FILE_SHORT
        else:
            output_file = FINAL_AUDIO_FILE_LONG
        
        success = AudioStitcher.stitch_chunks(self.chunks, output_file, self.script_type)
        
        if success and self.script_type == 'long':
            self._update_script_metadata()
        
        return success
    
    def generate_single_chunk_by_id(self, chunk_id: int) -> bool:
        """
        JOB B: Generate a single chunk by ID (CI-SAFE)
        
        Args:
            chunk_id: Chunk ID to generate
            
        Returns:
            True if successful
        """
        log("=" * 80)
        log(f"JOB B: SINGLE CHUNK GENERATION ({self.script_type.upper()} ID: {chunk_id})")
        log("=" * 80)
        
        if not SESSION_FILE.exists():
            log("❌ No session found - run JOB A first (--prepare)")
            return False
        
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Verify script type matches
        if session_data.get('script_type') != self.script_type:
            log(f"❌ Session type mismatch: expected {self.script_type}, found {session_data.get('script_type')}")
            log(f"   Please use --script-type {session_data.get('script_type')}")
            return False
        
        chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        
        if chunk_id < 0 or chunk_id >= len(chunks):
            log(f"❌ Invalid chunk ID: {chunk_id} (valid range: 0-{len(chunks)-1})")
            return False
        
        chunk = chunks[chunk_id]
        
        if chunk.status == ChunkStatus.COMPLETED.value:
            log(f"ℹ️ Chunk {chunk_id} already completed - skipping")
            return True
        
        self.session_manager = SessionManager(
            session_data['run_id'], 
            session_data['script_file'],
            session_data.get('script_type', self.script_type)
        )
        self.session_manager.session = SessionState(**session_data)
        
        self.xtts_generator = XTTSAudioGenerator(self.voice_preset)
        self.xtts_generator.load_model()
        
        return self._generate_single_chunk(chunk)
    
    def _generate_single_chunk(self, chunk: ChunkMetadata) -> bool:
        """
        Generate audio for a single chunk with CI-safe retry logic
        
        Args:
            chunk: Chunk metadata
            
        Returns:
            True if successful
        """
        log(f"\n🎙️ Chunk {chunk.chunk_id} ({self.script_type}):")
        log(f"   Estimated: {chunk.estimated_duration:.1f}s")
        log(f"   Text length: {len(chunk.text)} chars")
        log(f"   Retry count: {chunk.retries}")
        
        # SAFETY CHECK: Verify duration cap
        if chunk.estimated_duration > MAX_CHUNK_DURATION:
            log(f"   ⚠️ WARNING: Chunk exceeds {MAX_CHUNK_DURATION}s cap!")
        
        chunk.status = ChunkStatus.IN_PROGRESS.value
        chunk.timestamp = datetime.now().isoformat()
        self.session_manager.update_chunk_status(chunk)
        
        log(f"🚀 Starting generation (retry {chunk.retries})...")
        generation_start = time.time()
        
        audio_array = self.xtts_generator.generate_chunk_audio(chunk, retries=chunk.retries)
        
        generation_time = time.time() - generation_start
        
        if audio_array is not None:
            wav_path = CHUNKS_DIR / f"{self.script_type}_chunk_{chunk.chunk_id:03d}.wav"
            
            # Enforce XTTS sample rate
            write_wav(str(wav_path), XTTS_SAMPLE_RATE, audio_array)
            
            chunk.status = ChunkStatus.COMPLETED.value
            chunk.wav_path = str(wav_path)
            chunk.error = None
            
            actual_duration = len(audio_array) / XTTS_SAMPLE_RATE
            log(f"✅ Completed in {generation_time:.1f}s")
            log(f"   Chunk duration (pre-stitch): {actual_duration:.1f}s")
            log(f"   File: {wav_path.name}")
            
            self.session_manager.update_chunk_status(chunk)
            return True
        else:
            chunk.status = ChunkStatus.FAILED.value
            chunk.error = f"Generation failed after {MAX_RETRIES} attempts"
            chunk.retries += 1
            
            log(f"❌ Failed after {chunk.retries} total attempts")
            
            self.session_manager.update_chunk_status(chunk)
            return False
    
    def stitch_existing_chunks(self) -> bool:
        """
        JOB C: Stitch existing chunks into final audio with global speed optimization
        
        Returns:
            True if successful
        """
        log("=" * 80)
        log(f"JOB C: AUDIO STITCHING WITH GLOBAL SPEED OPTIMIZATION ({self.script_type.upper()})")
        log("=" * 80)
        
        if not SESSION_FILE.exists():
            log("❌ No session found - cannot stitch")
            return False
        
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Verify script type matches
        if session_data.get('script_type') != self.script_type:
            log(f"❌ Session type mismatch: expected {self.script_type}, found {session_data.get('script_type')}")
            return False
        
        chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        
        # Select appropriate output file
        if self.script_type == 'short':
            output_file = FINAL_AUDIO_FILE_SHORT
        else:
            output_file = FINAL_AUDIO_FILE_LONG
        
        success = AudioStitcher.stitch_chunks(chunks, output_file, self.script_type)
        
        if success and self.script_type == 'long':
            self.script_file = session_data['script_file']
            self.load_script()
            self._update_script_metadata()
        
        return success
    
    def print_status(self):
        """Print current session status"""
        if not SESSION_FILE.exists():
            log("ℹ️ No session found")
            return
        
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        script_type = session_data.get('script_type', 'unknown')
        
        log("\n" + "=" * 80)
        log(f"📊 SESSION STATUS ({script_type.upper()})")
        log("=" * 80)
        log(f"Run ID: {session_data['run_id']}")
        log(f"Script: {session_data['script_file']}")
        log(f"Script Type: {script_type}")
        log(f"Created: {session_data['created_at']}")
        log(f"Updated: {session_data['updated_at']}")
        log(f"\nProgress: {session_data['chunks_completed']}/{session_data['total_chunks']} completed")
        log(f"Failed: {session_data['chunks_failed']}")
        
        log("\nChunks:")
        for chunk in session_data['chunks']:
            status_icon = {
                'pending': '⏸️',
                'in_progress': '⏳',
                'completed': '✅',
                'failed': '❌'
            }.get(chunk['status'], '❓')
            
            log(f"  {status_icon} Chunk {chunk['chunk_id']}: {chunk['status']} "
                f"({chunk['estimated_duration']:.1f}s, retries: {chunk['retries']})")
            
            if chunk.get('wav_path') and Path(chunk['wav_path']).exists():
                log(f"      WAV: {chunk['wav_path']}")
        
        log("=" * 80 + "\n")
    
    def _update_script_metadata(self):
        """Update script JSON with audio metadata"""
        rate, audio = read_wav(FINAL_AUDIO_FILE_LONG)
        duration = len(audio) / rate
        
        self.script_data['audio_info'] = {
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}",
            'sample_rate': rate,
            'chunks_generated': len(self.chunks),
            'voice_preset': self.voice_preset,
            'generation_method': 'xtts_v2_sentence_based_chunking_ci_safe_studio_quality',
            'voice_clone_used': USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE),
            'generated_at': datetime.now().isoformat(),
            'fixes_applied': {
                'sentence_based_chunking': True,
                'strict_text_cleaning': True,
                'reduced_fragmentation': True,
                'forced_voice_cloning': True,
                'audio_smoothing': True,
                'sample_rate_enforced': True,
                'smooth_stitching': True,
                'youtube_optimized_pacing': True,
                'silence_trimmed': True,
                'global_speed_optimization': True,
                'global_speed_multiplier': FINAL_SPEED_MULTIPLIER,
                'chunk_level_speed_disabled': True,
                'voice_confidence_optimized': True
            }
        }
        
        with open(self.script_file, 'w', encoding='utf-8') as f:
            json.dump(self.script_data, f, ensure_ascii=False, indent=2)
        
        log(f"✅ Updated script metadata: {duration/60:.2f} minutes")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='CI/CD Safe XTTS v2 TTS Audio Generator - STUDIO QUALITY VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JOB A: Prepare chunks (fast, <1 min)
  python generate_audio.py --script-file script.json --run-id test_001 --prepare --script-type long
  python generate_audio.py --script-file script_short.json --run-id test_001 --prepare --script-type short
  
  # JOB B: Generate specific chunk (CI matrix job)
  python generate_audio.py --script-file script.json --run-id test_001 --chunk-id 0 --script-type long
  python generate_audio.py --script-file script_short.json --run-id test_001 --chunk-id 0 --script-type short
  
  # JOB C: Stitch existing chunks (applies global speed optimization)
  python generate_audio.py --stitch --script-type long
  python generate_audio.py --stitch --script-type short
  
  # Legacy: Generate all chunks (not CI-safe)
  python generate_audio.py --script-file script.json --run-id test_001 --script-type long
  python generate_audio.py --script-file script_short.json --run-id test_001 --script-type short
  
  # Resume from existing session
  python generate_audio.py --script-file script.json --run-id test_001 --resume --script-type long
  
  # Check status
  python generate_audio.py --status
        """
    )
    
    parser.add_argument('--script-file', type=str, help='Path to script.json')
    parser.add_argument('--run-id', type=str, help='Unique run identifier')
    parser.add_argument('--voice-preset', type=str, default='hi',
                       help='Voice preset/language (default: hi for Hindi)')
    parser.add_argument('--script-type', type=str, choices=['long', 'short'], default='long',
                       help='Script type: long (10-15 min) or short (45-60 sec)')
    
    # Mode selection
    parser.add_argument('--prepare', action='store_true',
                       help='JOB A: Prepare chunks only (fast)')
    parser.add_argument('--chunk-id', type=int, default=None,
                       help='JOB B: Generate specific chunk by ID (CI-safe)')
    parser.add_argument('--stitch', action='store_true',
                       help='JOB C: Stitch existing chunk WAVs with global speed optimization')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing session')
    parser.add_argument('--status', action='store_true',
                       help='Print session status')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.status:
        orchestrator = AudioGenerationOrchestrator('', '', script_type=args.script_type)
        orchestrator.print_status()
        return
    
    if args.stitch:
        orchestrator = AudioGenerationOrchestrator('', '', script_type=args.script_type)
        success = orchestrator.stitch_existing_chunks()
        sys.exit(0 if success else 1)
    
    # All other modes require script-file and run-id
    if not args.script_file or not args.run_id:
        parser.error('--script-file and --run-id are required (unless using --status or --stitch)')
    
    # Create orchestrator with script type
    orchestrator = AudioGenerationOrchestrator(
        args.script_file,
        args.run_id,
        args.voice_preset,
        args.script_type
    )
    
    # Execute based on mode
    try:
        if args.prepare:
            # JOB A: Chunk preparation
            success = orchestrator.prepare_chunks()
        elif args.chunk_id is not None:
            # JOB B: Single chunk generation
            success = orchestrator.generate_single_chunk_by_id(args.chunk_id)
        else:
            # Legacy: Full generation mode
            success = orchestrator.generate_all_chunks(resume=args.resume)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        log("\n\n⚠️ Generation interrupted by user")
        log("💾 Session saved - use --resume to continue")
        sys.exit(130)
    except Exception as e:
        log(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
