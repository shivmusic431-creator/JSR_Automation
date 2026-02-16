#!/usr/bin/env python3
"""
Enhanced Audio Generation with Coqui XTTS v2 - CI/CD Safe Version
Production-grade pause-aware, resumable audio generation system

Features:
- Intelligent pause-aware chunking (~2 min chunks)
- XTTS v2-safe micro-segmentation
- Resume-safe persistent session state
- Automatic retry logic (3 attempts per chunk)
- CLI modes: generate, resume, stitch, status, chunk-id
- CI/CD safe execution with heartbeat logging
- GitHub Actions watchdog protection
- Backwards compatible with existing pipeline
- Production-safe speaker_idx=0 for multi-speaker models
- Voice cloning support with my_voice.wav

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
# CONSTANTS
# ============================================================================

# Audio generation parameters - CRITICAL CI-SAFE LIMITS
TARGET_CHUNK_DURATION = 120  # seconds (~2 minutes)
MIN_CHUNK_DURATION = 90      # Lowered from 100 for more flexibility
MAX_CHUNK_DURATION = 150     # HARD CAP - NEVER EXCEED
WORDS_PER_MINUTE = 150       # Average speaking rate for Hindi
XTTS_MICRO_SEGMENT_CHARS = 500  # XTTS-safe character limit per generation
MAX_RETRIES = 3

# XTTS Configuration
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000  # XTTS v2 native sample rate
XTTS_LANGUAGE = "hi"  # Hindi language code

# Voice cloning configuration
VOICE_CLONE_FILE = "voices/my_voice.wav"
USE_VOICE_CLONE = True

# CI/CD Safety parameters
HEARTBEAT_INTERVAL = 20  # seconds - log activity every 20s
MAX_SILENT_TIME = 30  # seconds - never silent for more than 30s
MEMORY_CHECK_INTERVAL = 5  # Check memory every N micro-segments

# Pause markers (priority order - longer pauses preferred for chunking)
PAUSE_MARKERS = [
    r'\[PAUSE-3\]',  # 3 seconds - highest priority for chunking
    r'\[PAUSE-2\]',  # 2 seconds
    r'\[PAUSE-1\]',  # 1 second
]

# All markers to clean before synthesis
ALL_MARKERS = [
    r'\[PAUSE-1\]',
    r'\[PAUSE-2\]',
    r'\[PAUSE-3\]',
    r'\[EMPHASIS\]',
    r'\[WHISPER\]',
    r'\[EXCITED\]',
    r'\[SERIOUS\]',
    r'\[QUESTION\]',
    r'\[SCENE:[^\]]*\]',
    
    # REMOVE EMOTION INDICATORS
    r'\(धीरे से\)',
    r'\(गंभीर स्वर में\)',
    r'\(रहस्यमय स्वर में\)',
    r'\(उत्साह से\)',
    r'\(हल्की मुस्कान के साथ\)',
    r'\(फुसफुसाते हुए\)',
    r'\(आश्चर्य से\)',
    r'\(दुखी होकर\)',
    r'\(गुस्से में\)',
    r'\(प्यार से\)',
]


# Directories
OUTPUT_DIR = Path('output')
CHUNKS_DIR = OUTPUT_DIR / 'audio_chunks'
SESSION_FILE = CHUNKS_DIR / 'session.json'
FINAL_AUDIO_FILE = OUTPUT_DIR / 'audio.wav'


# ============================================================================
# DYNAMIC SPEAKER SELECTION
# ============================================================================

def select_best_male_speaker(tts):
    """
    Dynamically select best available male speaker.
    Never fails. Always returns valid speaker.
    Uses correct XTTS v2 speaker access path.
    """
    try:
        # Correct XTTS v2 speaker access path
        speaker_dict = tts.synthesizer.tts_model.speaker_manager.speakers
        available = list(speaker_dict.keys())
    except Exception as e:
        log(f"❌ Failed to fetch XTTS speakers: {e}")
        raise RuntimeError("XTTS speakers unavailable")
    
    if not available:
        raise RuntimeError("No XTTS speakers found")
    
    log(f"Available XTTS speakers: {available}")
    
    # Preferred male speakers priority
    preferred = [
        "Soft John",
        "Daniel",
        "Thomas",
        "Matthew",
        "David",
        "Ryan",
        "John"
    ]
    
    for p in preferred:
        if p in available:
            log(f"✅ Selected preferred speaker: {p}")
            return p
    
    fallback = available[0]
    log(f"⚠️ Using fallback speaker: {fallback}")
    return fallback


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

def estimate_duration_from_text(text: str) -> float:
    """
    Estimate audio duration from text using word count
    
    Args:
        text: Input text
        
    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    duration = (words / WORDS_PER_MINUTE) * 60
    return duration


def clean_text_for_synthesis(text: str) -> str:
    """
    Remove ALL non-spoken metadata safely before XTTS synthesis.
    This ensures emotion indicators are NOT spoken while preserving narration.
    """
    # Remove ALL emotion indicators like:
    # (गंभीर स्वर में), (धीरे से), (रहस्यमय स्वर में), etc.
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove scene markers like:
    # [SCENE: office_tension]
    text = re.sub(r'\[SCENE:[^\]]*\]', '', text)
    
    # Remove pause markers if present
    text = re.sub(r'\[PAUSE-[123]\]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def find_pause_markers(text: str) -> List[Tuple[int, str]]:
    """
    Find all pause marker positions in text
    
    Args:
        text: Script text
        
    Returns:
        List of (position, marker_type) tuples
    """
    markers = []
    
    for marker_pattern in PAUSE_MARKERS:
        for match in re.finditer(marker_pattern, text):
            marker_type = match.group(0)
            markers.append((match.start(), marker_type))
    
    markers.sort(key=lambda x: x[0])
    
    return markers


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


def increase_audio_speed(audio: np.ndarray, speed: float = 1.08) -> np.ndarray:
    """
    Increase voice speed slightly for more confident narration.
    Safe for XTTS output and preserves audio quality.
    """
    if speed <= 1.0:
        return audio
    
    indices = np.round(np.arange(0, len(audio), speed))
    indices = indices[indices < len(audio)].astype(int)
    
    return audio[indices]


# ============================================================================
# PRODUCTION-SAFE CHUNKING SYSTEM
# ============================================================================

class ScriptChunker:
    """
    Production-safe pause-aware script chunking system
    
    CRITICAL FIXES:
    - Deterministic iteration (no exponential accumulation)
    - Hard 150s duration cap enforcement
    - Safe pause-aware splitting with fallback
    - Chunk validation layer
    - Memory-safe operation
    """
    
    def __init__(self, script_text: str):
        self.script_text = script_text
        self.chunks: List[ChunkMetadata] = []
    
    def chunk_script(self) -> List[ChunkMetadata]:
        """
        Split script into pause-aware chunks with strict duration caps
        
        Returns:
            List of chunk metadata objects
        """
        log("📋 Analyzing script for production-safe chunking...")
        log(f"   Script length: {len(self.script_text)} chars")
        log(f"   Max chunk duration: {MAX_CHUNK_DURATION}s (HARD CAP)")
        
        # Find all pause markers
        pause_positions = find_pause_markers(self.script_text)
        
        if not pause_positions:
            log("⚠️ No pause markers found - using sentence-based chunking")
            return self._fallback_sentence_chunking()
        
        log(f"✓ Found {len(pause_positions)} pause markers")
        
        # Build chunks deterministically
        chunks = self._build_chunks_deterministic(pause_positions)
        
        # Validate all chunks
        chunks = self._validate_and_fix_chunks(chunks)
        
        log(f"✅ Created {len(chunks)} validated chunks")
        return chunks
    
    def _build_chunks_deterministic(self, pause_positions: List[Tuple[int, str]]) -> List[ChunkMetadata]:
        """
        CRITICAL: Deterministic chunk builder with no exponential accumulation
        
        Algorithm:
        1. Iterate through script ONCE
        2. Track current position (never go backwards)
        3. Accumulate text until pause marker + duration check
        4. Finalize chunk and reset accumulator
        5. NEVER re-append prior text
        
        Args:
            pause_positions: List of (position, marker) tuples
            
        Returns:
            List of chunks
        """
        chunks = []
        current_pos = 0  # Current position in script
        chunk_start_pos = 0  # Start of current chunk being built
        
        log("🔨 Building chunks deterministically...")
        
        for i, (marker_pos, marker_type) in enumerate(pause_positions):
            # Calculate end position (after the marker)
            marker_len = len(marker_type)
            segment_end = marker_pos + marker_len
            
            # Extract candidate chunk text (from chunk start to current marker)
            candidate_text = self.script_text[chunk_start_pos:segment_end].strip()
            
            if not candidate_text:
                continue
            
            # Estimate duration
            estimated_duration = estimate_duration_from_text(candidate_text)
            
            log(f"  Evaluating at marker {i+1}/{len(pause_positions)}: "
                f"{estimated_duration:.1f}s, {len(candidate_text)} chars")
            
            # Decision logic with HARD CAP enforcement
            should_finalize = False
            reason = ""
            
            # CRITICAL: Hard cap enforcement
            if estimated_duration > MAX_CHUNK_DURATION:
                should_finalize = True
                reason = f"HARD CAP EXCEEDED ({estimated_duration:.1f}s > {MAX_CHUNK_DURATION}s)"
            
            # Normal chunking logic
            elif estimated_duration >= MIN_CHUNK_DURATION:
                if marker_type in [r'\[PAUSE-3\]', r'\[PAUSE-2\]']:
                    should_finalize = True
                    reason = f"Good pause point ({marker_type})"
                elif estimated_duration >= TARGET_CHUNK_DURATION:
                    should_finalize = True
                    reason = f"Target duration reached"
            
            # Last marker - always finalize
            if i == len(pause_positions) - 1 and not should_finalize:
                should_finalize = True
                reason = "Last marker"
            
            # Finalize chunk if conditions met
            if should_finalize:
                chunk_id = len(chunks)
                
                # SAFETY: Final duration check before creating chunk
                if estimated_duration > MAX_CHUNK_DURATION:
                    log(f"  ⚠️ WARNING: Chunk {chunk_id} exceeds cap, attempting split...")
                    # Try to split at sentence boundary
                    split_chunks = self._emergency_split_chunk(
                        candidate_text, 
                        chunk_start_pos,
                        chunk_id
                    )
                    chunks.extend(split_chunks)
                else:
                    # Create normal chunk
                    chunk = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=candidate_text,
                        estimated_duration=estimated_duration,
                        status=ChunkStatus.PENDING.value,
                        retries=0,
                        error=None,
                        wav_path=None,
                        timestamp=None
                    )
                    chunks.append(chunk)
                    log(f"  ✅ Chunk {chunk_id}: {estimated_duration:.1f}s, "
                        f"{len(candidate_text)} chars - {reason}")
                
                # CRITICAL: Move to next chunk start (NO OVERLAP)
                chunk_start_pos = segment_end
                current_pos = segment_end
        
        # Handle any remaining text after last pause marker
        if chunk_start_pos < len(self.script_text):
            remaining_text = self.script_text[chunk_start_pos:].strip()
            if remaining_text:
                estimated = estimate_duration_from_text(remaining_text)
                
                # Check duration cap
                if estimated > MAX_CHUNK_DURATION:
                    log(f"  ⚠️ Remaining text exceeds cap, splitting...")
                    split_chunks = self._emergency_split_chunk(
                        remaining_text,
                        chunk_start_pos,
                        len(chunks)
                    )
                    chunks.extend(split_chunks)
                else:
                    chunk_id = len(chunks)
                    chunk = ChunkMetadata(
                        chunk_id=chunk_id,
                        text=remaining_text,
                        estimated_duration=estimated,
                        status=ChunkStatus.PENDING.value,
                        retries=0,
                        error=None,
                        wav_path=None,
                        timestamp=None
                    )
                    chunks.append(chunk)
                    log(f"  ✅ Final chunk {chunk_id}: {estimated:.1f}s")
        
        return chunks
    
    def _emergency_split_chunk(self, text: str, start_pos: int, base_chunk_id: int) -> List[ChunkMetadata]:
        """
        Emergency splitter for chunks that exceed MAX_CHUNK_DURATION
        Splits at sentence boundaries to stay under cap
        
        Args:
            text: Text to split
            start_pos: Starting position in original script
            base_chunk_id: Base chunk ID for numbering
            
        Returns:
            List of split chunks
        """
        log(f"  🚨 Emergency split activated for {len(text)} chars")
        
        sentences = re.split(r'([।?!]+)', text)
        split_chunks = []
        current_text = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            
            if not sentence:
                continue
            
            full_sentence = sentence + delimiter
            candidate = (current_text + " " + full_sentence) if current_text else full_sentence
            duration = estimate_duration_from_text(candidate)
            
            # If adding this sentence would exceed cap, finalize current chunk
            if duration > MAX_CHUNK_DURATION and current_text:
                chunk_id = base_chunk_id + len(split_chunks)
                split_chunks.append(ChunkMetadata(
                    chunk_id=chunk_id,
                    text=current_text.strip(),
                    estimated_duration=estimate_duration_from_text(current_text),
                    status=ChunkStatus.PENDING.value,
                    retries=0,
                    error=None,
                    wav_path=None,
                    timestamp=None
                ))
                log(f"    Split chunk {chunk_id}: {split_chunks[-1].estimated_duration:.1f}s")
                current_text = full_sentence
            else:
                current_text = candidate
        
        # Handle remaining text
        if current_text.strip():
            chunk_id = base_chunk_id + len(split_chunks)
            split_chunks.append(ChunkMetadata(
                chunk_id=chunk_id,
                text=current_text.strip(),
                estimated_duration=estimate_duration_from_text(current_text),
                status=ChunkStatus.PENDING.value,
                retries=0,
                error=None,
                wav_path=None,
                timestamp=None
            ))
            log(f"    Split chunk {chunk_id}: {split_chunks[-1].estimated_duration:.1f}s")
        
        return split_chunks
    
    def _validate_and_fix_chunks(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """
        Final validation pass - ensures no chunks exceed hard caps
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validated and fixed chunks
        """
        log("🔍 Validating chunks...")
        
        validated_chunks = []
        issues_found = 0
        
        for i, chunk in enumerate(chunks):
            # Check 1: Duration cap
            if chunk.estimated_duration > MAX_CHUNK_DURATION:
                log(f"  ❌ Chunk {i} exceeds cap: {chunk.estimated_duration:.1f}s > {MAX_CHUNK_DURATION}s")
                issues_found += 1
                # Re-split this chunk
                split_chunks = self._emergency_split_chunk(chunk.text, 0, len(validated_chunks))
                validated_chunks.extend(split_chunks)
                continue
            
            # Check 2: Empty text
            if not chunk.text.strip():
                log(f"  ⚠️ Chunk {i} is empty - skipping")
                issues_found += 1
                continue
            
            # Check 3: Reasonable text length (sanity check)
            if len(chunk.text) > 5000:  # ~5000 chars is very long
                log(f"  ⚠️ Chunk {i} has unusual length: {len(chunk.text)} chars")
            
            # Chunk is valid
            validated_chunks.append(chunk)
        
        # Re-index chunks sequentially
        for i, chunk in enumerate(validated_chunks):
            chunk.chunk_id = i
        
        if issues_found > 0:
            log(f"  ⚠️ Fixed {issues_found} issues during validation")
        else:
            log(f"  ✅ All chunks validated successfully")
        
        # Final sanity check - print chunk statistics
        total_duration = sum(c.estimated_duration for c in validated_chunks)
        max_duration = max(c.estimated_duration for c in validated_chunks) if validated_chunks else 0
        
        log(f"  📊 Chunk statistics:")
        log(f"     Total chunks: {len(validated_chunks)}")
        log(f"     Total estimated duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        log(f"     Max chunk duration: {max_duration:.1f}s")
        log(f"     Avg chunk duration: {total_duration/len(validated_chunks):.1f}s")
        
        return validated_chunks
    
    def _fallback_sentence_chunking(self) -> List[ChunkMetadata]:
        """
        Fallback to sentence-based chunking when no pause markers exist
        WITH STRICT DURATION CAP ENFORCEMENT
        """
        log("🔄 Using fallback sentence-based chunking...")
        
        sentences = re.split(r'([।?!]+)', self.script_text)
        chunks = []
        current_text = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            
            if not sentence:
                continue
            
            full_sentence = sentence + delimiter
            candidate = (current_text + " " + full_sentence) if current_text else full_sentence
            duration = estimate_duration_from_text(candidate)
            
            # CRITICAL: Hard cap check
            if duration > MAX_CHUNK_DURATION:
                # Finalize current chunk if it exists
                if current_text:
                    chunk_id = len(chunks)
                    chunks.append(ChunkMetadata(
                        chunk_id=chunk_id,
                        text=current_text.strip(),
                        estimated_duration=estimate_duration_from_text(current_text),
                        status=ChunkStatus.PENDING.value,
                        retries=0,
                        error=None,
                        wav_path=None,
                        timestamp=None
                    ))
                    log(f"  Chunk {chunk_id}: {chunks[-1].estimated_duration:.1f}s")
                
                # Start new chunk with current sentence
                current_text = full_sentence
            
            # Normal accumulation within cap
            elif duration >= MIN_CHUNK_DURATION:
                chunk_id = len(chunks)
                chunks.append(ChunkMetadata(
                    chunk_id=chunk_id,
                    text=candidate.strip(),
                    estimated_duration=duration,
                    status=ChunkStatus.PENDING.value,
                    retries=0,
                    error=None,
                    wav_path=None,
                    timestamp=None
                ))
                log(f"  Chunk {chunk_id}: {duration:.1f}s")
                current_text = ""
            else:
                current_text = candidate
        
        # Handle remaining text
        if current_text.strip():
            chunk_id = len(chunks)
            duration = estimate_duration_from_text(current_text)
            chunks.append(ChunkMetadata(
                chunk_id=chunk_id,
                text=current_text.strip(),
                estimated_duration=duration,
                status=ChunkStatus.PENDING.value,
                retries=0,
                error=None,
                wav_path=None,
                timestamp=None
            ))
            log(f"  Chunk {chunk_id}: {duration:.1f}s")
        
        return chunks


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

class SessionManager:
    """
    Persistent session state management for resume capability
    """
    
    def __init__(self, run_id: str, script_file: str):
        self.run_id = run_id
        self.script_file = script_file
        self.session: Optional[SessionState] = None
        
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_or_create_session(self, chunks: List[ChunkMetadata], voice_preset: str) -> SessionState:
        """Load existing session or create new one"""
        if SESSION_FILE.exists():
            log("📂 Loading existing session...")
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data['run_id'] == self.run_id:
                log("✓ Session loaded successfully")
                chunks_data = [ChunkMetadata(**c) for c in data['chunks']]
                
                self.session = SessionState(
                    run_id=data['run_id'],
                    script_file=data['script_file'],
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
                log("⚠️ Different run_id - creating new session")
        
        log("📝 Creating new session...")
        self.session = SessionState(
            run_id=self.run_id,
            script_file=self.script_file,
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
# XTTS v2 AUDIO GENERATION (CI-SAFE) WITH DYNAMIC SPEAKER SELECTION
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
    - Uses speaker_idx=0 for production-safe multi-speaker model inference
    - Supports voice cloning via my_voice.wav
    """
    
    def __init__(self, voice_preset: str = 'hi'):
        self.voice_preset = voice_preset
        self.language = XTTS_LANGUAGE
        self.tts: Optional[TTS] = None
        self.model_loaded = False
        self.heartbeat = HeartbeatLogger()
        self.speaker = None
    
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
        
        self.heartbeat.start("Loading XTTS v2 model...")
        
        try:
            self.tts = TTS(XTTS_MODEL_NAME)

            # CRITICAL SAFETY CHECKS
            if not hasattr(self.tts, "synthesizer"):
                raise RuntimeError("XTTS synthesizer missing")

            if not hasattr(self.tts.synthesizer, "tts_model"):
                raise RuntimeError("XTTS tts_model missing")

            if not hasattr(self.tts.synthesizer.tts_model, "speaker_manager"):
                raise RuntimeError("XTTS speaker manager missing")

            # Voice cloning takes priority over built-in speakers
            if USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE):
                self.speaker = None
                log(f"✅ Using voice clone file: {VOICE_CLONE_FILE}")
            else:
                # Dynamically select best available male speaker
                self.speaker = select_best_male_speaker(self.tts)
                log(f"✅ Using XTTS speaker: {self.speaker}")

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
        
        clean_text = clean_text_for_synthesis(chunk.text)
        
        for attempt in range(MAX_RETRIES):
            try:
                log(f"🎙️ Attempt {attempt + 1}/{MAX_RETRIES} for chunk {chunk.chunk_id}")
                
                audio_array = self._generate_with_heartbeat(clean_text, chunk.chunk_id)
                
                if audio_array is not None:
                    log(f"✅ Generation successful on attempt {attempt + 1}")
                    memory_cleanup()
                    return audio_array
                
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
        
        Args:
            text: Clean text to synthesize
            chunk_id: Chunk identifier
            
        Returns:
            Audio array or None
        """
        segments = self._split_into_micro_segments(text)
        log(f"📝 Split into {len(segments)} micro-segments")
        
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
                
                # XTTS v2 inference with dynamic speaker and language parameters
                try:
                    # Use cloned voice if available
                    if USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE):
                        wav = self.tts.tts(
                            text=segment,
                            speaker_wav=VOICE_CLONE_FILE,
                            language=self.language
                        )
                    else:
                        wav = self.tts.tts(
                            text=segment,
                            speaker=self.speaker,
                            language=self.language
                        )
                except KeyError:
                    if USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE):
                        log(f"⚠️ Voice clone failed. Attempting with built-in speaker...")
                        # Fallback to built-in speaker if cloning fails
                        self.speaker = select_best_male_speaker(self.tts)
                        wav = self.tts.tts(
                            text=segment,
                            speaker=self.speaker,
                            language=self.language
                        )
                    else:
                        log(f"⚠️ Speaker {self.speaker} missing. Selecting new speaker...")
                        self.speaker = select_best_male_speaker(self.tts)
                        # Use cloned voice if available (with retry)
                        if USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE):
                            wav = self.tts.tts(
                                text=segment,
                                speaker_wav=VOICE_CLONE_FILE,
                                language=self.language
                            )
                        else:
                            wav = self.tts.tts(
                                text=segment,
                                speaker=self.speaker,
                                language=self.language
                            )
                except Exception as e:
                    log(f"⚠️ Generation failed: {e}")

                    # Always reselect speaker to avoid broken speaker state
                    try:
                        new_speaker = select_best_male_speaker(self.tts)

                        if new_speaker != self.speaker:
                            log(f"🔁 Switching speaker: {self.speaker} → {new_speaker}")

                        self.speaker = new_speaker

                    except Exception as speaker_error:
                        log(f"❌ Speaker reselection failed: {speaker_error}")
                        raise

                    # Use cloned voice if available (with retry)
                    if USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE):
                        wav = self.tts.tts(
                            text=segment,
                            speaker_wav=VOICE_CLONE_FILE,
                            language=self.language
                        )
                    else:
                        wav = self.tts.tts(
                            text=segment,
                            speaker=self.speaker,
                            language=self.language
                        )

                # Convert to numpy array (XTTS returns list)
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)
                
                # Convert from float32 [-1, 1] to int16 for WAV format
                audio_int16 = (wav * 32767).astype(np.int16)
                audio_arrays.append(audio_int16)
                
                segment_time = time.time() - segment_start
                log(f"  ✓ Segment {i+1} completed in {segment_time:.1f}s")
                
                if (i + 1) % MEMORY_CHECK_INTERVAL == 0:
                    log(f"  🧹 Memory cleanup after {i+1} segments")
                    memory_cleanup()
            
            log(f"🔗 Concatenating {len(audio_arrays)} segments...")
            final_audio = np.concatenate(audio_arrays)
            
            # Apply confident voice speed boost
            final_audio = increase_audio_speed(final_audio, speed=1.08)
            
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
        
        Args:
            text: Text to split
            
        Returns:
            List of segments
        """
        sentences = re.split(r'([।?!]+)', text)
        segments = []
        current_segment = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            full_sentence = sentence + delimiter
            
            if len(current_segment) + len(full_sentence) > XTTS_MICRO_SEGMENT_CHARS:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = full_sentence
            else:
                current_segment += full_sentence
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments


# ============================================================================
# AUDIO STITCHING
# ============================================================================

class AudioStitcher:
    """Stitch multiple WAV chunks into final audio"""
    
    @staticmethod
    def stitch_chunks(chunks: List[ChunkMetadata], output_file: Path) -> bool:
        """
        Stitch chunk WAV files into single output
        
        Args:
            chunks: List of chunk metadata
            output_file: Output file path
            
        Returns:
            True if successful
        """
        if not XTTS_AVAILABLE:
            log("⚠️ XTTS not available - cannot stitch audio")
            return False
        
        log("🔗 Stitching audio chunks...")
        
        completed_chunks = [c for c in chunks if c.status == ChunkStatus.COMPLETED.value and c.wav_path]
        
        if not completed_chunks:
            log("❌ No completed chunks to stitch")
            return False
        
        log(f"   Found {len(completed_chunks)} chunks to stitch")
        
        audio_segments = []
        
        for chunk in sorted(completed_chunks, key=lambda x: x.chunk_id):
            wav_path = Path(chunk.wav_path)
            
            if not wav_path.exists():
                log(f"⚠️ Chunk {chunk.chunk_id} WAV not found: {wav_path}")
                continue
            
            rate, audio = read_wav(wav_path)
            audio_segments.append(audio)
            log(f"  ✓ Loaded chunk {chunk.chunk_id}: {len(audio)/rate:.1f}s")
        
        if not audio_segments:
            log("❌ No audio segments loaded")
            return False
        
        log("🔗 Concatenating segments...")
        final_audio = np.concatenate(audio_segments)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(output_file), XTTS_SAMPLE_RATE, final_audio)
        
        duration = len(final_audio) / XTTS_SAMPLE_RATE
        log(f"✅ Final audio: {duration:.1f}s ({duration/60:.2f} minutes)")
        log(f"   Saved to: {output_file}")
        
        return True


# ============================================================================
# ORCHESTRATION
# ============================================================================

class AudioGenerationOrchestrator:
    """
    Main orchestration layer for audio generation workflow
    """
    
    def __init__(self, script_file: str, run_id: str, voice_preset: str = 'hi'):
        self.script_file = script_file
        self.run_id = run_id
        self.voice_preset = voice_preset
        
        self.script_data: Optional[Dict] = None
        self.script_text: Optional[str] = None
        self.chunks: List[ChunkMetadata] = []
        
        self.session_manager: Optional[SessionManager] = None
        self.xtts_generator: Optional[XTTSAudioGenerator] = None
    
    def load_script(self):
        """Load script from JSON file (supports old and new formats)"""
        log(f"📖 Loading script: {self.script_file}")
        
        with open(self.script_file, 'r', encoding='utf-8') as f:
            self.script_data = json.load(f)
        
        # Check for old format (single hindi_script field)
        if 'hindi_script' in self.script_data:
            self.script_text = self.script_data['hindi_script']
            log(f"✓ Loaded script (old format): {len(self.script_text)} characters")
            return
        
        # Check for new format (structured script sections)
        if 'script' in self.script_data:
            log("📋 Detected new structured script format")
            self.script_text = self._assemble_script_from_sections(self.script_data['script'])
            log(f"✓ Assembled script from sections: {len(self.script_text)} characters")
            return
        
        # No valid format found
        raise ValueError("No valid script format found in JSON (expected 'hindi_script' or 'script' field)")
    
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
        log("JOB A: CHUNK PREPARATION")
        log("=" * 80)
        
        self.load_script()
        
        chunker = ScriptChunker(self.script_text)
        self.chunks = chunker.chunk_script()
        
        self.session_manager = SessionManager(self.run_id, self.script_file)
        self.session_manager.load_or_create_session(self.chunks, self.voice_preset)
        
        log(f"\n✅ Chunk preparation complete!")
        log(f"   Total chunks: {len(self.chunks)}")
        log(f"   Session saved to: {SESSION_FILE}")
        log(f"\nNext steps:")
        log(f"   1. Run JOB B for each chunk (parallel safe)")
        log(f"   2. Run JOB C to stitch final audio")
        
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
        log("LEGACY MODE: SEQUENTIAL CHUNK GENERATION")
        log("⚠️ WARNING: Not recommended for videos >10 minutes")
        log("=" * 80)
        
        self.load_script()
        
        if resume and SESSION_FILE.exists():
            log("📂 Resuming from existing session...")
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            self.chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        else:
            chunker = ScriptChunker(self.script_text)
            self.chunks = chunker.chunk_script()
        
        self.session_manager = SessionManager(self.run_id, self.script_file)
        self.session_manager.load_or_create_session(self.chunks, self.voice_preset)
        
        self.xtts_generator = XTTSAudioGenerator(self.voice_preset)
        self.xtts_generator.load_model()
        
        log(f"\n🎬 Starting generation of {len(self.chunks)} chunks...")
        
        for chunk in self.chunks:
            if chunk.status == ChunkStatus.COMPLETED.value:
                log(f"\n⏭️ Skipping chunk {chunk.chunk_id} (already completed)")
                continue
            
            success = self._generate_single_chunk(chunk)
            
            if not success:
                log(f"\n❌ Generation failed for chunk {chunk.chunk_id}")
                return False
        
        log("\n" + "=" * 80)
        log("🔗 STITCHING FINAL AUDIO")
        log("=" * 80)
        
        success = AudioStitcher.stitch_chunks(self.chunks, FINAL_AUDIO_FILE)
        
        if success:
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
        log(f"JOB B: SINGLE CHUNK GENERATION (ID: {chunk_id})")
        log("=" * 80)
        
        if not SESSION_FILE.exists():
            log("❌ No session found - run JOB A first (--prepare)")
            return False
        
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        
        if chunk_id < 0 or chunk_id >= len(chunks):
            log(f"❌ Invalid chunk ID: {chunk_id} (valid range: 0-{len(chunks)-1})")
            return False
        
        chunk = chunks[chunk_id]
        
        if chunk.status == ChunkStatus.COMPLETED.value:
            log(f"ℹ️ Chunk {chunk_id} already completed - skipping")
            return True
        
        self.session_manager = SessionManager(session_data['run_id'], session_data['script_file'])
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
        log(f"\n🎙️ Chunk {chunk.chunk_id}:")
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
            wav_path = CHUNKS_DIR / f"chunk_{chunk.chunk_id:03d}.wav"
            write_wav(str(wav_path), XTTS_SAMPLE_RATE, audio_array)
            
            chunk.status = ChunkStatus.COMPLETED.value
            chunk.wav_path = str(wav_path)
            chunk.error = None
            
            actual_duration = len(audio_array) / XTTS_SAMPLE_RATE
            log(f"✅ Completed in {generation_time:.1f}s")
            log(f"   Audio duration: {actual_duration:.1f}s")
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
        JOB C: Stitch existing chunks into final audio
        
        Returns:
            True if successful
        """
        log("=" * 80)
        log("JOB C: AUDIO STITCHING")
        log("=" * 80)
        
        if not SESSION_FILE.exists():
            log("❌ No session found - cannot stitch")
            return False
        
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        chunks = [ChunkMetadata(**c) for c in session_data['chunks']]
        
        success = AudioStitcher.stitch_chunks(chunks, FINAL_AUDIO_FILE)
        
        if success:
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
        
        log("\n" + "=" * 80)
        log("📊 SESSION STATUS")
        log("=" * 80)
        log(f"Run ID: {session_data['run_id']}")
        log(f"Script: {session_data['script_file']}")
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
        rate, audio = read_wav(FINAL_AUDIO_FILE)
        duration = len(audio) / rate
        
        self.script_data['audio_info'] = {
            'duration_seconds': duration,
            'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}",
            'sample_rate': rate,
            'chunks_generated': len(self.chunks),
            'voice_preset': self.voice_preset,
            'generation_method': 'xtts_v2_pause_aware_chunked_ci_safe',
            'voice_clone_used': USE_VOICE_CLONE and os.path.exists(VOICE_CLONE_FILE),
            'generated_at': datetime.now().isoformat()
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
        description='CI/CD Safe XTTS v2 TTS Audio Generator - PRODUCTION VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # JOB A: Prepare chunks (fast, <1 min)
  python generate_audio_xtts.py --script-file script.json --run-id test_001 --prepare
  
  # JOB B: Generate specific chunk (CI matrix job)
  python generate_audio_xtts.py --script-file script.json --run-id test_001 --chunk-id 0
  python generate_audio_xtts.py --script-file script.json --run-id test_001 --chunk-id 1
  
  # JOB C: Stitch existing chunks
  python generate_audio_xtts.py --stitch
  
  # Legacy: Generate all chunks (not CI-safe)
  python generate_audio_xtts.py --script-file script.json --run-id test_001
  
  # Resume from existing session
  python generate_audio_xtts.py --script-file script.json --run-id test_001 --resume
  
  # Check status
  python generate_audio_xtts.py --status
        """
    )
    
    parser.add_argument('--script-file', type=str, help='Path to script.json')
    parser.add_argument('--run-id', type=str, help='Unique run identifier')
    parser.add_argument('--voice-preset', type=str, default='hi',
                       help='Voice preset/language (default: hi for Hindi)')
    
    # Mode selection
    parser.add_argument('--prepare', action='store_true',
                       help='JOB A: Prepare chunks only (fast)')
    parser.add_argument('--chunk-id', type=int, default=None,
                       help='JOB B: Generate specific chunk by ID (CI-safe)')
    parser.add_argument('--stitch', action='store_true',
                       help='JOB C: Stitch existing chunk WAVs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing session')
    parser.add_argument('--status', action='store_true',
                       help='Print session status')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.status:
        orchestrator = AudioGenerationOrchestrator('', '')
        orchestrator.print_status()
        return
    
    if args.stitch:
        orchestrator = AudioGenerationOrchestrator('', '')
        success = orchestrator.stitch_existing_chunks()
        sys.exit(0 if success else 1)
    
    # All other modes require script-file and run-id
    if not args.script_file or not args.run_id:
        parser.error('--script-file and --run-id are required (unless using --status or --stitch)')
    
    # Create orchestrator
    orchestrator = AudioGenerationOrchestrator(
        args.script_file,
        args.run_id,
        args.voice_preset
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
