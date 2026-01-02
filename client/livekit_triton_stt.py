"""
LiveKit Triton STT Plugin
========================

LiveKit Agents plugin for using Whisper v3 Turbo via Triton Server.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from livekit.agents import stt, APIConnectOptions, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .triton_whisper_client import TritonWhisperClient

logger = logging.getLogger(__name__)


class TritonWhisperSTT(stt.STT):
    """
    LiveKit STT plugin using Whisper v3 Turbo on Triton Server.
    
    This plugin provides scalable speech recognition using NVIDIA Triton
    Inference Server, perfect for high-concurrency deployments.
    
    Args:
        server_url: Triton server URL (e.g., "localhost:8001" for gRPC)
        model_name: Model name on Triton (default: "whisper_v3_turbo")
        language: Language code or "auto" for detection
        protocol: Communication protocol ("grpc" or "http")
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """
    
    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "whisper_v3_turbo",
        language: Optional[str] = "auto",
        protocol: str = "grpc",
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False
            )
        )
        
        self.server_url = server_url
        self.model_name = model_name
        self.language = None if language == "auto" else language
        self.protocol = protocol
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Thread pool for synchronous Triton calls
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize Triton client
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Triton client with retry."""
        for attempt in range(self.max_retries):
            try:
                self._client = TritonWhisperClient(
                    server_url=self.server_url,
                    model_name=self.model_name,
                    protocol=self.protocol,
                    timeout=self.timeout
                )
                logger.info(f"Connected to Triton Whisper at {self.server_url}")
                return
            except Exception as e:
                logger.warning(f"Failed to connect (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        """Process audio buffer and return transcription."""
        
        # Convert AudioBuffer to numpy array
        if isinstance(buffer, list):
            all_data = []
            for frame in buffer:
                frame_data = np.frombuffer(frame.data, dtype=np.int16)
                all_data.append(frame_data)
            audio_data = np.concatenate(all_data)
        else:
            audio_data = np.frombuffer(buffer.data, dtype=np.int16)
        
        # Determine language
        lang = None
        if language is not NOT_GIVEN:
            lang = None if language == "auto" else language
        else:
            lang = self.language
        
        # Run inference with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.perf_counter()
                
                # Run blocking Triton call in thread pool
                loop = asyncio.get_running_loop()
                transcription, detected_language = await loop.run_in_executor(
                    self._executor,
                    self._client.transcribe,
                    audio_data,
                    lang
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                audio_duration = len(audio_data) / 16000
                
                logger.debug(
                    f"Triton Whisper latency: {elapsed_ms:.0f}ms for {audio_duration:.1f}s audio"
                )
                
                if transcription:
                    logger.debug(f"Transcribed ({detected_language}): {transcription[:100]}...")
                
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(
                        text=transcription,
                        start_time=0,
                        end_time=0,
                        language=detected_language
                    )],
                )
                
            except Exception as e:
                logger.warning(f"Triton inference failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Try to reinitialize client if connection failed
                    if "connect" in str(e).lower():
                        try:
                            self._init_client()
                        except:
                            pass
                    
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed: {e}")
                    # Return empty result instead of crashing
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[stt.SpeechData(
                            text="",
                            start_time=0,
                            end_time=0,
                            language=""
                        )],
                    )
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)