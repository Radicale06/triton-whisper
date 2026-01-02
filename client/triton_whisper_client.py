"""
Triton Client for Whisper v3 Turbo
==================================

Client implementation for communicating with Whisper v3 Turbo
running on NVIDIA Triton Inference Server.
"""

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TritonWhisperClient:
    """Client for Whisper v3 Turbo on Triton Server."""
    
    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "whisper_v3_turbo",
        protocol: str = "grpc",
        timeout: float = 120.0
    ):
        """
        Initialize Triton client.
        
        Args:
            server_url: Triton server URL
            model_name: Name of the model on Triton
            protocol: Communication protocol ("grpc" or "http")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.model_name = model_name
        self.protocol = protocol.lower()
        self.timeout = timeout
        
        # Initialize client
        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=self.server_url,
                verbose=False
            )
        else:
            self.client = httpclient.InferenceServerClient(
                url=self.server_url,
                verbose=False
            )
        
        # Verify connection
        if not self.client.is_server_live():
            raise ConnectionError(f"Cannot connect to Triton server at {server_url}")
        
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"Model '{model_name}' is not ready on server")
        
        logger.info(f"Connected to Triton server at {server_url}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        sample_rate: int = 16000
    ) -> Tuple[str, str]:
        """
        Transcribe audio using Whisper v3 Turbo.
        
        Args:
            audio: Audio data as numpy array (int16 or float32)
            language: Language code (e.g., "en", "es") or None for auto-detect
            sample_rate: Audio sample rate (must be 16000)
            
        Returns:
            Tuple of (transcription, detected_language)
        """
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio")
        
        # Convert audio to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Prepare inputs
        inputs = []
        outputs = []
        
        if self.protocol == "grpc":
            # Audio input
            audio_input = grpcclient.InferInput("audio", audio.shape, "FP32")
            audio_input.set_data_from_numpy(audio)
            inputs.append(audio_input)
            
            # Language input (optional)
            if language:
                lang_data = np.array([language], dtype=object)
                lang_input = grpcclient.InferInput("language", lang_data.shape, "BYTES")
                lang_input.set_data_from_numpy(lang_data)
                inputs.append(lang_input)
            
            # Outputs
            outputs.append(grpcclient.InferRequestedOutput("transcription"))
            outputs.append(grpcclient.InferRequestedOutput("detected_language"))
            
        else:  # http
            # Audio input
            audio_input = httpclient.InferInput("audio", audio.shape, "FP32")
            audio_input.set_data_from_numpy(audio)
            inputs.append(audio_input)
            
            # Language input (optional)
            if language:
                lang_data = np.array([language], dtype=object)
                lang_input = httpclient.InferInput("language", lang_data.shape, "BYTES")
                lang_input.set_data_from_numpy(lang_data)
                inputs.append(lang_input)
            
            # Outputs
            outputs.append(httpclient.InferRequestedOutput("transcription"))
            outputs.append(httpclient.InferRequestedOutput("detected_language"))
        
        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self.timeout
        )
        
        # Extract results
        transcription = response.as_numpy("transcription")[0].decode('utf-8')
        detected_language = response.as_numpy("detected_language")[0].decode('utf-8')
        
        return transcription, detected_language
    
    def health_check(self) -> dict:
        """Check server and model health."""
        return {
            "server_live": self.client.is_server_live(),
            "server_ready": self.client.is_server_ready(),
            "model_ready": self.client.is_model_ready(self.model_name)
        }


# Example usage
if __name__ == "__main__":
    import wave
    
    # Initialize client
    client = TritonWhisperClient(
        server_url="localhost:8001",
        model_name="whisper_v3_turbo",
        protocol="grpc"
    )
    
    # Check health
    print("Health check:", client.health_check())
    
    # Example: Load and transcribe audio
    # with wave.open("sample.wav", "rb") as wav_file:
    #     frames = wav_file.readframes(wav_file.getnframes())
    #     audio = np.frombuffer(frames, dtype=np.int16)
    #     
    #     transcription, language = client.transcribe(audio)
    #     print(f"Transcription: {transcription}")
    #     print(f"Language: {language}")