"""
Triton Client for XTTS v2
=========================

Client implementation for XTTS v2 running on Triton Server.
Supports multiple languages and voice cloning.
"""

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TritonXTTSClient:
    """Client for XTTS v2 on Triton Server."""
    
    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "xtts_v2",
        protocol: str = "grpc",
        timeout: float = 120.0
    ):
        """
        Initialize Triton client for XTTS.
        
        Args:
            server_url: Triton server URL
            model_name: Name of the model on Triton
            protocol: Communication protocol ("grpc" or "http")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url
        self.model_name = model_name
        self.protocol = protocol.lower()
        self.timeout = int(timeout)
        
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
        
        logger.info(f"Connected to XTTS v2 on Triton at {server_url}")
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_wav: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        length_penalty: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using XTTS v2.
        
        Args:
            text: Text to synthesize
            language: Language code (e.g., "en", "es", "fr", etc.)
            speaker_wav: Optional voice sample for cloning (16kHz mono)
            temperature: Generation temperature (0.0-1.0)
            length_penalty: Length penalty for generation
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Prepare inputs
        inputs = []
        
        if self.protocol == "grpc":
            # Text input
            text_data = np.array([text], dtype=object)
            text_input = grpcclient.InferInput("text", text_data.shape, "BYTES")
            text_input.set_data_from_numpy(text_data)
            inputs.append(text_input)
            
            # Language input
            lang_data = np.array([language], dtype=object)
            lang_input = grpcclient.InferInput("language", lang_data.shape, "BYTES")
            lang_input.set_data_from_numpy(lang_data)
            inputs.append(lang_input)
            
            # Optional speaker wav
            if speaker_wav is not None:
                # Ensure it's float32
                if speaker_wav.dtype != np.float32:
                    speaker_wav = speaker_wav.astype(np.float32) / 32768.0
                speaker_input = grpcclient.InferInput("speaker_wav", speaker_wav.shape, "FP32")
                speaker_input.set_data_from_numpy(speaker_wav)
                inputs.append(speaker_input)
            
            # Temperature
            temp_data = np.array([temperature], dtype=np.float32)
            temp_input = grpcclient.InferInput("temperature", temp_data.shape, "FP32")
            temp_input.set_data_from_numpy(temp_data)
            inputs.append(temp_input)
            
            # Length penalty
            len_data = np.array([length_penalty], dtype=np.float32)
            len_input = grpcclient.InferInput("length_penalty", len_data.shape, "FP32")
            len_input.set_data_from_numpy(len_data)
            inputs.append(len_input)
            
            # Outputs
            outputs = [
                grpcclient.InferRequestedOutput("audio"),
                grpcclient.InferRequestedOutput("sample_rate")
            ]
            
        else:  # http
            # Similar for HTTP protocol
            raise NotImplementedError("HTTP protocol not yet implemented")
        
        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self.timeout
        )
        
        # Extract results
        audio = response.as_numpy("audio")
        sample_rate = response.as_numpy("sample_rate")[0]
        
        return audio, sample_rate
    
    def health_check(self) -> dict:
        """Check server and model health."""
        return {
            "server_live": self.client.is_server_live(),
            "server_ready": self.client.is_server_ready(),
            "model_ready": self.client.is_model_ready(self.model_name)
        }


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = TritonXTTSClient(
        server_url="localhost:8001",
        model_name="xtts_v2"
    )
    
    # Check health
    print("Health check:", client.health_check())
    
    # Example synthesis
    audio, sample_rate = client.synthesize(
        text="Hello, this is a test of XTTS version 2.",
        language="en"
    )
    
    print(f"Generated audio: {len(audio)} samples at {sample_rate}Hz")