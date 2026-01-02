"""
Triton Python Backend Model for XTTS v2
=======================================

Multi-language, multi-voice TTS using Coqui XTTS v2.
Supports voice cloning and all major languages.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import logging
import os
import time
from threading import Lock


class TritonPythonModel:
    """XTTS v2 model for Triton Inference Server."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("xtts_v2")
        
        # Model loading lock
        self.model_lock = Lock()
        self.model = None
        
        # Model configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000  # XTTS v2 outputs 24kHz audio
        
        # Default voices for each language (can be overridden with speaker_wav)
        self.default_speakers = {
            "en": "Claribel Dervla",
            "es": "Alejandro Meszaros", 
            "fr": "Lucile Rosalie",
            "de": "Karlsson Gottlieb",
            "it": "Fabrizio Lombardi",
            "pt": "Catarina Valente",
            "pl": "Bartosz Kasper",
            "tr": "Ece Arda",
            "ru": "Dmitry Nikolaev",
            "nl": "Maarten Van Steen",
            "cs": "Bohuslav Dvorak",
            "ar": "Salma Hayek",
            "zh-cn": "Liu Xiaoming",
            "ja": "Tanaka Yuki"
        }
        
        self.logger.info(f"XTTS v2 initialized on {self.device}")
        self.logger.info("Model will be loaded on first inference request")

    def _load_model(self):
        """Load the model if not already loaded."""
        with self.model_lock:
            if self.model is None:
                self.logger.info("Loading XTTS v2 model...")
                start_time = time.time()
                
                try:
                    from TTS.api import TTS
                    
                    # Load XTTS v2
                    self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                    
                    load_time = time.time() - start_time
                    self.logger.info(f"XTTS v2 loaded successfully in {load_time:.2f} seconds")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load model: {e}")
                    raise

    def execute(self, requests):
        """Execute TTS inference requests."""
        # Ensure model is loaded
        self._load_model()
        
        responses = []
        
        for request in requests:
            try:
                # Get inputs
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = text_tensor.as_numpy()[0].decode('utf-8')
                
                language_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                language = language_tensor.as_numpy()[0].decode('utf-8')
                
                # Optional inputs
                speaker_wav = None
                try:
                    speaker_tensor = pb_utils.get_input_tensor_by_name(request, "speaker_wav")
                    if speaker_tensor:
                        speaker_wav = speaker_tensor.as_numpy()
                except:
                    pass
                
                temperature = 0.7
                try:
                    temp_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
                    if temp_tensor:
                        temperature = float(temp_tensor.as_numpy()[0])
                except:
                    pass
                
                length_penalty = 1.0
                try:
                    len_tensor = pb_utils.get_input_tensor_by_name(request, "length_penalty")
                    if len_tensor:
                        length_penalty = float(len_tensor.as_numpy()[0])
                except:
                    pass
                
                # Generate audio
                audio = self._synthesize(
                    text, 
                    language, 
                    speaker_wav, 
                    temperature, 
                    length_penalty
                )
                
                # Create output tensors
                audio_tensor = pb_utils.Tensor(
                    "audio",
                    audio
                )
                
                sample_rate_tensor = pb_utils.Tensor(
                    "sample_rate",
                    np.array([self.sample_rate], dtype=np.int32)
                )
                
                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor, sample_rate_tensor]
                )
                responses.append(response)
                
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"TTS failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses

    def _synthesize(self, text, language, speaker_wav=None, temperature=0.7, length_penalty=1.0):
        """Synthesize speech using XTTS v2."""
        try:
            self.logger.info(f"Synthesizing '{text[:50]}...' in {language}")
            
            # If speaker_wav provided, use voice cloning
            if speaker_wav is not None and len(speaker_wav) > 0:
                # Save temporary wav file for voice cloning
                import tempfile
                import wave
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Convert float32 audio to int16 for wav file
                    audio_int16 = (speaker_wav * 32767).astype(np.int16)
                    
                    with wave.open(tmp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)  # Assuming 16kHz input
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    # Generate with cloned voice
                    wav = self.model.tts(
                        text=text,
                        speaker_wav=tmp_file.name,
                        language=language,
                        temperature=temperature,
                        length_penalty=length_penalty
                    )
                    
                    # Clean up
                    os.unlink(tmp_file.name)
            else:
                # Use default speaker for the language
                speaker = self.default_speakers.get(language, "Claribel Dervla")
                self.logger.info(f"Using default speaker: {speaker}")
                
                wav = self.model.tts(
                    text=text,
                    speaker=speaker,
                    language=language,
                    temperature=temperature,
                    length_penalty=length_penalty
                )
            
            # Convert to numpy array
            if isinstance(wav, list):
                audio_data = np.array(wav, dtype=np.float32)
            else:
                audio_data = wav.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            self.logger.info(f"Generated {len(audio_data)/self.sample_rate:.1f}s of audio")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            raise

    def finalize(self):
        """Clean up resources."""
        self.logger.info("Cleaning up XTTS v2 model...")
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("Model cleanup complete")