"""
Triton Python Backend Model for Whisper v3 Turbo
================================================

This implements Whisper v3 Turbo using Triton's Python backend.
The model is loaded on first request to minimize container size.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
import logging
import os
from threading import Lock
import time


class TritonPythonModel:
    """Whisper v3 Turbo model for Triton Inference Server."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args["model_config"])
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("whisper_v3_turbo")
        
        # Model loading lock (for thread safety)
        self.model_lock = Lock()
        self.model = None
        self.processor = None
        
        # Model configuration
        self.model_name = "openai/whisper-large-v3-turbo"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Generation config
        self.generation_config = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
            "return_timestamps": False,
        }
        
        self.logger.info(f"Whisper v3 Turbo initialized on {self.device}")
        self.logger.info("Model will be loaded on first inference request")

    def _load_model(self):
        """Load the model if not already loaded."""
        with self.model_lock:
            if self.model is None:
                self.logger.info(f"Loading {self.model_name}...")
                start_time = time.time()
                
                try:
                    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                    
                    # Load processor
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    
                    # Load model
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        use_safetensors=True,
                        device_map=self.device if self.device == "cuda" else None
                    )
                    
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    
                    load_time = time.time() - start_time
                    self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load model: {e}")
                    raise

    def execute(self, requests):
        """Execute inference requests."""
        # Ensure model is loaded
        self._load_model()
        
        responses = []
        
        for request in requests:
            try:
                # Get inputs
                audio_tensor = pb_utils.get_input_tensor_by_name(request, "audio")
                audio_data = audio_tensor.as_numpy()
                
                # Get optional language
                language = None
                try:
                    lang_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                    if lang_tensor:
                        language = lang_tensor.as_numpy()[0].decode('utf-8')
                except:
                    pass
                
                # Process audio
                transcription, detected_lang = self._transcribe(audio_data, language)
                
                # Create output tensors
                transcription_tensor = pb_utils.Tensor(
                    "transcription",
                    np.array([transcription.encode('utf-8')], dtype=np.object_)
                )
                
                language_tensor = pb_utils.Tensor(
                    "detected_language",
                    np.array([detected_lang.encode('utf-8')], dtype=np.object_)
                )
                
                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[transcription_tensor, language_tensor]
                )
                responses.append(response)
                
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Inference failed: {str(e)}")
                )
                responses.append(error_response)
        
        return responses

    def _transcribe(self, audio_data, language=None):
        """Transcribe audio using Whisper v3 Turbo."""
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # Process audio
            inputs = self.processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            # Move to device
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
            
            # Configure language if specified
            generation_config = self.generation_config.copy()
            if language and language != "auto":
                generation_config["language"] = language
                generation_config["task"] = "transcribe"
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    **generation_config
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                normalize=True
            )[0]
            
            # Extract language if auto-detected
            if not language or language == "auto":
                # Try to extract language from special tokens
                with torch.no_grad():
                    # Get language token (usually the first special token)
                    lang_token_id = predicted_ids[0][1].item()  # Skip <|startoftranscript|>
                    lang_token = self.processor.tokenizer.decode([lang_token_id])
                    # Extract language code from token like "<|en|>"
                    if lang_token.startswith("<|") and lang_token.endswith("|>"):
                        detected_lang = lang_token[2:-2]
                    else:
                        detected_lang = "unknown"
            else:
                detected_lang = language
            
            return transcription.strip(), detected_lang
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            raise

    def finalize(self):
        """Clean up resources."""
        self.logger.info("Cleaning up Whisper model...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info("Model cleanup complete")