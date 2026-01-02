#!/usr/bin/env python3
"""Simple test for Triton Whisper"""

import numpy as np
import sys
sys.path.append('client')

from triton_whisper_client import TritonWhisperClient

# Connect to server
print("Connecting to Triton...")
client = TritonWhisperClient(
    server_url="4.tcp.ngrok.io:18723",
    model_name="whisper_v3_turbo",
    protocol="grpc"
)

# Create 3 seconds of silence (which is valid audio)
print("Creating test audio...")
audio = np.zeros(16000 * 3, dtype=np.int16)  # 3 seconds of silence

# Test transcription
print("Testing transcription...")
try:
    transcription, language = client.transcribe(audio)
    print(f"Success!")
    print(f"Transcription: '{transcription}'")
    print(f"Language: {language}")
    print("\nNote: Empty transcription is normal for silence.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()