#!/usr/bin/env python3
"""Test Triton with speech audio"""

import numpy as np
import sys
import wave
sys.path.append('client')

from triton_whisper_client import TritonWhisperClient

# Connect to server
print("Connecting to Triton...")
client = TritonWhisperClient(
    server_url="4.tcp.ngrok.io:18723",  # Your ngrok URL
    model_name="whisper_v3_turbo",
    protocol="grpc"
)

if len(sys.argv) > 1:
    # Load WAV file
    wav_file = sys.argv[1]
    print(f"Loading {wav_file}...")
    
    with wave.open(wav_file, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        duration = len(audio) / 16000
        print(f"Duration: {duration:.1f} seconds")
else:
    print("Usage: python test_with_speech.py your_audio.wav")
    print("\nGenerating test beep instead...")
    # Generate a beep pattern that might be recognized
    audio = np.zeros(16000 * 2, dtype=np.int16)
    # Add some beeps
    for i in range(0, 16000, 8000):
        audio[i:i+1000] = (np.sin(2 * np.pi * 440 * np.arange(1000) / 16000) * 10000).astype(np.int16)

# Test transcription
print("Sending to Whisper...")
try:
    transcription, language = client.transcribe(audio, language="en")  # Force English
    print(f"\n✅ Success!")
    print(f"Transcription: '{transcription}'")
    print(f"Language: {language}")
except Exception as e:
    print(f"Error: {e}")