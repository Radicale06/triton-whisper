#!/usr/bin/env python3
"""Test Triton with proper audio resampling"""

import numpy as np
import wave
import sys
from scipy import signal
sys.path.append('client')

from triton_whisper_client import TritonWhisperClient

def load_and_resample_audio(wav_file):
    """Load WAV file and resample to 16kHz mono."""
    print(f"Loading {wav_file}...")
    
    with wave.open(wav_file, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        print(f"Original: {framerate}Hz, {channels} channel(s)")
        
        # Read audio
        frames = wf.readframes(n_frames)
        audio = np.frombuffer(frames, dtype=np.int16)
        
        # Convert to mono if stereo
        if channels == 2:
            print("Converting stereo to mono...")
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Resample to 16kHz if needed
        if framerate != 16000:
            print(f"Resampling from {framerate}Hz to 16000Hz...")
            # Calculate resampling ratio
            resample_ratio = 16000 / framerate
            # Calculate new length
            new_length = int(len(audio) * resample_ratio)
            # Resample
            audio = signal.resample(audio, new_length)
        
        # Convert back to int16
        audio = audio.astype(np.int16)
        
        duration = len(audio) / 16000
        print(f"Resampled duration: {duration:.1f} seconds")
        
        return audio

# Load and resample audio
wav_file = sys.argv[1] if len(sys.argv) > 1 else "harvard.wav"
audio = load_and_resample_audio(wav_file)

# Connect to Triton
print("\nConnecting to Triton...")
client = TritonWhisperClient(
    server_url="4.tcp.ngrok.io:18723",
    model_name="whisper_v3_turbo",
    protocol="grpc"
)

# Transcribe
print("Sending properly formatted audio to Whisper...")
transcription, language = client.transcribe(audio, language="en")

print(f"\n✅ Results:")
print(f"Transcription: '{transcription}'")
print(f"Language: {language}")

# Save the resampled audio for verification
output_file = wav_file.replace('.wav', '_16khz.wav')
print(f"\nSaving resampled audio to: {output_file}")
with wave.open(output_file, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(audio.tobytes())