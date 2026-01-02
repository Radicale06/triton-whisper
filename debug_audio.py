#!/usr/bin/env python3
"""Debug audio processing"""

import numpy as np
import wave
import sys
sys.path.append('client')

from triton_whisper_client import TritonWhisperClient

# Load WAV file
wav_file = sys.argv[1] if len(sys.argv) > 1 else "harvard.wav"
print(f"Loading {wav_file}...")

with wave.open(wav_file, 'rb') as wf:
    # Check WAV properties
    channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    
    print(f"WAV properties:")
    print(f"  Channels: {channels}")
    print(f"  Sample width: {sample_width} bytes")
    print(f"  Sample rate: {framerate} Hz")
    print(f"  Total frames: {n_frames}")
    print(f"  Duration: {n_frames/framerate:.1f} seconds")
    
    # Read audio
    frames = wf.readframes(n_frames)
    audio = np.frombuffer(frames, dtype=np.int16)
    
    # If stereo, convert to mono
    if channels == 2:
        print("Converting stereo to mono...")
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    # If not 16kHz, we need to resample
    if framerate != 16000:
        print(f"WARNING: Audio is {framerate}Hz, but Whisper expects 16000Hz")
        print("Consider resampling the audio first")
    
    # Check audio statistics
    print(f"\nAudio statistics:")
    print(f"  Min value: {audio.min()}")
    print(f"  Max value: {audio.max()}")
    print(f"  Mean value: {audio.mean():.2f}")
    print(f"  Non-zero samples: {np.count_nonzero(audio)}")
    
    # Take only first 30 seconds for testing
    max_samples = 16000 * 30  # 30 seconds
    if len(audio) > max_samples:
        print(f"\nTrimming to first 30 seconds for testing...")
        audio = audio[:max_samples]

# Connect and test
print("\nConnecting to Triton...")
client = TritonWhisperClient(
    server_url="4.tcp.ngrok.io:18723",
    model_name="whisper_v3_turbo",
    protocol="grpc"
)

print("Sending audio to Whisper...")
transcription, language = client.transcribe(audio, language="en")

print(f"\nResults:")
print(f"  Transcription: '{transcription}'")
print(f"  Language: {language}")

# If still getting "you", try with forced prompt
if transcription == "you":
    print("\nThe model seems to be returning default output.")
    print("This might indicate:")
    print("1. The model needs to be restarted")
    print("2. The audio format needs conversion")
    print("3. The model is not loading correctly")