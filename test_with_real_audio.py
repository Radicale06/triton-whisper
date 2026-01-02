#!/usr/bin/env python3
"""
Test Triton Whisper with real audio
"""

import sys
import numpy as np
import wave
import json
sys.path.append('client')

from client.triton_whisper_client import TritonWhisperClient

def test_with_audio_file(audio_file_path=None):
    """Test Triton with an audio file or generate test audio."""
    
    # Initialize client
    print("🚀 Connecting to Triton server...")
    client = TritonWhisperClient(
        server_url="0.tcp.ngrok.io:10877",  # gRPC port
        model_name="whisper_v3_turbo"
    )
    
    # Check health
    health = client.health_check()
    print(f"✅ Server health: {health}")
    
    if audio_file_path and audio_file_path.endswith('.wav'):
        # Load real audio file
        print(f"📁 Loading audio file: {audio_file_path}")
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            print(f"   Duration: {len(audio)/16000:.1f} seconds")
    else:
        # Generate test audio (sine wave saying "hello")
        print("🎵 Generating test audio (sine wave)...")
        duration = 2.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a simple tone
        frequency = 440.0  # A4 note
        audio = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
    
    # Transcribe
    print("🎤 Sending audio to Whisper...")
    try:
        transcription, language = client.transcribe(audio)
        
        print("\n📝 Results:")
        print(f"   Transcription: '{transcription}'")
        print(f"   Language: {language}")
        
        if not transcription:
            print("\n💡 Tip: Empty transcription is normal for silence or tones.")
            print("   Try with a real speech audio file: python test_with_real_audio.py yourfile.wav")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        

if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_with_audio_file(audio_file)