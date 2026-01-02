# Triton Whisper v3 Turbo Deployment

Complete implementation of OpenAI's Whisper v3 Turbo on NVIDIA Triton Inference Server for high-concurrency voice agent deployments.

## Features

- ✅ Pure NVIDIA Triton Server (no third-party images)
- ✅ Whisper v3 Turbo with Python backend
- ✅ Automatic language detection
- ✅ GPU acceleration with FP16 support
- ✅ Production-ready with health checks
- ✅ LiveKit Agents integration

## Quick Start

1. **Build and Run Server:**
   ```bash
   cd triton-whisper
   ./build_and_run.sh
   ```

2. **Test the Server:**
   ```bash
   # Check health
   curl http://localhost:8000/v2/health/ready

   # Check model status
   curl http://localhost:8000/v2/models/whisper_v3_turbo
   ```

3. **Use in Your Voice Agent:**
   ```python
   from triton_whisper.client.livekit_triton_stt import TritonWhisperSTT
   
   stt = TritonWhisperSTT(
       server_url="localhost:8001",  # gRPC endpoint
       model_name="whisper_v3_turbo",
       language="auto"  # Auto-detect language
   )
   ```

## Architecture

```
┌─────────────────┐     gRPC/HTTP      ┌─────────────────────┐
│   Voice Agent   │ ←---------------→  │   Triton Server     │
│  (Python App)   │                    │                     │
│                 │                    │  ┌───────────────┐  │
│ TritonWhisperSTT│                    │  │ Whisper v3    │  │
│                 │                    │  │ Turbo Model   │  │
└─────────────────┘                    │  │ (Python Backend)│ │
                                       │  └───────────────┘  │
                                       └─────────────────────┘
```

## Directory Structure

```
triton-whisper/
├── model_repository/           # Triton model repository
│   └── whisper_v3_turbo/      # Model configuration
│       ├── config.pbtxt       # Triton config
│       └── 1/                 # Model version
│           └── model.py       # Python backend implementation
├── client/                    # Client implementations
│   ├── triton_whisper_client.py    # Base client
│   └── livekit_triton_stt.py       # LiveKit plugin
├── scripts/                   # Setup scripts
│   └── setup_whisper_model.py
├── Dockerfile                 # Triton server image
├── docker-compose.yml         # Deployment config
├── build_and_run.sh          # Quick start script
└── README.md                 # This file
```

## Configuration

### Model Configuration (`config.pbtxt`)

- **Max Batch Size**: 0 (dynamic batching disabled by default)
- **Instance Count**: 2 GPU instances
- **Backend**: Python (for flexibility)

### Docker Configuration

- **Base Image**: `nvcr.io/nvidia/tritonserver:24.10-py3`
- **GPU Memory**: ~6GB for model
- **Shared Memory**: 8GB for audio processing

## Performance Tuning

### 1. Enable Dynamic Batching

Edit `model_repository/whisper_v3_turbo/config.pbtxt`:

```protobuf
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [ 2, 4, 8 ]
  max_queue_delay_microseconds: 1000
}
```

### 2. Increase Model Instances

```protobuf
instance_group [
  {
    count: 4  # Increase for more parallelism
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### 3. Multi-GPU Support

```protobuf
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  },
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 1 ]
  }
]
```

## API Usage

### Python Client Example

```python
from triton_whisper.client import TritonWhisperClient
import numpy as np

# Initialize client
client = TritonWhisperClient(
    server_url="localhost:8001",
    protocol="grpc"
)

# Transcribe audio
audio_data = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
transcription, language = client.transcribe(audio_data)

print(f"Text: {transcription}")
print(f"Language: {language}")
```

### LiveKit Integration

```python
from livekit.agents import AgentSession
from triton_whisper.client.livekit_triton_stt import TritonWhisperSTT

# Create STT instance
stt = TritonWhisperSTT(
    server_url="triton-server:8001",
    model_name="whisper_v3_turbo",
    language="auto",
    timeout=120.0
)

# Use in agent session
session = AgentSession(stt=stt, llm=..., tts=...)
```

## Monitoring

### Prometheus Metrics
```bash
curl http://localhost:8002/metrics
```

Key metrics to monitor:
- `nv_inference_request_success` - Successful requests
- `nv_inference_request_failure` - Failed requests  
- `nv_inference_queue_duration_us` - Queue time
- `nv_inference_compute_infer_duration_us` - Inference time
- `nv_gpu_utilization` - GPU usage

### Logs
```bash
# View logs
docker-compose logs -f

# Filter for errors
docker-compose logs | grep ERROR
```

## Troubleshooting

### Model Not Loading
1. Check GPU memory: `nvidia-smi`
2. Verify CUDA installation: `nvidia-smi`
3. Check logs: `docker-compose logs`

### High Latency
1. Enable dynamic batching
2. Increase model instances
3. Check GPU utilization
4. Consider using multiple GPUs

### Connection Errors
1. Verify server is running: `docker ps`
2. Check firewall rules for ports 8000-8002
3. Test with curl: `curl http://localhost:8000/v2/health/ready`

## Production Deployment

### 1. Resource Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 8+ cores recommended
- **RAM**: 16GB+ system memory
- **Disk**: 50GB for model cache

### 2. Scaling Strategies
- **Vertical**: Use larger GPUs (A100, H100)
- **Horizontal**: Deploy multiple Triton instances
- **Load Balancing**: Use nginx/HAProxy for distribution

### 3. High Availability
```yaml
# docker-compose-ha.yml
services:
  triton-1:
    extends:
      file: docker-compose.yml
      service: triton-whisper
    ports:
      - "8000:8000"
      
  triton-2:
    extends:
      file: docker-compose.yml
      service: triton-whisper
    ports:
      - "8010:8000"
```

## License

This implementation uses OpenAI's Whisper model. Please refer to OpenAI's model license for usage terms.