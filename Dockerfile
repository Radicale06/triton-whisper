FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Install Python dependencies for Whisper
RUN pip install --upgrade pip && \
    pip install transformers>=4.39.0 \
                torch>=2.1.0 \
                torchaudio \
                accelerate \
                safetensors \
                sentencepiece

# Create directories
RUN mkdir -p /workspace/model_repository

# Copy model repository
COPY model_repository /workspace/model_repository

# Copy setup script
COPY scripts/setup_whisper_model.py /workspace/setup_whisper_model.py

# Environment variables for better performance
ENV CUDA_MODULE_LOADING=LAZY
ENV TRANSFORMERS_CACHE=/workspace/.cache
ENV HF_HOME=/workspace/.cache
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0

# Create cache directory
RUN mkdir -p /workspace/.cache

# Run setup script
RUN python /workspace/setup_whisper_model.py

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton Server
CMD ["tritonserver", \
     "--model-repository=/workspace/model_repository", \
     "--allow-gpu-metrics=true", \
     "--gpu-metrics-interval=1000", \
     "--log-verbose=1", \
     "--strict-model-config=false"]