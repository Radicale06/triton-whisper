#!/bin/bash
# Build and run Triton Whisper v3 Turbo server

set -e

echo "🚀 Building Triton Whisper v3 Turbo Docker image..."

# Build the Docker image
docker-compose build

echo "✅ Build complete!"
echo ""
echo "🔧 Starting Triton server..."

# Start the server
docker-compose up -d

echo ""
echo "⏳ Waiting for server to be ready (this may take a few minutes on first run)..."

# Wait for server to be healthy
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/v2/health/ready > /dev/null; then
        echo "✅ Server is ready!"
        break
    fi
    
    echo -n "."
    sleep 5
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo ""
    echo "❌ Server failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo ""
echo "📊 Server status:"
curl -s http://localhost:8000/v2/health/ready | python -m json.tool

echo ""
echo "🎯 Model status:"
curl -s http://localhost:8000/v2/models/whisper_v3_turbo | python -m json.tool

echo ""
echo "✅ Triton Whisper v3 Turbo is running!"
echo ""
echo "📍 Endpoints:"
echo "   - HTTP: http://localhost:8000"
echo "   - gRPC: localhost:8001"
echo "   - Metrics: http://localhost:8002/metrics"
echo ""
echo "📝 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"