FROM ultralytics/ultralytics:latest-cpu

# Install ffmpeg and other deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir openvino filterpy scipy

WORKDIR /app

COPY tracker.py .

RUN mkdir -p /output/hls

CMD ["python", "all_int8_openvino_model.py", \
     "--weights", "/app/all_int8_openvino_model", \
     "--source", "rtmp://nginx-rtmp:1935/live/stream", \
     "--device", "cpu", \
     "--conf", "0.5", \
     "--output-hls", "/output/hls/tracked.m3u8"]