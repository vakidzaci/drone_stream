# YOLO RTMP Docker Setup

## Quick Start

1. **Put your YOLO model in `models/` folder**

2. **Start everything:**
```bash
   docker-compose up --build
```

3. **Stream input to:**
```
   rtmp://localhost:1935/live/stream
```

4. **View output at:**
```
   http://localhost:8888
```

## Test with FFmpeg
```bash
# Stream a video file
ffmpeg -re -i video.mp4 -c:v libx264 -f flv rtmp://localhost:1935/live/stream

# Stream webcam (Linux)
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -f flv rtmp://localhost:1935/live/stream
```

## Ports

- `1935` - RTMP input
- `8080` - HLS stream (direct)
- `8888` - Web player

## Stop
```bash
docker-compose down
```