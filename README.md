# Real-Time Voice Spoof Detection Gateway

A production-grade, low-latency audio biometrics security gateway for real-time voice anti-spoofing and deepfake voice detection.

---

## Architecture Overview

```
+---------------------+        WebSocket (16kHz PCM Binary)        +-------------------------------+
|  Browser / Client   | =========================================> | FastAPI WebSocket Ingestion   |
| (Web Audio API Res) | <----------------------------------------- |        (/ws/audio)            |
+---------------------+         JSON Classification Events         +---------------+---------------+
                                                                                   |
                                                                                   v
                                                                  +--------------------------------+
                                                                  | Circular Sliding Window Buffer |
                                                                  |  (2.0s Window, 500ms Hop)      |
                                                                  +----------------+---------------+
                                                                                   |
                                                                                   v
                                                                  +--------------------------------+
                                                                  | Voice Activity Detection (VAD) |
                                                                  |  (Silero ONNX / Adaptive ZCR)  |
                                                                  +----------------+---------------+
                                                                                   |
                                                                   [Speech Active] | [Silence: Drop]
                                                                                   v
                                                                  +--------------------------------+
                                                                  |  ONNX Runtime Inference Pool   |
                                                                  |  (Input: [1, 32000] -> Sub50ms)|
                                                                  +--------------------------------+
```

### Key Engineering Features
1. **Ultra Low Latency Pipeline**: Sub-50ms target inference duration using optimized ONNX Runtime CPU execution provider and background worker thread pool (`ThreadPoolExecutor`).
2. **Audio Buffer**: Thread-safe circular FIFO buffer ingesting 16-bit linear PCM little-endian audio, extracting 32,000-sample (2.0s) windows every 8,000 samples (500ms hop).
3. **Voice Activity Detection (VAD)**: Drops silent or unvoiced frames to eliminate inference on empty acoustic environments. Supports Silero VAD ONNX with adaptive energy & zero-crossing rate fallback.
4. **Client-Side Resampling**: Web Audio API linear resampling downsamples host microphone streams (44.1kHz / 48kHz) to standardized 16,000 Hz before transmission.
5. **Zero-Dependency Mock Model Generator**: `model_helper.py` exports a standalone ONNX graph (`spoof_detector.onnx`) without requiring external multi-gigabyte downloads.

---

## File Structure

| File | Purpose |
| :--- | :--- |
| `server.py` | FastAPI application, WebSocket audio ingestion, session manager, and worker pool |
| `model_helper.py` | Standalone ONNX model generator, pure ONNX / PyTorch graph exporter & benchmark |
| `audio_buffer.py` | Circular sliding audio buffer (2.0s window, 500ms slide hop) |
| `vad_engine.py` | Voice Activity Detection engine (Silero ONNX + Adaptive RMS/ZCR) |
| `test_client.py` | Automated headless streaming test client (generates speech/silence/spoof signals) |
| `static/index.html` | Cyberpunk / FinTech styled real-time UI with Web Audio API & Waveform Visualizer |
| `Dockerfile` | Production container definition with pre-compiled ONNX model |
| `docker-compose.yml` | Container orchestration service configuration |
| `requirements.txt` | Pinned Python production dependencies |

---

## Quickstart Guide

### Option 1: Local Development Mode (Offline / Python)

#### 1. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. (Optional) Generate and Benchmark the ONNX Model
```bash
python model_helper.py --test
```

#### 4. Launch the Server
```bash
python server.py
# Or with uvicorn directly:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

#### 5. Open the Web Dashboard
Navigate to [http://localhost:8000](http://localhost:8000) in your browser. Click **Start Audio Streaming** and speak into your microphone.

---

### Option 2: Docker Container (Online Mode)

#### 1. Build and Run via Docker Compose
```bash
docker compose up --build
```

#### 2. Or Build & Run using Docker CLI
```bash
# Build the image
docker build -t voice-spoof-gateway:latest .

# Run the container
docker run -d -p 8000:8000 --name voice-spoof-gateway voice-spoof-gateway:latest
```

Access the UI at [http://localhost:8000](http://localhost:8000) or check health status at [http://localhost:8000/health](http://localhost:8000/health).

---

## Automated Headless Verification

You can verify the entire WebSocket audio pipeline programmatically without a microphone by running `test_client.py`:

```bash
python test_client.py 6.0
```

### Expected Output
```text
======================================================================
  VOICE SPOOF GATEWAY - AUTOMATED TEST CLIENT
  Target URI       : ws://localhost:8000/ws/audio
  Stream Duration  : 6.0s
  Chunk Size       : 50ms (800 samples @ 16000Hz)
======================================================================
[✓] Connected to WebSocket Gateway successfully.
[SERVER HANDSHAKE] Client ID: client_1717200012_a3f9c2

>>> Phase 1: Streaming 2.5s Synthetic Speech Audio...
[SCORE EVENT] Win #1: [BONAFIDE] | Spoof: 8.2% | Bonafide: 91.8% | Inf Latency: 4.8ms (PASS (<50ms)) | Total: 8.1ms
[SCORE EVENT] Win #2: [BONAFIDE] | Spoof: 9.1% | Bonafide: 90.9% | Inf Latency: 4.2ms (PASS (<50ms)) | Total: 7.4ms

>>> Phase 2: Streaming 1.5s Silence / Background Audio...
[VAD EVENT] Win #3: SILENCE DROPPED (VAD Prob: 0.02)
[VAD EVENT] Win #4: SILENCE DROPPED (VAD Prob: 0.01)

>>> Phase 3: Streaming 2.0s High-Frequency Modulated Audio...
[SCORE EVENT] Win #5: [SPOOF] | Spoof: 94.6% | Bonafide: 5.4% | Inf Latency: 4.5ms (PASS (<50ms)) | Total: 7.9ms

======================================================================
  TEST RUN METRICS & SUMMARY
======================================================================
Total Server Events Received : 6
Classification Windows Run   : 3
Avg ONNX Inference Latency   : 4.50 ms
Min ONNX Inference Latency   : 4.20 ms
Max ONNX Inference Latency   : 4.80 ms
Sub-50ms Target Status       : PASSED [OK]
======================================================================
```

---

## WebSocket API Specification

- **Endpoint**: `ws://<HOST>:8000/ws/audio`
- **Inbound Stream**: Binary frames containing 16-bit Linear PCM Little-Endian mono audio samples at 16,000 Hz.
- **Outbound Frames**: JSON strings.

### Sample Classification Payload (Speech Active)
```json
{
  "event": "classification",
  "timestamp": 1717200000.123,
  "vad_active": true,
  "vad_prob": 0.89,
  "is_spoof": false,
  "label": "BONAFIDE",
  "spoof_confidence": 0.082,
  "bonafide_confidence": 0.918,
  "inference_latency_ms": 4.5,
  "total_pipeline_latency_ms": 7.9,
  "window_index": 1,
  "samples_processed": 32000,
  "buffer_duration_sec": 2.0
}
```

### Sample Silence Drop Payload (VAD Filtered)
```json
{
  "event": "vad_silence",
  "timestamp": 1717200000.623,
  "vad_active": false,
  "vad_prob": 0.02,
  "label": "SILENCE",
  "message": "Silent/unvoiced audio frame dropped by VAD",
  "window_index": 2,
  "buffer_duration_sec": 2.0
}
```
