"""
Real-Time Voice Spoof Detection Gateway - FastAPI WebSocket Server.
Ingests 16kHz mono PCM binary streams over persistent WebSockets, buffers 2.0s sliding
audio windows, applies Voice Activity Detection (VAD), and runs sub-50ms ONNX inference.
"""

import os
import time
import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from audio_buffer import CircularAudioBuffer
from vad_engine import VoiceActivityDetector
from model_helper import ensure_model_exists, DEFAULT_MODEL_PATH, WINDOW_SAMPLES

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("voice_gateway")

# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize FastAPI application
app = FastAPI(
    title="Real-Time Voice Spoof Detection Gateway",
    description="Low-Latency Audio Biometrics & Deepfake Voice Protection Gateway",
    version="1.0.0"
)

# Enable CORS for full compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference session and thread pool for async execution
onnx_session: Optional[ort.InferenceSession] = None
onnx_input_name: str = "input_audio"
onnx_output_name: str = "spoof_prob"
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="onnx_worker")


def get_client_ip(websocket: WebSocket) -> str:
    """
    Extracts the client IP address from WebSocket connection headers or socket info.
    Supports reverse proxy configurations (Render, Cloudflare, Nginx) by inspecting
    'x-forwarded-for' (parsing the first IP in proxy chains) and 'x-real-ip'.
    Falls back safely to direct socket client.host or '127.0.0.1'.
    """
    try:
        # 1. Check X-Forwarded-For header (handles comma-separated proxy chains)
        forwarded_for = websocket.headers.get("x-forwarded-for")
        if forwarded_for:
            first_ip = forwarded_for.split(",")[0].strip()
            if first_ip:
                return first_ip

        # 2. Check X-Real-IP header
        real_ip = websocket.headers.get("x-real-ip")
        if real_ip and real_ip.strip():
            return real_ip.strip()

        # 3. Check direct socket client host info
        if websocket.client and websocket.client.host:
            host = websocket.client.host.strip()
            if host:
                return host
    except Exception as exc:
        logger.debug(f"Could not resolve client IP from WebSocket: {exc}")

    # 4. Default fallback
    return "127.0.0.1"


def init_onnx_model():
    """Initializes the ONNX runtime inference session with optimized threading."""
    global onnx_session, onnx_input_name, onnx_output_name
    model_path = os.path.join(BASE_DIR, DEFAULT_MODEL_PATH)
    model_path = ensure_model_exists(model_path)
    logger.info(f"Loading ONNX Model from '{model_path}'...")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    onnx_session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )

    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    logger.info(f"ONNX Model initialized. Input: '{onnx_input_name}', Output: '{onnx_output_name}'")


@app.on_event("startup")
async def startup_event():
    """Server startup lifecycle hook."""
    init_onnx_model()
    logger.info("Voice Spoof Detection Gateway online and ready for WebSocket streaming.")


@app.on_event("shutdown")
async def shutdown_event():
    """Server teardown lifecycle hook."""
    inference_executor.shutdown(wait=False)
    logger.info("Voice Spoof Detection Gateway shutdown completed.")


def run_model_inference_sync(audio_window: np.ndarray) -> Dict[str, Any]:
    """
    Synchronous ONNX inference worker executed inside ThreadPoolExecutor.
    Audio input shape: (32000,) -> reshaped to (1, 32000)
    """
    t_start = time.perf_counter()

    # Ensure shape is [1, 32000] and float32
    input_tensor = np.ascontiguousarray(np.expand_dims(audio_window, axis=0), dtype=np.float32)

    # Execute ONNX runtime inference
    outputs = onnx_session.run([onnx_output_name], {onnx_input_name: input_tensor})
    raw_output = np.asarray(outputs[0]).flatten()

    # Support single-sigmoid [1, 1] or multi-class softmax [1, 2]
    if len(raw_output) == 1:
        spoof_score = float(raw_output[0])
        bonafide_score = 1.0 - spoof_score
    else:
        if np.sum(raw_output) > 1.05 or np.min(raw_output) < 0:
            exp_vals = np.exp(raw_output - np.max(raw_output))
            probs = exp_vals / np.sum(exp_vals)
        else:
            probs = raw_output
        bonafide_score = float(probs[0])
        spoof_score = float(probs[1])

    is_spoof = spoof_score >= 0.50
    label = "SPOOF" if is_spoof else "BONAFIDE"

    inference_duration_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "is_spoof": is_spoof,
        "label": label,
        "spoof_confidence": round(spoof_score, 4),
        "bonafide_confidence": round(bonafide_score, 4),
        "inference_latency_ms": round(inference_duration_ms, 2)
    }


class ConnectionManager:
    """Manages active streaming WebSocket connections and connection telemetry."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.total_connections_served: int = 0

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.total_connections_served += 1
        logger.info(f"Client connected: [{client_id}] | Active streams: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client disconnected: [{client_id}] | Active streams: {len(self.active_connections)}")


manager = ConnectionManager()


@app.get("/", response_class=FileResponse)
async def get_index():
    """Serves the frontend single-page dashboard directly."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    root_index = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(root_index):
        return FileResponse(root_index, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found.")


@app.get("/health")
async def healthcheck():
    """Gateway health check and performance diagnostic telemetry."""
    return JSONResponse(
        content={
            "status": "HEALTHY",
            "model_loaded": onnx_session is not None,
            "active_clients": len(manager.active_connections),
            "total_connections_served": manager.total_connections_served,
            "model_input": onnx_input_name,
            "model_output": onnx_output_name,
            "target_sample_rate_hz": 16000,
            "window_samples": WINDOW_SAMPLES,
            "hop_samples": 8000,
            "timestamp": time.time()
        }
    )


async def handle_audio_websocket(websocket: WebSocket):
    """
    Common handler for WebSocket audio streaming ingestion with client IP extraction & telemetry.
    """
    client_ip = get_client_ip(websocket)
    client_id = f"client_{int(time.time() * 1000)}_{os.urandom(3).hex()}"
    await manager.connect(client_id, websocket)
    logger.info(f"[+] Stream connected from Client IP: {client_ip} (ID: {client_id})")

    audio_buffer = CircularAudioBuffer(sample_rate=16000, window_duration=2.0, hop_duration=0.5)
    vad = VoiceActivityDetector(silero_onnx_path="silero_vad.onnx", threshold=0.5, energy_threshold=0.015)
    loop = asyncio.get_running_loop()

    # Initial connection handshake
    await websocket.send_json({
        "event": "connected",
        "client_id": client_id,
        "client_ip": client_ip,
        "message": "Gateway ready for 16kHz PCM audio stream.",
        "config": {
            "sample_rate": 16000,
            "window_sec": 2.0,
            "hop_sec": 0.5,
            "sub_50ms_target": True
        }
    })

    total_bytes_received = 0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                chunk_bytes = message["bytes"]
                total_bytes_received += len(chunk_bytes)
                t_chunk_received = time.perf_counter()

                # Ingest into circular buffer
                audio_buffer.append_bytes_pcm16(chunk_bytes)

                # Process sliding windows (2.0s duration, triggered every 500ms)
                for window in audio_buffer.get_windows():
                    # 1. Voice Activity Detection (VAD)
                    is_speech, vad_prob = vad.process_chunk(window)

                    if not is_speech:
                        response_payload = {
                            "event": "vad_silence",
                            "timestamp": time.time(),
                            "client_ip": client_ip,
                            "vad_active": False,
                            "vad_prob": round(float(vad_prob), 4),
                            "label": "SILENCE",
                            "message": "Silent/unvoiced audio frame dropped by VAD",
                            "window_index": audio_buffer.window_counter,
                            "buffer_duration_sec": round(audio_buffer.current_buffer_duration, 2)
                        }
                        await websocket.send_json(response_payload)
                        continue

                    # 2. Asynchronous ONNX Model Inference
                    inference_res = await loop.run_in_executor(
                        inference_executor,
                        run_model_inference_sync,
                        window
                    )

                    total_pipeline_latency_ms = (time.perf_counter() - t_chunk_received) * 1000.0

                    # 3. Broadcast classification result back to client
                    response_payload = {
                        "event": "classification",
                        "timestamp": time.time(),
                        "client_ip": client_ip,
                        "vad_active": True,
                        "vad_prob": round(float(vad_prob), 4),
                        "is_spoof": inference_res["is_spoof"],
                        "label": inference_res["label"],
                        "spoof_confidence": inference_res["spoof_confidence"],
                        "bonafide_confidence": inference_res["bonafide_confidence"],
                        "inference_latency_ms": inference_res["inference_latency_ms"],
                        "total_pipeline_latency_ms": round(total_pipeline_latency_ms, 2),
                        "window_index": audio_buffer.window_counter,
                        "samples_processed": len(window),
                        "buffer_duration_sec": round(audio_buffer.current_buffer_duration, 2)
                    }

                    await websocket.send_json(response_payload)

            elif "text" in message and message["text"]:
                try:
                    text_data = json.loads(message["text"])
                    cmd = text_data.get("command")
                    if cmd == "reset":
                        audio_buffer.reset()
                        vad.reset_state()
                        await websocket.send_json({"event": "buffer_reset", "status": "ok", "client_ip": client_ip})
                    elif cmd == "ping":
                        await websocket.send_json({"event": "pong", "time": time.time(), "client_ip": client_ip})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"[-] Client {client_ip} ({client_id}) disconnected cleanly.")
    except Exception as exc:
        logger.warning(f"[-] WebSocket exception on Client {client_ip} ({client_id}): {exc}", exc_info=False)
    finally:
        manager.disconnect(client_id)
        audio_buffer.reset()


# Support both /ws/stream and /ws/audio endpoints
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    await handle_audio_websocket(websocket)


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await handle_audio_websocket(websocket)


# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
elif os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  REAL-TIME VOICE SPOOF DETECTION GATEWAY")
    print("  Open UI at: http://localhost:8000")
    print("  Target Sample Rate: 16 kHz Mono | Latency Target: < 50ms")
    print("=" * 60)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
