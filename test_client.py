"""
Automated Headless Test Client for Real-Time Voice Spoof Gateway.
Simulates a live 16kHz microphone stream by generating synthetic speech, silence,
and artifact-injected audio frames, streaming them over WebSockets, and measuring latency.
"""

import sys
import time
import json
import asyncio
import numpy as np
import websockets
from typing import List

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 0.05  # 50ms streaming chunk size (800 samples)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
WS_URI = "ws://localhost:8000/ws/stream"


def generate_audio_signal(duration_sec: float, signal_type: str = "speech") -> bytes:
    """
    Generates synthetic 16kHz mono audio and converts to 16-bit Little-Endian PCM bytes.
    """
    total_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)

    if signal_type == "silence":
        # Pure digital silence / very subtle line noise
        samples = 0.0005 * np.random.randn(total_samples).astype(np.float32)

    elif signal_type == "speech":
        # Multi-harmonic voiced signal with pitch modulation mimicking human vocal cords (F0 ~ 130 Hz)
        f0 = 130.0 + 15.0 * np.sin(2 * np.pi * 1.5 * t)
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        formant1 = 0.6 * np.sin(phase)
        formant2 = 0.3 * np.sin(2 * phase)
        formant3 = 0.15 * np.sin(4 * phase)
        formant4 = 0.1 * np.sin(8 * phase)
        # Envelope envelope modulation (syllables)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))
        samples = (formant1 + formant2 + formant3 + formant4) * envelope * 0.7
        samples += 0.01 * np.random.randn(total_samples)

    elif signal_type == "spoof_synthetic":
        # High-frequency phase artifacts and robotic modulation typical of synthetic vocoders
        carrier = np.sin(2 * np.pi * 440 * t)
        modulator = np.sin(2 * np.pi * 880 * t * 1.5)
        artifacts = 0.2 * np.sin(2 * np.pi * 3500 * t)
        samples = (0.5 * carrier * modulator + artifacts) * 0.8
    else:
        samples = np.random.uniform(-0.1, 0.1, total_samples).astype(np.float32)

    # Clamp to [-1.0, 1.0] and convert to 16-bit signed integer PCM
    samples = np.clip(samples, -1.0, 1.0)
    int16_samples = (samples * 32767.0).astype(np.int16)
    return int16_samples.tobytes()


async def run_streaming_test(duration_sec: float = 6.0):
    """
    Connects to the WebSocket gateway and streams chunks in real-time.
    """
    print("=" * 70)
    print("  VOICE SPOOF GATEWAY - AUTOMATED TEST CLIENT")
    print(f"  Target URI       : {WS_URI}")
    print(f"  Stream Duration  : {duration_sec}s")
    print(f"  Chunk Size       : {CHUNK_DURATION_SEC * 1000:.0f}ms ({CHUNK_SAMPLES} samples @ {SAMPLE_RATE}Hz)")
    print("=" * 70)

    try:
        async with websockets.connect(WS_URI) as ws:
            print("[✓] Connected to WebSocket Gateway successfully.")

            # Task 1: Receiver loop for server events
            received_events: List[dict] = []
            latencies: List[float] = []

            async def receive_worker():
                try:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        received_events.append(data)

                        event_type = data.get("event")
                        if event_type == "connected":
                            print(f"[SERVER HANDSHAKE] Client ID: {data.get('client_id')}")

                        elif event_type == "vad_silence":
                            print(f"[VAD EVENT] Win #{data.get('window_index')}: SILENCE DROPPED (VAD Prob: {data.get('vad_prob'):.2f})")

                        elif event_type == "classification":
                            inf_lat = data.get("inference_latency_ms", 0)
                            tot_lat = data.get("total_pipeline_latency_ms", 0)
                            latencies.append(inf_lat)
                            label = data.get("label")
                            spoof_conf = data.get("spoof_confidence", 0) * 100
                            bonafide_conf = data.get("bonafide_confidence", 0) * 100

                            sub50 = "PASS (<50ms)" if inf_lat <= 50.0 else "WARN (>50ms)"
                            print(f"[SCORE EVENT] Win #{data.get('window_index')}: [{label}] | Spoof: {spoof_conf:.1f}% | Bonafide: {bonafide_conf:.1f}% | Inf Latency: {inf_lat}ms ({sub50}) | Total: {tot_lat}ms")

                except asyncio.CancelledError:
                    pass
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receive_worker())

            # Task 2: Stream audio sequence (2.5s speech -> 1.5s silence -> 2.0s synthetic)
            print("\n>>> Phase 1: Streaming 2.5s Synthetic Speech Audio...")
            speech_bytes = generate_audio_signal(2.5, "speech")
            for i in range(0, len(speech_bytes), CHUNK_SAMPLES * 2):
                chunk = speech_bytes[i:i + CHUNK_SAMPLES * 2]
                await ws.send(chunk)
                await asyncio.sleep(CHUNK_DURATION_SEC)

            print("\n>>> Phase 2: Streaming 1.5s Silence / Background Audio...")
            silence_bytes = generate_audio_signal(1.5, "silence")
            for i in range(0, len(silence_bytes), CHUNK_SAMPLES * 2):
                chunk = silence_bytes[i:i + CHUNK_SAMPLES * 2]
                await ws.send(chunk)
                await asyncio.sleep(CHUNK_DURATION_SEC)

            print("\n>>> Phase 3: Streaming 2.0s High-Frequency Modulated Audio...")
            spoof_bytes = generate_audio_signal(2.0, "spoof_synthetic")
            for i in range(0, len(spoof_bytes), CHUNK_SAMPLES * 2):
                chunk = spoof_bytes[i:i + CHUNK_SAMPLES * 2]
                await ws.send(chunk)
                await asyncio.sleep(CHUNK_DURATION_SEC)

            # Wait briefly for lingering server classification responses
            await asyncio.sleep(0.5)
            recv_task.cancel()

            # Summary Metrics
            print("\n" + "=" * 70)
            print("  TEST RUN METRICS & SUMMARY")
            print("=" * 70)
            print(f"Total Server Events Received : {len(received_events)}")
            print(f"Classification Windows Run   : {len(latencies)}")

            if latencies:
                avg_inf_lat = np.mean(latencies)
                min_inf_lat = np.min(latencies)
                max_inf_lat = np.max(latencies)
                print(f"Avg ONNX Inference Latency   : {avg_inf_lat:.2f} ms")
                print(f"Min ONNX Inference Latency   : {min_inf_lat:.2f} ms")
                print(f"Max ONNX Inference Latency   : {max_inf_lat:.2f} ms")
                print(f"Sub-50ms Target Status       : {'PASSED [OK]' if avg_inf_lat < 50 else 'EXCEEDED'}")
            print("=" * 70 + "\n")

    except Exception as exc:
        print(f"[!] Test Client Connection Failed: {exc}")
        print("    Ensure the backend server is running: `python server.py`")


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    asyncio.run(run_streaming_test(dur))
