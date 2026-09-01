"""
Circular Sliding Window Audio Buffer for Real-Time Streaming Ingestion.
Maintains a FIFO ring buffer for 16kHz mono audio, outputting 2.0s windows (32,000 samples)
every 500ms (8,000 samples hop/slide).
"""

import threading
import numpy as np
from typing import List, Generator, Optional


class CircularAudioBuffer:
    """
    Thread-safe circular audio buffer for streaming 16kHz PCM audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration: float = 2.0,
        hop_duration: float = 0.5,
        max_buffer_duration: float = 10.0
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_duration)  # 32,000 samples (2.0s)
        self.hop_samples = int(sample_rate * hop_duration)        # 8,000 samples (0.5s)
        self.max_buffer_samples = int(sample_rate * max_buffer_duration)

        # Internal buffer holding float32 samples in [-1.0, 1.0]
        self._buffer: np.ndarray = np.empty((0,), dtype=np.float32)
        self._new_samples_count: int = 0
        self._lock = threading.Lock()
        self.window_counter: int = 0

    def append_bytes_pcm16(self, pcm_bytes: bytes) -> int:
        """
        Appends raw 16-bit signed integer Little-Endian PCM byte chunk.
        Converts int16 values to float32 in range [-1.0, 1.0].

        Args:
            pcm_bytes: Raw binary bytes from WebSocket.

        Returns:
            int: Number of audio samples appended.
        """
        if not pcm_bytes:
            return 0

        # Convert bytes to int16 numpy array
        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(int16_samples) == 0:
            return 0

        # Normalize to float32 [-1.0, 1.0]
        float32_samples = int16_samples.astype(np.float32) / 32768.0

        with self._lock:
            self._buffer = np.concatenate((self._buffer, float32_samples))
            self._new_samples_count += len(float32_samples)

            # Prevent unbounded memory growth if client streams without consumption
            if len(self._buffer) > self.max_buffer_samples:
                overflow = len(self._buffer) - self.max_buffer_samples
                self._buffer = self._buffer[overflow:]

        return len(float32_samples)

    def append_float32(self, float_samples: np.ndarray) -> int:
        """
        Appends float32 audio samples directly.
        """
        if len(float_samples) == 0:
            return 0

        with self._lock:
            self._buffer = np.concatenate((self._buffer, float_samples.astype(np.float32)))
            self._new_samples_count += len(float_samples)
            if len(self._buffer) > self.max_buffer_samples:
                overflow = len(self._buffer) - self.max_buffer_samples
                self._buffer = self._buffer[overflow:]

        return len(float_samples)

    def get_windows(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields available 2.0-second (32,000 samples) windows
        whenever at least 500ms (8,000 samples) of new audio has been received.

        Yields:
            np.ndarray: 1D Float32 array of shape (32000,)
        """
        with self._lock:
            # Yield windows while we have at least window_samples and enough new hop samples
            while len(self._buffer) >= self.window_samples and self._new_samples_count >= self.hop_samples:
                # Extract the most recent 2.0s window ending at the current accumulation point
                # Or extract from position (len(buffer) - window_samples)
                start_idx = len(self._buffer) - self.window_samples
                window = self._buffer[start_idx:start_idx + self.window_samples].copy()

                self._new_samples_count -= self.hop_samples
                self.window_counter += 1

                # Clean up old samples that won't be needed for future windows beyond max retention
                if len(self._buffer) > self.window_samples * 2:
                    retain = self.window_samples + self.hop_samples
                    self._buffer = self._buffer[-retain:]

                yield window

    def reset(self):
        """Clears the buffer and counters."""
        with self._lock:
            self._buffer = np.empty((0,), dtype=np.float32)
            self._new_samples_count = 0
            self.window_counter = 0

    @property
    def current_buffer_duration(self) -> float:
        """Returns the current duration in seconds stored in the buffer."""
        with self._lock:
            return len(self._buffer) / float(self.sample_rate)

    @property
    def total_buffered_samples(self) -> int:
        with self._lock:
            return len(self._buffer)
