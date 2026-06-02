import os
import queue
import sys
import threading
import time

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._closed = False
        self._cancelled = False
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        if self._closed or self._cancelled:
            return
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                if self._cancelled:
                    continue
                self._assert_dimensions(frame)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                if self._first_frame:
                    stderr_fd = sys.stderr.fileno()
                    old_stderr = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    try:
                        packets = self.stream.encode(frame)
                        for packet in packets:
                            self.container.mux(packet)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                        os.close(devnull)
                        self._first_frame = False
                else:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
            finally:
                self.queue.task_done()

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Blocking call. Waits for queue to drain, flushes, and closes the container."""
        if self._closed:
            return self.output_path

        print("Waiting for video writer queue to empty...")
        self.queue.join()
        if not self._cancelled:
            print("Video writer queue is empty, flushing stream...")
            self._flush_stream()
        self._closed = True
        self.queue.put(None)
        self._thread.join(timeout=2.0)
        self.container.close()
        return self.output_path

    def cancel(self) -> None:
        """Best-effort cancellation for discarded episodes.

        Discarded episodes do not need a valid mp4 trailer, so avoid flushing the
        encoder. This keeps the skip path responsive and prevents orphan videos.
        """
        if self._closed:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            return

        self._cancelled = True
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.queue.task_done()
        self.queue.put(None)
        self._thread.join(timeout=2.0)
        self._closed = True
        try:
            self.container.close()
        finally:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.container.close()
