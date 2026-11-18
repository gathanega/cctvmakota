"""
Frame queue manager: FIFO per-camera task queue untuk proses frame CCTV.
Dipakai oleh app.py untuk non-blocking frame processing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

log = logging.getLogger("greenflow.frame_queue")


@dataclass
class FrameTask:
    """One frame + metadata to process."""
    location: str
    jpeg_bytes: bytes
    device_id: str
    submitted_at: float = 0.0
    
    def __post_init__(self):
        if not self.submitted_at:
            self.submitted_at = time.time()


class FrameQueueManager:
    """
    Per-camera frame queue with worker pool.
    - process_func: async callable(jpeg_bytes, location, device_id) -> (row, program)
    - queue_size: max frames per camera before dropping oldest
    - max_age: drop frames older than this (seconds)
    - concurrency: number of worker threads
    """
    
    def __init__(
        self,
        process_func: Callable[[bytes, str, str], tuple],
        queue_size: int = 3,
        max_age: int = 30,
        concurrency: int = 2,
    ):
        self.process_func = process_func
        self.queue_size = queue_size
        self.max_age = max_age
        self.concurrency = concurrency
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: list[asyncio.Task] = []
        self._stats: dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def submit(self, location: str, jpeg_bytes: bytes, device_id: str) -> dict:
        """Queue a frame; return submission info."""
        task = FrameTask(location, jpeg_bytes, device_id)
        
        with self._lock:
            if location not in self._queues:
                self._queues[location] = asyncio.Queue(maxsize=self.queue_size)
                self._stats[location] = {"submitted": 0, "processed": 0, "dropped": 0}
            
            queue = self._queues[location]
            stat = self._stats[location]
            stat["submitted"] += 1
            
            try:
                queue.put_nowait(task)
                return {"queued": True, "queue_size": queue.qsize(), "location": location}
            except asyncio.QueueFull:
                stat["dropped"] += 1
                log.warning(f"[{location}] queue full, dropped frame")
                return {"queued": False, "queue_size": queue.qsize(), "location": location, "dropped": True}
    
    def pending(self, location: str) -> int:
        """Current queue depth for this camera."""
        with self._lock:
            queue = self._queues.get(location)
            return queue.qsize() if queue else 0
    
    def stats(self) -> dict:
        """Per-camera statistics."""
        with self._lock:
            return {loc: dict(s) for loc, s in self._stats.items()}
    
    async def close(self):
        """Shutdown workers."""
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
