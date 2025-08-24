"""
Cross-platform audio management with playback, queuing, and device control.
"""

import logging
import threading
import queue
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

import sounddevice as sd
import numpy as np

from .tts_engine import AudioData


logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """Audio playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass 
class AudioDevice:
    """Audio device information."""
    id: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False


class AudioQueue:
    """Queue for managing audio playback requests."""
    
    def __init__(self, maxsize: int = 10):
        self._queue = queue.Queue(maxsize=maxsize)
        self._current_item = None
    
    def put(self, audio_data: AudioData, priority: int = 0):
        """Add audio to queue with optional priority."""
        try:
            item = {"audio": audio_data, "priority": priority, "timestamp": time.time()}
            self._queue.put(item, block=False)
        except queue.Full:
            logger.warning("Audio queue is full, dropping oldest item")
            try:
                self._queue.get_nowait()  # Remove oldest
                self._queue.put(item, block=False)
            except queue.Empty:
                pass
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get next audio item from queue."""
        try:
            self._current_item = self._queue.get(timeout=timeout)
            return self._current_item
        except queue.Empty:
            return None
    
    def clear(self):
        """Clear all items from queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._current_item = None
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class AudioManager:
    """Manages audio playback with device control and queuing."""
    
    def __init__(self):
        self._audio_queue = AudioQueue()
        self._playback_thread = None
        self._playback_state = PlaybackState.STOPPED
        self._current_stream = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        self._volume = 1.0
        self._output_device = None
        self._sample_rate = 22050  # Default sample rate
        
        # Callbacks
        self._state_callbacks = []
        self._playback_finished_callbacks = []
        
        # Thread lock for state management
        self._state_lock = threading.Lock()
        
        logger.info("Audio manager initialized")
        self._start_playback_thread()
    
    def play_audio(self, audio_data: AudioData, interrupt: bool = True) -> bool:
        """Play audio data, optionally interrupting current playback."""
        if interrupt:
            self.stop_playback()
        
        try:
            priority = 1 if interrupt else 0
            self._audio_queue.put(audio_data, priority)
            
            # Wake up playback thread
            self._stop_event.clear()
            
            logger.debug(f"Added audio to queue (duration: {audio_data.duration():.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue audio: {e}")
            return False
    
    def stop_playback(self):
        """Stop current playback and clear queue."""
        logger.debug("Stopping audio playback")
        
        with self._state_lock:
            self._stop_event.set()
            self._pause_event.clear()
            
            # Stop current stream
            if self._current_stream and self._current_stream.active:
                self._current_stream.stop()
                self._current_stream.close()
                self._current_stream = None
            
            # Clear queue
            self._audio_queue.clear()
            
            self._set_playback_state(PlaybackState.STOPPED)
    
    def pause_playback(self):
        """Pause current playback."""
        if self._playback_state == PlaybackState.PLAYING:
            logger.debug("Pausing audio playback")
            self._pause_event.set()
            self._set_playback_state(PlaybackState.PAUSED)
    
    def resume_playback(self):
        """Resume paused playback."""
        if self._playback_state == PlaybackState.PAUSED:
            logger.debug("Resuming audio playback")
            self._pause_event.clear()
            self._set_playback_state(PlaybackState.PLAYING)
    
    def set_volume(self, volume: float):
        """Set output volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        logger.debug(f"Set volume to {self._volume:.2f}")
    
    def get_volume(self) -> float:
        """Get current volume."""
        return self._volume
    
    def set_output_device(self, device_id: Optional[int]):
        """Set audio output device."""
        self._output_device = device_id
        logger.debug(f"Set output device to {device_id}")
    
    def get_output_devices(self) -> List[AudioDevice]:
        """Get list of available output devices."""
        devices = []
        
        try:
            device_info = sd.query_devices()
            default_device = sd.default.device[1]  # Output device
            
            for i, info in enumerate(device_info):
                if info['max_output_channels'] > 0:  # Only output devices
                    device = AudioDevice(
                        id=i,
                        name=info['name'],
                        channels=info['max_output_channels'],
                        sample_rate=info['default_samplerate'],
                        is_default=(i == default_device)
                    )
                    devices.append(device)
                    
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
        
        return devices
    
    def get_playback_state(self) -> PlaybackState:
        """Get current playback state."""
        return self._playback_state
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playback_state == PlaybackState.PLAYING
    
    def is_paused(self) -> bool:
        """Check if playback is paused."""
        return self._playback_state == PlaybackState.PAUSED
    
    def get_queue_size(self) -> int:
        """Get number of items in playback queue."""
        return self._audio_queue.size()
    
    def add_state_callback(self, callback: Callable[[PlaybackState], None]):
        """Add callback for playback state changes."""
        self._state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[PlaybackState], None]):
        """Remove playback state callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)
    
    def add_finished_callback(self, callback: Callable[[], None]):
        """Add callback for when playback finishes."""
        self._playback_finished_callbacks.append(callback)
    
    def remove_finished_callback(self, callback: Callable[[], None]):
        """Remove playback finished callback."""
        if callback in self._playback_finished_callbacks:
            self._playback_finished_callbacks.remove(callback)
    
    def _start_playback_thread(self):
        """Start the audio playback worker thread."""
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        logger.debug("Started audio playback thread")
    
    def _playback_worker(self):
        """Main playback worker thread."""
        logger.debug("Audio playback worker started")
        
        while True:
            try:
                # Wait for audio in queue
                audio_item = self._audio_queue.get(timeout=1.0)
                
                if audio_item is None:
                    continue
                
                audio_data = audio_item["audio"]
                
                # Check if we should stop
                if self._stop_event.is_set():
                    continue
                
                logger.debug(f"Starting playback of {audio_data.duration():.2f}s audio")
                self._set_playback_state(PlaybackState.PLAYING)
                
                # Play the audio
                success = self._play_audio_data(audio_data)
                
                if success:
                    logger.debug("Audio playback completed")
                else:
                    logger.warning("Audio playback failed")
                    self._set_playback_state(PlaybackState.ERROR)
                
                # Notify finished callbacks
                for callback in self._playback_finished_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in playback finished callback: {e}")
                
                # Set state to stopped if queue is empty
                if self._audio_queue.is_empty():
                    self._set_playback_state(PlaybackState.STOPPED)
                    
            except queue.Empty:
                # Timeout - check if we should continue
                continue
            except Exception as e:
                logger.error(f"Error in playback worker: {e}")
                self._set_playback_state(PlaybackState.ERROR)
                time.sleep(0.1)  # Brief pause before retrying
    
    def _play_audio_data(self, audio_data: AudioData) -> bool:
        """Play a single audio data object."""
        try:
            # Prepare audio data
            audio = audio_data.data.copy()
            
            # Apply volume
            if self._volume != 1.0:
                audio = audio * self._volume
                audio = np.clip(audio, -1.0, 1.0)  # Prevent clipping
            
            # Ensure audio is in correct format for sounddevice
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)  # Convert to column vector
            
            # Create output stream
            stream = sd.OutputStream(
                samplerate=audio_data.sample_rate,
                channels=audio_data.channels,
                device=self._output_device,
                callback=None,
                dtype=audio.dtype
            )
            
            self._current_stream = stream
            
            # Start stream and play audio
            with stream:
                # Write audio data in chunks to allow for pause/stop
                chunk_size = int(audio_data.sample_rate * 0.1)  # 100ms chunks
                
                for i in range(0, len(audio), chunk_size):
                    # Check for stop signal
                    if self._stop_event.is_set():
                        logger.debug("Playback stopped by stop event")
                        break
                    
                    # Handle pause
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)
                    
                    # Get chunk
                    chunk = audio[i:i + chunk_size]
                    
                    # Write chunk to stream
                    stream.write(chunk)
            
            self._current_stream = None
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            self._current_stream = None
            return False
    
    def _set_playback_state(self, state: PlaybackState):
        """Set playback state and notify callbacks."""
        if self._playback_state != state:
            old_state = self._playback_state
            self._playback_state = state
            
            logger.debug(f"Playback state changed: {old_state.value} -> {state.value}")
            
            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    def cleanup(self):
        """Clean up resources and stop playback."""
        logger.info("Cleaning up audio manager...")
        
        self.stop_playback()
        
        # Stop playback thread
        if self._playback_thread and self._playback_thread.is_alive():
            # Signal thread to stop
            self._stop_event.set()
            self._playback_thread.join(timeout=2.0)
        
        # Clear callbacks
        self._state_callbacks.clear()
        self._playback_finished_callbacks.clear()
        
        logger.info("Audio manager cleanup complete")