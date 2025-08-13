#!/usr/bin/env python3
"""
Voice AI Client - Complete GUI application for voice-to-voice AI communication
Features:
- Real-time voice recording and playback
- WebSocket connection with auto-reconnection
- Session management
- Streaming support
- Modern GUI with dark theme
- Audio visualization
- Message history
"""

import asyncio
import websockets
import json
import base64
import threading
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Callable
from pathlib import Path
import queue
import io

# GUI and audio dependencies
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    import pyaudio
    import wave
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyaudio numpy matplotlib")
    exit(1)

# Note: Removed simpleaudio dependency, as playback now uses PyAudio for consistency and device selection.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioManager:
    """Handle audio recording and playback"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_playing = False
        self.recording_data = []
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
        # Find all audio devices
        self.input_devices = []
        self.output_devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                self.input_devices.append((i, info['name']))
                logger.info(f"Found input device: {info['name']}")
            if info['maxOutputChannels'] > 0:
                self.output_devices.append((i, info['name']))
                logger.info(f"Found output device: {info['name']}")
        
        # Default to first available devices
        self.input_device_index = self.input_devices[0][0] if self.input_devices else None
        self.output_device_index = self.output_devices[0][0] if self.output_devices else None
    
    def start_recording(self, callback: Optional[Callable] = None):
        """Start audio recording"""
        if self.is_recording or self.input_device_index is None:
            return
        
        self.is_recording = True
        self.recording_data = []
        
        def record_audio():
            try:
                stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.input_device_index,
                    frames_per_buffer=self.chunk_size
                )
                
                while self.is_recording:
                    data = stream.read(self.chunk_size)
                    self.recording_data.append(data)
                    
                    # Call callback with audio level for visualization
                    if callback:
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        level = np.abs(audio_data).mean()
                        callback(level)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Recording error: {e}")
                self.is_recording = False
        
        # Start recording in separate thread
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data"""
        if not self.is_recording:
            return b""
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1.0)
        
        # Convert recorded data to WAV format
        if self.recording_data:
            audio_data = b''.join(self.recording_data)
            return self._to_wav_bytes(audio_data)
        
        return b""
    
    def play_audio(self, wav_data: bytes):
        """Play audio data using PyAudio"""
        if not wav_data or self.output_device_index is None:
            return
        
        def play_thread():
            try:
                logger.info("Starting audio playback")
                wf = wave.open(io.BytesIO(wav_data), 'rb')
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.output_device_index
                )
                
                data = wf.readframes(self.chunk_size)
                while data:
                    stream.write(data)
                    data = wf.readframes(self.chunk_size)
                
                stream.stop_stream()
                stream.close()
                wf.close()
                logger.info("Playback finished")
            except Exception as e:
                logger.error(f"Playback error: {e}")
        
        # Start playback in a daemon thread (GUI mainloop keeps app alive)
        threading.Thread(target=play_thread, daemon=True).start()
    
    def _to_wav_bytes(self, raw_audio: bytes) -> bytes:
        """Convert raw audio to WAV format"""
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.audio.get_sample_size(self.format))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_audio)
        
        return wav_buffer.getvalue()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.is_recording = False
        self.is_playing = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1.0)
        self.audio.terminate()

class WebSocketClient:
    """Handle WebSocket connection with auto-reconnection"""
    
    def __init__(self, server_url: str, message_callback: Callable):
        self.server_url = server_url
        self.message_callback = message_callback
        self.websocket = None
        self.session_id = str(uuid.uuid4())
        self.user_id = "user_" + str(uuid.uuid4())[:8]
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2
        self.message_queue = queue.Queue()
        self.loop = None
        
        # Start connection in background
        self.connection_task = None
        self.start_connection()
    
    def start_connection(self):
        """Start WebSocket connection"""
        if self.connection_task and self.connection_task.is_alive():
            return

        def run_connection():
            # Create and bind an event loop to THIS background thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            try:
                loop.run_until_complete(self._connect_and_listen())
            except Exception as e:
                logger.error(f"Connection task error: {e}")
            finally:
                # ensure no further cross-thread scheduling happens
                self.loop = None
                try:
                    loop.run_until_complete(asyncio.sleep(0))
                finally:
                    loop.close()

        self.connection_task = threading.Thread(target=run_connection, daemon=True)
        self.connection_task.start()
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to {self.server_url}...")
                
                async with websockets.connect(
                    self.server_url,
                    ping_interval=30,
                    ping_timeout=20,
                    close_timeout=20,
                    max_size=16 * 1024 * 1024
                ) as websocket:
                    self.websocket = websocket
                    self.is_connected = True
                    self.reconnect_attempts = 0
                    self.reconnect_delay = 2
                    
                    logger.info("Connected to server!")
                    self.message_callback({
                        "type": "connection_status",
                        "connected": True
                    })
                    
                    # Send queued messages
                    while not self.message_queue.empty():
                        try:
                            message = self.message_queue.get_nowait()
                            await websocket.send(message)
                        except queue.Empty:
                            break
                    
                    # Listen for messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            self.message_callback(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Message handling error: {e}")
            
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Connection closed by server (code={getattr(e, 'code', 'unknown')}, reason={getattr(e, 'reason', '')})")
            except websockets.exceptions.InvalidURI:
                logger.error("Invalid server URL")
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            # Connection lost - attempt reconnection
            self.is_connected = False
            self.websocket = None
            self.reconnect_attempts += 1
            
            self.message_callback({
                "type": "connection_status",
                "connected": False
            })
            
            if self.reconnect_attempts < self.max_reconnect_attempts:
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds... (Attempt {self.reconnect_attempts})")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, 30)  # Exponential backoff
        
        logger.error("Max reconnection attempts reached")

    async def _send_json(self, message_json: str):
        """Send a JSON string over the active websocket on the connection loop.
        If not connected, queue the message for later."""
        if self.is_connected and self.websocket:
            try:
                await self.websocket.send(message_json)
                return
            except Exception as e:
                logger.error(f"Send error: {e}")
        # If we got here, queue for later
        self.message_queue.put(message_json)
    
    def send_message(self, message_data: Dict):
        """Send message to server"""
        message_data["session_id"] = self.session_id
        message_data["user_id"] = self.user_id
        message_json = json.dumps(message_data)
        
        # Schedule send on the connection loop if available; otherwise queue
        if self.loop is not None:
            future = asyncio.run_coroutine_threadsafe(self._send_json(message_json), self.loop)
            # capture send errors and re-queue
            def _handle_result(fut):
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Send scheduling error: {e}")
                    self.message_queue.put(message_json)
            future.add_done_callback(_handle_result)
        else:
            self.message_queue.put(message_json)
    
    def disconnect(self):
        """Disconnect from server"""
        self.is_connected = False
        self.reconnect_attempts = self.max_reconnect_attempts  # Prevent reconnection
        if self.websocket and self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)

class VoiceAIClientGUI:
    """Main GUI application for Voice AI Client"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice AI Client")
        self.root.minsize(height=600, width=800)
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")
        
        # Initialize components
        self.audio_manager = AudioManager()
        self.websocket_client = None
        
        # GUI state
        self.is_recording = False
        self.connection_status = False
        self.messages = []
        self.current_audio_level = 0
        
        # Server configuration
        self.server_url = "ws://51.159.146.41:8765"  # Default server
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.setup_audio_visualization()
        
        # Connect to server
        self.connect_to_server()
        
        # Setup cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Setup dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme colors
        style.configure('Dark.TFrame', background='#1e1e1e')
        style.configure('Dark.TLabel', background='#1e1e1e', foreground='#ffffff')
        style.configure('Dark.TButton', background='#333333', foreground='#ffffff')
        style.map('Dark.TButton', background=[('active', '#555555')])
        style.configure('Success.TButton', background='#28a745', foreground='#ffffff')
        style.map('Success.TButton', background=[('active', '#34ce57')])
        style.configure('Danger.TButton', background='#dc3545', foreground='#ffffff')
        style.map('Danger.TButton', background=[('active', '#e4606d')])
    
    def create_widgets(self):
        """Create main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame - Connection and controls
        top_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        self.status_label = ttk.Label(
            top_frame, 
            text="Disconnected", 
            style='Dark.TLabel',
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Server URL entry
        ttk.Label(top_frame, text="Server:", style='Dark.TLabel').pack(side=tk.LEFT, padx=(20, 5))
        self.server_entry = tk.Entry(
            top_frame, 
            width=30,
            bg="#333333",
            fg="#ffffff",
            insertbackground="#ffffff"
        )
        self.server_entry.insert(0, self.server_url)
        self.server_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Input device dropdown
        ttk.Label(top_frame, text="Input:", style='Dark.TLabel').pack(side=tk.LEFT, padx=(10, 5))
        input_names = [d[1] for d in self.audio_manager.input_devices]
        self.input_combo = ttk.Combobox(top_frame, values=input_names, state="readonly", width=20)
        self.input_combo.pack(side=tk.LEFT, padx=(0, 10))
        if input_names:
            self.input_combo.current(0)
        self.input_combo.bind("<<ComboboxSelected>>", self.update_input_device)
        
        # Output device dropdown
        ttk.Label(top_frame, text="Output:", style='Dark.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        output_names = [d[1] for d in self.audio_manager.output_devices]
        self.output_combo = ttk.Combobox(top_frame, values=output_names, state="readonly", width=20)
        self.output_combo.pack(side=tk.LEFT, padx=(0, 10))
        if output_names:
            self.output_combo.current(0)
        self.output_combo.bind("<<ComboboxSelected>>", self.update_output_device)
        
        # Connect/Disconnect button
        self.connect_button = ttk.Button(
            top_frame,
            text="Connect",
            command=self.toggle_connection,
            style='Success.TButton'
        )
        self.connect_button.pack(side=tk.LEFT)
        
        # Audio controls frame
        audio_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        audio_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Record button
        self.record_button = ttk.Button(
            audio_frame,
            text="🎤 Start Recording",
            command=self.toggle_recording,
            style='Success.TButton'
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Send text button
        self.send_text_button = ttk.Button(
            audio_frame,
            text="Send Text",
            command=self.send_text_message,
            style='Dark.TButton'
        )
        self.send_text_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear history button
        self.clear_button = ttk.Button(
            audio_frame,
            text="Clear History",
            command=self.clear_history,
            style='Dark.TButton'
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # Audio level indicator
        self.audio_level_label = ttk.Label(
            audio_frame,
            text="Audio Level: 0%",
            style='Dark.TLabel'
        )
        self.audio_level_label.pack(side=tk.RIGHT)
        
        # Middle frame - Split between chat and visualization
        middle_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chat frame (left side)
        chat_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(chat_frame, text="Conversation:", style='Dark.TLabel', font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            height=20,
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#555555",
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        
        # Text input
        self.text_input = tk.Text(
            chat_frame,
            height=3,
            bg="#333333",
            fg="#ffffff",
            insertbackground="#ffffff",
            wrap=tk.WORD
        )
        self.text_input.pack(fill=tk.X)
        self.text_input.bind('<Control-Return>', lambda e: self.send_text_message())
        
        # Visualization frame (right side)
        viz_frame = ttk.Frame(middle_frame, style='Dark.TFrame')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        ttk.Label(viz_frame, text="Audio Visualization:", style='Dark.TLabel', font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        # Audio visualization placeholder
        self.viz_frame = ttk.Frame(viz_frame, style='Dark.TFrame')
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Bottom frame - Status and info
        bottom_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        bottom_frame.pack(fill=tk.X)
        
        self.info_label = ttk.Label(
            bottom_frame,
            text="Ready to connect. Press 'Connect' to start.",
            style='Dark.TLabel'
        )
        self.info_label.pack(side=tk.LEFT)
        
        # Session ID display
        self.session_label = ttk.Label(
            bottom_frame,
            text="",
            style='Dark.TLabel',
            font=('Arial', 8)
        )
        self.session_label.pack(side=tk.RIGHT)
    
    def update_input_device(self, event):
        """Update selected input device"""
        if self.input_combo.current() >= 0:
            self.audio_manager.input_device_index = self.audio_manager.input_devices[self.input_combo.current()][0]
            logger.info(f"Selected input device: {self.input_combo.get()}")
    
    def update_output_device(self, event):
        """Update selected output device"""
        if self.output_combo.current() >= 0:
            self.audio_manager.output_device_index = self.audio_manager.output_devices[self.output_combo.current()][0]
            logger.info(f"Selected output device: {self.output_combo.get()}")
    
    def setup_audio_visualization(self):
        """Setup real-time audio visualization"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(4, 3), dpi=100, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Setup plot
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('Audio Level', color='white')
        self.ax.tick_params(colors='white')
        
        # Audio level line
        self.audio_levels = [0] * 100
        self.line, = self.ax.plot(self.audio_levels, color='#00ff00', linewidth=2)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start visualization update
        self.update_visualization()
    
    def update_visualization(self):
        """Update audio visualization"""
        try:
            # Update audio levels
            self.audio_levels.pop(0)
            self.audio_levels.append(self.current_audio_level / 1000 * 100)  # Scale to 0-100
            
            # Update plot
            self.line.set_ydata(self.audio_levels)
            self.canvas.draw_idle()
            
            # Update audio level label
            level_percent = min(int(self.current_audio_level / 1000 * 100), 100)
            self.audio_level_label.config(text=f"Audio Level: {level_percent}%")
            
        except Exception as e:
            logger.error(f"Visualization update error: {e}")
        
        # Schedule next update
        self.root.after(50, self.update_visualization)
    
    def audio_level_callback(self, level: float):
        """Callback for audio level updates"""
        self.current_audio_level = level
    
    def connect_to_server(self):
        """Connect to WebSocket server"""
        if self.websocket_client:
            self.websocket_client.disconnect()
        
        self.server_url = self.server_entry.get().strip()
        if not self.server_url.startswith('ws://'):
            self.server_url = 'ws://' + self.server_url
        
        self.websocket_client = WebSocketClient(self.server_url, self.handle_server_message)
        self.info_label.config(text="Connecting to server...")
    
    def toggle_connection(self):
        """Toggle connection to server"""
        if self.connection_status:
            # Disconnect
            if self.websocket_client:
                self.websocket_client.disconnect()
            self.connection_status = False
            self.update_connection_status(False)
        else:
            # Connect
            self.connect_to_server()
    
    def update_connection_status(self, connected: bool):
        """Update connection status display"""
        self.connection_status = connected
        if connected:
            self.status_label.config(text="Connected", foreground="#28a745")
            self.connect_button.config(text="Disconnect", style='Danger.TButton')
            self.info_label.config(text="Connected! You can now start talking.")
            if self.websocket_client:
                self.session_label.config(text=f"Session: {self.websocket_client.session_id[:8]}")
        else:
            self.status_label.config(text="Disconnected", foreground="#dc3545")
            self.connect_button.config(text="Connect", style='Success.TButton')
            self.info_label.config(text="Disconnected. Click 'Connect' to reconnect.")
            self.session_label.config(text="")
    
    def toggle_recording(self):
        """Toggle audio recording"""
        if not self.connection_status:
            messagebox.showwarning("Not Connected", "Please connect to server first.")
            return
        
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.record_button.config(text="🎤 Start Recording", style='Success.TButton')
            
            # Get recorded audio
            audio_data = self.audio_manager.stop_recording()
            
            if audio_data:
                # Send audio to server
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                self.websocket_client.send_message({
                    "type": "audio_message",
                    "audio_data": audio_b64
                })
                
                self.add_message("user", "[Voice Message Sent]", message_type="audio")
                self.info_label.config(text="Processing voice message...")
        else:
            # Start recording
            self.is_recording = True
            self.record_button.config(text="🛑 Stop Recording", style='Danger.TButton')
            self.info_label.config(text="Recording... Click 'Stop Recording' when done.")
            
            # Start audio recording
            self.audio_manager.start_recording(self.audio_level_callback)
    
    def send_text_message(self):
        """Send text message to server"""
        if not self.connection_status:
            messagebox.showwarning("Not Connected", "Please connect to server first.")
            return
        
        text_content = self.text_input.get("1.0", tk.END).strip()
        if not text_content:
            return
        
        # Clear input
        self.text_input.delete("1.0", tk.END)
        
        # Send to server
        self.websocket_client.send_message({
            "type": "text_message",
            "content": text_content
        })
        
        # Add to chat display
        self.add_message("user", text_content)
        self.info_label.config(text="Processing message...")
    
    def handle_server_message(self, message_data: Dict):
        """Handle incoming server messages"""
        try:
            message_type = message_data.get("type")
            
            if message_type == "connection_status":
                connected = message_data.get("connected", False)
                self.root.after(0, lambda: self.update_connection_status(connected))
            
            elif message_type == "connected":
                self.root.after(0, lambda: self.add_message("system", "Connected to Voice AI Server"))
                # Request conversation history
                if self.websocket_client:
                    self.websocket_client.send_message({"type": "get_history"})
            
            elif message_type == "transcription":
                content = message_data.get("content", "")
                self.root.after(0, lambda: self.add_message("user", f"[Transcribed]: {content}", message_type="transcription"))
            
            elif message_type == "text_response":
                content = message_data.get("content", "")
                self.root.after(0, lambda: self.add_message("assistant", content))
                self.root.after(0, lambda: self.info_label.config(text="Response received."))
            
            elif message_type == "audio_response":
                audio_b64 = message_data.get("audio_data", "")
                if audio_b64:
                    try:
                        audio_data = base64.b64decode(audio_b64)
                        self.audio_manager.play_audio(audio_data)
                        self.root.after(0, lambda: self.add_message("assistant", "[Voice Response Playing]", message_type="audio"))
                    except Exception as e:
                        logger.error(f"Audio playback error: {e}")
            
            elif message_type == "history":
                messages = message_data.get("messages", [])
                self.root.after(0, lambda: self.load_history(messages))
            
            elif message_type == "error":
                error_msg = message_data.get("message", "Unknown error")
                self.root.after(0, lambda: self.add_message("system", f"Error: {error_msg}"))
                self.root.after(0, lambda: self.info_label.config(text="Error occurred."))
        
        except Exception as e:
            logger.error(f"Error handling server message: {e}")
    
    def add_message(self, role: str, content: str, message_type: str = "text"):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format message based on role
        if role == "user":
            prefix = f"[{timestamp}] You: "
            color = "#4CAF50"
        elif role == "assistant":
            prefix = f"[{timestamp}] AI: "
            color = "#2196F3"
        else:  # system
            prefix = f"[{timestamp}] System: "
            color = "#FF9800"
        
        # Add message with color formatting
        start_pos = self.chat_display.index(tk.END)
        self.chat_display.insert(tk.END, prefix)
        
        # Configure tag for colored prefix
        tag_name = f"{role}_{len(self.messages)}"
        prefix_end = self.chat_display.index(tk.END)
        self.chat_display.tag_add(tag_name, start_pos, prefix_end)
        self.chat_display.tag_config(tag_name, foreground=color, font=('Arial', 10, 'bold'))
        
        # Add content
        self.chat_display.insert(tk.END, content + "\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Store message
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "message_type": message_type
        })
    
    def load_history(self, messages: List[Dict]):
        """Load conversation history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.messages.clear()
        
        for msg in messages:
            self.add_message(
                msg.get("role", "unknown"),
                msg.get("content", ""),
                msg.get("message_type", "text")
            )
        
        self.info_label.config(text="Conversation history loaded.")
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.messages.clear()
        self.info_label.config(text="Chat history cleared.")
    
    def on_closing(self):
        """Handle application closing"""
        logger.info("GUI closing triggered")
        try:
            # Stop recording if active
            if self.is_recording:
                self.audio_manager.stop_recording()
            
            # Disconnect from server
            if self.websocket_client:
                self.websocket_client.disconnect()
            
            # Cleanup audio
            self.audio_manager.cleanup()
            
            # Close application
            self.root.destroy()
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()

def main():
    """Main entry point"""
    try:
        app = VoiceAIClientGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
