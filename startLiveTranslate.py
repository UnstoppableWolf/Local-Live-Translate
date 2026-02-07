import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import threading
import queue
import json
import os
import sys
import time
import requests
from faster_whisper import WhisperModel
import torch
import subprocess
import base64
import io

# Auto-install dependencies if needed
def install_package(package_name, import_name=None):
    """Install a package if it's not available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} installed successfully!")
            return True
        except Exception as e:
            print(f"Failed to install {package_name}: {e}")
            return False

# Try to import noisereduce for better noise suppression
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    print("noisereduce library loaded - advanced noise suppression enabled")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("noisereduce not available - using basic noise filtering")
    print("Install with: pip install noisereduce")

# Try to import Flask for web server
try:
    from flask import Flask, render_template_string, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
    print("Flask available - web server enabled")
except ImportError:
    print("Flask not available - installing...")
    if install_package("flask") and install_package("flask-cors"):
        from flask import Flask, render_template_string, jsonify, request
        from flask_cors import CORS
        FLASK_AVAILABLE = True
        print("Flask installed and loaded!")
    else:
        FLASK_AVAILABLE = False
        print("Web server disabled - Flask installation failed")

# Real-time configuration for ultra-fast response
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Larger chunks for better efficiency
ENERGY_THRESHOLD = 0.001
MODEL_SIZE = "medium"  # Medium model - best balance (769M parameters)
LM_STUDIO_URL = "http://youripaddresshere:1234/v1/chat/completions"
LM_STUDIO_API_KEY = "YOURAPIKEYHERE"
PREF_FILE = "mic_pref.json"

# Real-time processing settings - truly live
TRANSCRIBE_INTERVAL = 0.5  # Transcribe every 0.5s during speech
MIN_AUDIO_LENGTH = 0.4     # Minimum 0.4s of audio (lower for faster response)
ROLLING_WINDOW = 3.0       # 3 second buffer
TRANSLATION_DEBOUNCE = 0.2 # Very fast translation
MIN_TRANSLATION_LENGTH = 1
MAX_ACCUMULATED_LENGTH = 180
CORRECTION_WINDOW = 3
MAX_SOURCE_TEXT_LENGTH = 500  # Auto-clear source text after this many characters
MAX_TRANSLATION_TEXT_LENGTH = 800  # Auto-clear translation after this many characters

# Voice Activity Detection - less sensitive (increased by 0.3)
VAD_ENERGY_THRESHOLD = 0.0005  # Lowered from 0.0008 (more sensitive)
VAD_SILENCE_DURATION = 0.4  # Very quick end detection (was 0.5)
VAD_SPEECH_DURATION = 0.2   # Lowered from 0.3 (more sensitive)
VAD_NOISE_GATE = 0.0001     # Lowered from 0.0002 (more sensitive)
VAD_SMOOTHING_WINDOW = 2

# Real-time queues with higher throughput
audio_queue = queue.Queue(maxsize=10)  # Reduced from 100 to prevent backlog
ui_queue = queue.Queue(maxsize=200)
translate_queue = queue.Queue(maxsize=50)
level_queue = queue.Queue(maxsize=30)

# Web server queues for broadcasting
web_source_queue = queue.Queue(maxsize=50)
web_translation_queue = queue.Queue(maxsize=50)
web_clients = []  # List of connected web clients
web_audio_queue = queue.Queue(maxsize=50)  # Queue for web microphone audio
web_current_translation = ""  # Store current translation for persistent access
web_current_source = ""  # Store current source text

class RealtimeTranslator:
    def __init__(self):
        print("Creating real-time translator...")
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Real-time Voice Translator")
        self.root.geometry("700x600")
        
        # Initialize variables for real-time processing
        self.model = None
        self.silero_model = None  # Silero VAD model
        self.silero_utils = None  # Silero utilities
        self.client = None
        self.stream = None
        self.current_level = 0.0
        self.is_running = True
        self.mic_gain = 1.0  # Microphone gain multiplier
        self.noise_profile = None  # For adaptive noise reduction
        self.noise_sample_count = 0  # Count samples for noise profiling
        
        # Web server
        self.web_server = None
        self.web_server_thread = None
        self.web_server_enabled = False
        self.web_is_active = False  # Track if web client is speaking
        
        # Real-time audio processing
        self.audio_buffer = np.zeros(int(SAMPLE_RATE * ROLLING_WINDOW), dtype=np.float32)
        self.buffer_pos = 0
        self.last_transcribe_time = 0
        self.last_translate_time = 0
        self.current_text = ""
        self.last_translated_text = ""
        self.accumulated_text = ""  # Store accumulated Chinese text
        self.last_clean_text = ""   # Store last clean transcription for comparison
        self.translated_length = 0  # Track how much text has been translated
        self.recent_transcriptions = []  # Store recent transcriptions to prevent spam
        self.last_significant_text = ""  # Store last significant text
        self.correction_history = []     # Store recent transcriptions for correction detection
        self.pending_translation = ""    # Store text waiting for translation
        self.translation_in_progress = False  # Flag to track translation state
        self.last_audio_hash = None      # Track last audio chunk to prevent duplicate transcription
        
        # Voice Activity Detection (VAD) variables
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.silence_start_time = None  # Initialize as None
        self.potential_speech_start = None  # Initialize as None
        self.energy_history = []
        self.noise_level = VAD_NOISE_GATE
        self.adaptive_threshold = VAD_ENERGY_THRESHOLD
        
        # Create UI
        self.create_ui()
        print("UI created")
        
        # Start real-time processing
        self.start_realtime_processing()
        print("Real-time processing started")
        
        # Start UI updates
        self.update_ui()
        print("UI updates started")

    def create_ui(self):
        """Create the user interface"""
        # Status
        self.status_label = tk.Label(self.root, text="Starting real-time mode...", fg="blue")
        self.status_label.pack(pady=5)
        
        # Model selection
        model_frame = tk.Frame(self.root)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(model_frame, text="Whisper Model:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.available_models = [
            {"name": "Tiny (fastest, least accurate)", "code": "tiny"},
            {"name": "Base (fast, basic accuracy)", "code": "base"},
            {"name": "Small (balanced)", "code": "small"},
            {"name": "Medium (good accuracy)", "code": "medium"},
            {"name": "Large-v2 (best accuracy)", "code": "large-v2"}
        ]
        model_names = [m["name"] for m in self.available_models]
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=model_names, state="readonly", width=30)
        self.model_combo.grid(row=0, column=1, padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)
        self.model_combo.current(3)  # Default to Medium
        
        # LLM Correction checkbox
        self.llm_correction_var = tk.BooleanVar(value=False)
        self.llm_correction_check = tk.Checkbutton(
            model_frame, 
            text="LLM Correction (slower, more accurate)", 
            variable=self.llm_correction_var,
            font=("Arial", 9)
        )
        self.llm_correction_check.grid(row=0, column=2, padx=10, sticky="w")
        
        # Language selection frame
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(fill="x", padx=10, pady=5)
        
        # From Language
        tk.Label(lang_frame, text="From:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.languages = [
            {"name": "Chinese", "code": "zh"},
            {"name": "Russian", "code": "ru"},
            {"name": "English", "code": "en"}
        ]
        language_names = [lang["name"] for lang in self.languages]
        
        self.from_language_var = tk.StringVar()
        self.from_language_combo = ttk.Combobox(lang_frame, textvariable=self.from_language_var, values=language_names, state="readonly", width=15)
        self.from_language_combo.grid(row=0, column=1, padx=5)
        self.from_language_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        self.from_language_combo.current(0)  # Default to Chinese
        
        # To Language
        tk.Label(lang_frame, text="To:", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky="w", padx=(20, 5))
        self.to_language_var = tk.StringVar()
        self.to_language_combo = ttk.Combobox(lang_frame, textvariable=self.to_language_var, values=language_names, state="readonly", width=15)
        self.to_language_combo.grid(row=0, column=3, padx=5)
        self.to_language_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        self.to_language_combo.current(2)  # Default to English
        
        # Microphone selection
        tk.Label(self.root, text="Microphone:", font=("Arial", 10, "bold")).pack()
        
        # Get available mics
        try:
            devices = sd.query_devices()
            self.mics = [d for d in devices if d['max_input_channels'] > 0]
            mic_names = [f"{i}: {d['name']}" for i, d in enumerate(self.mics)]
            print(f"Found {len(self.mics)} microphones:")
            for i, name in enumerate(mic_names):
                print(f"  {name}")
        except Exception as e:
            print(f"Error getting devices: {e}")
            self.mics = []
            mic_names = ["No microphones found"]
        
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(self.root, textvariable=self.mic_var, values=mic_names, state="readonly")
        self.mic_combo.pack(fill="x", padx=10, pady=5)
        self.mic_combo.bind("<<ComboboxSelected>>", self.on_mic_change)
        
        # Try to select Yeti mic by default, otherwise use first mic
        default_mic_index = 0
        if mic_names and mic_names[0] != "No microphones found":
            for i, name in enumerate(mic_names):
                if "yeti" in name.lower():
                    default_mic_index = i
                    print(f"Found Yeti mic at index {i}, selecting as default")
                    break
            self.mic_combo.current(default_mic_index)
        
        # Microphone gain slider
        gain_frame = tk.Frame(self.root)
        gain_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(gain_frame, text="Mic Gain:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 5))
        self.gain_slider = tk.Scale(
            gain_frame,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            orient="horizontal",
            command=self.on_gain_change,
            length=200
        )
        self.gain_slider.set(1.0)  # Default gain
        self.gain_slider.pack(side="left", padx=5)
        
        self.gain_label = tk.Label(gain_frame, text="1.0x", font=("Arial", 9))
        self.gain_label.pack(side="left", padx=5)
        
        # Noise suppression checkbox
        self.noise_suppression_var = tk.BooleanVar(value=NOISEREDUCE_AVAILABLE)
        self.noise_suppression_check = tk.Checkbutton(
            gain_frame,
            text="Noise Suppression" + (" ✓" if NOISEREDUCE_AVAILABLE else " (install noisereduce)"),
            variable=self.noise_suppression_var,
            font=("Arial", 9),
            state="normal" if NOISEREDUCE_AVAILABLE else "disabled"
        )
        self.noise_suppression_check.pack(side="left", padx=10)
        
        # Web server checkbox
        self.web_server_var = tk.BooleanVar(value=False)
        self.web_server_check = tk.Checkbutton(
            self.root,
            text="Host Translation Web Server (http://0.0.0.0:1235)" + (" ✓" if FLASK_AVAILABLE else " (installing Flask...)"),
            variable=self.web_server_var,
            font=("Arial", 10, "bold"),
            command=self.toggle_web_server,
            state="normal" if FLASK_AVAILABLE else "disabled"
        )
        self.web_server_check.pack(pady=5)
        
        # Level bar
        tk.Label(self.root, text="Audio Level:", font=("Arial", 10, "bold")).pack()
        self.level_canvas = tk.Canvas(self.root, height=20, bg="black")
        self.level_canvas.pack(fill="x", padx=10, pady=5)
        
        # Source language text (dynamic label)
        self.source_language_name = "Chinese"
        self.target_language_name = "English"
        self.source_label = tk.Label(self.root, text=f"{self.source_language_name} (Voice Recognition):", font=("Arial", 10, "bold"))
        self.source_label.pack()
        self.source_text = tk.Text(self.root, height=6, font=("Arial", 11), bg="#f0f0f0")
        self.source_text.pack(fill="x", padx=10, pady=5)
        
        # Target language text
        self.target_label = tk.Label(self.root, text=f"{self.target_language_name} (Live Translation):", font=("Arial", 10, "bold"), fg="blue")
        self.target_label.pack()
        self.english_text = tk.Text(self.root, height=8, font=("Arial", 11), fg="blue", bg="#f8f8ff")
        self.english_text.pack(fill="both", expand=True, padx=10, pady=5)

    def start_realtime_processing(self):
        """Start all real-time processing threads"""
        # Model loading
        threading.Thread(target=self.load_models, daemon=True).start()
        
        # Real-time audio capture
        threading.Thread(target=self.start_audio_stream, daemon=True).start()
        
        # Single transcription thread (multiple threads cause lag on RTX 3050)
        threading.Thread(target=self.continuous_transcription, daemon=True).start()
        
        # Single translation thread
        threading.Thread(target=self.realtime_translation, daemon=True).start()
        
        # Web audio translation thread
        threading.Thread(target=self.web_audio_translation, daemon=True).start()
        
        # Level processing
        threading.Thread(target=self.process_audio_levels, daemon=True).start()

    def load_models(self):
        """Load AI models with optimized settings"""
        try:
            # Load Silero VAD first (lightweight, fast)
            ui_queue.put(("status", "Loading Silero VAD..."))
            print("Loading Silero VAD model...")
            
            try:
                # Load Silero VAD model
                self.silero_model, self.silero_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                
                # Extract utility functions
                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.silero_utils
                
                print("Silero VAD loaded successfully!")
                ui_queue.put(("status", "Silero VAD loaded ✓"))
            except Exception as e:
                print(f"Warning: Could not load Silero VAD: {e}")
                print("Continuing without Silero VAD (will use energy-based VAD only)")
                self.silero_model = None
                self.silero_utils = None
            
            ui_queue.put(("status", f"Loading Whisper {MODEL_SIZE} model..."))
            print(f"Loading Whisper {MODEL_SIZE} model...")
            
            # Add CUDA to PATH if not already there
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
            ]
            
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    if cuda_path not in os.environ['PATH']:
                        os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']
                        print(f"Added CUDA to PATH: {cuda_path}")
                    break
            
            # Check if CuPy can find CUDA
            cuda_available = False
            try:
                import cupy as cp
                # Try to create a simple array on GPU
                test_array = cp.array([1, 2, 3])
                cuda_available = True
                print(f"CuPy CUDA available! Device: {cp.cuda.Device()}")
                del test_array
            except Exception as e:
                print(f"CuPy CUDA not available: {e}")
                print("Will use CPU (slower)")
            
            # Load model
            device = "cuda" if cuda_available else "cpu"
            compute_type = "float16" if cuda_available else "int8"
            
            print(f"Loading model on: {device} with {compute_type}")
            
            self.model = WhisperModel(
                MODEL_SIZE, 
                device=device,
                compute_type=compute_type,
                num_workers=1,
                download_root=None
            )
            
            print(f"Model loaded successfully on {device.upper()}!")
            
            silero_status = " + Silero VAD" if self.silero_model else ""
            ui_queue.put(("status", "Connecting to LM Studio..."))
            try:
                response = requests.post(
                    LM_STUDIO_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {LM_STUDIO_API_KEY}"
                    },
                    json={
                        "model": "local-model",
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                        "stream": False
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    ui_queue.put(("status", f"Ready! {MODEL_SIZE} on {device.upper()}{silero_status}"))
                    print(f"System ready!")
                else:
                    ui_queue.put(("status", f"LM Studio connected{silero_status}"))
            except Exception as e:
                ui_queue.put(("status", f"LM Studio warning: {e}"))
                
        except Exception as e:
            ui_queue.put(("status", f"Model error: {e}"))
            print(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()

    def start_audio_stream(self):
        """Start continuous audio stream"""
        try:
            time.sleep(1)  # Wait for UI
            
            if not self.mics:
                ui_queue.put(("status", "No microphones available"))
                print("No microphones found")
                return
            
            device_index = self.mic_combo.current()
            if device_index < 0 or device_index >= len(self.mics):
                device_index = 0
                print(f"Using default microphone index: {device_index}")
                
            ui_queue.put(("status", "Starting real-time audio..."))
            print(f"Starting audio with device index: {device_index}")
            
            # Close existing stream if any
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            # Create new stream
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_index,
                blocksize=CHUNK_SIZE,
                callback=self.realtime_audio_callback
            )
            
            # Start the stream
            self.stream.start()
            ui_queue.put(("status", "Real-time audio active"))
            print("Audio stream started successfully")
            
            # Keep stream alive
            while self.is_running:
                if self.stream is None:
                    print("Stream became None, breaking")
                    break
                if not self.stream.active:
                    print("Stream became inactive, breaking")
                    break
                time.sleep(0.1)
                
        except Exception as e:
            ui_queue.put(("status", f"Audio error: {e}"))
            print(f"Audio stream error: {e}")
            import traceback
            traceback.print_exc()

    def realtime_audio_callback(self, indata, frames, time_info, status):
        """Real-time audio callback with advanced VAD and noise filtering"""
        try:
            if status:
                print(f"Audio status: {status}")
            
            if indata is None or len(indata) == 0:
                return
            
            # Skip PC audio processing if web client is active
            if self.web_is_active:
                return
            
            audio = indata[:, 0]
            
            # Apply microphone gain
            audio = audio * self.mic_gain
            
            current_time = time.time()
            
            # Apply noise reduction
            filtered_audio = self.apply_noise_reduction(audio)
            
            # Calculate RMS energy on filtered audio
            rms = np.sqrt(np.mean(filtered_audio**2))
            
            # Send level update (use original audio for visual feedback)
            original_rms = np.sqrt(np.mean(audio**2))
            try:
                level_queue.put_nowait(original_rms)
            except queue.Full:
                pass
            
            # Voice Activity Detection
            speech_ended = self.voice_activity_detection(rms, current_time)
            
            # Only process audio during active speech
            if self.is_speaking:
                # Add filtered audio to rolling buffer
                chunk_size = len(filtered_audio)
                
                # Handle buffer wraparound
                if self.buffer_pos + chunk_size > len(self.audio_buffer):
                    # Shift buffer left
                    shift_size = self.buffer_pos + chunk_size - len(self.audio_buffer)
                    self.audio_buffer[:-shift_size] = self.audio_buffer[shift_size:]
                    self.buffer_pos = len(self.audio_buffer) - chunk_size
                
                # Add new filtered audio
                self.audio_buffer[self.buffer_pos:self.buffer_pos + chunk_size] = filtered_audio
                self.buffer_pos += chunk_size
            
            # Trigger transcription when speech ends OR during long speech
            should_transcribe = False
            
            if speech_ended:
                # Speech just ended - transcribe accumulated audio
                should_transcribe = True
                print("Speech ended - triggering transcription")
            elif self.is_speaking and (current_time - self.last_transcribe_time) >= TRANSCRIBE_INTERVAL:
                # Long speech - transcribe periodically for live feedback
                should_transcribe = True
                print("Live transcription during speech")
            
            if should_transcribe and self.buffer_pos > 0:
                # Get accumulated audio for transcription
                if self.buffer_pos < len(self.audio_buffer):
                    speech_audio = self.audio_buffer[:self.buffer_pos].copy()
                else:
                    speech_audio = self.audio_buffer.copy()
                
                # Only transcribe if we have enough audio
                min_samples = int(SAMPLE_RATE * MIN_AUDIO_LENGTH)
                if len(speech_audio) >= min_samples:
                    # Create a hash of the audio to detect duplicates
                    import hashlib
                    audio_hash = hashlib.md5(speech_audio.tobytes()).hexdigest()
                    
                    # Skip if this is the same audio we just transcribed
                    if audio_hash != self.last_audio_hash:
                        print(f"Queuing audio for transcription: {len(speech_audio)} samples ({len(speech_audio)/SAMPLE_RATE:.2f}s)")
                        
                        # Clear queue if it's full to prevent lag
                        if audio_queue.full():
                            print("Audio queue full! Clearing old audio...")
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                        
                        try:
                            audio_queue.put_nowait(speech_audio)
                            self.last_transcribe_time = current_time
                            self.last_audio_hash = audio_hash  # Store hash
                        except queue.Full:
                            print("Audio queue still full after clearing, skipping this chunk")
                    else:
                        print(f"Skipping duplicate audio chunk (same hash)")
                    
                    # Clear buffer if speech ended, otherwise keep accumulating
                    if speech_ended:
                        self.buffer_pos = 0
                        self.last_audio_hash = None  # Reset hash when speech ends
                else:
                    print(f"Audio too short for transcription: {len(speech_audio)} samples ({len(speech_audio)/SAMPLE_RATE:.2f}s < {MIN_AUDIO_LENGTH}s)")
                        
        except Exception as e:
            print(f"Audio callback error: {e}")

    def apply_noise_reduction(self, audio):
        """Apply noise reduction and filtering"""
        try:
            # Use noisereduce if available and enabled
            if NOISEREDUCE_AVAILABLE and self.noise_suppression_var.get():
                # Build noise profile from initial silence if not yet built
                if self.noise_profile is None and self.noise_sample_count < 10:
                    # Collect first few samples as noise profile
                    if not self.is_speaking:
                        if self.noise_profile is None:
                            self.noise_profile = audio.copy()
                        else:
                            self.noise_profile = np.concatenate([self.noise_profile, audio])
                        self.noise_sample_count += 1
                        
                        if self.noise_sample_count >= 10:
                            print("Noise profile built from background samples")
                
                # Apply stationary noise reduction
                if len(audio) > 512:  # Need minimum length
                    try:
                        # Use stationary noise reduction (faster, works well for constant background noise)
                        reduced = nr.reduce_noise(
                            y=audio,
                            sr=SAMPLE_RATE,
                            stationary=True,
                            prop_decrease=0.8  # Reduce noise by 80%
                        )
                        return reduced
                    except Exception as e:
                        print(f"Noisereduce error: {e}")
                        # Fall through to basic filtering
            
            # Basic noise gate (fallback or if noisereduce disabled)
            audio_filtered = np.where(np.abs(audio) > self.noise_level, audio, 0)
            
            # Simple high-pass filter to remove low-frequency noise
            # This removes rumble and low-frequency background noise
            if len(audio_filtered) > 1:
                # Simple difference filter (high-pass)
                filtered = np.diff(audio_filtered, prepend=audio_filtered[0])
                # Normalize
                if np.max(np.abs(filtered)) > 0:
                    filtered = filtered * 0.7  # Reduce amplitude slightly
                return filtered
            
            return audio_filtered
            
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio

    def check_speech_with_silero(self, audio_chunk):
        """Use Silero VAD to check if audio contains speech"""
        if self.silero_model is None:
            return True  # If Silero not available, assume speech (fallback to energy VAD)
        
        try:
            # Silero expects exactly 512 samples at 16kHz
            chunk_size = 512
            
            # If audio is shorter than 512 samples, pad it
            if len(audio_chunk) < chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)), mode='constant')
            
            # For speed, sample only a few chunks instead of processing all
            # Process beginning, middle, and end chunks
            num_chunks = len(audio_chunk) // chunk_size
            
            if num_chunks <= 3:
                # Short audio, check all chunks
                chunks_to_check = range(0, len(audio_chunk), chunk_size)
            else:
                # Long audio, sample 5 chunks evenly distributed
                step = num_chunks // 5
                chunks_to_check = [i * chunk_size for i in range(0, num_chunks, max(1, step))][:5]
            
            speech_probs = []
            for i in chunks_to_check:
                chunk = audio_chunk[i:i+chunk_size]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(chunk).float()
                
                # Get speech probability (0.0 to 1.0)
                speech_prob = self.silero_model(audio_tensor, SAMPLE_RATE).item()
                speech_probs.append(speech_prob)
            
            # Use max speech probability (if ANY chunk has speech, consider it speech)
            max_speech_prob = np.max(speech_probs)
            
            # Threshold: 0.15 means 15% confidence of speech (very lenient for better detection)
            speech_threshold = 0.15
            
            has_speech = max_speech_prob > speech_threshold
            
            if has_speech:
                print(f"Silero VAD: Speech detected (confidence: {max_speech_prob:.2f})")
            else:
                print(f"Silero VAD: No speech (confidence: {max_speech_prob:.2f})")
            
            return has_speech
            
        except Exception as e:
            print(f"Silero VAD error: {e}")
            return True  # On error, assume speech (fallback)

    def get_initial_prompt(self):
        """Get initial prompt to guide Whisper transcription"""
        # Get current language
        from_index = self.from_language_combo.current()
        if from_index >= 0:
            language_code = self.languages[from_index]["code"]
        else:
            language_code = "zh"
        
        # Language-specific prompts to improve accuracy
        prompts = {
            "zh": "以下是普通话的句子。",  # "The following is a Mandarin sentence."
            "ru": "Следующее предложение на русском языке.",  # "The following sentence is in Russian."
            "en": "The following is a clear English sentence."
        }
        
        return prompts.get(language_code, "")

    def correct_transcription_with_llm(self, text, language):
        """Use LLM to correct transcription errors"""
        if not text or len(text.strip()) < 2:
            return text
        
        try:
            print(f"LLM correction: Correcting '{text}'")
            
            # Create correction prompt
            correction_prompts = {
                "zh": f"Correct any transcription errors in this Chinese text. Output ONLY the corrected text, nothing else:\n{text}",
                "ru": f"Correct any transcription errors in this Russian text. Output ONLY the corrected text, nothing else:\n{text}",
                "en": f"Correct any transcription errors in this English text. Output ONLY the corrected text, nothing else:\n{text}"
            }
            
            prompt = correction_prompts.get(language, f"Correct any errors in: {text}")
            
            # Quick LLM correction
            response = requests.post(
                LM_STUDIO_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {LM_STUDIO_API_KEY}"
                },
                json={
                    "model": "local-model",
                    "messages": [
                        {"role": "system", "content": "You are a transcription correction assistant. Output ONLY the corrected text."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "max_tokens": 200,
                    "temperature": 0.1
                },
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    corrected = result['choices'][0]['message']['content'].strip()
                    print(f"LLM correction result: '{corrected}'")
                    return corrected
            
            return text  # Return original if correction fails
            
        except Exception as e:
            print(f"LLM correction error: {e}")
            return text  # Return original on error

    def update_noise_estimation(self, rms):
        """Update background noise estimation"""
        try:
            # Update noise level estimation during silence
            if not self.is_speaking:
                # Slowly adapt noise level during non-speech
                self.noise_level = self.noise_level * 0.95 + rms * 0.05  # Faster adaptation to changing noise
                # Keep it within reasonable bounds
                self.noise_level = max(VAD_NOISE_GATE, min(self.noise_level, VAD_ENERGY_THRESHOLD * 0.5))
            
            # Update adaptive threshold - speech must be above noise
            # Require speech to be at least 2x the noise level (reduced from 3x)
            self.adaptive_threshold = max(VAD_ENERGY_THRESHOLD * 0.7, self.noise_level * 2.0)
            
        except Exception as e:
            print(f"Noise estimation error: {e}")

    def voice_activity_detection(self, rms, current_time):
        """Advanced voice activity detection with noise filtering"""
        try:
            # Add to energy history for smoothing
            self.energy_history.append(rms)
            if len(self.energy_history) > VAD_SMOOTHING_WINDOW:
                self.energy_history.pop(0)
            
            # Calculate smoothed energy
            smoothed_energy = np.mean(self.energy_history)
            
            # Update noise estimation
            self.update_noise_estimation(smoothed_energy)
            
            # Voice activity detection - speech must be SIGNIFICANTLY above background noise
            # Main detection: energy must be above adaptive threshold (2x noise level)
            speech_detected = smoothed_energy > self.adaptive_threshold
            
            # Additional check: energy must also be above absolute minimum
            # This prevents triggering on slight noise increases
            absolute_minimum = VAD_ENERGY_THRESHOLD * 0.5  # Lowered from 0.8 to be more sensitive
            speech_detected = speech_detected and (smoothed_energy > absolute_minimum)
            
            # Fallback: also detect if energy is significantly above noise floor
            fallback_detected = smoothed_energy > (self.noise_level * 2.0)  # Lowered from 2.5 to 2.0
            
            # Use either detection method, but prefer the stricter one
            if speech_detected or fallback_detected:
                if not self.is_speaking:
                    # Check if we have enough continuous speech to start
                    if self.potential_speech_start is not None:
                        if current_time - self.potential_speech_start >= VAD_SPEECH_DURATION:
                            self.is_speaking = True
                            self.speech_start_time = self.potential_speech_start
                            ui_queue.put(("status", "🎤 Speaking detected"))
                            print(f"Speech started! Energy: {smoothed_energy:.4f}, Noise: {self.noise_level:.4f}, Threshold: {self.adaptive_threshold:.4f}")
                    else:
                        self.potential_speech_start = current_time
                        print(f"Potential speech start. Energy: {smoothed_energy:.4f}, Noise: {self.noise_level:.4f}")
                else:
                    # Continue speaking
                    self.last_speech_time = current_time
                    
            else:
                # No speech detected - energy is at background noise level
                self.potential_speech_start = None
                
                if self.is_speaking:
                    # Check if silence duration is long enough to stop
                    if self.silence_start_time is None:
                        self.silence_start_time = current_time
                        print(f"Speech paused. Energy: {smoothed_energy:.4f}, Noise: {self.noise_level:.4f}")
                    elif current_time - self.silence_start_time >= VAD_SILENCE_DURATION:
                        self.is_speaking = False
                        self.silence_start_time = None
                        ui_queue.put(("status", "🔇 Speech ended"))
                        print(f"Speech ended. Energy: {smoothed_energy:.4f}, Noise: {self.noise_level:.4f}")
                        return True  # Signal end of speech
                else:
                    self.silence_start_time = None
            
            return False  # No end of speech
            
        except Exception as e:
            print(f"VAD error: {e}")
            return False

    def clean_transcribed_text(self, text):
        """Clean and filter transcribed text - optimized for Chinese/Russian"""
        if not text:
            return ""
        
        # Enhanced subtitle artifacts list (English and Chinese)
        subtitle_keywords = [
            "subtitle", "subtitles", "editor", "редактор", "субтитров", 
            "revers", "ревер", "headlines", "возгласы", "заголовки",
            "novkovic", "новикова", "zakomoldin", "закомолдина",
            "happy", "радостные", "so lan ya", "down to earth",
            "downed", "once down", "coconut egg", "ten dollars",
            "scared you", "he scared", "subtitles by",
            "字幕", "索兰娅", "索蘭婭", "字幕by",
            "谢谢观看", "謝謝觀看", "thank you for watching", "thanks for watching",
            "视频就到此为止", "視頻就到此為止", "this is where the video ends",
            "下期再见", "下期見", "see you next time",
            "lyrics by", "歌词", "請按讚", "请按赞", "like, subscribe", "share",
            "support the", "明镜", "点点", "mingjing", "dian dian",
            "李宗恒", "李宗盛", "li zongheng", "li zongsheng", "burp",
            "the following is", "以下是普通话", "以下是", "mandarin chinese",
            "clear english sentence", "следующее предложение"
        ]
        
        # Convert to lowercase for checking (and check original for Chinese)
        text_lower = text.lower().strip()
        
        # Skip if contains subtitle keywords (check both cases)
        for keyword in subtitle_keywords:
            if keyword in text_lower or keyword in text:
                print(f"Filtered out common video/spam phrase: '{keyword}'")
                return ""
        
        # Skip if text looks like subtitle formatting
        if any(phrase in text_lower for phrase in ["subtitles by", "down to earth", "coconut egg", "lyrics by"]):
            return ""
        
        # Skip Chinese subtitle patterns or video ending phrases
        if any(phrase in text for phrase in ["字幕by", "索兰娅", "索蘭婭", "谢谢观看", "視頻就到此為止", "請按讚", "明镜"]):
            return ""
        
        # Get current language
        from_index = self.from_language_combo.current()
        language_code = "zh"
        if from_index >= 0:
            language_code = self.languages[from_index]["code"]
        
        # For Chinese/Russian, be more lenient with repetition
        is_cjk = language_code in ["zh", "ru"]
        
        # Count non-ASCII characters (Chinese/Russian characters)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        total_chars = len(text.replace(" ", ""))
        
        # If it's mostly Chinese/Russian characters, allow it through with minimal filtering
        if is_cjk and non_ascii_count > total_chars * 0.5:
            # Only filter EXTREMELY repetitive text (same character 20+ times in a row)
            if len(text) > 20:
                # Check for same character repeated many times
                for i in range(len(text) - 19):
                    if len(set(text[i:i+20])) == 1:
                        return ""  # Same character 20+ times
            
            # Allow the text through - it's legitimate Chinese/Russian
            return text.strip()
        
        # For English or mixed text, apply aggressive filtering for repetition
        words = text.split()
        
        # If it's a single word repeated multiple times, keep only one
        if len(words) >= 2:
            unique_words = set(w.lower().strip('.,!?') for w in words)
            # If there's only 1 unique word repeated, return just that word once
            if len(unique_words) == 1:
                return words[0].strip('.,!?')
        
        if len(words) == 1 and len(words[0]) <= 3:
            return ""  # Skip very short single words
        
        # Check for excessive word repetition in English
        if len(words) >= 3:
            word_counts = {}
            for word in words:
                clean_word = word.lower().strip('.,!?')
                if len(clean_word) > 0:
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
            
            # If any single word appears more than 50% of the time, it's spam
            if word_counts:
                max_count = max(word_counts.values())
                if max_count > len(words) * 0.5:
                    # Return just the word once instead of filtering completely
                    most_common_word = max(word_counts, key=word_counts.get)
                    return most_common_word
        
        # Remove repeated characters and words for English
        cleaned_words = []
        prev_word = ""
        
        for word in words:
            clean_word = word.strip('.,!?').lower()
            
            # Skip very short words or single characters repeated
            if len(clean_word) <= 1:
                continue
            
            # Skip if word contains too many dots or special chars
            special_count = sum(1 for c in word if not c.isalnum())
            if special_count > len(word) * 0.3:
                continue
            
            # Skip consecutive duplicate words completely
            if clean_word == prev_word:
                continue
            
            cleaned_words.append(word)
            prev_word = clean_word
        
        cleaned_text = " ".join(cleaned_words)
        
        # Additional filtering
        if len(cleaned_text.strip()) < 2:
            return ""
        
        return cleaned_text.strip()

    def detect_runaway_speech(self, text):
        """Detect if Whisper is hallucinating repeated words/phrases (runaway speech)"""
        if not text or len(text.strip()) < 10:
            return False
        
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for same word repeated consecutively 3+ times
        consecutive_count = 1
        for i in range(len(words) - 1):
            word1 = words[i].lower().strip('.,!?')
            word2 = words[i + 1].lower().strip('.,!?')
            
            if word1 == word2 and len(word1) > 0:
                consecutive_count += 1
                if consecutive_count >= 3:
                    print(f"Runaway speech detected: '{word1}' repeated {consecutive_count} times consecutively")
                    return True
            else:
                consecutive_count = 1
        
        # Check for same phrase repeated (2-3 word sequences)
        for phrase_len in [2, 3]:
            if len(words) < phrase_len * 2:
                continue
            
            for i in range(len(words) - phrase_len * 2 + 1):
                phrase1 = " ".join(words[i:i+phrase_len]).lower()
                phrase2 = " ".join(words[i+phrase_len:i+phrase_len*2]).lower()
                
                if phrase1 == phrase2 and len(phrase1) > 3:
                    print(f"Runaway speech detected: phrase '{phrase1}' repeated")
                    return True
        
        # Check for word appearing too many times overall (not just consecutive)
        word_counts = {}
        for word in words:
            clean_word = word.lower().strip('.,!?')
            if len(clean_word) > 2:  # Only count words longer than 2 chars
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        # If any word appears more than 50% of the time, it's runaway
        if word_counts:
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.5 and len(words) >= 5:
                most_common = max(word_counts, key=word_counts.get)
                print(f"Runaway speech detected: '{most_common}' appears {max_count}/{len(words)} times ({max_count/len(words)*100:.0f}%)")
                return True
        
        return False
                
    def detect_correction(self, new_text):
        """Detect if new text is a correction of previous transcription"""
        if not self.correction_history or not new_text:
            return False, ""
        
        # Check if new text corrects recent transcription
        for i, (old_text, timestamp) in enumerate(self.correction_history):
            # Only check recent corrections (within 5 seconds)
            if time.time() - timestamp > 5.0:
                continue
            
            old_words = old_text.lower().split()
            new_words = new_text.lower().split()
            
            # Check if new text extends or corrects old text
            if len(new_words) > len(old_words):
                # Check if old text is a prefix of new text
                if old_words == new_words[:len(old_words)]:
                    # This is an extension/correction
                    correction_part = " ".join(new_words[len(old_words):])
                    return True, correction_part
            
            # Check if new text replaces part of old text
            common_words = set(old_words) & set(new_words)
            if len(common_words) > 0 and len(common_words) < len(old_words):
                # Partial replacement detected
                return True, new_text
        
        return False, ""

    def add_to_correction_history(self, text):
        """Add text to correction history with timestamp"""
        if text and text.strip():
            self.correction_history.append((text.strip(), time.time()))
            # Keep only recent corrections
            if len(self.correction_history) > CORRECTION_WINDOW:
                self.correction_history.pop(0)

    def should_translate_now(self):
        """Determine if we should translate immediately based on context"""
        if not self.accumulated_text or self.translation_in_progress:
            return False
        
        # Check if we have enough new content
        new_content_length = len(self.accumulated_text) - self.translated_length
        if new_content_length < MIN_TRANSLATION_LENGTH:
            return False
        
        # Fast translation for short phrases
        if len(self.accumulated_text.split()) <= 5:
            return True
        
        # Check if we hit natural break points
        text = self.accumulated_text.strip()
        if text.endswith(('。', '，', '？', '！', '.', ',', '?', '!')):
            return True
        
        # Translate if we have significant new content
        if new_content_length >= 8:  # About 2-3 words
            return True
        
        return False

    def is_similar_text(self, text1, text2, threshold=0.7):
        """Check if two texts are similar to avoid duplicates"""
        if not text1 or not text2:
            return False
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold

    def detect_correction(self, new_text):
        """Detect if new text is a correction of previous transcription"""
        if not self.correction_history or not new_text:
            return False, ""
        
        # Check if new text corrects recent transcription
        for i, (old_text, timestamp) in enumerate(self.correction_history):
            # Only check recent corrections (within 5 seconds)
            if time.time() - timestamp > 5.0:
                continue
            
            old_words = old_text.lower().split()
            new_words = new_text.lower().split()
            
            # Check if new text extends or corrects old text
            if len(new_words) > len(old_words):
                # Check if old text is a prefix of new text
                if old_words == new_words[:len(old_words)]:
                    # This is an extension/correction
                    correction_part = " ".join(new_words[len(old_words):])
                    return True, correction_part
            
            # Check if new text replaces part of old text
            common_words = set(old_words) & set(new_words)
            if len(common_words) > 0 and len(common_words) < len(old_words):
                # Partial replacement detected
                return True, new_text
        
        return False, ""

    def add_to_correction_history(self, text):
        """Add text to correction history with timestamp"""
        if text and text.strip():
            self.correction_history.append((text.strip(), time.time()))
            # Keep only recent corrections
            if len(self.correction_history) > CORRECTION_WINDOW:
                self.correction_history.pop(0)

    def should_translate_now(self):
        """Determine if we should translate immediately based on context"""
        if not self.accumulated_text or self.translation_in_progress:
            return False
        
        # Check if we have enough new content
        new_content_length = len(self.accumulated_text) - self.translated_length
        if new_content_length < MIN_TRANSLATION_LENGTH:
            return False
        
        # Fast translation for short phrases
        if len(self.accumulated_text.split()) <= 5:
            return True
        
        # Check if we hit natural break points
        text = self.accumulated_text.strip()
        if text.endswith(('。', '，', '？', '！', '.', ',', '?', '!')):
            return True
        
        # Translate if we have significant new content
        if new_content_length >= 8:  # About 2-3 words
            return True
        
        return False

    def is_similar_text(self, text1, text2, threshold=0.7):
        """Check if two texts are similar to avoid duplicates"""
        if not text1 or not text2:
            return False
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold
    def continuous_transcription(self):
        """Continuous transcription thread with debug output"""
        last_transcribed_text = ""  # Track last transcription to avoid duplicates
        last_transcribed_time = 0   # Track when last transcription happened
        
        while self.is_running:
            try:
                if not self.model:
                    time.sleep(0.1)
                    continue
                
                audio_chunk = audio_queue.get(timeout=0.5)
                current_time = time.time()
                print(f"Processing audio chunk of length: {len(audio_chunk)} ({len(audio_chunk)/SAMPLE_RATE:.2f}s)")
                
                # Skip if too short
                if len(audio_chunk) < int(SAMPLE_RATE * MIN_AUDIO_LENGTH):
                    print(f"Audio too short: {len(audio_chunk)} < {int(SAMPLE_RATE * MIN_AUDIO_LENGTH)}")
                    continue
                
                # Get current language
                from_index = self.from_language_combo.current()
                if from_index >= 0:
                    language_code = self.languages[from_index]["code"]
                else:
                    language_code = "zh"
                
                print(f"Transcribing with language: {language_code}")
                
                # Normalize audio with better preprocessing and amplification
                audio_chunk_original = audio_chunk.copy()  # Keep original for Silero
                
                if np.max(np.abs(audio_chunk)) > 0:
                    # Normalize to -1 to 1 range
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                    
                    # Apply pre-emphasis filter to boost high frequencies (improves speech clarity)
                    pre_emphasis = 0.97
                    audio_chunk = np.append(audio_chunk[0], audio_chunk[1:] - pre_emphasis * audio_chunk[:-1])
                    
                    # Apply amplification for better detection
                    audio_chunk = np.clip(audio_chunk * 1.8, -1.0, 1.0)
                
                # Calculate audio energy to detect if this is mostly silence/noise
                audio_energy = np.sqrt(np.mean(audio_chunk**2))
                audio_duration = len(audio_chunk) / SAMPLE_RATE
                
                # If audio energy is very low, skip transcription (likely just background noise)
                if audio_energy < 0.01:  # Lowered from 0.02 to be more sensitive
                    print(f"Audio energy too low ({audio_energy:.4f}), skipping transcription (likely background noise)")
                    continue
                
                # SILERO VAD CHECK: Use ML model to verify speech presence (use ORIGINAL audio, not pre-emphasized)
                # Only check Silero if energy is borderline - if energy is high, trust it
                if audio_energy < 0.1:  # Increased from 0.08 - use Silero for more cases
                    if not self.check_speech_with_silero(audio_chunk_original):
                        print("Silero VAD: No speech detected, skipping transcription")
                        continue
                else:
                    print(f"High energy detected ({audio_energy:.4f}), bypassing Silero VAD")
                
                # Get initial prompt for better context (helps with accuracy)
                # Disabled - prompts were leaking into transcription output
                initial_prompt = None  # Was causing "The following is..." to appear in output
                
                # Optimized transcription settings - balanced speed and accuracy
                gpu_start = time.time()
                segments, info = self.model.transcribe(
                    audio_chunk,
                    language=language_code,
                    beam_size=5,  # Reduced from 10 - good balance (5 is sweet spot)
                    best_of=1,     # Disabled - was causing multiple versions (changed from 5)
                    temperature=0.0,  # Single temperature for speed (was multiple)
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.6,  # Increased from 0.5 - more strict
                        min_speech_duration_ms=300,  # Increased from 250
                        min_silence_duration_ms=500
                    ),
                    condition_on_previous_text=False,  # Disabled - prevents hallucinations
                    initial_prompt=initial_prompt,     # Keep - helps guide model
                    no_speech_threshold=0.8,  # Increased from 0.7 - much more strict
                    compression_ratio_threshold=2.4,   # Keep - prevents hallucinations
                    log_prob_threshold=-0.8,           # Increased from -1.0 - reject lower confidence
                    no_repeat_ngram_size=3,            # Prevent repeating 3-grams
                    word_timestamps=False
                )
                gpu_time = time.time() - gpu_start
                
                print(f"GPU transcription took: {gpu_time:.3f}s for {len(audio_chunk)/SAMPLE_RATE:.2f}s of audio")
                print(f"Transcription completed, processing segments...")
                
                # Process segments with minimal filtering
                new_text_parts = []
                total_segment_text_length = 0
                
                for segment in segments:
                    text = segment.text.strip()
                    print(f"Raw segment text: '{text}'")
                    if text and len(text) > 0:
                        total_segment_text_length += len(text)
                        
                        # Apply LLM correction if enabled
                        if self.llm_correction_var.get():
                            text = self.correct_transcription_with_llm(text, language_code)
                            print(f"After LLM correction: '{text}'")
                        
                        # Apply only basic cleaning
                        cleaned_text = self.clean_transcribed_text(text)
                        print(f"Cleaned text: '{cleaned_text}'")
                        if cleaned_text:
                            new_text_parts.append(cleaned_text)
                
                # Check for hallucination: if output text is way too long for audio duration, it's hallucinating
                # Typical speech rate: ~150 words/min = 2.5 words/sec
                # For Chinese: ~4-5 characters/sec, English: ~12-15 characters/sec
                expected_max_chars = audio_duration * 20  # Very generous estimate (20 chars/sec)
                
                if total_segment_text_length > expected_max_chars:
                    print(f"Hallucination detected: {total_segment_text_length} chars for {audio_duration:.1f}s audio (expected max {expected_max_chars:.0f})")
                    print("Skipping this transcription - likely background noise causing hallucination")
                    continue
                
                if new_text_parts:
                    new_text = " ".join(new_text_parts)
                    print(f"Final new text: '{new_text}'")
                    
                    # FIRST: Check for runaway speech (Whisper hallucination)
                    if self.detect_runaway_speech(new_text):
                        print("Runaway speech detected, skipping this transcription")
                        continue
                    
                    # Smart duplicate detection for live transcription
                    if last_transcribed_text:
                        new_lower = new_text.lower().strip()
                        last_lower = last_transcribed_text.lower().strip()
                        
                        # Exact duplicate check
                        if new_lower == last_lower and current_time - last_transcribed_time < 1.0:
                            print("Exact duplicate within 1s, skipping")
                            continue
                        
                        # Check if new text is just the last text repeated
                        # e.g., "do not" -> "do not do not"
                        if last_lower and new_lower.startswith(last_lower + " " + last_lower[:10]):
                            print(f"Detected repeated phrase at start, skipping")
                            continue
                        
                        # Check if new text contains last text repeated multiple times
                        if last_lower and len(last_lower) > 5:
                            count = new_lower.count(last_lower)
                            if count >= 2:
                                print(f"Detected phrase repeated {count} times, skipping")
                                continue
                        
                        # Check if new text is a subset of last text (shorter version)
                        if last_lower and len(new_lower) < len(last_lower):
                            if new_lower in last_lower:
                                print(f"New text is subset of last text, skipping")
                                continue
                        
                        # Check if last text is a subset of new text (new is longer version)
                        if last_lower and len(last_lower) < len(new_lower):
                            if last_lower in new_lower:
                                print(f"Last text is subset of new text, skipping (likely variation)")
                                continue
                        
                        # Check for very similar text (>70% word overlap)
                        if last_lower and len(last_lower) > 5 and len(new_lower) > 5:
                            last_words = set(last_lower.split())
                            new_words = set(new_lower.split())
                            if last_words and new_words:
                                overlap = len(last_words & new_words) / max(len(last_words), len(new_words))
                                if overlap > 0.7:
                                    print(f"Text too similar to last ({overlap*100:.0f}% overlap), skipping")
                                    continue
                        
                        # Check for word-level duplication (e.g., "penis penis")
                        new_words = new_lower.split()
                        if len(new_words) >= 2:
                            # Check if all words are the same
                            unique_words = set(new_words)
                            if len(unique_words) == 1:
                                print(f"All words are identical: '{new_words[0]}', skipping")
                                continue
                            
                            # Check if first and second half are identical
                            mid = len(new_words) // 2
                            if mid > 0:
                                first_half = " ".join(new_words[:mid])
                                second_half = " ".join(new_words[mid:mid*2])
                                if first_half == second_half and len(first_half) > 3:
                                    print(f"First and second half identical: '{first_half}', skipping")
                                    continue
                    
                    # Minimal spam checking - only check for obvious spam
                    is_obvious_spam = False
                    
                    # Only check for subtitle artifacts (English and Chinese)
                    text_lower = new_text.lower().strip()
                    spam_phrases = ["subtitles by", "so lan ya", "down to earth", "字幕by", "索兰娅", "索蘭婭"]
                    
                    for phrase in spam_phrases:
                        if phrase in text_lower or phrase in new_text:
                            is_obvious_spam = True
                            print(f"Detected spam phrase: {phrase}")
                            break
                    
                    # Check for excessive repetition within the text itself
                    words = new_text.split()
                    if len(words) >= 3:
                        # Count consecutive repeated words
                        consecutive_repeats = 0
                        for i in range(len(words) - 1):
                            if words[i].lower() == words[i+1].lower():
                                consecutive_repeats += 1
                        
                        # If more than 40% of words are consecutive repeats, it's spam
                        if consecutive_repeats / len(words) > 0.4:
                            is_obvious_spam = True
                            print(f"Excessive repetition detected: {consecutive_repeats}/{len(words)} consecutive repeats")
                    
                    if not is_obvious_spam:
                        print(f"Adding text to accumulated: '{new_text}'")
                        
                        # Auto-clear if source text is getting too long
                        if len(self.accumulated_text) > MAX_ACCUMULATED_LENGTH:
                            print(f"Source text too long ({len(self.accumulated_text)} chars), clearing old text")
                            # Keep only the most recent portion
                            words = self.accumulated_text.split()
                            if len(words) > 20:
                                # Calculate how much we're keeping
                                kept_text = " ".join(words[-20:])
                                removed_length = len(self.accumulated_text) - len(kept_text)
                                
                                # Update accumulated text
                                self.accumulated_text = kept_text
                                
                                # Adjust translated_length to account for removed text
                                # If we removed text that was already translated, adjust the counter
                                if self.translated_length > removed_length:
                                    self.translated_length -= removed_length
                                else:
                                    self.translated_length = 0
                                
                                # Mark that we've already translated what we kept
                                # This prevents re-translation of old text
                                self.translated_length = len(self.accumulated_text)
                                
                                ui_queue.put(("source_replace", self.accumulated_text))
                                print(f"Adjusted translated_length to {self.translated_length} after clearing")
                        
                        # Simple accumulation - just add the text
                        if self.accumulated_text:
                            # Don't add if it's exactly the same as what we just added
                            if not self.accumulated_text.endswith(new_text):
                                self.accumulated_text += " " + new_text
                            else:
                                print("Text already at end, skipping")
                                continue
                        else:
                            self.accumulated_text = new_text
                        
                        print(f"Accumulated text now: '{self.accumulated_text}'")
                        
                        # Update UI
                        ui_queue.put(("source_append", self.accumulated_text))
                        print(f"UI queue size: {ui_queue.qsize()}")
                        
                        # Only queue for translation if we have enough NEW text since last translation
                        new_content_length = len(self.accumulated_text) - self.translated_length
                        if new_content_length >= 2:  # Just 2 characters for Chinese (1-2 words)
                            # Get only the new text that hasn't been translated yet
                            new_text_to_translate = self.accumulated_text[self.translated_length:].strip()
                            
                            if len(new_text_to_translate) >= 1:  # Even 1 character is meaningful in Chinese
                                print(f"Queuing NEW text for translation: '{new_text_to_translate}'")
                                try:
                                    translate_queue.put_nowait(new_text_to_translate)
                                except queue.Full:
                                    print("Translation queue full, clearing...")
                                    while not translate_queue.empty():
                                        try:
                                            translate_queue.get_nowait()
                                        except queue.Empty:
                                            break
                                    translate_queue.put_nowait(new_text_to_translate)
                        
                        # Store this as the last transcribed text with timestamp
                        last_transcribed_text = new_text
                        last_transcribed_time = current_time
                    else:
                        print("Text marked as spam, skipping")
                else:
                    print("No text parts after processing")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()

    def realtime_translation(self):
        """Simplified real-time translation using requests"""
        while self.is_running:
            try:
                if not self.model:
                    time.sleep(0.1)
                    continue
                
                text_to_translate = translate_queue.get(timeout=0.5)
                
                # Skip if text is too short (but allow even 1 char for Chinese)
                if len(text_to_translate.strip()) < 1:
                    continue
                
                # Skip if same as last translated text (avoid redundant translations)
                if text_to_translate == self.last_translated_text:
                    print("Skipping duplicate translation")
                    continue
                
                # Faster translation timing - reduced debounce
                current_time = time.time()
                time_since_last = current_time - self.last_translate_time
                if time_since_last < TRANSLATION_DEBOUNCE:
                    sleep_time = TRANSLATION_DEBOUNCE - time_since_last
                    time.sleep(sleep_time)
                
                # Get current languages for prompt
                from_index = self.from_language_combo.current()
                to_index = self.to_language_combo.current()
                
                if from_index >= 0 and to_index >= 0:
                    source_language = self.languages[from_index]["name"]
                    target_language = self.languages[to_index]["name"]
                else:
                    source_language = "Chinese"
                    target_language = "English"
                
                # Check if translation text box is getting too full
                # Get current translation text length from UI
                try:
                    current_translation = self.english_text.get(1.0, tk.END).strip()
                    if len(current_translation) > MAX_TRANSLATION_TEXT_LENGTH:
                        print(f"Translation text too long ({len(current_translation)} chars), clearing")
                        ui_queue.put(("translation_clear", None))
                        time.sleep(0.05)  # Brief pause to let UI clear
                except:
                    pass
                
                # Add space before new translation if there's existing text
                ui_queue.put(("translation_append_start", None))
                
                print(f"Translating: '{text_to_translate}'")
                
                # Optimized translation with streaming using requests
                try:
                    response = requests.post(
                        LM_STUDIO_URL,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {LM_STUDIO_API_KEY}"
                        },
                        json={
                            "model": "local-model",
                            "messages": [
                                {"role": "system", "content": f"Translate {source_language} to {target_language}. Be concise and natural. Output only the translation."},
                                {"role": "user", "content": text_to_translate}
                            ],
                            "stream": True,
                            "max_tokens": 150,
                            "temperature": 0.1,
                            "top_p": 0.95
                        },
                        stream=True,
                        timeout=15
                    )
                    
                    # Stream translation and append to existing translation
                    if response.status_code == 200:
                        translation_buffer = ""
                        for line in response.iter_lines():
                            if line:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    data_str = line_text[6:]
                                    if data_str.strip() == '[DONE]':
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        if 'choices' in data and len(data['choices']) > 0:
                                            delta = data['choices'][0].get('delta', {})
                                            if 'content' in delta:
                                                content = delta['content']
                                                translation_buffer += content
                                                ui_queue.put(("translation_stream", content))
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Update translated length - mark current accumulated text as translated
                        self.translated_length = len(self.accumulated_text)
                        self.last_translated_text = text_to_translate
                        self.last_translate_time = time.time()
                        print(f"Translation complete. Translated length now: {self.translated_length}")
                    else:
                        print(f"Translation API error: {response.status_code}")
                        ui_queue.put(("translation_append", " [Translation Error]"))
                    
                except Exception as translation_error:
                    print(f"Translation API error: {translation_error}")
                    ui_queue.put(("translation_append", " [Translation Error]"))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Translation thread error: {e}")

    def process_audio_levels(self):
        """Process audio levels for smooth visualization"""
        while self.is_running:
            try:
                # Get latest level
                latest_level = None
                while not level_queue.empty():
                    try:
                        latest_level = level_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_level is not None:
                    # Smooth level changes
                    self.current_level = self.current_level * 0.7 + latest_level * 0.3
                
                time.sleep(0.016)  # ~60fps
                
            except Exception as e:
                print(f"Level processing error: {e}")

    def on_model_change(self, event=None):
        """Handle model selection change"""
        try:
            model_index = self.model_combo.current()
            if model_index >= 0:
                new_model_code = self.available_models[model_index]["code"]
                new_model_name = self.available_models[model_index]["name"]
                
                ui_queue.put(("status", f"Loading {new_model_name}..."))
                print(f"Changing model to: {new_model_code}")
                
                # Load new model in background thread
                def load_new_model():
                    try:
                        global MODEL_SIZE
                        
                        # Unload old model to free memory
                        if self.model is not None:
                            print("Unloading old model...")
                            del self.model
                            self.model = None
                            import gc
                            gc.collect()
                            
                            # Clear CUDA cache if using GPU
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    print("CUDA cache cleared")
                            except:
                                pass
                        
                        # Load new model
                        MODEL_SIZE = new_model_code
                        print(f"Loading new model: {new_model_code}")
                        self.model = WhisperModel(new_model_code, device="cuda", compute_type="float16")
                        ui_queue.put(("status", f"Model changed to {new_model_name}"))
                        print(f"Model loaded: {new_model_code}")
                    except Exception as e:
                        ui_queue.put(("status", f"Model load error: {e}"))
                        print(f"Model load error: {e}")
                
                threading.Thread(target=load_new_model, daemon=True).start()
                
        except Exception as e:
            print(f"Model change error: {e}")

    def on_language_change(self, event=None):
        """Handle language selection change"""
        try:
            from_index = self.from_language_combo.current()
            to_index = self.to_language_combo.current()
            
            if from_index >= 0 and to_index >= 0:
                # Prevent same language selection
                if from_index == to_index:
                    ui_queue.put(("status", "Cannot translate to the same language!"))
                    # Reset to previous valid selection
                    if hasattr(self, 'source_language_name'):
                        # Find the index of the previous source language
                        for i, lang in enumerate(self.languages):
                            if lang["name"] == self.source_language_name:
                                self.from_language_combo.current(i)
                                break
                    return
                
                self.source_language_name = self.languages[from_index]["name"]
                self.target_language_name = self.languages[to_index]["name"]
                
                self.source_label.config(text=f"{self.source_language_name} (Voice Recognition):")
                self.target_label.config(text=f"{self.target_language_name} (Live Translation):")
                
                # Clear text areas when language changes
                self.source_text.delete(1.0, tk.END)
                self.english_text.delete(1.0, tk.END)
                
                # Reset accumulated text and translation tracking
                self.accumulated_text = ""
                self.last_clean_text = ""
                self.translated_length = 0
                self.recent_transcriptions = []  # Clear spam detection history
                
                # Reset VAD state
                self.is_speaking = False
                self.speech_start_time = 0
                self.last_speech_time = 0
                self.silence_start_time = None
                self.potential_speech_start = None
                self.energy_history = []
                
                ui_queue.put(("status", f"Translating {self.source_language_name} → {self.target_language_name}"))
                print(f"Language changed: {self.source_language_name} → {self.target_language_name}")
        except Exception as e:
            print(f"Language change error: {e}")

    def on_mic_change(self, event=None):
        """Handle microphone change"""
        try:
            print("Microphone change detected")
            
            # Stop current stream safely
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    self.stream.close()
                    print("Previous stream closed")
                except Exception as e:
                    print(f"Error closing stream: {e}")
                self.stream = None
            
            # Validate selected device
            device_index = self.mic_combo.current()
            if device_index >= 0 and device_index < len(self.mics):
                device_info = self.mics[device_index]
                if device_info['max_input_channels'] < 1:
                    ui_queue.put(("status", "Selected device has no input channels!"))
                    print(f"Error: Device {device_index} has no input channels")
                    return
            
            # Start new stream after a short delay
            time.sleep(0.5)
            threading.Thread(target=self.start_audio_stream, daemon=True).start()
            
        except Exception as e:
            print(f"Mic change error: {e}")
            ui_queue.put(("status", f"Mic change error: {e}"))

    def on_gain_change(self, value):
        """Handle microphone gain change"""
        try:
            self.mic_gain = float(value)
            self.gain_label.config(text=f"{self.mic_gain:.1f}x")
            print(f"Mic gain set to: {self.mic_gain:.1f}x")
        except Exception as e:
            print(f"Gain change error: {e}")

    def is_repetitive_or_spam(self, text):
        """Check if text is repetitive or spam"""
        if not text or len(text.strip()) < 2:
            return True
        
        text_clean = text.strip()
        
        # Check against recent transcriptions
        for recent_text in self.recent_transcriptions:
            if self.is_similar_text(text_clean, recent_text, threshold=0.7):
                return True
        
        # Check for internal repetition
        words = text_clean.split()
        if len(words) > 1:
            # Check if more than 60% of words are repeated
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            if repeated_words / len(words) > 0.6:
                return True
        
        # Check for very short repeated phrases
        if len(words) <= 2 and text_clean in self.accumulated_text:
            return True
        
        return False

    def add_to_recent_transcriptions(self, text):
        """Add text to recent transcriptions list"""
        if text and text.strip():
            self.recent_transcriptions.append(text.strip())
            # Keep only last 5 transcriptions
            if len(self.recent_transcriptions) > 5:
                self.recent_transcriptions.pop(0)

    def update_ui(self):
        """Update UI elements"""
        try:
            # Process ALL pending UI updates immediately (no limit)
            updates_processed = 0
            while updates_processed < 100:  # Higher limit to clear backlog faster
                try:
                    cmd, payload = ui_queue.get_nowait()
                    updates_processed += 1
                    
                    if cmd == "status":
                        self.status_label.config(text=payload)
                    elif cmd == "source_append":
                        # Replace source text with accumulated text
                        self.source_text.delete(1.0, tk.END)
                        self.source_text.insert(1.0, payload)
                        self.source_text.see(tk.END)
                    elif cmd == "source_replace":
                        # Replace source text (used when clearing old text)
                        self.source_text.delete(1.0, tk.END)
                        self.source_text.insert(1.0, payload)
                        self.source_text.see(tk.END)
                    elif cmd == "translation_clear":
                        self.english_text.delete(1.0, tk.END)
                    elif cmd == "translation_append_start":
                        # Add a space before new translation if there's existing text
                        current_content = self.english_text.get(1.0, tk.END).strip()
                        if current_content:
                            self.english_text.insert(tk.END, " ")
                    elif cmd == "translation_stream":
                        self.english_text.insert(tk.END, payload)
                        self.english_text.see(tk.END)
                    elif cmd == "translation_append":
                        # Append translation text
                        self.english_text.insert(tk.END, payload)
                        self.english_text.see(tk.END)
                    elif cmd == "translation_complete":
                        # Handle non-streaming translation fallback - append instead of replace
                        current_content = self.english_text.get(1.0, tk.END).strip()
                        if current_content:
                            self.english_text.insert(tk.END, " " + payload)
                        else:
                            self.english_text.insert(1.0, payload)
                        self.english_text.see(tk.END)
                        
                except queue.Empty:
                    break
            
            # Update level bar
            self.update_level_bar()
            
        except Exception as e:
            print(f"UI update error: {e}")
        
        # Schedule next update - faster refresh
        self.root.after(8, self.update_ui)  # ~120fps for more responsive UI

    def update_level_bar(self):
        """Update audio level visualization"""
        try:
            self.level_canvas.delete("all")
            width = self.level_canvas.winfo_width()
            
            if width > 1:
                level_width = int(width * min(1.0, self.current_level * 20))
                if level_width > 0:
                    # Dynamic color based on level
                    if self.current_level < 0.3:
                        color = "lime"
                    elif self.current_level < 0.7:
                        color = "yellow"
                    else:
                        color = "red"
                    
                    self.level_canvas.create_rectangle(0, 0, level_width, 20, fill=color, outline="")
        except Exception as e:
            print(f"Level bar error: {e}")

    def toggle_web_server(self):
        """Toggle web server on/off"""
        if self.web_server_var.get():
            self.start_web_server()
        else:
            self.stop_web_server()

    def web_audio_translation(self):
        """Process web microphone audio and translate"""
        accumulated_text = ""
        last_translation_time = 0
        screen_accumulated_text = ""  # Separate accumulator for screen share
        last_activity_time = time.time()
        
        while self.is_running:
            try:
                # Get web audio text from queue
                data = web_audio_queue.get(timeout=0.5)
                text = data.get('text', '')
                from_lang = data.get('from_lang', 'zh')
                to_lang = data.get('to_lang', 'en')
                is_screen = data.get('is_screen', False)
                
                if not text:
                    continue
                
                # Mark web as active (pause PC audio)
                self.web_is_active = True
                last_activity_time = time.time()
                
                # For screen share, use the text directly without accumulation
                # This ensures each chunk is translated independently
                if is_screen:
                    accumulated_text = text.strip()
                    print(f"Screen share text (no accumulation): '{accumulated_text}'")
                else:
                    # For web microphone, accumulate text
                    accumulated_text += " " + text
                    accumulated_text = accumulated_text.strip()
                    print(f"Web mic text accumulated: '{accumulated_text}'")
                
                # Translate immediately (real-time)
                current_time = time.time()
                
                # For screen share, don't throttle - translate every chunk
                # For web mic, throttle to every 0.5 seconds
                if not is_screen and (current_time - last_translation_time < 0.5):
                    continue
                
                last_translation_time = current_time
                
                print(f"Translating {'screen' if is_screen else 'web mic'} text: '{accumulated_text}' from {from_lang} to {to_lang}")
                
                # Get language names
                lang_map = {'zh': 'Chinese', 'ru': 'Russian', 'en': 'English'}
                source_language = lang_map.get(from_lang, 'Chinese')
                target_language = lang_map.get(to_lang, 'English')
                
                # Translate using LM Studio
                try:
                    response = requests.post(
                        LM_STUDIO_URL,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {LM_STUDIO_API_KEY}"
                        },
                        json={
                            "model": "local-model",
                            "messages": [
                                {"role": "system", "content": f"Translate {source_language} to {target_language}. Be concise and natural. Output only the translation."},
                                {"role": "user", "content": accumulated_text}
                            ],
                            "stream": False,
                            "max_tokens": 200,
                            "temperature": 0.1
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            translation = result['choices'][0]['message']['content'].strip()
                            print(f"{'Screen' if is_screen else 'Web mic'} translation result: '{translation}'")
                            
                            # Store globally for persistent access
                            global web_current_translation, web_current_source
                            web_current_translation = translation
                            web_current_source = accumulated_text
                            
                            # Clear old translations and add new one
                            while not web_translation_queue.empty():
                                try:
                                    web_translation_queue.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Put new translation
                            web_translation_queue.put_nowait(translation)
                            print(f"Translation queued for web display")
                            
                            # For screen share, clear accumulated text after translation
                            # This ensures each chunk is independent
                            if is_screen:
                                screen_accumulated_text = ""
                                print("Screen accumulator cleared for next chunk")
                    else:
                        print(f"Translation API error: {response.status_code}")
                        web_translation_queue.put_nowait(f"[Translation Error: {response.status_code}]")
                        
                except requests.exceptions.Timeout:
                    print(f"Web translation timeout")
                    web_translation_queue.put_nowait("[Translation Timeout]")
                except Exception as e:
                    print(f"Web translation error: {e}")
                    web_translation_queue.put_nowait(f"[Error: {str(e)}]")
                    
            except queue.Empty:
                # Only mark inactive after 5 seconds of no activity (not 0.5s)
                if self.web_is_active:
                    current_time = time.time()
                    if current_time - last_activity_time > 5.0:
                        print("Web client inactive for 5s - resuming PC audio")
                        self.web_is_active = False
                        accumulated_text = ""  # Reset for next session
                        screen_accumulated_text = ""  # Reset screen accumulator
                continue
            except Exception as e:
                print(f"Web audio translation thread error: {e}")
                import traceback
                traceback.print_exc()

    def start_web_server(self):
        """Start the Flask web server"""
        if not FLASK_AVAILABLE:
            print("Flask not available, cannot start web server")
            self.web_server_var.set(False)
            return
        
        if self.web_server_enabled:
            print("Web server already running")
            return
        
        print("Starting web server on http://0.0.0.0:1235")
        self.web_server_enabled = True
        
        # Start web server in separate thread
        self.web_server_thread = threading.Thread(target=self.run_web_server, daemon=True)
        self.web_server_thread.start()
        
        ui_queue.put(("status", "Web server started on http://0.0.0.0:1235"))

    def stop_web_server(self):
        """Stop the Flask web server"""
        if not self.web_server_enabled:
            return
        
        print("Stopping web server...")
        self.web_server_enabled = False
        
        # Try to shutdown Flask gracefully
        try:
            # Send shutdown request to Flask
            requests.post('http://127.0.0.1:1235/shutdown', timeout=2)
        except:
            pass
        
        ui_queue.put(("status", "Web server stopped"))

    def run_web_server(self):
        """Run Flask web server"""
        app = Flask(__name__)
        CORS(app)  # Enable CORS for cross-origin requests
        
        # HTML template for the web interface
        HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Voice Translator</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: #4CAF50;
            color: white;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .mode-selector {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .mode-button {
            padding: 12px 24px;
            margin: 0 10px;
            border: 2px solid #2196F3;
            background: white;
            color: #2196F3;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .mode-button:hover {
            background: #E3F2FD;
        }
        .mode-button.active {
            background: #2196F3;
            color: white;
        }
        .controls {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .control-group {
            margin: 10px 0;
        }
        label {
            font-weight: bold;
            margin-right: 10px;
        }
        select {
            padding: 5px 10px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .mic-control {
            text-align: center;
            margin: 20px 0;
        }
        .mic-button {
            padding: 20px 40px;
            font-size: 18px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .mic-button.inactive {
            background: #4CAF50;
            color: white;
        }
        .mic-button.inactive:hover {
            background: #45a049;
        }
        .mic-button.active {
            background: #f44336;
            color: white;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .text-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            min-height: 150px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .text-box h2 {
            margin-top: 0;
            color: #555;
        }
        .source-text {
            border-left: 4px solid #2196F3;
        }
        .translation-text {
            border-left: 4px solid #4CAF50;
        }
        .text-content {
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .footer {
            text-align: center;
            color: #777;
            margin-top: 20px;
        }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>🎤 Real-time Voice Translator</h1>
    
    <div class="status" id="status">
        Connecting to server...
    </div>
    
    <div class="mode-selector">
        <h3>Select Mode:</h3>
        <button class="mode-button active" id="watchMode" onclick="setMode('watch')">
            👀 Watch Mode (View PC Transcription)
        </button>
        <button class="mode-button" id="speakMode" onclick="setMode('speak')">
            🎤 Speak Mode (Use Web Microphone)
        </button>
        <button class="mode-button" id="typeMode" onclick="setMode('type')">
            ⌨️ Type Mode (Text Translation)
        </button>
        <button class="mode-button" id="screenMode" onclick="setMode('screen')">
            🖥️ Screen Share (Capture Tab Audio)
        </button>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>From Language:</label>
            <select id="fromLang">
                <option value="zh">Chinese</option>
                <option value="ru">Russian</option>
                <option value="en">English</option>
            </select>
            
            <label style="margin-left: 20px;">To Language:</label>
            <select id="toLang">
                <option value="en" selected>English</option>
                <option value="zh">Chinese</option>
                <option value="ru">Russian</option>
            </select>
        </div>
        
        <div class="control-group" style="text-align: center; margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
            <button onclick="showOBSLink()" style="padding: 10px 20px; background: #9C27B0; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 14px;">
                📺 Get OBS Captions Link
            </button>
            <div id="obsLinkContainer" style="display: none; margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                <p style="margin: 5px 0; font-weight: bold;">OBS Browser Source URL:</p>
                <input type="text" id="obsLink" readonly style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; font-size: 12px;">
                <button onclick="copyOBSLink()" style="margin-top: 8px; padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    📋 Copy Link
                </button>
                <p style="margin: 10px 0 5px 0; font-size: 12px; color: #666;">
                    <strong>How to use in OBS:</strong><br>
                    1. Add a &quot;Browser&quot; source in OBS<br>
                    2. Paste the URL above<br>
                    3. Set width: 1920, height: 200<br>
                    4. Check &quot;Shutdown source when not visible&quot;<br>
                    5. Captions will appear automatically!
                </p>
            </div>
        </div>
    </div>
    
    <div id="speakControls" style="display: none;">
        <div class="mic-control">
            <button class="mic-button inactive" id="micButton" onclick="toggleMicrophone()">
                🎤 Start Speaking
            </button>
        </div>
        <div class="warning" id="browserWarning" style="display: none;">
            ⚠️ Speech recognition not available. This may be because:<br>
            • You're using an unsupported browser (use Chrome, Edge, or Safari)<br>
            • You're on mobile Safari without HTTPS (iOS requires secure connection)<br>
            <br>
            <strong>For mobile users:</strong> Access via HTTPS or use desktop browser
        </div>
    </div>
    
    <div id="typeControls" style="display: none;">
        <div class="controls">
            <div class="control-group">
                <label>Type your text:</label>
                <textarea id="typeInput" rows="4" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd; font-size: 16px;" placeholder="Type text here and press Translate..."></textarea>
            </div>
            <div class="control-group" style="text-align: center; margin-top: 10px;">
                <button onclick="translateTypedText()" style="padding: 12px 30px; background: #4CAF50; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; font-weight: bold;">
                    🔄 Translate
                </button>
                <button onclick="clearTypedText()" style="padding: 12px 30px; background: #f44336; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; font-weight: bold; margin-left: 10px;">
                    🗑️ Clear
                </button>
            </div>
        </div>
    </div>
    
    <div id="screenControls" style="display: none;">
        <div class="mic-control">
            <button class="mic-button inactive" id="screenButton" onclick="toggleScreenShare()">
                🖥️ Start Screen Share
            </button>
        </div>
        <div class="warning" id="screenWarning" style="display: none;">
            ⚠️ Screen sharing not available or permission denied.
        </div>
        <div class="warning" style="background: #e3f2fd; border-color: #2196F3;">
            ℹ️ <strong>How to use:</strong><br>
            1. Click "Start Screen Share"<br>
            2. Select "Chrome Tab" (not entire screen)<br>
            3. Check "Share tab audio" checkbox<br>
            4. Select the tab you want to capture<br>
            5. Audio from that tab will be transcribed and translated
        </div>
    </div>
    
    <div class="text-box source-text">
        <h2 id="sourceLabel">Source (Voice Recognition)</h2>
        <div class="text-content" id="sourceText">Waiting for speech...</div>
    </div>
    
    <div class="text-box translation-text">
        <h2 id="translationLabel">Translation (Live)</h2>
        <div class="text-content" id="translationText">Waiting for translation...</div>
    </div>
    
    <div class="footer">
        <p>Connected to: <strong>{{ host }}</strong></p>
        <p id="modeStatus">Mode: Watch (viewing PC transcription)</p>
    </div>
    
    <script>
        let sourceText = '';
        let translationText = '';
        let currentMode = 'watch';
        let recognition = null;
        let isListening = false;
        let webSourceText = '';
        let screenStream = null;
        let audioContext = null;
        let mediaRecorder = null;
        let isScreenSharing = false;
        
        // Initialize speech recognition
        function initSpeechRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            // Check if we're on mobile Safari
            const isMobileSafari = /iPhone|iPad|iPod/.test(navigator.userAgent) && /Safari/.test(navigator.userAgent);
            const isHTTPS = window.location.protocol === 'https:';
            
            if (!SpeechRecognition) {
                document.getElementById('browserWarning').style.display = 'block';
                console.log('Speech Recognition not available');
                console.log('User Agent:', navigator.userAgent);
                console.log('Protocol:', window.location.protocol);
                return null;
            }
            
            // Mobile Safari requires HTTPS
            if (isMobileSafari && !isHTTPS) {
                document.getElementById('browserWarning').innerHTML = 
                    '⚠️ <strong>iOS Safari requires HTTPS for speech recognition.</strong><br>' +
                    'Please access this page via:<br>' +
                    '• <strong>https://' + window.location.host + '</strong><br>' +
                    '• Or use a desktop browser';
                document.getElementById('browserWarning').style.display = 'block';
                return null;
            }
            
            try {
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                
                // Get selected language
                const fromLang = document.getElementById('fromLang').value;
                recognition.lang = fromLang === 'zh' ? 'zh-CN' : fromLang === 'ru' ? 'ru-RU' : 'en-US';
                
                recognition.onresult = function(event) {
                    let interimTranscript = '';
                    let finalTranscript = '';
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript + ' ';
                        } else {
                            interimTranscript += transcript;
                        }
                    }
                    
                    if (finalTranscript) {
                        webSourceText += finalTranscript;
                        document.getElementById('sourceText').textContent = webSourceText;
                        
                        console.log('Final transcript:', finalTranscript);
                        console.log('Sending to server for translation...');
                        
                        // Send to server for translation
                        fetch('/api/web_transcribe', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                text: finalTranscript.trim(),
                                from_lang: document.getElementById('fromLang').value,
                                to_lang: document.getElementById('toLang').value
                            })
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Server response:', data);
                            if (!data.success) {
                                console.error('Translation failed:', data.error);
                            }
                        })
                        .catch(error => {
                            console.error('Failed to send transcription:', error);
                        });
                    } else if (interimTranscript) {
                        document.getElementById('sourceText').textContent = webSourceText + interimTranscript;
                    }
                };
                
                recognition.onerror = function(event) {
                    console.error('Speech recognition error:', event.error);
                    if (event.error === 'not-allowed') {
                        document.getElementById('browserWarning').innerHTML = 
                            '⚠️ <strong>Microphone access denied.</strong><br>' +
                            'Please allow microphone access in your browser settings.';
                        document.getElementById('browserWarning').style.display = 'block';
                        isListening = false;
                        const button = document.getElementById('micButton');
                        button.textContent = '🎤 Start Speaking';
                        button.classList.remove('active');
                        button.classList.add('inactive');
                    } else if (event.error === 'no-speech') {
                        // Restart if no speech detected
                        if (isListening) {
                            recognition.start();
                        }
                    }
                };
                
                recognition.onend = function() {
                    if (isListening) {
                        recognition.start(); // Restart for continuous listening
                    }
                };
                
                return recognition;
            } catch (error) {
                console.error('Error initializing speech recognition:', error);
                document.getElementById('browserWarning').style.display = 'block';
                return null;
            }
        }
        
        // Set mode
        function setMode(mode) {
            currentMode = mode;
            
            // Update button styles
            document.getElementById('watchMode').classList.toggle('active', mode === 'watch');
            document.getElementById('speakMode').classList.toggle('active', mode === 'speak');
            document.getElementById('typeMode').classList.toggle('active', mode === 'type');
            document.getElementById('screenMode').classList.toggle('active', mode === 'screen');
            
            // Show/hide controls
            document.getElementById('speakControls').style.display = mode === 'speak' ? 'block' : 'none';
            document.getElementById('typeControls').style.display = mode === 'type' ? 'block' : 'none';
            document.getElementById('screenControls').style.display = mode === 'screen' ? 'block' : 'none';
            
            // Update status
            let statusText = '';
            if (mode === 'watch') {
                statusText = 'Mode: Watch (viewing PC transcription)';
            } else if (mode === 'speak') {
                statusText = 'Mode: Speak (using web microphone)';
            } else if (mode === 'type') {
                statusText = 'Mode: Type (text translation)';
            } else if (mode === 'screen') {
                statusText = 'Mode: Screen Share (capturing tab audio)';
            }
            document.getElementById('modeStatus').textContent = statusText;
            
            // Stop listening if switching away from speak mode
            if (mode !== 'speak' && isListening) {
                toggleMicrophone();
            }
            
            // Stop screen sharing if switching away
            if (mode !== 'screen' && isScreenSharing) {
                toggleScreenShare();
            }
            
            // Clear text when switching modes
            if (mode === 'speak') {
                webSourceText = '';
                document.getElementById('sourceText').textContent = 'Click "Start Speaking" to begin...';
                document.getElementById('translationText').textContent = 'Waiting for speech...';
            } else if (mode === 'type') {
                document.getElementById('sourceText').textContent = 'Type text and click Translate...';
                document.getElementById('translationText').textContent = 'Translation will appear here...';
            } else if (mode === 'screen') {
                document.getElementById('sourceText').textContent = 'Click "Start Screen Share" to capture tab audio...';
                document.getElementById('translationText').textContent = 'Waiting for audio...';
            }
        }
        
        // Toggle screen share
        async function toggleScreenShare() {
            const button = document.getElementById('screenButton');
            
            if (isScreenSharing) {
                // Stop screen sharing
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                if (screenStream) {
                    screenStream.getTracks().forEach(track => track.stop());
                    screenStream = null;
                }
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                
                isScreenSharing = false;
                button.textContent = '🖥️ Start Screen Share';
                button.classList.remove('active');
                button.classList.add('inactive');
                console.log('Screen sharing stopped');
            } else {
                // Start screen sharing
                try {
                    // Request screen share with audio
                    screenStream = await navigator.mediaDevices.getDisplayMedia({
                        video: true,
                        audio: {
                            echoCancellation: false,
                            noiseSuppression: false,
                            autoGainControl: false
                        }
                    });
                    
                    // Check if audio track exists
                    const audioTrack = screenStream.getAudioTracks()[0];
                    if (!audioTrack) {
                        alert('No audio track found. Make sure to check "Share tab audio" when selecting the tab.');
                        screenStream.getTracks().forEach(track => track.stop());
                        return;
                    }
                    
                    console.log('Screen share started with audio');
                    
                    // Create audio-only stream for MediaRecorder
                    const audioStream = new MediaStream([audioTrack]);
                    
                    // Try different MIME types
                    let mimeType = 'audio/webm';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = 'audio/webm;codecs=opus';
                        if (!MediaRecorder.isTypeSupported(mimeType)) {
                            mimeType = 'audio/ogg;codecs=opus';
                            if (!MediaRecorder.isTypeSupported(mimeType)) {
                                mimeType = '';  // Let browser choose
                            }
                        }
                    }
                    
                    console.log('Using MIME type:', mimeType || 'default');
                    
                    // Create AudioContext for processing
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(audioStream);
                    
                    // Create ScriptProcessor for capturing raw audio data
                    const bufferSize = 4096;
                    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
                    
                    let audioBuffers = [];
                    let isProcessing = false;
                    
                    processor.onaudioprocess = (e) => {
                        // If processing, skip this chunk to prevent memory buildup
                        if (isProcessing) {
                            console.log('Skipping audio chunk (still processing previous)');
                            return;
                        }
                        
                        // Get audio data
                        const inputData = e.inputBuffer.getChannelData(0);
                        const buffer = new Float32Array(inputData);
                        audioBuffers.push(buffer);
                        
                        // Process every 3 seconds worth of audio (16000 samples/sec * 3 = 48000 samples)
                        const totalSamples = audioBuffers.reduce((sum, buf) => sum + buf.length, 0);
                        
                        // Also prevent buffer from growing too large (max 5 seconds)
                        if (totalSamples > 80000) {
                            console.warn('Audio buffer overflow, clearing old data');
                            audioBuffers = audioBuffers.slice(-10); // Keep only last 10 chunks
                        }
                        
                        if (totalSamples >= 48000) {
                            isProcessing = true;
                            
                            // Combine all buffers
                            const combinedBuffer = new Float32Array(totalSamples);
                            let offset = 0;
                            for (const buf of audioBuffers) {
                                combinedBuffer.set(buf, offset);
                                offset += buf.length;
                            }
                            
                            // Clear buffers for next batch IMMEDIATELY
                            audioBuffers = [];
                            
                            console.log(`Processing ${totalSamples} audio samples (${(totalSamples/16000).toFixed(1)}s)`);
                            
                            // Convert Float32Array to Int16Array (WAV format)
                            const int16Buffer = new Int16Array(combinedBuffer.length);
                            for (let i = 0; i < combinedBuffer.length; i++) {
                                const s = Math.max(-1, Math.min(1, combinedBuffer[i]));
                                int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                            }
                            
                            // Create WAV file
                            const wavBuffer = createWavFile(int16Buffer, 16000);
                            const base64Audio = arrayBufferToBase64(wavBuffer);
                            
                            console.log(`Sending WAV audio to server (${wavBuffer.byteLength} bytes)...`);
                            
                            fetch('/api/screen_audio', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    audio: base64Audio,
                                    from_lang: document.getElementById('fromLang').value,
                                    to_lang: document.getElementById('toLang').value,
                                    mime_type: 'audio/wav',
                                    is_wav: true
                                })
                            })
                            .then(response => response.json())
                            .then(data => {
                                console.log('Server response:', data);
                                if (data.text) {
                                    // Accumulate source text
                                    const currentSource = document.getElementById('sourceText').textContent;
                                    if (currentSource.includes('Click "Start Screen Share"') || currentSource.includes('Waiting')) {
                                        document.getElementById('sourceText').textContent = data.text;
                                    } else {
                                        document.getElementById('sourceText').textContent = currentSource + ' ' + data.text;
                                    }
                                }
                                if (data.error) {
                                    console.error('Server error:', data.error);
                                }
                                isProcessing = false;
                            })
                            .catch(error => {
                                console.error('Error sending audio:', error);
                                isProcessing = false;
                            });
                        }
                    };
                    
                    // Connect audio pipeline
                    source.connect(processor);
                    processor.connect(audioContext.destination);
                    
                    // Helper function to create WAV file
                    function createWavFile(samples, sampleRate) {
                        const buffer = new ArrayBuffer(44 + samples.length * 2);
                        const view = new DataView(buffer);
                        
                        // WAV header
                        writeString(view, 0, 'RIFF');
                        view.setUint32(4, 36 + samples.length * 2, true);
                        writeString(view, 8, 'WAVE');
                        writeString(view, 12, 'fmt ');
                        view.setUint32(16, 16, true); // fmt chunk size
                        view.setUint16(20, 1, true); // PCM format
                        view.setUint16(22, 1, true); // mono
                        view.setUint32(24, sampleRate, true);
                        view.setUint32(28, sampleRate * 2, true); // byte rate
                        view.setUint16(32, 2, true); // block align
                        view.setUint16(34, 16, true); // bits per sample
                        writeString(view, 36, 'data');
                        view.setUint32(40, samples.length * 2, true);
                        
                        // Write samples
                        const offset = 44;
                        for (let i = 0; i < samples.length; i++) {
                            view.setInt16(offset + i * 2, samples[i], true);
                        }
                        
                        return buffer;
                    }
                    
                    function writeString(view, offset, string) {
                        for (let i = 0; i < string.length; i++) {
                            view.setUint8(offset + i, string.charCodeAt(i));
                        }
                    }
                    
                    function arrayBufferToBase64(buffer) {
                        let binary = '';
                        const bytes = new Uint8Array(buffer);
                        for (let i = 0; i < bytes.byteLength; i++) {
                            binary += String.fromCharCode(bytes[i]);
                        }
                        return btoa(binary);
                    }
                    
                    // Store references for cleanup
                    mediaRecorder = { 
                        stop: () => {
                            processor.disconnect();
                            source.disconnect();
                            audioContext.close();
                            console.log('Audio processing stopped');
                        },
                        state: 'recording'
                    };
                    
                    isScreenSharing = true;
                    button.textContent = '🛑 Stop Screen Share';
                    button.classList.remove('inactive');
                    button.classList.add('active');
                    
                    // Handle stream end (user stops sharing)
                    screenStream.getVideoTracks()[0].onended = () => {
                        console.log('Screen share ended by user');
                        if (isScreenSharing) {
                            toggleScreenShare();
                        }
                    };
                    
                } catch (error) {
                    console.error('Screen share error:', error);
                    document.getElementById('screenWarning').textContent = 
                        '⚠️ ' + (error.message || 'Screen sharing failed. Make sure to allow screen sharing and select "Share tab audio".');
                    document.getElementById('screenWarning').style.display = 'block';
                }
            }
        }
        
        // Translate typed text
        function translateTypedText() {
            const text = document.getElementById('typeInput').value.trim();
            
            if (!text) {
                alert('Please enter some text to translate');
                return;
            }
            
            // Show source text
            document.getElementById('sourceText').textContent = text;
            document.getElementById('translationText').textContent = 'Translating...';
            
            // Send to server for translation
            fetch('/api/web_transcribe', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    text: text,
                    from_lang: document.getElementById('fromLang').value,
                    to_lang: document.getElementById('toLang').value
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Translation will be fetched by updateDisplay
                    console.log('Translation queued');
                } else {
                    document.getElementById('translationText').textContent = 'Translation error: ' + (data.error || 'Unknown error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('translationText').textContent = 'Connection error';
            });
        }
        
        // Clear typed text
        function clearTypedText() {
            document.getElementById('typeInput').value = '';
            document.getElementById('sourceText').textContent = 'Type text and click Translate...';
            document.getElementById('translationText').textContent = 'Translation will appear here...';
        }
        
        // Allow Enter key to translate (Shift+Enter for new line)
        document.addEventListener('DOMContentLoaded', function() {
            const typeInput = document.getElementById('typeInput');
            if (typeInput) {
                typeInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        translateTypedText();
                    }
                });
            }
        });
        
        // Toggle microphone
        function toggleMicrophone() {
            if (!recognition) {
                recognition = initSpeechRecognition();
                if (!recognition) return;
            }
            
            const button = document.getElementById('micButton');
            
            if (isListening) {
                recognition.stop();
                isListening = false;
                button.textContent = '🎤 Start Speaking';
                button.classList.remove('active');
                button.classList.add('inactive');
            } else {
                // Update language before starting
                const fromLang = document.getElementById('fromLang').value;
                recognition.lang = fromLang === 'zh' ? 'zh-CN' : fromLang === 'ru' ? 'ru-RU' : 'en-US';
                
                recognition.start();
                isListening = true;
                button.textContent = '🛑 Stop Speaking';
                button.classList.remove('inactive');
                button.classList.add('active');
                webSourceText = '';
            }
        }
        
        // Update display
        function updateDisplay() {
            if (currentMode === 'watch') {
                // Watch mode - get data from PC
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                        document.getElementById('status').style.background = data.is_running ? '#4CAF50' : '#f44336';
                        
                        // Update source text if changed
                        if (data.source_text && data.source_text !== sourceText) {
                            sourceText = data.source_text;
                            document.getElementById('sourceText').textContent = sourceText;
                        }
                        
                        // Update translation if changed
                        if (data.translation_text && data.translation_text !== translationText) {
                            translationText = data.translation_text;
                            document.getElementById('translationText').textContent = translationText;
                        }
                        
                        // Update labels
                        document.getElementById('sourceLabel').textContent = 
                            data.from_language + ' (Voice Recognition)';
                        document.getElementById('translationLabel').textContent = 
                            data.to_language + ' (Live Translation)';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('status').textContent = 'Connection error';
                        document.getElementById('status').style.background = '#f44336';
                    });
            } else if (currentMode === 'speak' || currentMode === 'type' || currentMode === 'screen') {
                // Speak/Type/Screen mode - get translation from server
                fetch('/api/web_translation')
                    .then(response => response.json())
                    .then(data => {
                        // Always update translation if available
                        if (data.translation) {
                            document.getElementById('translationText').textContent = data.translation;
                        }
                        // Update source text for screen mode
                        if (currentMode === 'screen' && data.source) {
                            const sourceBox = document.getElementById('sourceText');
                            const currentText = sourceBox.textContent;
                            // Only update if it's different and not empty
                            if (data.source && data.source !== currentText && !currentText.includes('Click "Start Screen Share"')) {
                                sourceBox.textContent = data.source;
                            }
                        }
                    })
                    .catch(error => console.error('Translation error:', error));
            }
        }
        
        // Update every 500ms
        setInterval(updateDisplay, 500);
        updateDisplay();
        
        // Language change handlers
        document.getElementById('fromLang').addEventListener('change', function() {
            if (currentMode === 'watch') {
                fetch('/api/set_language', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        from: this.value,
                        to: document.getElementById('toLang').value
                    })
                });
            } else {
                // Update speech recognition language
                if (recognition && isListening) {
                    recognition.stop();
                    isListening = false;
                    setTimeout(() => {
                        toggleMicrophone();
                    }, 100);
                }
            }
        });
        
        document.getElementById('toLang').addEventListener('change', function() {
            if (currentMode === 'watch') {
                fetch('/api/set_language', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        from: document.getElementById('fromLang').value,
                        to: this.value
                    })
                });
            }
        });
        
        // OBS Captions Link Functions
        function showOBSLink() {
            const container = document.getElementById('obsLinkContainer');
            const input = document.getElementById('obsLink');
            
            // Generate the OBS link
            const protocol = window.location.protocol;
            const host = window.location.host;
            const obsUrl = protocol + '//' + host + '/obs-captions';
            
            input.value = obsUrl;
            container.style.display = 'block';
        }
        
        function copyOBSLink() {
            const input = document.getElementById('obsLink');
            input.select();
            input.setSelectionRange(0, 99999); // For mobile
            
            try {
                document.execCommand('copy');
                alert('OBS link copied to clipboard! ✓\\n\\nPaste it into OBS Browser source.');
            } catch (err) {
                alert('Failed to copy. Please manually copy the link.');
            }
        }
    </script>
</body>
</html>
        '''
        
        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE, host=request.host)
        
        @app.route('/obs-captions')
        @app.route('/obs-captions/')
        def obs_captions():
            """OBS Browser Source page - shows only translation text"""
            OBS_TEMPLATE = '''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>OBS Captions</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background: transparent;
                        font-family: Arial, sans-serif;
                        overflow: hidden;
                    }
                    #captions {
                        color: white;
                        font-size: 32px;
                        font-weight: bold;
                        text-align: center;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.8),
                                   -1px -1px 2px rgba(0,0,0,0.8),
                                    1px -1px 2px rgba(0,0,0,0.8),
                                   -1px 1px 2px rgba(0,0,0,0.8);
                        line-height: 1.4;
                        word-wrap: break-word;
                    }
                </style>
            </head>
            <body>
                <div id="captions">Waiting for translation...</div>
                
                <script>
                    let lastTranslation = '';
                    
                    async function updateCaptions() {
                        try {
                            const response = await fetch('/api/obs-translation');
                            const data = await response.json();
                            
                            if (data.translation && data.translation !== lastTranslation) {
                                document.getElementById('captions').textContent = data.translation;
                                lastTranslation = data.translation;
                            }
                        } catch (error) {
                            console.error('Error fetching captions:', error);
                        }
                    }
                    
                    // Update every 200ms for smooth captions
                    setInterval(updateCaptions, 200);
                    updateCaptions();
                </script>
            </body>
            </html>
            '''
            return render_template_string(OBS_TEMPLATE)
        
        @app.route('/api/obs-translation')
        def api_obs_translation():
            """Get current translation for OBS"""
            # Get translation from queue or stored value
            translation = ""
            try:
                if not web_translation_queue.empty():
                    translation = web_translation_queue.get_nowait()
                elif web_current_translation:
                    translation = web_current_translation
            except:
                pass
            
            # Also check PC translation if web is not active
            if not translation and not self.web_is_active:
                translation = self.english_text.get(1.0, tk.END).strip()
            
            return jsonify({'translation': translation})
        
        @app.route('/api/status')
        def api_status():
            """Get current status and text"""
            from_index = self.from_language_combo.current()
            to_index = self.to_language_combo.current()
            
            from_lang = self.languages[from_index]["name"] if from_index >= 0 else "Chinese"
            to_lang = self.languages[to_index]["name"] if to_index >= 0 else "English"
            
            return jsonify({
                'status': self.status_label.cget('text'),
                'is_running': self.is_running,
                'source_text': self.accumulated_text,
                'translation_text': self.english_text.get(1.0, tk.END).strip(),
                'from_language': from_lang,
                'to_language': to_lang
            })
        
        @app.route('/api/set_language', methods=['POST'])
        def api_set_language():
            """Change language settings"""
            data = request.json
            from_code = data.get('from', 'zh')
            to_code = data.get('to', 'en')
            
            # Find indices
            from_idx = next((i for i, lang in enumerate(self.languages) if lang['code'] == from_code), 0)
            to_idx = next((i for i, lang in enumerate(self.languages) if lang['code'] == to_code), 2)
            
            # Update UI (must be done in main thread)
            self.root.after(0, lambda: self.from_language_combo.current(from_idx))
            self.root.after(0, lambda: self.to_language_combo.current(to_idx))
            self.root.after(0, self.on_language_change)
            
            return jsonify({'success': True})
        
        @app.route('/api/web_transcribe', methods=['POST'])
        def api_web_transcribe():
            """Receive transcription from web microphone and translate"""
            data = request.json
            text = data.get('text', '')
            from_lang = data.get('from_lang', 'zh')
            to_lang = data.get('to_lang', 'en')
            
            if not text:
                return jsonify({'success': False, 'error': 'No text provided'})
            
            print(f"Web transcription received: '{text}'")
            
            # Queue for translation
            try:
                web_audio_queue.put_nowait({
                    'text': text,
                    'from_lang': from_lang,
                    'to_lang': to_lang
                })
                return jsonify({'success': True})
            except queue.Full:
                return jsonify({'success': False, 'error': 'Queue full'})
        
        @app.route('/api/web_translation')
        def api_web_translation():
            """Get latest web translation"""
            try:
                global web_current_translation, web_current_source
                
                # Try to get latest from queue first
                translation = web_current_translation
                while not web_translation_queue.empty():
                    try:
                        translation = web_translation_queue.get_nowait()
                        web_current_translation = translation  # Update global
                    except queue.Empty:
                        break
                
                return jsonify({
                    'translation': translation,
                    'source': web_current_source
                })
            except Exception as e:
                return jsonify({'translation': '', 'source': '', 'error': str(e)})
        
        @app.route('/api/screen_audio', methods=['POST'])
        def api_screen_audio():
            """Receive screen share audio and transcribe"""
            try:
                data = request.json
                audio_base64 = data.get('audio', '')
                from_lang = data.get('from_lang', 'zh')
                to_lang = data.get('to_lang', 'en')
                is_wav = data.get('is_wav', False)
                
                if not audio_base64:
                    return jsonify({'success': False, 'error': 'No audio data'})
                
                print(f"Received screen audio ({'WAV' if is_wav else 'WebM'}), processing...")
                
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_base64)
                print(f"Decoded audio: {len(audio_bytes)} bytes")
                
                if is_wav:
                    # Audio is already in WAV format from browser, load directly
                    print("Loading WAV audio directly (no FFmpeg needed)...")
                    try:
                        import wave
                        import io
                        
                        # Load WAV from bytes
                        wav_io = io.BytesIO(audio_bytes)
                        with wave.open(wav_io, 'rb') as wf:
                            # Verify it's the right format
                            channels = wf.getnchannels()
                            sample_rate = wf.getframerate()
                            sample_width = wf.getsampwidth()
                            
                            print(f"WAV format: {channels} channels, {sample_rate}Hz, {sample_width} bytes/sample")
                            
                            # Read audio data
                            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                            audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        print(f"Audio loaded: {len(audio_float)} samples ({len(audio_float)/sample_rate:.1f}s)")
                        
                    except Exception as wav_err:
                        print(f"Error loading WAV: {wav_err}")
                        import traceback
                        traceback.print_exc()
                        return jsonify({'success': False, 'error': f'Failed to load WAV: {str(wav_err)}'})
                else:
                    # Old WebM path (kept for compatibility, but shouldn't be used anymore)
                    print("WARNING: Received WebM format, this path is deprecated")
                    return jsonify({'success': False, 'error': 'WebM format no longer supported, please refresh page'})
                
                # Transcribe the audio
                if self.model:
                    print("Starting transcription...")
                    try:
                        segments, info = self.model.transcribe(
                            audio_float,
                            language=from_lang,
                            beam_size=5,
                            temperature=0.0,
                            vad_filter=True
                        )
                        
                        transcribed_text = ""
                        for segment in segments:
                            transcribed_text += segment.text + " "
                        
                        transcribed_text = transcribed_text.strip()
                        
                        if transcribed_text:
                            print(f"Screen audio transcribed: '{transcribed_text}'")
                            
                            # Queue for translation with screen flag
                            try:
                                web_audio_queue.put_nowait({
                                    'text': transcribed_text,
                                    'from_lang': from_lang,
                                    'to_lang': to_lang,
                                    'is_screen': True
                                })
                                print(f"Queued for translation successfully")
                            except queue.Full:
                                print(f"Warning: web_audio_queue is full, clearing old items")
                                while not web_audio_queue.empty():
                                    try:
                                        web_audio_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                web_audio_queue.put_nowait({
                                    'text': transcribed_text,
                                    'from_lang': from_lang,
                                    'to_lang': to_lang,
                                    'is_screen': True
                                })
                            
                            return jsonify({'success': True, 'text': transcribed_text})
                        else:
                            print("No text transcribed (silence or no speech)")
                            return jsonify({'success': True, 'text': '', 'message': 'No speech detected'})
                    
                    except Exception as transcribe_err:
                        print(f"Transcription error: {transcribe_err}")
                        import traceback
                        traceback.print_exc()
                        return jsonify({'success': False, 'error': f'Transcription failed: {str(transcribe_err)}'})
                else:
                    print("Model not loaded yet")
                    return jsonify({'success': False, 'error': 'Model not loaded'})
                
            except Exception as e:
                print(f"Screen audio error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)})
        
        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            """Shutdown the Flask server"""
            print("Shutdown request received")
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                # For production servers, just return success
                # The server will stop when self.web_server_enabled becomes False
                return jsonify({'success': True, 'message': 'Shutdown signal sent'})
            func()
            return jsonify({'success': True, 'message': 'Server shutting down...'})
        
        try:
            print("Starting HTTP server on http://0.0.0.0:1235")
            # Run with threaded=True to allow shutdown
            app.run(host='0.0.0.0', port=1235, debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            print(f"Web server error: {e}")
        finally:
            self.web_server_enabled = False
            print("Web server thread ended")

    def cleanup(self):
        """Cleanup resources"""
        try:
            print("Cleaning up resources...")
            self.is_running = False
            
            # Stop web server
            if self.web_server_enabled:
                self.stop_web_server()
            
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    self.stream.close()
                    print("Audio stream closed")
                except Exception as e:
                    print(f"Error closing stream: {e}")
                self.stream = None
                
        except Exception as e:
            print(f"Cleanup error: {e}")

    def run(self):
        """Run the application"""
        try:
            def on_closing():
                self.cleanup()
                self.root.destroy()
            
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Run error: {e}")

if __name__ == "__main__":
    try:
        print("Starting real-time translator...")
        
        # Test audio devices first
        try:
            print("Testing audio devices...")
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            print(f"Available input devices: {len(input_devices)}")
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")
        except Exception as e:
            print(f"Error testing audio devices: {e}")
        
        app = RealtimeTranslator()
        app.run()
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()