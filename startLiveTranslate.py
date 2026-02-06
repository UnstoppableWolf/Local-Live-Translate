import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import threading
import queue
import json
import os
import time
import requests
from faster_whisper import WhisperModel

# Real-time configuration for ultra-fast response
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # Larger chunks for better efficiency
ENERGY_THRESHOLD = 0.001
MODEL_SIZE = "medium"  # Medium model - best balance (769M parameters)
LM_STUDIO_URL = "http://192.168.1.208:1234/v1/chat/completions"
LM_STUDIO_API_KEY = "sk-lm-xCIDFvdn:Iy8t1zs1vhqrkT0t4i2e"
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
VAD_ENERGY_THRESHOLD = 0.0008  # Increased from 0.0005 (0.3 less sensitive)
VAD_SILENCE_DURATION = 0.4  # Very quick end detection (was 0.5)
VAD_SPEECH_DURATION = 0.3   # Increased from 0.2 (0.3 less sensitive)
VAD_NOISE_GATE = 0.0002     # Increased from 0.0001 (0.3 less sensitive)
VAD_SMOOTHING_WINDOW = 2

# Real-time queues with higher throughput
audio_queue = queue.Queue(maxsize=10)  # Reduced from 100 to prevent backlog
ui_queue = queue.Queue(maxsize=200)
translate_queue = queue.Queue(maxsize=50)
level_queue = queue.Queue(maxsize=30)

class RealtimeTranslator:
    def __init__(self):
        print("Creating real-time translator...")
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Real-time Voice Translator")
        self.root.geometry("700x600")
        
        # Initialize variables for real-time processing
        self.model = None
        self.client = None
        self.stream = None
        self.current_level = 0.0
        self.is_running = True
        
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
        
        # Level processing
        threading.Thread(target=self.process_audio_levels, daemon=True).start()

    def load_models(self):
        """Load AI models with optimized settings"""
        try:
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
                    ui_queue.put(("status", f"Ready! {MODEL_SIZE} on {device.upper()}."))
                    print(f"System ready!")
                else:
                    ui_queue.put(("status", f"LM Studio connected"))
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
            
            audio = indata[:, 0]
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
            # Simple noise gate
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
            absolute_minimum = VAD_ENERGY_THRESHOLD * 1.2  # Increased from 0.8 to be more strict
            speech_detected = speech_detected and (smoothed_energy > absolute_minimum)
            
            # Fallback: also detect if energy is significantly above noise floor
            fallback_detected = smoothed_energy > (self.noise_level * 3.0)  # Increased from 2.0 to 3.0
            
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
            "字幕", "索兰娅", "索蘭婭", "字幕by"  # Chinese subtitle artifacts
        ]
        
        # Convert to lowercase for checking (and check original for Chinese)
        text_lower = text.lower().strip()
        
        # Skip if contains subtitle keywords (check both cases)
        for keyword in subtitle_keywords:
            if keyword in text_lower or keyword in text:
                return ""
        
        # Skip if text looks like subtitle formatting
        if any(phrase in text_lower for phrase in ["subtitles by", "down to earth", "coconut egg"]):
            return ""
        
        # Skip Chinese subtitle patterns
        if any(phrase in text for phrase in ["字幕by", "索兰娅", "索蘭婭"]):
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
                if np.max(np.abs(audio_chunk)) > 0:
                    # Normalize to -1 to 1 range
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                    # Apply stronger amplification for better detection (increased from 1.2 to 1.5)
                    audio_chunk = np.clip(audio_chunk * 1.5, -1.0, 1.0)
                
                # Calculate audio energy to detect if this is mostly silence/noise
                audio_energy = np.sqrt(np.mean(audio_chunk**2))
                audio_duration = len(audio_chunk) / SAMPLE_RATE
                
                # If audio energy is very low, skip transcription (likely just background noise)
                if audio_energy < 0.02:  # Very low energy threshold
                    print(f"Audio energy too low ({audio_energy:.4f}), skipping transcription (likely background noise)")
                    continue
                
                # Optimized transcription settings - batch processing is faster
                gpu_start = time.time()
                segments, info = self.model.transcribe(
                    audio_chunk,
                    language=language_code,
                    beam_size=5,  # Better accuracy, batched efficiently
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=500
                    ),
                    condition_on_previous_text=False,
                    no_speech_threshold=0.7,  # Increased from 0.6 to be more strict
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
            
            # Start new stream after a short delay
            time.sleep(0.5)
            threading.Thread(target=self.start_audio_stream, daemon=True).start()
            
        except Exception as e:
            print(f"Mic change error: {e}")
            ui_queue.put(("status", f"Mic change error: {e}"))

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

    def cleanup(self):
        """Cleanup resources"""
        try:
            print("Cleaning up resources...")
            self.is_running = False
            
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