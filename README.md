# Real-Time Local AI Voice Translator (Python)

This project is a **local, real-time voice transcription and translation application** written in Python.

It uses **Whisper (via faster-whisper)** for live speech recognition and **LM Studio** for translation through a local LLM API. The focus is low latency and full local control — no cloud services required.

---

## Overview

The application is split into two workloads to keep latency as low as possible.

### Dictation (This Python App)

Runs on the system where audio is captured.

**Responsibilities:**
- Microphone input  
- Voice Activity Detection (VAD)  
- Noise filtering  
- Real-time Whisper transcription  

This part benefits heavily from **CUDA** and an **NVIDIA GPU**.

### Translation (LM Studio)

Runs separately and handles translation only.

**Responsibilities:**
- Exposes an OpenAI-compatible local API  
- Receives text from the Python app  
- Returns translated output in real time  

You *can* run both on one machine, but performance will be noticeably worse.

---

## Hardware Requirements

### Recommended Setup

- **Two desktops or laptops**
- **NVIDIA GPU on each system**
- CUDA installed and working

One machine runs the Python dictation app.  
The other runs LM Studio for translation.

### CPU Fallback

If CUDA is not available:

- Whisper runs on CPU  
- Translation still works  
- Latency increases significantly  

**CPU-only usage notes:**
- Use the **tiny** or **base** Whisper models  
- Larger models will be very slow  

CPU mode is functional but not suitable for real-time use.

---

## Software Requirements

### Operating System

- Windows 10 or Windows 11

### Python

- Python **3.9 – 3.11** recommended

### CUDA (For GPU Acceleration)

Install **one** of the following:

- CUDA 11.8  
- CUDA 12.0  
- CUDA 12.1  

CUDA must be available in the system `PATH`.

---

## Python Dependencies

Install required packages:

```bash
pip install sounddevice numpy requests faster-whisper torch torchvision torchaudio
Optional (used to verify CUDA availability):

pip install cupy-cuda12x
faster-whisper will automatically use CUDA if available.
```
LM Studio Setup (Required for Translation)
1. Install LM Studio
https://lmstudio.ai

2. Load a Model
Recommended model size:

4B – 8B parameters

Smaller models are faster.
Larger models improve translation quality but increase latency.

3. Enable Local API Server
In LM Studio:

Enable the OpenAI-compatible API server

Note the IP address and port

The Python app sends requests to this endpoint.

Example from the code:

LM_STUDIO_URL = "http://192.168.1.208:1234/v1/chat/completions"
API Key Note
LM Studio does not require a real API key, but the OpenAI-style endpoint expects one.

Any placeholder value will work unless you restrict access manually.

Running the Application
python startLiveTranslate.py
On startup:

Select your microphone

Choose a Whisper model

Select source and target languages

Wait for Whisper and LM Studio to finish loading

Status messages in the UI indicate readiness and connection state.

Whisper Model Notes
Model	Speed	Accuracy	CPU Usable
tiny	Very fast	Low	Yes
base	Fast	Basic	Yes
small	Moderate	Good	Limited
medium	Slower	Very good	No
large-v2	Very slow	Best	No
The default model is medium, which assumes a GPU is available.

Common Issues
CUDA Not Detected
Whisper falls back to CPU

Expect high latency

Verify GPU access with:

nvidia-smi
LM Studio Not Running
Transcription still works

Translation will not function

Status bar will show warnings

Audio Noise or False Speech Detection
Reduce microphone gain

Use a directional microphone

Avoid system audio bleed or desktop capture

Notes
This is a fully local system

No cloud APIs are used

No OpenAI services are required

All processing stays on your hardware
