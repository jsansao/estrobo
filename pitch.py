import pyaudio
import numpy as np
import math

# Parameters
RATE = 44100  # Sampling rate
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono
device_index = 4

def calculate_pitch(data, rate):
    """Estimate pitch using Autocorrelation Method."""
    # Convert audio data to numpy array and normalize
    data = np.frombuffer(data, dtype=np.int16)
    data = data - np.mean(data)  # Zero-center the data

    # Autocorrelation of data
    corr = np.correlate(data, data, mode='full')[len(data)-1:]
    d = np.diff(corr)

    # Find the first positive difference
    start = np.where(d > 0)[0][0]

    # Find the next peak after the first positive difference
    peak = np.argmax(corr[start:]) + start
    r = rate / peak if peak != 0 else 0

    return r if 50 < r < 5000 else None  # Filter frequencies outside human pitch range

# Setup PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=device_index)

print("Listening...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        pitch = calculate_pitch(data, RATE)
        if pitch:
            print(f"Detected pitch: {pitch:.2f} Hz")
except KeyboardInterrupt:
    print("Stopping...")

# Close stream
stream.stop_stream()
stream.close()
p.terminate()
