import numpy as np
import pyaudio
import queue
import threading
from scipy.signal import butter, lfilter

class PitchDetector:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue()
        
        # Audio stream parameters
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.input_device_index = 4
        self.running = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Pitch detection parameters
        self.min_frequency = 50  # Hz
        self.max_frequency = 1000  # Hz
        self.min_period = int(self.sample_rate / self.max_frequency)
        self.max_period = int(self.sample_rate / self.min_frequency)

    def butter_bandpass(self, lowcut=50, highcut=1000, order=5):
        """Create butterworth bandpass filter"""
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut=50, highcut=1000, order=5):
        """Apply bandpass filter to the signal"""
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y

    def normalize(self, data):
        """Normalize the audio data"""
        return data / np.max(np.abs(data))

    def autocorrelation(self, data):
        """Compute autocorrelation of the signal"""
        corr = np.correlate(data, data, mode='full')
        corr = corr[len(corr)//2:]  # Keep only the positive lags
        return corr

    def get_pitch(self, data):
        """Detect pitch using autocorrelation method"""
        # Preprocess the signal
        filtered_data = self.bandpass_filter(data)
        normalized_data = self.normalize(filtered_data)
        
        # Compute autocorrelation
        corr = self.autocorrelation(normalized_data)
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(self.min_period, self.max_period):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                peaks.append((i, corr[i]))
        
        if not peaks:
            return None
        
        # Find the highest peak
        period = max(peaks, key=lambda x: x[1])[0]
        
        # Convert period to frequency
        frequency = self.sample_rate / period
        return frequency

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        """Start the pitch detection"""
        self.running = True
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=False,
            frames_per_buffer=self.buffer_size,
            input_device_index=self.input_device_index,
            stream_callback=self.audio_callback
        )
        
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.start()

    def stop(self):
        """Stop the pitch detection"""
        self.running = False
        self.detection_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                pitch = self.get_pitch(audio_data)
                
                if pitch is not None:
                    print(f"Detected pitch: {pitch:.1f} Hz")
                    
            except queue.Empty:
                continue

# Usage example
if __name__ == "__main__":
    detector = PitchDetector()
    try:
        print("Starting pitch detection... (Press Ctrl+C to stop)")
        detector.start()
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping pitch detection...")
        detector.stop()
