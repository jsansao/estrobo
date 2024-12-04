import numpy as np
from scipy.io import wavfile
import pyaudio
import struct

class YINPitchDetector:
    def __init__(self, sample_rate=44100, buffer_size=2048, threshold=0.1):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        # Initialize PyAudio for real-time acquisition
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=buffer_size
        )

    def difference_function(self, data):
        """
        Calculate the difference function as described in the YIN paper
        """
        length = len(data)
        diff = np.zeros(length)
        
        for tau in range(length):
            for j in range(length - tau):
                diff[tau] += (data[j] - data[j + tau]) ** 2
        
        return diff

    def cumulative_mean_normalized_difference(self, diff):
        """
        Compute the cumulative mean normalized difference function
        """
        length = len(diff)
        cmnd = np.zeros(length)
        cmnd[0] = 1.0
        
        running_sum = 0.0
        for tau in range(1, length):
            running_sum += diff[tau]
            cmnd[tau] = diff[tau] * tau / running_sum if running_sum != 0 else 1.0
        
        return cmnd

    def absolute_threshold(self, cmnd):
        """
        Find the first minimum below the threshold
        """
        tau = 0
        length = len(cmnd)
        
        while tau < length:
            if cmnd[tau] < self.threshold:
                while tau + 1 < length and cmnd[tau + 1] < cmnd[tau]:
                    tau += 1
                return tau
            tau += 1
            
        return -1  # No pitch found

    def get_pitch(self, data):
        """
        Estimate the pitch using the YIN algorithm
        """
        diff = self.difference_function(data)
        cmnd = self.cumulative_mean_normalized_difference(diff)
        tau = self.absolute_threshold(cmnd)
        
        if tau == -1:
            return -1  # No pitch found
            
        # Interpolation to refine the pitch estimation
        if tau > 0 and tau < len(cmnd) - 1:
            alpha = cmnd[tau-1]
            beta = cmnd[tau]
            gamma = cmnd[tau+1]
            peak = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            tau = tau + peak
            
        return self.sample_rate / tau if tau != 0 else -1

    def process_realtime(self):
        """
        Process real-time audio input
        """
        try:
            while True:
                # Read audio data
                data = self.stream.read(self.buffer_size, exception_on_overflow=False)
                # Convert bytes to float32 array
                samples = np.frombuffer(data, dtype=np.float32)
                
                # Get pitch
                pitch = self.get_pitch(samples)
                
                if pitch != -1:
                    print(f"Detected pitch: {pitch:.2f} Hz")
                else:
                    print("No pitch detected")
                    
        except KeyboardInterrupt:
            print("\nStopping pitch detection...")
            self.cleanup()

    def cleanup(self):
        """
        Clean up audio resources
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

# Usage example
if __name__ == "__main__":
    # Initialize pitch detector
    detector = YINPitchDetector(
        sample_rate=44100,
        buffer_size=2048,
        threshold=0.1
    )
    
    # Start real-time processing
    print("Starting pitch detection (Press Ctrl+C to stop)...")
    detector.process_realtime()
