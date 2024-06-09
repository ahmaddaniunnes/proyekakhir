import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, lfilter, freqz
import librosa
import soundfile as sf
import os

def cheby_lowpass(cutoff, fs, order, rp):
    nyquist = 0.5 * fs
    wp = cutoff / nyquist
    b, a = cheby1(order, rp, wp, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order, rp):
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    y = lfilter(b, a, data)
    return y

def plot_signal(time, signal, filtered_signal, fs):
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(signal)
    plt.title('Sinyal Asli')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(filtered_signal)
    plt.title('Sinyal Hasil Low-pass Chebyshev Filter')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def plot_impulse_response(b, a):
    impulse_response = np.zeros(1000)
    impulse_response[0] = 1
    filtered_impulse_response = lfilter(b, a, impulse_response)

    plt.figure()
    plt.plot(filtered_impulse_response)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Respons Impuls Filter')
    plt.show()

def plot_frequency_response(b, a, fs):
    w, h = freqz(b, a, worN=8000)
    plt.figure()
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Respons Frekuensi Filter')
    plt.show()

def process_audio(input_file, output_file, cutoff, order, rp):
    # Read the audio file using librosa
    data, fs = librosa.load(input_file)

    # Apply the low-pass filter
    filtered_data = lowpass_filter(data, cutoff, fs, order, rp)

    # Write the filtered data to the output file
    output_path = os.path.join(output_file)
    sf.write(output_path, filtered_data, fs)

    # Plot the signals and frequency response
    time = np.arange(len(data)) / fs
    plot_signal(time, data, filtered_data, fs)
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    plot_impulse_response(b, a)
    plot_frequency_response(b, a, fs)

if __name__ == "__main__":
    input_file = r'C:\Users\Ahmad Dani\OneDrive\Documents\060591718.wav'  # Path to your input audio file
    output_file = r'C:\Users\Ahmad Dani\OneDrive\Documents\output_audio.wav'  # Path to save the output audio file
    cutoff_frequency = 1000  # Cutoff frequency in Hz
    filter_order = 5  # Order of the filter
    ripple = 1  # Maximum ripple in passband (dB)

    process_audio(input_file, output_file, cutoff_frequency, filter_order, ripple)
