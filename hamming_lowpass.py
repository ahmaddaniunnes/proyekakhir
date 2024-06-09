import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os

# Path file audio input
file_path = r'C:\Users\Ahmad Dani\OneDrive\Documents\060591718.wav'

# Path folder untuk menyimpan hasil
output_folder = r'C:\Users\Ahmad Dani\OneDrive\Documents'

# Cek apakah folder output ada
if not os.path.exists(output_folder):
    print(f"Folder tidak ditemukan: {output_folder}")
else:
    # Cek apakah file audio ada
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan: {file_path}")
    else:
        # Muat file audio
        y, sr = librosa.load(file_path, sr=None)

        # Desain band-pass FIR filter dengan jendela rectangular
        numtaps = 101  # Jumlah tap filter
        lowcut = 1000  # Frekuensi cut-off bawah (Hz)
        fir_coeff = signal.firwin(numtaps, lowcut, window='hamming', fs=sr)

        # Terapkan filter FIR pada sinyal
        filtered_signal_fir = signal.lfilter(fir_coeff, 1.0, y)

        # Path file output
        output_path = os.path.join(output_folder, 'filtered_hamming_bandpass_fir.wav')
        sf.write(output_path, filtered_signal_fir, sr)

        # Plot sinyal asli dan sinyal hasil filter
        plt.figure(figsize=(15, 6))

        plt.subplot(4, 1, 1)
        plt.plot(y)
        plt.title('Sinyal Asli')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(4, 1, 2)
        plt.plot(filtered_signal_fir)
        plt.title('Sinyal Hasil Low-pass FIR Filter dengan Jendela Hamming')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

         # Plot respons impuls
        plt.subplot(4, 2, 5)
        plt.stem(fir_coeff)
        plt.title('Respons Impuls Filter')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        # Plot respons frekuensi
        plt.subplot(4, 2, 6)
        w, h = signal.freqz(fir_coeff, worN=8000)
        plt.plot(0.5 * sr * w / np.pi, np.abs(h), 'b')
        plt.title('Respons Frekuensi Filter')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')

        plt.tight_layout()
        plt.show()

        print(f"Sinyal hasil filter disimpan di: {output_path}")
