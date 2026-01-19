import matplotlib
matplotlib.use('TkAgg')

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data

    except Exception as e:
        print(f"Error: {e}")

def time_domain_plot(sig, t_s):
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t_s, sig.real, color='blue')
    plt.plot(t_s, sig.imag, color='red')

    plt.title(f"Time Domain Signal (True Time)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sig.real, color='blue')
    plt.plot(sig.imag, color='red')

    plt.title(f"Time Domain Signal (Valid Only)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)


def analyze(fname):
    # get data
    data = load_data(fname)

    # get data properties of interest:
    [sig, t_s, fs_hz] = parse_ops243_datastruct(data)

    # Viz
    time_domain_plot(sig, t_s)
    spectrogram(sig, fs_hz, window_size=4096, overlap=0)

def parse_ops243_datastruct(data):
    # extract meaningful info and create a simple dict to capture it
    fs_hz = data.get("metadata").get("sample_rate")
    ts_s = 1/fs_hz
    tmp_sig = []
    tmp_time = []
    t_start_tot = []

    # construct sample and sample time vectors
    captures = data.get("captures")

    for capture in captures:
        sig = capture.get("complex_signal")
        tmp_sig.append(sig)
        t_start = capture.get("sample_time")
        t_start_tot.append(t_start)
        t_end = t_start + (ts_s * len(sig))
        tmp_time.append(np.arange(t_start, t_end, ts_s))

    sig_tot = np.concatenate(tmp_sig)
    t_tot = np.concatenate(tmp_time)
    t_tot = t_tot - t_tot[0]
    
    return sig_tot, t_tot, fs_hz

def spectrogram(sig, fs_hz, window_size, overlap):
    # full fft for comparison
    sig_fft = np.fft.fftshift(np.fft.fft(sig))
    fft_freqs = np.linspace(-fs_hz/2, fs_hz/2, len(sig), endpoint=False)
    fft_mph = dopp_to_mph(fft_freqs)

    # rough spectral estimate
    hop_size = window_size - overlap
    num_segments = (len(sig) - overlap) // hop_size
    
    window = np.hanning(window_size)
    
    spectrogram = np.zeros((window_size, num_segments), dtype=complex)
    
    for i in range(num_segments):
        # get segment
        start = i * hop_size
        segment = sig[start : start + window_size]
        
        # windowing
        windowed_segment = segment * window
        
        # fft 
        spectrum = np.fft.fft(windowed_segment)
        
        # fft shift
        spectrogram[:, i] = np.fft.fftshift(spectrum)

    # power in db
    spectrogram_db = db20(spectrogram)
    freqs_hz = np.linspace(-fs_hz/2, fs_hz/2, window_size, endpoint=False)
    times_s = np.arange(num_segments) * (hop_size / fs_hz)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram_db, aspect='auto', origin='lower', 
               extent=[times_s[0], times_s[-1], freqs_hz[0], freqs_hz[-1]])

    plt.title("Spectrogram (Valid Signal w/ Discontinuity)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.plot(fft_mph, sig_fft)
    plt.title("Full Res FFT")
    plt.xlabel("Derived Velocity (mph)")
    plt.ylabel("Amplitude (linear)")
    plt.grid(True)


def dopp_to_mph(freqs, fc=24.125e9):
    c = 299792458  # speed of light
    m_per_s = freqs * c / (2 * fc)
    mph = m_per_s * 2.23693629
    return mph
    

def db20(data):
    # accept voltage measurement and return power in db
    return 20 * np.log10(np.abs(data) + 1e-10) # prevent zeros from sneaking in

def boldify():
    plt.rcParams.update({
    'font.weight': 'bold',              
    'axes.labelweight': 'bold',         
    'axes.titleweight': 'bold',         
    'axes.linewidth': 2.0,              
    'lines.linewidth': 2.5,            
    'xtick.major.width': 1.5,           
    'ytick.major.width': 1.5,
    'font.size': 16                     
    })

if __name__=="__main__":
    # Relative pathing (run from this dir)
    data_dir = "data"
    dataset_h1 = "iq_captures_club_swinging.pkl"
    dataset_h0 = "iq_captures_noise.pkl"

    # Basic signal analysis
    boldify()
    analyze(os.path.join(data_dir, dataset_h0))
    analyze(os.path.join(data_dir, dataset_h1))

    plt.show()
