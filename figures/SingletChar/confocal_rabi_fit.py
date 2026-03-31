import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def rabi_oscillation(t, amp, freq, phase, decay, offset):
    """Damped cosine function for Rabi oscillations."""
    return amp * np.exp(-t / decay) * np.cos(2 * np.pi * freq * t + phase) + offset

def main():
    # --- Update these paths to your exact file locations ---
    npz_file = r"2026_01_07-13_49_03-(Wu).npz"
    txt_file = r"2026_01_07-13_49_03-(Wu).txt"
    
    # 1. Load the taus from the text file
    with open(txt_file, 'r') as f:
        config = json.load(f)
    taus_ns = np.array(config['taus_ns'])
    
    # 2. Load raw data and SLICE strictly to the 10 valid runs
    npz = np.load(npz_file)
    raw_data = npz[npz.files[0]]  # Expected shape: (2, 16, 51, 20000)
    
    valid_runs = 10
    # Dimensions: (Gates, Runs, Taus, Reps) -> slice runs dimension to grab :10
    valid_data = raw_data[:, :valid_runs, :, :]
    
    # 3. Separate gates and sum across runs (axis 0) and reps (axis 2)
    # Based on Rabi.py: gate 0 = reference, gate 1 = signal
    ref_data = valid_data[0] # Shape: (10, 51, 20000)
    sig_data = valid_data[1] # Shape: (10, 51, 20000)
    
    ref_counts = np.sum(ref_data, axis=(0, 2))
    sig_counts = np.sum(sig_data, axis=(0, 2))
    
    # Calculate Normalized Contrast
    counts = sig_counts / ref_counts
    
    # 4. Fit the data
    offset_guess = np.mean(counts)
    amp_guess = (np.max(counts) - np.min(counts)) / 2.0
    
    counts_zero_mean = counts - offset_guess
    fft_vals = np.fft.rfft(counts_zero_mean)
    fft_freqs = np.fft.rfftfreq(len(taus_ns), d=(taus_ns[1] - taus_ns[0]))
    freq_guess = fft_freqs[np.argmax(np.abs(fft_vals[1:])) + 1] 
    
    if freq_guess == 0:
        freq_guess = 1.0 / (taus_ns[-1] - taus_ns[0])
        
    decay_guess = taus_ns[-1] / 2.0 
    phase_guess = np.pi 

    p0 = [amp_guess, freq_guess, phase_guess, decay_guess, offset_guess]
    bounds = ([0, 0, -2*np.pi, 0, -np.inf], [np.inf, np.inf, 2*np.pi, np.inf, np.inf])

    try:
        popt, _ = curve_fit(rabi_oscillation, taus_ns, counts, p0=p0, bounds=bounds)
        fit_y = rabi_oscillation(taus_ns, *popt)
        pi_pulse = 1.0 / (2.0 * popt[1])
        print(f"Fit Success! Pi-pulse time: {pi_pulse:.2f} ns, Rabi Freq: {popt[1]*1000:.2f} MHz")
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = p0
        fit_y = rabi_oscillation(taus_ns, *popt)

    # 5. Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(taus_ns, counts, 'o', label='Valid Data (10 Runs)', color='navy')
    
    t_smooth = np.linspace(taus_ns[0], taus_ns[-1], 500)
    plt.plot(t_smooth, rabi_oscillation(t_smooth, *popt), '-', label='Rabi Fit', linewidth=2, color='darkorange')
    
    plt.xlabel('Microwave Pulse Duration (ns)')
    plt.ylabel('Normalized Contrast')
    plt.title(f'Rabi Oscillation (Runs 1-{valid_runs} only)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()