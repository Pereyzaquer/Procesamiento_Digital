import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.io as sio


# Detecta la frecuencia pico
def detector_fp(fs, Y):
    i_max = np.argmax(Y)
    return i_max * fs / 2 / len(Y)

# Ancho de banda pasa bajos
def bandwidth_lp(fs, Y, percentage):
    total_power = np.sum(Y)
    cumulative_power = np.cumsum(Y)
    idx = np.searchsorted(cumulative_power, total_power * percentage)
    return idx * fs / 2 / len(Y)

# Ancho de banda pasa banda
def bandwidth_bp(fs, Y, percentage):
    peak_idx = np.argmax(Y)
    total_power = np.sum(Y)
    accumulated = Y[peak_idx]
    i = 1
    while accumulated < total_power * percentage and peak_idx + i < len(Y) and peak_idx - i >= 0:
        accumulated += Y[peak_idx + i] + Y[peak_idx - i]
        i += 1
    return 2 * i * fs / 2 / len(Y)

# Ancho de banda pasa banda desde una frecuencia mínima
def bandwidth_bp2(fs, Y, percentage, f_min):
    freqs = np.linspace(0, fs / 2, len(Y))
    idx_min = np.searchsorted(freqs, f_min)
    
    total_power = np.sum(Y[idx_min:])
    cumulative_power = np.cumsum(Y[idx_min:])
    idx = np.searchsorted(cumulative_power, total_power * percentage)
    
    return (freqs[idx_min + idx] - f_min)

# Función principal
def analizar_ventanas(fs, signal, percentage=0.99, mode='lp', title="Análisis de Welch", f_min=0):
    windows = {
        'Flattop': 'r',
        'Blackman': 'g',
        'Hann': 'b'
    }

    N = len(signal)
    T = N / fs

    plt.figure(figsize=(10, 6))

    # En esta variable se acumula la potencia total medida por cada ventana
    power_t = 0

    for name, color in windows.items():
        f, Y = sig.welch(signal, fs, window=name.lower())

        # Pico
        fp = detector_fp(fs, Y)

        # Potencia estimada por la ventana (área bajo PSD)
        power = np.trapz(Y, f)
        power_t += power
        # Ancho de banda
        if mode == 'lp':
            bw = bandwidth_lp(fs, Y, percentage)
            bw_range = (0, bw)
        elif mode == 'bp':
            bw = bandwidth_bp(fs, Y, percentage)
            bw_range = (fp - bw/2, fp + bw/2)
        elif mode == 'bp2':
            bw = bandwidth_bp2(fs, Y, percentage, f_min)
            bw_range = (f_min, f_min + bw)
        else:
            raise ValueError("Modo inválido. Usa 'lp', 'bp' o 'bp2'.")

        # Etiqueta con datos
        label = f"{name} - fp={fp:.1f}Hz - bw={bw:.1f}Hz - P={power/1e3:.0f}.e3"

        # Graficar PSD
        plt.semilogy(f, Y, color=color, label=label)

        # Marcar frecuencia pico
        plt.axvline(fp, color=color, linestyle='--')

        # Marcar ancho de banda
        plt.axvspan(*bw_range, color=color, alpha=0.2)

    power_t = power_t / len(windows)  # Promedio de potencia total
    power_t = power_t*percentage
    plt.title(f"{title} - Potencia concatenada {power_t/1e3:.0f}.e3")
    plt.xlabel(f"Frecuencia [Hz] | N = {N} muestras | fs = {fs} Hz | T = {T:.2f} s")
    plt.ylabel("Densidad espectral [W/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

##################################################################

fs_ecg = 1000 # Hz
ecg_one_lead = np.load('../Mariano_lib/pdstestbench-master/ecg_sin_ruido.npy')

# Pasa bajos
analizar_ventanas(fs=fs_ecg, signal=ecg_one_lead, percentage=0.99, mode='lp', title="Low-Pass PSD - Ecg")

# Pasa banda
analizar_ventanas(fs=fs_ecg, signal=ecg_one_lead, percentage=0.99, mode='bp', title="Band-Pass PSD - Ecg")

analizar_ventanas(fs=fs_ecg, signal=ecg_one_lead, percentage=0.99, mode='bp2', f_min=3, title="Band-Pass - Ecg - f_min = 3 Hz")