import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.signal import iirdesign, sosfilt, sosfreqz, group_delay, sos2tf, welch

def max_bw_valido(fs, fmin):
    """
    Calcula el máximo ancho de banda (BW) permitido dado un fs y una frecuencia mínima (fmin)
    para que las frecuencias de corte del filtro pasabanda sean válidas (0 < Wn < 1).

    Retorna el BW máximo que se puede usar.
    """
    nyq = fs / 2

    # Para que (fmin - bw/2)/nyq > 0  --> bw/2 < fmin
    # Para que (fmin + bw + bw/2)/nyq < 1 --> fmin + 1.5*bw < nyq --> bw < (nyq - fmin) / 1.5

    bw_max_izq = 2 * fmin
    bw_max_der = (nyq - fmin) / 1.5

    bw_max = min(bw_max_izq, bw_max_der)
    return bw_max

# Detecta la frecuencia pico
def detector_fp(fs, Y):
    i_max = np.argmax(Y)
    return i_max * fs / 2 / len(Y)

# Ancho de banda pasa bajos
def bandwidth_lp(fs, Y, percentage):
    total_power = np.sum(Y)
    power_bt = 0                        # Potencia acumulada
    i = 0
    while power_bt < total_power*percentage and i < len(Y):
        power_bt += Y[i]               # Acumula potencia
        i += 1
    #print(f"Power Lp: {power_bt:.0f}, Bw: {i* fs / 2 / len(Y)}")
    return i * fs / 2 / len(Y)

# Ancho de banda pasa banda
def bandwidth_bp(fs, Y, percentage):
    total_power = np.sum(Y)
    fp_index = np.argmax(Y)
    power_acc = Y[fp_index]
    
    left = fp_index - 1
    right = fp_index + 1

    min_index = fp_index
    max_index = fp_index

    while power_acc < total_power * percentage:
        # Chequea si se puede avanzar en ambos lados
        power_left = Y[left] if left >= 0 else -np.inf
        power_right = Y[right] if right < len(Y) else -np.inf

        # Elegir el lado con más potencia
        if power_left >= power_right:
            if left >= 0:
                power_acc += Y[left]
                min_index = left
                left -= 1
            else:
                if right < len(Y):
                    power_acc += Y[right]
                    max_index = right
                    right += 1
                else:
                    break  # Ambos límites alcanzados
        else:
            if right < len(Y):
                power_acc += Y[right]
                max_index = right
                right += 1
            else:
                if left >= 0:
                    power_acc += Y[left]
                    min_index = left
                    left -= 1
                else:
                    break  # Ambos límites alcanzados

    bw = (max_index - min_index+1) * fs / (2 * len(Y))  # Resolución en Hz
    #print(f"Power Pb: {power_acc:.0f}, BW: {bw:.2f} Hz")
    return bw

    return bw


# Ancho de banda pasa banda desde una frecuencia mínima
def bandwidth_bp2(fs, Y, percentage, f_min):
    total_power = np.sum(Y)

    f_min_index = int(f_min * len(Y) / (fs / 2))  # Índice de la frecuencia mínima
    i = f_min_index
    power_bt = 0                     # No sumes antes del while

    while power_bt < total_power * percentage and i < len(Y):
        power_bt += Y[i]
        i += 1

    bw = (i - f_min_index) * fs / (2 * len(Y))
    #print(f"Power Pb2: {power_bt:.0f}, BW: {bw:.2f} Hz")
    return bw

# Función principal
def analizar_ventanas(fs, signal, percentage=0.99, mode='lp', title="Análisis de Welch", f_min=0):
    windows = {
        'Flattop': 'r',
        'Blackman': 'g',
        'Hann': 'b'
    }

    N = len(signal)
    T = N / fs

    plt.figure(figsize=(10, 4))

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
            if  fp - bw/2 > 0:
                bw_range = (fp - bw/2, fp + bw/2)
            else:
                bw_range = (0, bw-fp)
        elif mode == 'bp2':
            bw = bandwidth_bp2(fs, Y, percentage, f_min)
            bw_range = (f_min, f_min + bw)
        else:
            raise ValueError("Modo inválido. Usa 'lp', 'bp' o 'bp2'.")

        # Etiqueta con datos
        label = f"{name} - fp={fp:.1f}Hz - bw={bw:.1f}Hz - P={power:.2e}"

        # Graficar PSD
        plt.semilogy(f, Y, color=color, label=label)

        # Marcar frecuencia pico
        plt.axvline(fp, color=color, linestyle='--')

        # Marcar ancho de banda
        plt.axvspan(*bw_range, color=color, alpha=0.2)

    power_t = power_t / len(windows)  # Promedio de potencia total
    power_t = power_t*percentage

    plt.title(f"{title} - Potencia concatenada {power_t:.2e}")
    plt.xlabel(f"Frecuencia [Hz] | N = {N} muestras | fs = {fs} Hz | T = {T:.2f} s")
    plt.ylabel("Densidad espectral [W/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def diseñar_filtrar_plotear(fs, bw, Ap, As, señal, tipo='ellip', mode='lp', fc_min=None, plot_espectro=True, porcentaje=0.99, bi_lateral=False, label="senal", zoom=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import iirdesign, sosfilt, sosfiltfilt, sosfreqz, group_delay, sos2tf, welch
    from scipy.io.wavfile import write
    import os
    import pathlib

    # Función interna para guardar señales
    def guardar_senal(senal, label, fc, bw, mode, bilateral=False):
        carpeta = "señales_filtradas"
        pathlib.Path(carpeta).mkdir(parents=True, exist_ok=True)

        tipo = 'filtfilt' if bilateral else 'sosfilt'

        # Guardar como .npy
        nombre_archivo = f"{label}_{mode}_{tipo}.npy"
        path_completo = os.path.join(carpeta, nombre_archivo)
        np.save(path_completo, senal)

        # Guardar como .wav
        nombre_archivo_wav = f"{label}_{mode}_{tipo}.wav"
        path_completo_wav = os.path.join(carpeta, nombre_archivo_wav)
        write(path_completo_wav, fs, senal.astype(np.float32))  # asegurate del tipo

    nyq = fs / 2
    assert tipo in ['butter', 'cheby1', 'cheby2', 'ellip'], "Filtro no soportado"

    if mode == 'lp':
        fp = fc_min + bw  # frecuencia de paso
        fs_ = fp + (bw / 2)  # frecuencia de corte
        wp = fp / nyq
        ws = fs_ / nyq
        if not (0 < wp < ws < 1):
            raise ValueError("Error: verifica frecuencias en modo 'lp'")

    elif mode == 'bp':
        if fc_min is None:
            raise ValueError("Falta fc_min para modo 'bp'")
        fmin = fc_min
        fmax = fmin + bw
        wp = [fmin / nyq, fmax / nyq]
        ws = [(fmin - bw/2) / nyq, (fmax + bw/2) / nyq]
        if any(x <= 0 or x >= 1 for x in ws + wp):
            print(f"ws: {ws}, wp: {wp}, nyq: {nyq}")
            raise ValueError("Error: verifica frecuencias, deben ser > 0 y < Nyquist")

    else:
        raise ValueError("Modo inválido. Usa 'lp' o 'bp'")

    sos = iirdesign(wp=wp, ws=ws, gpass=Ap, gstop=As, output='sos', ftype=tipo)
    señal_filtrada = sosfilt(sos, señal)
    guardar_senal(señal_filtrada, label, fc_min, bw, mode, bilateral=False)

    señal_filtrada_bi = None
    if bi_lateral:
        señal_filtrada_bi = sosfiltfilt(sos, señal)
        guardar_senal(señal_filtrada_bi, label, fc_min, bw, mode, bilateral=True)

    plt.figure(figsize=(16, 9))
    plt.plot(señal, label='Original', alpha=0.4)
    plt.plot(señal_filtrada, label='Filtrado causal', alpha=0.8)
    if bi_lateral:
        plt.plot(señal_filtrada_bi, label='Filtrado bilateral', alpha=0.6)
    plt.title("Señal en el tiempo")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid()
    plt.show()

    if zoom is not None and len(zoom) == 2:
        tmin, tmax = zoom
        plt.figure(figsize=(16, 9))
        plt.plot(range(tmin, tmax), señal[tmin:tmax], label='Original', alpha=0.4)
        plt.plot(range(tmin, tmax), señal_filtrada[tmin:tmax], label='Filtrado causal', alpha=0.8)
        if bi_lateral:
            plt.plot(range(tmin, tmax), señal_filtrada_bi[tmin:tmax], label='Filtrado bilateral', alpha=0.6)
        plt.title(f"Zoom de la señal entre muestras {tmin} y {tmax}")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.grid()
        plt.show()

    w, h = sosfreqz(sos, worN=2048, fs=fs)
    plt.figure(figsize=(16, 9))
    plt.plot(w, 20 * np.log10(np.abs(h)), label='Respuesta en frecuencia')
    if mode == 'lp':
        plt.axvline(fp, color='g', linestyle='--', label=f'fp = {fp:.1f} Hz')
        plt.axvline(fs_, color='r', linestyle='--', label=f'fs = {fs_:.1f} Hz')
    else:
        plt.axvline(fmin, color='g', linestyle='--', label=f'fmin = {fmin:.1f} Hz')
        plt.axvline(fmax, color='g', linestyle='--', label=f'fmax = {fmax:.1f} Hz')
        plt.axvline(ws[0]*nyq, color='r', linestyle='--', label=f'fs1 = {ws[0]*nyq:.1f} Hz')
        plt.axvline(ws[1]*nyq, color='r', linestyle='--', label=f'fs2 = {ws[1]*nyq:.1f} Hz')
    plt.title("Respuesta en frecuencia del filtro")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Ganancia (dB)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(w, np.unwrap(np.angle(h)), label='Fase del filtro')
    plt.title("Fase del filtro")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Fase (radianes)")
    plt.grid()
    plt.legend()
    plt.show()

    b, a = sos2tf(sos)
    w_gd, gd = group_delay((b, a), fs=fs)
    plt.figure(figsize=(16, 9))
    plt.plot(w_gd, gd, label='Retardo en grupo')
    plt.title("Retardo en grupo del filtro")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Retardo [muestras]")
    plt.grid()
    plt.legend()
    plt.show()

    if plot_espectro:
        windows = {'Flattop': 'r', 'Blackman': 'g', 'Hann': 'b'}
        from funciones import detector_fp, bandwidth_lp
        N = len(señal_filtrada)
        T = N / fs
        plt.figure(figsize=(16, 9))
        power_t = 0

        for name, color in windows.items():
            f, Y = welch(señal_filtrada, fs, window=name.lower())
            fp = detector_fp(fs, Y)
            power = np.trapz(Y, f)
            power_t += power
            bw_medida = bandwidth_lp(fs, Y, porcentaje)
            label_welch = f"{name} - fp={fp:.1f}Hz - bw={bw_medida:.1f}Hz - P={power:.2e}"
            plt.semilogy(f, Y, color=color, label=label_welch)
            plt.axvline(fp, color=color, linestyle='--')
            plt.axvspan(0, bw_medida, color=color, alpha=0.2)

        power_t /= len(windows)
        power_t *= porcentaje
        plt.title(f"Análisis tipo Welch - Potencia concat. {power_t:.2e}")
        plt.xlabel(f"Frecuencia [Hz] | N = {N} | fs = {fs} Hz | T = {T:.2f} s")
        plt.ylabel("Densidad espectral [W/Hz]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()