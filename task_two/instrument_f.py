import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, rfft, rfftfreq
from scipy.signal import medfilt
import hapi as hp

# ============== ПАРАМЕТРЫ ==============
CONST_C = 299792458.0
CONST_TAU = 50e-15
CONST_LAMBDA0 = 6.3e-7
CONST_W0 = 2 * np.pi * CONST_C / CONST_LAMBDA0
CONST_LAMBDA_PHASES = 6.3e-7
CONST_P_SAT = 0.0276
CONST_WINDOW_SIZE = 21

# Пути к файлам
FILE_SAMPLE = r"C:\Users\Андрей\PycharmProjects\Lab\2026.03.19\POS_SCAN_20.15.20.txt"
FILE_REFERENCE = r"C:\Users\Андрей\PycharmProjects\Lab\2026.03.19\POS_SCAN_20.28.07.txt"

# ============== 1. РАБОТА С API (HITRAN) ==============

def fetch_and_process_h2o_from_api(freq, nu_min=1500.0, nu_max=2500.0, 
                                    db_path='32', path_length_cm=150.0, 
                                    humidity_fraction=0.5):
    """Загружает данные поглощения H2O из HITRAN и применяет их к спектру.
    
    Args:
        freq: массив частот для применения поглощения
        nu_min, nu_max: диапазон волновых чисел для HITRAN
        db_path: путь к базе данных HITRAN
        path_length_cm: длина оптического пути в см
        humidity_fraction: доля насыщения влажности
    
    Returns:
        nu_hitran: волновые числа из HITRAN
        k_ref: коэффициент поглощения
        transmission: коэффициент пропускания на частотах freq
    """
    # Загрузка из HITRAN
    hp.db_begin(db_path)
    hp.fetch('H2O', 1, 1, nu_min, nu_max)
    
    nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
        SourceTables='H2O',
        Environment={'p': 1.0, 'T': 296.0},
        Diluent={'air': 1.0, 'self': 0.0},
        WavenumberRange=[nu_min, nu_max],
        WavenumberStep=0.01,
        HITRAN_units=False
    )
    
    # Применение поглощения
    p_h2o = humidity_fraction * CONST_P_SAT
    k_h2o = k_ref * (p_h2o / 1.0)
    
    f_cm1 = freq / CONST_C * 1e-2
    interp_T = interp1d(nu_hitran, np.exp(-k_h2o * path_length_cm),
                        kind='linear', bounds_error=False, fill_value=1.0)
    transmission = interp_T(f_cm1)
    
    return nu_hitran, k_ref, transmission




def process_signal_from_file(filename, delay_min=-3e-11, delay_max=3e-11):
    """Обработка одного файла: загрузка → сортировка → фильтрация → очистка."""
    data = np.loadtxt(filename)
    phases, signal = data[:, 0], data[:, 2]

    delay = phases / (2 * np.pi) * CONST_LAMBDA_PHASES / CONST_C
    print(f"Delay range: {delay.min()} to {delay.max()}")
    idx = np.argsort(delay)
    delay, signal = delay[idx], signal[idx]

    print(f"{filename}: {len(data)} → {len(delay)} точек")

    mask = (delay >= delay_min) & (delay <= delay_max)
    delay, signal = delay[mask], signal[mask]

    print(delay, signal)
    
    # Удаление фона медианным фильтром
    background = medfilt(signal, kernel_size=CONST_WINDOW_SIZE)
    signal_clean = signal - background 
    
    print(len(delay), len(signal_clean))
    return delay, signal_clean


def interpolate_signals(delay1, signal1, delay2, signal2):
    """Интерполирует два сигнала на общую сетку и обрезает края."""
    n_common = max(len(delay1), len(delay2))

    start = max(delay1.min(), delay2.min())
    stop = min(delay1.max(), delay2.max())
    if stop <= start:
        start = min(delay1.min(), delay2.min())
        stop = max(delay1.max(), delay2.max())

    delay_uniform = np.linspace(start, stop, n_common)

    interp1 = interp1d(delay1, signal1, kind='linear', bounds_error=False, fill_value=0.0)
    interp2 = interp1d(delay2, signal2, kind='linear', bounds_error=False, fill_value=0.0)

    signal1_unif = interp1(delay_uniform)
    signal2_unif = interp2(delay_uniform)

    cut = CONST_WINDOW_SIZE // 2
    delay_cut = delay_uniform[cut:-cut]
    signal1_cut = signal1_unif[cut:-cut]
    signal2_cut = signal2_unif[cut:-cut]
    dt = delay_cut[1] - delay_cut[0]

    return delay_cut, signal1_cut, signal2_cut, dt


def process_signal_from_files(filename1, filename2):
    """Полная обработка двух файлов с переиспользуемой функцией для одного файла."""
    delay1, signal1 = process_signal_from_file(filename1)
    delay2, signal2 = process_signal_from_file(filename2)

    delay_cut, signal1_cut, signal2_cut, dt = interpolate_signals(delay1, signal1, delay2, signal2)

    return delay_cut, signal1_cut, delay_cut, signal2_cut, dt, delay1, signal1, delay2, signal2


# ============== 3. ОБРАБОТКА И ВИЗУАЛИЗАЦИЯ СПЕКТРОВ ==============

def process_and_visualize_spectra(delay1_cut, signal1_cut, delay2_cut, signal2_cut, dt,
                                   delay1_raw, signal1_raw, delay2_raw, signal2_raw):
    """Вычисляет спектры, применяет теорию и H2O, построение графиков.
    
    Args:
        delay*_cut: обрезанные временные задержки
        signal*_cut: обрезанные сигналы
        dt: временной шаг
        delay*_raw, signal*_raw: исходные очищенные данные для временных графиков
    
    Returns:
        результаты спектрального анализа
    """
    # Спектральный анализ
    n = len(signal1_cut)
    freq = rfftfreq(n, dt)
    
    spec1_complex = rfft(signal1_cut)
    spec2_complex = rfft(signal2_cut)
    
    spectrum1 = np.abs(spec1_complex)
    spectrum2 = np.abs(spec2_complex)
    spectrum_diff = spectrum1 - spectrum2
    
    print(f"Спектры готовы: {len(freq)} точек")
    print(f"Спектр 1 max: {np.max(spectrum1):.2e}")
    print(f"Спектр 2 max: {np.max(spectrum2):.2e}")
    print(f"Разность max: {np.max(spectrum_diff):.2e}")

    plot_time_domain(delay1_raw, signal1_raw, delay2_raw, signal2_raw,
                     delay1_cut, signal1_cut, delay2_cut, signal2_cut)
    
    f_min, f_max = 1.5e13, 7.5e13
    mask_band = (freq >= f_min) & (freq <= f_max)
    
    freq_band = freq[mask_band]
    spec1_band = spectrum1[mask_band]
    spec2_band = spectrum2[mask_band]
    spec_diff_band = spectrum_diff[mask_band]
    
    # Нормировка
    spec1_norm = spec1_band / np.max(spec1_band)
    spec2_norm = spec2_band / np.max(spec2_band)
    spec_diff_norm = spec_diff_band / np.max(np.abs(spec_diff_band))
    
    # Теоретический импульс и спектр
    t_theory = np.arange(-n // 2, n // 2) * dt
    pulse_theory = np.exp(-t_theory ** 2 / (2 * (2.35 / CONST_TAU) ** 2)) * np.sin(CONST_W0 * t_theory)
    
    spec_th_full = fftshift(fft(pulse_theory))
    freq_full = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    
    mask_pos = freq_full >= 0
    freq_pos = freq_full[mask_pos]
    spec_th_pos = np.abs(spec_th_full[mask_pos])
    
    interp_th = interp1d(freq_pos, spec_th_pos, kind='linear',
                         bounds_error=False, fill_value=0.0)
    spec_th_on_freq = interp_th(freq)
    spec_th_band = spec_th_on_freq[mask_band]
    spec_th_norm = spec_th_band / np.max(spec_th_band)
    
    # H2O поглощение из HITRAN
    nu_min, nu_max = 1500.0, 2500.0
    nu_hitran, k_ref, tr_h2o = fetch_and_process_h2o_from_api(
        freq, nu_min, nu_max, path_length_cm=150.0, humidity_fraction=0.5
    )
    
    spec_abs_band = tr_h2o[mask_band]
    spec_abs_norm = spec_abs_band / np.max(spec_abs_band)
    
    # Финальные графики спектров
    plot_spectra(freq_band, spec1_norm, spec2_norm, spec_abs_norm, spec_diff_norm)
    
    return {
        'freq': freq,
        'spectrum1': spectrum1,
        'spectrum2': spectrum2,
        'spectrum_diff': spectrum_diff,
        'freq_band': freq_band,
        'spec1_band': spec1_band,
        'spec2_band': spec2_band,
        'spec_diff_band': spec_diff_band,
    }


# ============== ФУНКЦИИ ВИЗУАЛИЗАЦИИ ==============

def plot_time_domain(delay1, signal1_clean, delay2, signal2_clean,
                     delay1_cut, signal1_cut, delay2_cut, signal2_cut):
    """Построение графиков в временной области."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(delay1, signal1_clean, label='S1 очищенный', linewidth=2)
    plt.plot(delay2, signal2_clean, label='S2 очищенный', alpha=0.7)
    plt.legend()
    plt.grid()
    plt.ylabel("Signal")
    plt.title("Очищенные сигналы")
    
    plt.subplot(3, 1, 2)
    plt.plot(delay1_cut, signal1_cut, label='S1 на общей сетке', linewidth=2)
    plt.plot(delay2_cut, signal2_cut, label='S2 на общей сетке', alpha=0.7)
    plt.legend()
    plt.grid()
    plt.ylabel("Uniform signals")
    plt.xlabel("Delay (s)")
    
    plt.subplot(3, 1, 3)
    plt.plot(delay1_cut, signal1_cut - signal2_cut, label='Разность во времени', linewidth=2)
    plt.legend()
    plt.grid()
    plt.ylabel("Difference")
    plt.xlabel("Delay (s)")
    plt.tight_layout()
    plt.show()


def plot_spectra(freq_band, spec1_norm, spec2_norm, spec_abs_norm, spec_diff_norm):
    """Построение финальных спектров."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(freq_band / 1e12, spec1_norm, label="Спектр образца", linewidth=2)
    plt.plot(freq_band / 1e12, spec2_norm, label="Спектр референса", linewidth=2)
    plt.plot(freq_band / 1e12, spec_abs_norm, 'k--', label="Теория + H₂O", linewidth=2)
    plt.xlabel("Частота (THz)")
    plt.ylabel("Нормированная амплитуда")
    plt.xlim(45, 65)
    plt.grid(True)
    plt.legend()
    plt.title("Исходные спектры")
    
    plt.subplot(2, 1, 2)
    plt.plot(freq_band / 1e12, spec_diff_norm, label="Спектр1 - Спектр2", linewidth=3)
    plt.plot(freq_band / 1e12, spec_abs_norm, label="Теория + H₂O", linewidth=2.5)
    plt.xlabel("Частота (THz)")
    plt.ylabel("Нормированная разность")
    plt.xlim(45, 55)
    plt.grid(True)
    plt.legend()
    plt.title("ВЫЧИТАНИЕ СПЕКТРОВ")
    plt.tight_layout()
    plt.show()



def main():
    

    result = process_signal_from_files(FILE_SAMPLE, FILE_REFERENCE)
    delay1_cut, signal1_cut, delay2_cut, signal2_cut, dt, delay1_raw, signal1_raw, delay2_raw, signal2_raw = result
    

    specs = process_and_visualize_spectra(
        delay1_cut, signal1_cut, delay2_cut, signal2_cut, dt,
        delay1_raw, signal1_raw, delay2_raw, signal2_raw
    )
    
    return specs


if __name__ == "__main__":
    result = main()

