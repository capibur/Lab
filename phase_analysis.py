import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, rfft, rfftfreq, ifft
from scipy.signal import medfilt
from scipy.optimize import minimize_scalar
import task_two.hapi as hp

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


# ============== 1. АНАЛИЗ СПЕКТРАЛЬНОЙ ФАЗЫ ==============

def compute_spectrum_with_phase(signal, dt):
    """Вычисляет амплитуду и фазу спектра.
    
    Args:
        signal: временной сигнал
        dt: временной шаг
    
    Returns:
        freq: положительные частоты
        spectrum: амплитуда спектра
        phase: фаза спектра (в радианах)
    """
    n = len(signal)
    freq = rfftfreq(n, dt)
    spec_complex = rfft(signal)
    spectrum = np.abs(spec_complex)
    phase = np.angle(spec_complex)
    
    return freq, spectrum, phase


def extract_linear_phase(freq, phase, freq_band=None):
    """Извлекает линейную часть фазы (наклон) методом наименьших квадратов.
    
    Args:
        freq: массив частот
        phase: массив фазы
        freq_band: кортеж (f_min, f_max) для ограничения диапазона
    
    Returns:
        slope: наклон фазы (радиан/Гц)
        intercept: пересечение
        phase_linear: восстановленная линейная фаза
    """
    if freq_band is not None:
        mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
        freq_fit = freq[mask]
        phase_fit = phase[mask]
    else:
        # Используем только положительные частоты с ненулевой амплитудой
        mask = freq > 0
        freq_fit = freq[mask]
        phase_fit = phase[mask]
    
    # Линейная регрессия
    coeffs = np.polyfit(freq_fit, phase_fit, 1)
    slope = coeffs[0]  # радиан/Гц
    intercept = coeffs[1]
    
    phase_linear = slope * freq + intercept
    
    return slope, intercept, phase_linear


def remove_linear_phase(phase, freq, slope, intercept):
    """Удаляет линейную фазу из спектра.
    
    Returns:
        phase_residual: остаточная фаза после удаления наклона
        delay: эквивалентная задержка (slope / 2π)
    """
    phase_linear = slope * freq + intercept
    phase_residual = phase - phase_linear
    
    # Оборачиваем фазу в диапазон [-π, π]
    phase_residual = np.angle(np.exp(1j * phase_residual))
    
    # Рассчитываем задержку
    delay = slope / (2 * np.pi)  # перевод из радиан/Гц в секунды
    
    return phase_residual, delay


def find_optimal_delay(freq, phase, spectrum, freq_band=None):
    """Подбирает оптимальную задержку для флаттирования фазы (минимизирует остаточную дисперсию).
    
    Args:
        freq: массив частот
        phase: спектральная фаза
        spectrum: амплитуда спектра (для взвешивания)
        freq_band: диапазон для анализа
    
    Returns:
        optimal_delay: оптимальная задержка в секундах
        phase_residual: остаточная фаза после коррекции
        variance: минимальная дисперсия фазы
    """
    if freq_band is not None:
        mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
        freq_opt = freq[mask]
        phase_opt = phase[mask]
        spec_opt = spectrum[mask]
    else:
        mask = freq > 0
        freq_opt = freq[mask]
        phase_opt = phase[mask]
        spec_opt = spectrum[mask]
    
    # Нормируем амплитуду для взвешивания
    weights = spec_opt / np.max(spec_opt)
    
    def phase_variance(delay_sec):
        """Вычисляет взвешенную дисперсию фазы после коррекции задержки."""
        phase_corrected = phase_opt - 2 * np.pi * freq_opt * delay_sec
        phase_corrected = np.angle(np.exp(1j * phase_corrected))
        variance = np.sum(weights * phase_corrected ** 2) / np.sum(weights)
        return variance
    
    # Оптимизация
    result = minimize_scalar(phase_variance, bounds=(-1e-14, 1e-14), method='bounded')
    optimal_delay = result.x
    min_variance = result.fun
    
    # Вычисляем остаточную фазу
    phase_residual = phase_opt - 2 * np.pi * freq_opt * optimal_delay
    phase_residual = np.angle(np.exp(1j * phase_residual))
    
    return optimal_delay, phase_residual, min_variance


# ============== 2. АНАЛИЗ ИМПУЛЬСА С ПРЕДЫМПУЛЬСОМ ==============

def generate_pulse_with_prepulse(n_points, dt, prepulse_delay_fs=0, prepulse_amplitude=0.1):
    """Генерирует импульс с опциональным предымпульсом.
    
    Args:
        n_points: количество точек
        dt: временной шаг
        prepulse_delay_fs: задержка предымпульса в фемтосекундах
        prepulse_amplitude: амплитуда предымпульса (доля от основного)
    
    Returns:
        pulse: импульс с предымпульсом
        t: временная ось
    """
    t = np.arange(-n_points // 2, n_points // 2) * dt
    
    # Основной импульс
    gaussian = np.exp(-t ** 2 / (2 * (2.35 / CONST_TAU) ** 2))
    modulation = np.sin(CONST_W0 * t)
    main_pulse = gaussian * modulation
    
    # Предымпульс (скопированный импульс с задержкой)
    if prepulse_amplitude > 0 and prepulse_delay_fs != 0:
        prepulse_delay_s = prepulse_delay_fs * 1e-15
        main_pulse_delayed = np.exp(-(t - prepulse_delay_s) ** 2 / (2 * (2.35 / CONST_TAU) ** 2)) * \
                            np.sin(CONST_W0 * (t - prepulse_delay_s))
        pulse = main_pulse + prepulse_amplitude * main_pulse_delayed
    else:
        pulse = main_pulse
    
    return pulse, t


def analyze_pulse_spectrum(pulse, t, label=""):
    """Анализирует спектр импульса.
    
    Returns:
        freq, spectrum, phase, t
    """
    dt = t[1] - t[0]
    n = len(pulse)
    
    # Полный спектр для получения обоих положительных и отрицательных частот
    spec_full = fftshift(fft(pulse))
    freq_full = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    
    # Положительные частоты
    mask_pos = freq_full >= 0
    freq = freq_full[mask_pos]
    spec_complex = spec_full[mask_pos]
    spectrum = np.abs(spec_complex)
    phase = np.angle(spec_complex)
    
    # Нормировка спектра
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)
    
    return freq, spectrum, phase, t


# ============== 3. ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ ИЗ ФАЙЛОВ ==============

def load_and_process_signal(filename):
    """Загружает и обрабатывает сигнал из файла.
    
    Returns:
        delay, signal_clean, dt
    """
    data = np.loadtxt(filename)
    phases = data[:, 0]
    signal = data[:, 2]
    
    # Конвертирование в задержку
    delay = phases / (2 * np.pi) * CONST_LAMBDA_PHASES / CONST_C
    
    # Сортировка
    idx = np.argsort(delay)
    delay = delay[idx]
    signal = signal[idx]
    
    # Удаление фона
    background = medfilt(signal, kernel_size=CONST_WINDOW_SIZE)
    signal_clean = signal - background
    
    # Интерполяция на равномерную сетку
    delay_min, delay_max = delay.min(), delay.max()
    n_uniform = len(delay)
    delay_uniform = np.linspace(delay_min, delay_max, n_uniform)
    
    interp_func = interp1d(delay, signal_clean, kind='linear',
                          bounds_error=False, fill_value=0.0)
    signal_uniform = interp_func(delay_uniform)
    
    # Обрезка краёв
    cut = CONST_WINDOW_SIZE // 2
    delay_cut = delay_uniform[cut:-cut]
    signal_cut = signal_uniform[cut:-cut]
    
    dt = delay_cut[1] - delay_cut[0]
    
    return delay_cut, signal_cut, dt


def apply_h2o_absorption_simple(freq, humidity_fraction=0.5, path_length_cm=150.0):
    """Упрощённое применение H2O поглощения (для демонстрации).
    
    Returns:
        transmission: коэффициент пропускания
    """
    nu_min, nu_max = 1500.0, 2500.0
    hp.db_begin('32')
    hp.fetch('H2O', 1, 1, nu_min, nu_max)
    
    nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
        SourceTables='H2O',
        Environment={'p': 1.0, 'T': 296.0},
        Diluent={'air': 1.0, 'self': 0.0},
        WavenumberRange=[nu_min, nu_max],
        WavenumberStep=0.01,
        HITRAN_units=False
    )
    
    p_h2o = humidity_fraction * CONST_P_SAT
    k_h2o = k_ref * (p_h2o / 1.0)
    
    f_cm1 = freq / CONST_C * 1e-2
    interp_T = interp1d(nu_hitran, np.exp(-k_h2o * path_length_cm),
                        kind='linear', bounds_error=False, fill_value=1.0)
    transmission = interp_T(f_cm1)
    
    return transmission


# ============== 4. ВИЗУАЛИЗАЦИЯ ==============

def plot_phase_analysis(freq_band_1, freq_band_2, phase_sample, phase_reference, 
                       phase_sample_corrected, phase_reference_corrected,
                       delay_sample, delay_reference):
    """Построение графиков анализа фазы."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Исходные фазы
    ax = axes[0, 0]
    ax.plot(freq_band_1 / 1e12, np.unwrap(phase_sample), 'b-', label='Образец', linewidth=1.5)
    ax.plot(freq_band_2 / 1e12, np.unwrap(phase_reference), 'r-', label='Референс', linewidth=1.5)
    ax.set_xlabel('Частота (THz)')
    ax.set_ylabel('Фаза (рад)')
    ax.set_title('Исходная спектральная фаза')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Скорректированные фазы
    ax = axes[0, 1]
    ax.plot(freq_band_1 / 1e12, phase_sample_corrected, 'b-', label='Образец', linewidth=1.5)
    ax.plot(freq_band_2 / 1e12, phase_reference_corrected, 'r-', label='Референс', linewidth=1.5)
    ax.set_xlabel('Частота (THz)')
    ax.set_ylabel('Остаточная фаза (рад)')
    ax.set_title(f'После коррекции задержки')
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Подобранные задержки
    ax = axes[1, 0]
    delays = [delay_sample * 1e15, delay_reference * 1e15]  # в фемтосекундах
    labels = ['Образец', 'Референс']
    bars = ax.bar(labels, delays, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Задержка (fs)')
    ax.set_title('Подобранные задержки фазовой коррекции')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, delay in zip(bars, delays):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{delay:.2f} fs',
                ha='center', va='bottom')
    
    # Дисперсия фазы
    ax = axes[1, 1]
    phase_var_sample = np.var(phase_sample_corrected)
    phase_var_reference = np.var(phase_reference_corrected)
    vars_before = [np.var(phase_sample), np.var(phase_reference)]
    vars_after = [phase_var_sample, phase_var_reference]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, vars_before, width, label='До коррекции', alpha=0.7)
    bars2 = ax.bar(x + width/2, vars_after, width, label='После коррекции', alpha=0.7)
    
    ax.set_ylabel('Дисперсия фазы (рад²)')
    ax.set_title('Уменьшение дисперсии фазы после коррекции')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_prepulse_analysis(freq, spec_main, spec_prepulse_small, spec_prepulse_large,
                          t, pulse_main, pulse_prepulse_small, pulse_prepulse_large):
    """Построение графиков импульса с предымпульсом."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    freq_thz = freq / 1e12
    
    # === СПЕКТРЫ ===
    ax = axes[0, 0]
    ax.plot(freq_thz, spec_main, 'b-', label='Без предымпульса', linewidth=2)
    ax.set_xlabel('Частота (THz)')
    ax.set_ylabel('Амплитуда спектра')
    ax.set_title('Основной импульс')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(freq_thz, spec_prepulse_small, 'g-', label='Предымпульс 5%', linewidth=2)
    ax.set_xlabel('Частота (THz)')
    ax.set_ylabel('Амплитуда спектра')
    ax.set_title('С малым предымпульсом (20 fs, 5%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 2]
    ax.plot(freq_thz, spec_prepulse_large, 'r-', label='Предымпульс 30%', linewidth=2)
    ax.set_xlabel('Частота (THz)')
    ax.set_ylabel('Амплитуда спектра')
    ax.set_title('С большим предымпульсом (50 fs, 30%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # === ИМПУЛЬСЫ ВО ВРЕМЕНИ ===
    t_ps = t * 1e12  # перевод в пикосекунды
    
    ax = axes[1, 0]
    ax.plot(t_ps, pulse_main, 'b-', linewidth=1.5)
    ax.set_xlabel('Время (ps)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('Основной импульс (время)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 0.2)
    
    ax = axes[1, 1]
    ax.plot(t_ps, pulse_prepulse_small, 'g-', linewidth=1.5)
    ax.set_xlabel('Время (ps)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('С малым предымпульсом (время)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 0.2)
    
    ax = axes[1, 2]
    ax.plot(t_ps, pulse_prepulse_large, 'r-', linewidth=1.5)
    ax.set_xlabel('Время (ps)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('С большим предымпульсом (время)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 0.2)
    
    plt.tight_layout()
    plt.show()


# ============== ГЛАВНАЯ ФУНКЦИЯ ==============

def main():
    """Главный анализ спектральной фазы и импульсов с предымпульсом."""
    
    print("=" * 80)
    print("АНАЛИЗ СПЕКТРАЛЬНОЙ ФАЗЫ И ИМПУЛЬСОВ С ПРЕДЫМПУЛЬСОМ")
    print("=" * 80)
    
    # ========== ЧАСТЬ 1: АНАЛИЗ СПЕКТРАЛЬНОЙ ФАЗЫ ==========
    print("\n--- ЧАСТЬ 1: Анализ спектральной фазы ---\n")
    
    # Загрузка данных
    print("Загрузка сигналов...")
    delay1, signal1, dt1 = load_and_process_signal(FILE_SAMPLE)
    delay2, signal2, dt2 = load_and_process_signal(FILE_REFERENCE)
    
    # Вычисление спектра и фазы
    freq1, spec1, phase1 = compute_spectrum_with_phase(signal1, dt1)
    freq2, spec2, phase2 = compute_spectrum_with_phase(signal2, dt2)
    
    # Выделение частотной полосы
    f_min, f_max = 1.5e13, 7.5e13
    mask1 = (freq1 >= f_min) & (freq1 <= f_max)
    mask2 = (freq2 >= f_min) & (freq2 <= f_max)
    
    freq_band_1 = freq1[mask1]
    freq_band_2 = freq2[mask2]
    phase1_band = phase1[mask1]
    phase2_band = phase2[mask2]
    spec1_band = spec1[mask1]
    spec2_band = spec2[mask2]
    
    print(f"Образец: {len(signal1)} точек, dt={dt1:.3e} s")
    print(f"Референс: {len(signal2)} точек, dt={dt2:.3e} s")
    print(f"Частотная полоса образец: {f_min/1e12:.1f} - {f_max/1e12:.1f} THz ({len(freq_band_1)} точек)")
    print(f"Частотная полоса референс: {f_min/1e12:.1f} - {f_max/1e12:.1f} THz ({len(freq_band_2)} точек)")
    
    # Подбор оптимальной задержки для коррекции фазы
    print("\nПодбор оптимальной задержки для коррекции фазы...")
    delay1_opt, phase1_residual, var1_min = find_optimal_delay(freq1, phase1, spec1, (f_min, f_max))
    delay2_opt, phase2_residual, var2_min = find_optimal_delay(freq2, phase2, spec2, (f_min, f_max))
    
    print(f"Образец: оптимальная задержка = {delay1_opt*1e15:.4f} fs, дисперсия = {var1_min:.6f} рад²")
    print(f"Референс: оптимальная задержка = {delay2_opt*1e15:.4f} fs, дисперсия = {var2_min:.6f} рад²")
    
    # Коррекция фазы в полосе интереса
    phase1_residual_band = phase1_band - 2 * np.pi * freq_band_1 * delay1_opt
    phase1_residual_band = np.angle(np.exp(1j * phase1_residual_band))
    
    phase2_residual_band = phase2_band - 2 * np.pi * freq_band_2 * delay2_opt
    phase2_residual_band = np.angle(np.exp(1j * phase2_residual_band))
    
    # Использование первой полосы для визуализации (применим интерполяцию если нужно)
    freq_band = freq_band_1
    
    # Визуализация анализа фазы
    plot_phase_analysis(freq_band_1, freq_band_2, phase1_band, phase2_band,
                       phase1_residual_band, phase2_residual_band,
                       delay1_opt, delay2_opt)
    
    # ========== ЧАСТЬ 2: АНАЛИЗ ИМПУЛЬСА С ПРЕДЫМПУЛЬСОМ ==========
    print("\n--- ЧАСТЬ 2: Анализ импульса с предымпульсом ---\n")
    
    # Генерирование импульсов с разными предымпульсами
    n_points = 8192
    dt_pulse = 5e-16  # 0.5 fs
    
    # Основной импульс
    pulse_main, t_main = generate_pulse_with_prepulse(n_points, dt_pulse, 0, 0)
    freq_main, spec_main, _, t_main = analyze_pulse_spectrum(pulse_main, t_main, "Основной")
    
    # Импульс с малым предымпульсом (20 fs, 5% амплитуды)
    pulse_prepulse_small, t_small = generate_pulse_with_prepulse(n_points, dt_pulse, 20, 0.05)
    freq_small, spec_prepulse_small, _, t_small = analyze_pulse_spectrum(pulse_prepulse_small, t_small, "Малый предымпульс")
    
    # Импульс с большим предымпульсом (50 fs, 30% амплитуды)
    pulse_prepulse_large, t_large = generate_pulse_with_prepulse(n_points, dt_pulse, 50, 0.30)
    freq_large, spec_prepulse_large, _, t_large = analyze_pulse_spectrum(pulse_prepulse_large, t_large, "Большой предымпульс")
    
    print(f"Основной импульс: τ = {CONST_TAU*1e15:.1f} fs, ν₀ = {CONST_W0/(2*np.pi)/1e12:.1f} THz")
    print(f"Сгенерировано импульсов с предымпульсами:")
    print(f"  - Без предымпульса")
    print(f"  - С малым предымпульсом: задержка=20 fs, амплитуда=5%")
    print(f"  - С большим предымпульсом: задержка=50 fs, амплитуда=30%")
    
    # Визуализация импульсов
    plot_prepulse_analysis(freq_main, spec_main, spec_prepulse_small, spec_prepulse_large,
                          t_main, pulse_main, pulse_prepulse_small, pulse_prepulse_large)
    
    print("\n" + "=" * 80)
    print("Анализ завершён!")
    print("=" * 80)
    
    return {
        'phase_analysis': {
            'freq': freq_band,
            'phase1_raw': phase1_band,
            'phase2_raw': phase2_band,
            'phase1_corrected': phase1_residual_band,
            'phase2_corrected': phase2_residual_band,
            'delay1': delay1_opt,
            'delay2': delay2_opt,
        },
        'prepulse_analysis': {
            'freq': freq_main,
            'spec_main': spec_main,
            'spec_prepulse_small': spec_prepulse_small,
            'spec_prepulse_large': spec_prepulse_large,
        }
    }


if __name__ == "__main__":
    results = main()
