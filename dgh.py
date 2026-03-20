import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, rfft, rfftfreq
from scipy.signal import medfilt
import hapi as hp

# ---------------- Параметры ----------------
c = 299792458.0
tau = 50e-15
lambda0 = 6.3e-7
w0 = 2 * np.pi * c / lambda0
lambda_phases = 6.3e-7
p_sat = 0.0276
w_size = 21  # окно для сглаживания

# Два файла
filename1 = r"C:\Users\Андрей\PycharmProjects\Lab\2026.03.19\POS_SCAN_20.15.20.txt"  # образец
filename2 = r"C:\Users\Андрей\PycharmProjects\Lab\2026.03.19\POS_SCAN_20.28.07.txt"  # референс

# ---------------- Загрузка и обработка сигналов ----------------
data1 = np.loadtxt(filename1)
data2 = np.loadtxt(filename2)

phases1, signal1 = data1[:, 0], data1[:, 2]
phases2, signal2 = data2[:, 0], data2[:, 2]

# Перевод фаз в задержку
delay1 = phases1 / (2 * np.pi) * lambda_phases / c
delay2 = phases2 / (2 * np.pi) * lambda_phases / c

# Сортировка
idx1 = np.argsort(delay1)
idx2 = np.argsort(delay2)
delay1, signal1 = delay1[idx1], signal1[idx1]
delay2, signal2 = delay2[idx2], signal2[idx2]

# Общее окно
delay_min = max(delay1.min(), delay2.min())
delay_max = min(delay1.max(), delay2.max())
mask1 = (delay1 >= delay_min) & (delay1 <= delay_max)
mask2 = (delay2 >= delay_min) & (delay2 <= delay_max)

delay1, signal1 = delay1[mask1], signal1[mask1]
delay2, signal2 = delay2[mask2], signal2[mask2]

print(f"{filename1}: {len(data1)} → {len(delay1)} точек")
print(f"{filename2}: {len(data2)} → {len(delay2)} точек")

background1 = medfilt(signal1, kernel_size=w_size)
background2 = medfilt(signal2, kernel_size=w_size)
signal1_clean = signal1 - background1
signal2_clean = signal2 - background2


N1 = len(delay1)
N2 = len(delay2)
N_common = max(N1, N2)

delay_uniform = np.linspace(-1e-11, 1e-11, N_common)

# Интерполяция очищенных сигналов на общую сетку
interp1 = interp1d(delay1, signal1_clean, kind='linear',
                   bounds_error=False, fill_value=0.0)
interp2 = interp1d(delay2, signal2_clean, kind='linear',
                   bounds_error=False, fill_value=0.0)

signal1_uniform = interp1(delay_uniform)
signal2_uniform = interp2(delay_uniform)

# Обрезаем края
cut = w_size // 2
delay_cut = delay_uniform[cut:-cut]
signal1_cut = signal1_uniform[cut:-cut]
signal2_cut = signal2_uniform[cut:-cut]

dt = delay_cut[1] - delay_cut[0]
freq = rfftfreq(len(delay_cut), dt)

spec1 = rfft(signal1_cut)
spec2 = rfft(signal2_cut)

spectrum1 = np.abs(spec1)
spectrum2 = np.abs(spec2)

# 🔥 ВЫЧИТАНИЕ СПЕКТРОВ 🔥
spectrum_diff = spectrum1 - spectrum2

print(f"Спектры готовы: {len(freq)} точек")
print(f"Спектр 1 max: {np.max(spectrum1):.2e}")
print(f"Спектр 2 max: {np.max(spectrum2):.2e}")
print(f"Разность max: {np.max(spectrum_diff):.2e}")

# ---------------- Графики во времени ----------------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(delay1, signal1_clean, label='S1 очищенный', linewidth=2)
plt.plot(delay2, signal2_clean, label='S2 очищенный', alpha=0.7)
plt.legend();
plt.grid();
plt.ylabel("Signal");
plt.title("Очищенные сигналы")

plt.subplot(3, 1, 2)
plt.plot(delay_cut, signal1_cut, label='S1 на общей сетке', linewidth=2)
plt.plot(delay_cut, signal2_cut, label='S2 на общей сетке', alpha=0.7)
plt.legend();
plt.grid();
plt.ylabel("Uniform signals");
plt.xlabel("Delay (s)")

plt.subplot(3, 1, 3)
plt.plot(delay_cut, signal1_cut - signal2_cut, label='Разность во времени', linewidth=2)
plt.legend();
plt.grid();
plt.ylabel("Difference");
plt.xlabel("Delay (s)")
plt.tight_layout()
plt.show()

f_min, f_max = 1.5e13, 7.5e13
band = (freq >= f_min) & (freq <= f_max)

freq_band = freq[band]
spec1_band = spectrum1[band]
spec2_band = spectrum2[band]
spec_diff_band = spectrum_diff[band]

spec1_norm = spec1_band / np.max(spec1_band)
spec2_norm = spec2_band / np.max(spec2_band)
spec_diff_norm = spec_diff_band / np.max(np.abs(spec_diff_band))

N = len(delay_cut)
t_theory = np.arange(-N // 2, N // 2) * dt
pulse = np.exp(-t_theory ** 2 / (2 * (2.35 / tau) ** 2)) * np.sin(w0 * t_theory)

spec_th_full = fftshift(fft(pulse))
freq_full = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
mask_pos = freq_full >= 0
freq_pos = freq_full[mask_pos]
spec_th_pos = np.abs(spec_th_full[mask_pos])

interp_th = interp1d(freq_pos, spec_th_pos, kind='linear',
                     bounds_error=False, fill_value=0.0)
spec_th_on_freq = interp_th(freq)
spec_th_band = spec_th_on_freq[band]
spec_th_norm = spec_th_band / np.max(spec_th_band)


hp.db_begin('32')
nu_min, nu_max = 1500.0, 2500.0
hp.fetch('H2O', 1, 1, nu_min, nu_max)

nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
    SourceTables='H2O',
    Environment={'p': 1.0, 'T': 296.0},
    Diluent={'air': 1.0, 'self': 0.0},
    WavenumberRange=[nu_min, nu_max],
    WavenumberStep=0.01,
    HITRAN_units=False
)

p_h2o = 0.5 * p_sat  # 50% влажность
k_hitran = k_ref * (p_h2o / 1.0)
tr_h = np.exp(-k_hitran * 150.0)  # 150 см путь

f_cm1 = freq / c * 1e-2
interp_T = interp1d(nu_hitran, tr_h, kind='linear',
                    bounds_error=False, fill_value=1.0)
T_on_freq = interp_T(f_cm1)
spec_th_abs =  T_on_freq
spec_th_abs_band = spec_th_abs[band]
spec_th_abs_norm = spec_th_abs_band / np.max(spec_th_abs_band)

# Диагностика пиков
f_exp_peak = freq_band[np.argmax(spec_diff_band)]
f_th_peak = freq_band[np.argmax(spec_th_abs_band)]

# ---------------- Финальные спектры ----------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(freq_band / 1e12, spec1_norm, label="Спектр образца", linewidth=2)
plt.plot(freq_band / 1e12, spec2_norm, label="Спектр референса", linewidth=2)
plt.plot(freq_band / 1e12, spec_th_abs_norm, 'k--', label="Теория + H₂O", linewidth=2)
plt.xlabel("Частота (THz)");
plt.ylabel("Нормированная амплитуда")
plt.xlim(45, 65);
plt.grid(True);
plt.legend();
plt.title("Исходные спектры")

plt.subplot(2, 1, 2)
plt.plot(freq_band / 1e12, spec_diff_norm, label="Спектр1 - Спектр2", linewidth=3)
plt.plot(freq_band / 1e12, spec_th_abs_norm, label="Теория + H₂O", linewidth=2.5)
plt.xlabel("Частота (THz)");
plt.ylabel("Нормированная разность")
plt.xlim(45, 55);
plt.grid(True);
plt.legend();
plt.title("ВЫЧИТАНИЕ СПЕКТРОВ")
plt.tight_layout()
plt.show()
