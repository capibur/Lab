import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
# Предполагаем, что ваша библиотека доступна
from instrument_f import process_signal_from_file as ps

# --- Константы ---
CONST_C = 299792458.0  
CONST_TAU = 50e-15  
CONST_LAMBDA0 = 6.3e-7  
CONST_W0 = 2 * np.pi * CONST_C / CONST_LAMBDA0  
CONST_WINDOW_SIZE = 21 

FILE_SAMPLE_WATER = r"2026.03.19\POS_SCAN_20.15.20.txt"
FILE_SAMPLE_NO_WATER = r"POS_SCAN_20.28.07.txt" 

# --- Обработка данных ---
res_water = ps(FILE_SAMPLE_WATER)
delay_water, signal_water = res_water
res_nowater = ps(FILE_SAMPLE_NO_WATER)
delay_nowater, signal_nowater = res_nowater

# Интерполяция
delay_min = max(delay_water.min(), delay_nowater.min())
delay_max = min(delay_water.max(), delay_nowater.max())
delay_common = np.linspace(delay_min, delay_max, min(len(delay_water), len(delay_nowater)))

interp_water = interp1d(delay_water, signal_water, kind='linear', bounds_error=False, fill_value=0)
interp_nowater = interp1d(delay_nowater, signal_nowater, kind='linear', bounds_error=False, fill_value=0)

signal_water = interp_water(delay_common)
signal_nowater = interp_nowater(delay_common)
delay_water = delay_common
delay_nowater = delay_common
dt = delay_water[1] - delay_water[0]

# --- Расчет FFT с Zero-padding ---
n_original = len(signal_water)
n_fft = 2**int(np.ceil(np.log2(n_original * 8))) 
freq = rfftfreq(n_fft, dt)

spec_water_complex = rfft(signal_water, n=n_fft)
spec_nowater_complex = rfft(signal_nowater, n=n_fft)

# Фазы в радианах
spec_arg_water = np.angle(spec_water_complex)
spec_arg_nowater = np.angle(spec_nowater_complex)

# Развертка
spec_arg_water_unw = np.unwrap(spec_arg_water)
spec_arg_nowater_unw = np.unwrap(spec_arg_nowater)

# --- Групповая задержка и коррекция ---
f_low_mask = (freq > 4.2e13) & (freq < 6.0e13)
dphi_water_df = np.gradient(spec_arg_water_unw, freq)
dphi_nowater_df = np.gradient(spec_arg_nowater_unw, freq)

tau_water = -np.mean(dphi_water_df[f_low_mask]) / (2 * np.pi)
tau_nowater = -np.mean(dphi_nowater_df[f_low_mask]) / (2 * np.pi)

# Убираем линейный наклон
phi_corr_water = spec_arg_water_unw + 2 * np.pi * freq * tau_water
phi_corr_nowater = spec_arg_nowater_unw + 2 * np.pi * freq * tau_nowater

# Убираем постоянную компоненту (центрируем в рабочем диапазоне)
mean_phi_w = np.mean(phi_corr_water[f_low_mask])
mean_phi_nw = np.mean(phi_corr_nowater[f_low_mask])
phi_final_water = phi_corr_water - mean_phi_w
phi_final_nowater = phi_corr_nowater - mean_phi_nw

# --- ПЕРВЫЙ НАБОР ГРАФИКОВ (ФАЗА) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(freq/1e12, spec_arg_water_unw, label='С водой', alpha=0.7)
axes[0,0].plot(freq/1e12, spec_arg_nowater_unw, label='Без воды', alpha=0.7)
axes[0,0].set_title('Исходная развернутая фаза (рад)')
axes[0,0].set_xlim(10, 80)
axes[0,0].grid(True)
axes[0,0].legend()

axes[0,1].scatter(freq/1e12, spec_arg_water, label='С водой', s=1)
axes[0,1].set_title('Исходная фаза (без развертки)')
axes[0,1].set_xlim(10, 80)
axes[0,1].grid(True)

axes[1,0].plot(freq/1e12, phi_final_water, label='С водой')
axes[1,0].plot(freq/1e12, phi_final_nowater, label='Без воды')
axes[1,0].set_title('Скорректированная фаза (без наклона и DC)')
axes[1,0].set_xlim(10, 80)
axes[1,0].grid(True)
axes[1,0].legend()

axes[1,1].plot(freq/1e12, phi_final_water - phi_final_nowater, label='Δφ')
axes[1,1].set_title('Разница фаз (рад)')
axes[1,1].set_xlim(10, 80)
axes[1,1].grid(True)
axes[1,1].legend()

plt.tight_layout()
plt.show()

# --- ВТОРОЙ НАБОР ГРАФИКОВ (ВОЗВРАЩЕН К ОРИГИНАЛУ) ---
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# График сигналов
axes2[0].plot(delay_water*1e12, signal_water, label='С водой', alpha=0.7)
axes2[0].plot(delay_nowater*1e12, signal_nowater, label='Без воды', alpha=0.7)
axes2[0].set_title('Сигналы после скользящего среднего и обрезки')
axes2[0].set_xlabel('Задержка (пс)')
axes2[0].set_ylabel('Амплитуда сигнала')
axes2[0].grid(True)
axes2[0].legend()

# График амплитуд спектров (линейный масштаб)
axes2[1].plot(freq/1e12, np.abs(spec_water_complex), label='С водой')
axes2[1].plot(freq/1e12, np.abs(spec_nowater_complex), label='Без воды')
axes2[1].set_title('Амплитуды спектров')
axes2[1].set_xlabel('Частота (ТГц)')
axes2[1].set_ylabel('Амплитуда')
axes2[1].grid(True)
# Исправлено: вместо thicklines используем linewidth
for line in axes2[1].get_lines():
    line.set_linewidth(1.0) 
axes2[1].legend()
axes2[1].set_xlim(10, 80)

plt.tight_layout()
plt.show()

print(f"Средняя разность фаз в диапазоне: {np.mean(phi_final_water[f_low_mask] - phi_final_nowater[f_low_mask]):.4f} рад")