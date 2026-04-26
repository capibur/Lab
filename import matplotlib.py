import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, find_peaks

# Загрузка данных
print("Загрузка данных...")
data = np.loadtxt(r'c:\Users\Андрей\Desktop\data_2\01_step8_speed10_CaseOpen_FiltersON.txt')
sin_sig = data[:, 0]
cos_sig = data[:, 1]
ref_sig = data[:, 2]

time = np.arange(len(sin_sig)) / 1.25e6

# ✅ ИСПРАВЛЕНИЕ: Улучшенный поиск частоты лазера
print("Поиск частоты лазера...")
N = len(sin_sig)

# 1. Убираем тренд и среднее
sin_detrended = detrend(sin_sig, type='linear')

# 2. Вычисляем FFT
yf = fft(sin_detrended)
xf = fftfreq(N, 1/1.25e6)[:N//2]
power = np.abs(yf[:N//2])**2

# 3. Ищем ПИК НЕ в нуле! Игнорируем первые 5 точек (DC и низкие частоты)
mask = xf > 1.0  # Частоты > 1 Гц
xf_clean = xf[mask]
power_clean = power[mask]

# 4. Находим максимум, но проверяем, что не 0
peak_idx = np.argmax(power_clean)
laser_freq = xf_clean[peak_idx]

# Если частота слишком низкая, берем следующий пик
if laser_freq < 10:  # порог, например 10 Гц
    power_clean[peak_idx] = 0  # убираем этот пик
    peak_idx = np.argmax(power_clean)
    laser_freq = xf_clean[peak_idx]
    print(f"Исправлено: частота была слишком низкой, взята следующая: {laser_freq:.2f} Hz")

print(f"Частота лазера: {laser_freq:.2f} Hz")

# Диагностика - показываем топ-3 пика
top_indices = np.argsort(power_clean)[-3:][::-1]
print("Топ-3 частоты:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}: {xf_clean[idx]:.2f} Hz")

# Показываем спектр сразу после расчета
plt.figure(figsize=(10, 6))
plt.semilogy(xf[:5000], power[:5000], label='Спектр sin-сигнала')
plt.axvline(laser_freq, color='red', lw=2, label=f'f={laser_freq:.1f} Hz')
plt.xlim(0, 5000)
plt.xlabel('Частота, Гц')
plt.ylabel('Мощность')
plt.title('Спектр sin-сигнала')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Параметры разбиения
bin_size = 1024
overlap = 0.5
step = int(bin_size * (1 - overlap))

def sine_fit(t, A, f, phi, offset):
    return A * np.sin(2 * np.pi * f * t + phi) + offset

phases = []
bin_centers = []

print("Аппроксимация фаз...")
for i in range(0, len(ref_sig) - bin_size, step):
    bin_data = detrend(ref_sig[i:i+bin_size], type='linear')
    t_bin = time[i:i+bin_size] - time[i]
    
    p0 = [np.std(bin_data), laser_freq, 0, np.mean(bin_data)]
    
    try:
        popt, _ = curve_fit(sine_fit, t_bin, bin_data, p0=p0, maxfev=5000)
        phases.append(popt[2])
        bin_centers.append(i // step)
    except:
        continue

# Нормализация фаз
if len(phases) > 1:
    phases = np.unwrap(phases)
    phases = (phases + np.pi) % (2*np.pi) - np.pi

# Графики
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Сигналы
t_short = time[:10000]*1e3
ax1.plot(t_short, sin_sig[:10000], label='Sin', alpha=0.8)
ax1.plot(t_short, cos_sig[:10000], label='Cos', alpha=0.8)
ax1.plot(t_short, ref_sig[:10000], label='Ref', alpha=0.8)
ax1.legend()
ax1.set_title('Сигналы')
ax1.grid(True, alpha=0.3)

# 2. Спектр с пиками
ax2.semilogy(xf[:5000], power[:5000], label='Спектр')
ax2.axvline(laser_freq, color='red', lw=2, label=f'f={laser_freq:.1f} Hz')
ax2.set_xlim(0, 5000)
ax2.set_title('Спектр sin-сигнала')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Фазы
ax3.plot(bin_centers[:200], phases[:200], '.-')
ax3.set_xlabel('Номер бина')
ax3.set_ylabel('Фаза, рад')
ax3.set_title('Фазовый сдвиг')
ax3.grid(True, alpha=0.3)

# 4. Спектр ref для сравнения
yf_ref = fft(detrend(ref_sig))
power_ref = np.abs(yf_ref[:N//2])**2
ax4.semilogy(xf[:5000], power_ref[:5000], label='Спектр Ref')
ax4.axvline(laser_freq, color='red', ls='--', label=f'f={laser_freq:.1f} Hz')
ax4.set_xlim(0, 5000)
ax4.set_title('Спектр ref-сигнала')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_analysis_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# Отдельный график спектра sin-сигнала
plt.figure(figsize=(10, 6))
plt.semilogy(xf[:5000], power[:5000], label='Спектр sin-сигнала')
plt.axvline(laser_freq, color='red', lw=2, label=f'f={laser_freq:.1f} Hz')
plt.xlim(0, 5000)
plt.xlabel('Частота, Гц')
plt.ylabel('Мощность')
plt.title('Спектр sin-сигнала')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Сохранение
np.savetxt('phase_results.txt', np.column_stack([bin_centers, phases]), 
           header='bin phase_rad', fmt='%d %.6e')

print(f"Готово! Бинов: {len(phases)}, f_лазер: {laser_freq:.2f} Hz")