
#2. Построить фурье-спектр для импульса с предымпульсом.
#  Задать примерно правильную задержку, скажем 20 пс, и какую-то амплитуду, скажем 10% от основного импульса.
#  Сравнить 3 спектра: 
#  идеальный (без предымпульса),
#  плохой (с предымпульсом, диапазон сканирования больше чем задержка между ними), 
#  реалистичный (с предымпульсом, но диапазон сканирования меньше, чем задержка между импульсами).


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, rfft, rfftfreq
from scipy.signal import medfilt
import hapi as hp


CONST_C = 299792458.0
CONST_TAU = 50e-15
CONST_LAMBDA0 = 6.3e-7
CONST_W0 = 2 * np.pi * CONST_C / CONST_LAMBDA0
delay = 20e-12  # 20 ps
amp_pre = 0.1  # 10% от основного
amp_main = 1.0
sigma = CONST_TAU / (2 * np.sqrt(2 * np.log(2)))  
omega0 = CONST_W0

# Временная сетка
dt = 1e-15  # 1 fs
t_full = np.arange(-100e-12, 100e-12, dt) 
t_real = np.arange(-5e-12, 5e-12, dt)  

# Функция для генерации импульса
def generate_pulse(t, t0, amp, sigma, omega0):
    envelope = amp * np.exp(- (t - t0)**2 / (2 * sigma**2))
    carrier = np.cos(omega0 * (t - t0))
    return envelope * carrier

# Идеальный
signal_ideal = generate_pulse(t_full, 0, amp_main, sigma, omega0)

# Плохой
signal_bad = generate_pulse(t_full, 0, amp_main, sigma, omega0) + generate_pulse(t_full, -delay, amp_pre, sigma, omega0)

# Реалистичный
signal_real = generate_pulse(t_real, 0, amp_main, sigma, omega0) + generate_pulse(t_real, -delay, amp_pre, sigma, omega0)

# FFT
def compute_spectrum(signal, dt):
    N = len(signal)
    df = 1 /  (2 * N * dt)  
    freq = np.arange(N // 2 ) * df  
    spectrum = rfft(signal)
    return freq, np.abs(spectrum)**2

freq_full, spec_ideal = compute_spectrum(signal_ideal, dt)
_, spec_bad = compute_spectrum(signal_bad, dt)
freq_real, spec_real = compute_spectrum(signal_real, dt)

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(t_full * 1e12, signal_ideal)
plt.title('Идеальный сигнал (время)')
plt.xlabel('Время (ps)')
plt.ylabel('Амплитуда')

plt.subplot(2, 3, 2)
plt.plot(t_full * 1e12, signal_bad)
plt.title('Плохой сигнал (время)')
plt.xlabel('Время (ps)')
plt.ylabel('Амплитуда')

plt.subplot(2, 3, 3)
plt.plot(t_real * 1e12, signal_real)
plt.title('Реалистичный сигнал (время)')
plt.xlabel('Время (ps)')
plt.ylabel('Амплитуда')

plt.subplot(2, 3, 4)
plt.plot(freq_full / 1e12, spec_ideal)
plt.title('Идеальный спектр')
plt.xlabel('Частота (THz)')
plt.ylabel('Интенсивность')

plt.subplot(2, 3, 5)
plt.plot(freq_full / 1e12, spec_bad)
plt.title('Плохой спектр')
plt.xlabel('Частота (THz)')
plt.ylabel('Интенсивность')

plt.subplot(2, 3, 6)
plt.plot(freq_real / 1e12, spec_real)
plt.title('Реалистичный спектр')
plt.xlabel('Частота (THz)')
plt.ylabel('Интенсивность')

plt.tight_layout()
plt.show()