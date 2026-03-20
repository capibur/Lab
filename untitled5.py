import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, rfft, rfftfreq
from scipy.interpolate import interp1d

import hapi as hp

c = 299792458  # скорость света, м/с
tau = 50e-15  # длительность импульса, с
lambda0 = 6.2e-6  # центральная длина волны, м
w0 = 2 * np.pi * c / lambda0
lambda_phases = 6.23e-7
p_sat = 0.0276
filename = 0

w_size = 100
data = np.loadtxt(filename)

phases = data[:, 0]
signal = data[:, 1]

# задержка в секундах
delay = phases / (2 * np.pi) * lambda_phases / c

idx = np.argsort(delay)
delay = delay[idx]
signal = signal[idx]

# обрезка по задержке
delay_min = -1e-11
delay_max = 1e-11
mask = (delay >= delay_min) & (delay <= delay_max)
delay = delay[mask]
signal = signal[mask]
print(f"Обрезано с {len(data)} до {len(delay)} точек")

# усреднение фона
ker = np.ones(w_size) / w_size
background = np.convolve(signal, ker, mode='same')
signal = signal - background

plt.figure()
plt.plot(delay, signal)
plt.xlabel("Delay (s)")
plt.ylabel("Signal")
plt.title("Signal after moving average subtraction")
plt.grid()
plt.show()

# эксперимент: равномерная временная сетка
N = len(delay)
dt = (delay.max() - delay.min()) / (N - 1)  # шаг по времени
delay_uniform = np.linspace(delay.min(), delay.max(), N)

interp = interp1d(delay, signal, kind='linear', bounds_error=False, fill_value=0.0)
signal_uniform = interp(delay_uniform)

# окно Ханна
window = np.hanning(N)
signal_uniform *= window

# спектр эксперимента (положительные частоты соответствуют freq)
spec_exp = rfft(signal_uniform)
freq = rfftfreq(N, dt)
spectrum = np.abs(spec_exp)

# хотим, чтобы T было ≈ 2T_exp, dt_model = dt (та же дискретизация)
T = 10e-12  # полуширина окна, с
dt_model = dt  # шаг по времени = шаг эксперимента
t_model = np.arange(-T, T + dt_model, dt_model)
N_model = len(t_model)

# импульс
pulse1 = np.exp(-t_model ** 2 / (2 * tau ** 2)) * np.sin(w0 * t_model)

# спектр (полный FFT, затем rfft‑аналог по положительным частотам)
spec_model_full = fft(pulse1)
spec_model_intensity = np.abs(spec_model_full) ** 2

# положительные частоты для модели
df_model = 1 / (2 * T)
f_model = np.fft.fftfreq(N_model, dt_model)
f_model = np.fft.fftshift(f_model)[N_model // 2:]

temperature = 296.0
pressure = 1.0
relative_humidity = 50
p_sat = 0.0276
path_l = 150.0
p_ref = 1.0

hp.db_begin('hitran_ata')
nu_min = 1500.0
nu_max = 2500.0

hp.fetch('H2O', 1, 1, nu_min, nu_max)

nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
    SourceTables='H2O',
    Environment={'p': p_ref, 'T': temperature},
    Diluent={'air': 1.0, 'self': 0.0},
    WavenumberRange=[nu_min, nu_max],
    WavenumberStep=0.01,
    HITRAN_units=False
)

p_h2o = (relative_humidity / 100.0) * p_sat
k_hitran = k_ref * (p_h2o / p_ref)
tr_h = np.exp(-k_hitran * path_l)

# частота в Гц → волновое число в см⁻¹
f_cm1 = f_model / c * 1e2

interp_T = interp1d(
    nu_hitran,
    tr_h,
    kind='linear',
    bounds_error=False,
    fill_value=1.0
)
T_on_f_model = interp_T(f_cm1)

# выбранный диапазон частот
f_min = 5e13
f_max = 7e13
mask_model = (f_model >= f_min) & (f_model <= f_max)
f_min_idx = np.argmax(f_model >= f_min)  # первый индекс >= f_min
f_max_idx = np.argmax(f_model >= f_max)  # первый индекс >= f_max
spec_slice = spec_model_intensity[f_min_idx:f_max_idx]
T_on_slice = T_on_f_model[f_min_idx:f_max_idx]
spec_with_abs = spec_slice * T_on_slice

# теоретический спектр на частотной сетке f_model (в диапазоне [f_min, f_max])
f_interp = f_model[f_min_idx:f_max_idx]
y_interp = spec_with_abs

# интерполяция на сетку freq эксперимента
mask_freq = (freq >= f_min) & (freq <= f_max)
freq_small = freq[mask_freq]

# интерполяция теоретического спектра на freq_small
interp_model = interp1d(
    f_interp,
    y_interp / np.max(y_interp),  # нормируем
    kind='linear',
    bounds_error=False,
    fill_value=0.0
)

# теоретический спектр на сетке эксперимента
spec_model_on_freq = interp_model(freq_small)

# нормируем экспериментальный спектр
spectrum_norm = spectrum[mask_freq] / np.max(spectrum[mask_freq])

plt.figure(figsize=(10, 4))
plt.plot(freq[mask_freq], spectrum_norm, label="Экспериментальный спектр")
plt.plot(freq_small, spec_model_on_freq, label="Теоретический спектр (с поглощением)", ls="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Norm. Amplitude")
plt.xlim(0.45e14, 0.65e14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
