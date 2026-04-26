import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift
from scipy.signal import hilbert
import task_two.hapi as hp

# ---------------- Блок: загрузка и подготовка экспериментальных данных ----------------
c = 299792458
tau = 50e-15
lambda0 = 5e-6
w0 = 2 * np.pi * c / lambda0
lambda_phases = 6.23e-7
p_sat = 0.0276
filename = "POS_SCAN_19.23.34.txt"

w_size = 100
data = np.loadtxt(filename)

phases = data[:, 0]
signal = data[:, 1]

delay = phases / (2 * np.pi) * lambda_phases / c  # время в сек

# сортировка по времени
idx = np.argsort(delay)
delay = delay[idx]
signal = signal[idx]

# окно по задержкам
delay_min = -1e-11
delay_max = 1e-11
mask = (delay >= delay_min) & (delay <= delay_max)
delay = delay[mask]
signal = signal[mask]
print(f"Обрезано с {len(data)} до {len(delay)} точек")

# вычитание фона (скользящее среднее)
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

# ---------------- Эксперимент: равномерная сетка и FFT ----------------
N = len(delay)
delay_uniform = np.linspace(delay.min(), delay.max(), N)
interp_sig = interp1d(delay, signal, kind='linear', fill_value="extrapolate")
signal_uniform = interp_sig(delay_uniform)

dt = delay_uniform[1] - delay_uniform[0]           # шаг по времени
freq = np.fft.rfftfreq(N, dt)                      # частотная сетка эксперимента [Гц] [web:8][web:14]
spec_exp = np.fft.rfft(signal_uniform)
spectrum = np.abs(spec_exp)

plt.figure()
plt.plot(delay_uniform, signal_uniform)
plt.xlim(delay_min, delay_max)
plt.xlabel("Delay (s)")
plt.ylabel("Signal (uniform)")
plt.grid()
plt.show()

# ---------------- Теоретический импульс на той же временной сетке ----------------
# важно: используем ТОТ ЖЕ dt и ТО ЖЕ N, что и для эксперимента
t_theory = np.arange(-N//2, N//2) * dt  # центрируем вокруг нуля
pulse = np.exp(-t_theory**2 / (2 * tau**2)) * np.sin(w0 * t_theory)

# FFT + сдвиг, затем переведём на ту же rfft-сетку freq
spec_theory_full = fftshift(fft(pulse))
# частоты для полного FFT (как справочная сетка)
freq_full = np.fft.fftshift(np.fft.fftfreq(N, d=dt))  # Гц [web:13]

# модуль спектра теории на freq_full
spec_theory_full_abs = np.abs(spec_theory_full)

# ---------------- Приведение теоретического спектра к сетке freq ----------------
# частоты freq_full могут быть и отрицательные, а freq >= 0, поэтому интерполируем только по |f|
freq_full_pos = freq_full[freq_full >= 0]
spec_theory_pos = spec_theory_full_abs[freq_full >= 0]

interp_theory_on_exp = interp1d(freq_full_pos,
                                spec_theory_pos,
                                kind='linear',
                                bounds_error=False,
                                fill_value=0.0)

spec_theory_on_freq = interp_theory_on_exp(freq)  # теоретический спектр на сетке freq (Гц)

# ---------------- HITRAN/HAPI: поглощение и перенос на freq ----------------
temperature = 296.0  # K
pressure = 1.0       # атм (общее)
relative_humidity = 50.0
p_ref = 1.0
path_l = 150.0       # см

hp.db_begin('hitran_ata')
nu_min = 1500.0  # см^-1   ~ 6.67 мкм
nu_max = 2500.0  # см^-1   ~ 4.0 мкм

hp.fetch('H2O', 1, 1, nu_min, nu_max)

nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
    SourceTables='H2O',
    Environment={'p': p_ref, 'T': temperature},
    Diluent={'air': 1.0, 'self': 0.0},
    WavenumberRange=[nu_min, nu_max],
    WavenumberStep=0.01,
    HITRAN_units=False
)  # nu_hitran в см^-1, k_ref в см^-1 [web:9][web:12]

p_h2o = (relative_humidity / 100.0) * p_sat
k_hitran = k_ref * (p_h2o / p_ref)
tr_h = np.exp(-k_hitran * path_l)  # пропускание

# Перевод freq (Гц) -> волновое число (см^-1): nu = f / c [м^-1] * 1e-2 -> [см^-1]
f_cm1 = freq / c * 1e-2

interp_T = interp1d(
    nu_hitran,
    tr_h,
    kind='linear',
    bounds_error=False,
    fill_value=1.0
)

T_on_freq = interp_T(f_cm1)  # пропускание на той же сетке freq

spec_theory_abs = spec_theory_on_freq * T_on_freq

# ---------------- Ограничение по частоте + нормировка -------к---------
f_min = 5e13
f_max = 7e13

mask_band = (freq >= f_min) & (freq <= f_max)

freq_band = freq[mask_band]
spec_exp_band = spectrum[mask_band]
spec_theory_band = spec_theory_abs[mask_band]

# нормируем по максимуму в этом диапазоне, чтобы удобно сравнивать
spec_exp_band_norm = spec_exp_band / np.max(spec_exp_band) if np.max(spec_exp_band) != 0 else spec_exp_band
spec_theory_band_norm = spec_theory_band / np.max(spec_theory_band) if np.max(spec_theory_band) != 0 else spec_theory_band

# ---------------- Графики: одна частотная сетка freq ----------------
plt.figure(figsize=(8, 4))
plt.plot(freq_band, spec_exp_band_norm, label="Эксперимент (FFT)", lw=1.5)
plt.plot(freq_band, spec_theory_band_norm, label="Теория + H2O", lw=1.5)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized amplitude")
plt.xlim(0.45e14, 0.65e14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
