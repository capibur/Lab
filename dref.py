import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
import hapi as hp
tau = 50e-15  #сдлительность импульса
lambda0 = 5e-6  # центральная длина волны
c = 299792458
w0 = 2 * np.pi * c / lambda0  # несущая частота
dt = 1e-15  #шаг по времени
T = 10e-12  # полуширина окна
t = np.arange(-T, T + dt, dt)

df = 1 / (2 * T)  #Гц шаг по частоте
F = 1 / (2 * dt)  #Гц макс. частота
f = np.arange(-F, F + df, df)

#xастотное окно
f1 = 5.8e13
f2 = 6.1e13
f_width = 0.05e13


temperature = 296.0  # K (23 °C)
pressure = 1.0  #атм, общее давление
relative_humidity = 24
# Давление насыщенного пара при 23°C ~ 0.0276 атм
p_sat = 0.0276  #атм
path_l = 300.0  #длина пути в среде, см
p_ref = 1.0
hp.db_begin('hitran_ata')
nu_min = 1500.0  # см^-1   ~ 6.67 мкм
nu_max = 2500.0  # см^-1   ~ 4.0 мкм

hp.fetch('H2O', 1, 1, nu_min, nu_max)

# Коэффициент поглощения k(nu) в см^-1 при заданных p, T
# В Environment задаём общее давление, в Diluent — состав газа
nu_hitran, k_ref = hp.absorptionCoefficient_Lorentz(
    SourceTables='H2O',
    Environment={'p': p_ref, 'T': temperature},
    Diluent={'air': 1.0, 'self': 0.0},
    WavenumberRange=[nu_min, nu_max],
    WavenumberStep=0.01,
    HITRAN_units=False
)
p_h2o = (relative_humidity / 100.0) * p_sat
k_hitran = k_ref *(p_h2o/ p_ref)
tr_h = np.exp(-k_hitran * path_l)
f_cm1 = f / c * 1e-2
interp_T = interp1d(
    nu_hitran,
    tr_h,
    kind='linear',
    bounds_error=False,
    fill_value=1.0  #вне дапазона считаем нет поглощения
)

# Пропускание на частотной сетке сигнала
T_on_f_grid = interp_T(f_cm1)

f_min = 5e13
f_max = 7e13
f_min_idx = np.argmin(np.abs(f - f_min)) #ищем самое близкое значение
f_max_idx = np.argmin(np.abs(f - f_max))


res_spec = []
pulse1 = np.exp(-t ** 2 / (2 * tau ** 2)) * np.sin(w0 * t)
spec = fftshift(fft(pulse1))
spec_intensity = np.abs(spec) ** 2
spec_slice = spec_intensity[f_min_idx:f_max_idx] * T_on_f_grid[f_min_idx:f_max_idx]
print(len(t), len(spec_intensity))
plt.figure(figsize=(8, 3))
plt.plot(f[f_min_idx:f_max_idx] , spec_slice)
plt.xlabel('Частота (ТГц)')
plt.ylabel('Интенсивность (с учётом поглощения)')
plt.title('Спектр импульсов (последняя задержка)')
plt.grid(True)
plt.tight_layout()
plt.show()