import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
import hapi as hp
import os


c = 299792458  # м/с
tau = 50e-15
lambda0 = 5e-6
w0 = 2 * np.pi * c / lambda0

data = np.loadtxt('POS_SCAN_19.47.54.txt')
phases = data[:, 0]  # первый столбец - фазы [рад]
lambda_phases = 6.23e-7  # м (из твоего описания)
t = phases / (2 * np.pi) * lambda_phases / c  # t = φ/(2π) × λ/c

# сортируем по времени
sort_idx = np.argsort(t)
t = t[sort_idx]
if data.shape[1] > 1:
    amplitude = data[sort_idx, 1]
else:
    amplitude = np.ones_like(t)

dt_weights = np.zeros_like(t)
dt_weights[1:-1] = 0.5 * (t[2:] - t[:-2])
dt_weights[0] = t[1] - t[0]
dt_weights[-1] = t[-1] - t[-2]

t_mask = np.abs(t) < 9 * tau
t_cropped = t[t_mask]
dt_weights_crop = dt_weights[t_mask]
delays = t_cropped.copy()
T_total = t.max() - t.min()
df = 1 / T_total
F = 1 / (2 * np.mean(np.diff(t)))
f = np.arange(-F, F, df)

f_min = 5e13
f_max = 7e13
f_slice = f[(f >= f_min) & (f <= f_max)]
omega_slice = 2 * np.pi * f_slice
print(f"Частоты: {len(f_slice)} точек ({f_min*1e-12:.1f}-{f_max*1e-12:.1f} ТГц)")

# ================== 5. ПАРАМЕТРЫ СРЕДЫ ==================
temperature = 296.0
p_total = 1.0
p_sat = 0.0276
path_l = 300.0
p_ref = 1.0

# ================== 6. HITRAN / HAPI ==================
hp.db_begin('hitra_data')
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



def nonuniform_fourier_integral_fast(t_crop, f_t_crop, omega_crop, dt_weights_crop):
    phase = np.exp(-1j * np.outer(omega_crop, t_crop))
    return phase @ (f_t_crop * dt_weights_crop)

def compute_and_plot_for_RH(relative_humidity, frame_id):
    p_h2o = (relative_humidity / 100.0) * p_sat
    k_hitran = k_ref * (p_h2o / p_ref)
    tr_h = np.exp(-k_hitran * path_l)

    interp_T = interp1d(nu_hitran, tr_h, kind='linear', bounds_error=False, fill_value=1.0)
    T_on_f_slice = interp_T(f_slice / c * 1e-2)

    res_spec = []
    for delay in delays:
        # ИМПУЛЬСЫ на реальных задержках из txt!
        pulse1_crop = amplitude[t_mask] * np.exp(-t_cropped ** 2 / (2 * tau ** 2)) * np.sin(w0 * t_cropped)
        pulse2_crop = amplitude[t_mask] * np.exp(-(t_cropped - delay) ** 2 / (2 * tau ** 2)) * np.sin(w0 * (t_cropped - delay))
        pulse_crop = pulse1_crop + pulse2_crop

        # ИНТЕГРАЛЬНЫЙ ФУРЬЕ
        spec_integral = nonuniform_fourier_integral_fast(t_cropped, pulse_crop, omega_slice, dt_weights_crop)
        spec_intensity = np.abs(spec_integral) ** 2 / len(t_cropped)

        spec_slice = spec_intensity * T_on_f_slice
        res_spec.append(spec_slice)

    res_spec = np.array(res_spec)

    df_delay = 1.0 / (delays[-1] - delays[0])
    F_delay = 1.0 / (2 * (delays[1] - delays[0]))
    f_delay = np.arange(-F_delay, F_delay, df_delay)
    f_min_delay_idx = np.argmin(np.abs(f_delay - 5e13))
    f_max_delay_idx = np.argmin(np.abs(f_delay - 7e13))

    res_spec2d = []
    for k in range(res_spec.shape[1]):
        spec_delay = fftshift(fft(res_spec[:, k]))
        spec_delay_slice = np.abs(spec_delay[f_min_delay_idx:f_max_delay_idx])
        res_spec2d.append(spec_delay_slice)
    res_spec2d = np.array(res_spec2d).T

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1D спектр
    axes[0].plot(f_slice * 1e-12, res_spec[-1])
    axes[0].set_title(f'Спектр (RH={relative_humidity}%)')
    axes[0].set_xlabel('Частота (ТГц)')
    axes[0].grid(True)
    
    # 2D FFT
    im = axes[1].imshow(res_spec2d, extent=[f_slice[0]*1e-12, f_slice[-1]*1e-12,
                                            f_delay[f_min_delay_idx]*1e-12, f_delay[f_max_delay_idx]*1e-12],
                        aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title('2D FFT по задержке')
    axes[1].set_xlabel('Частота (ТГц)')
    fig.colorbar(im, ax=axes[1])
    
    
    plt.tight_layout()
    out_dir = "frames_rh_txt"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"fr_{frame_id:03d}.png"), dpi=100)
    plt.close(fig)


frame_id = 0
for RH in range(20,21 , 1):
    print(f"RH = {RH}%, сетка из txt")
    compute_and_plot_for_RH(RH, frame_id)
    frame_id += 1


phases = data[:, 0]  # 1-й столбец - фазы
second_col = data[:, 1]  # 2-й столбец (сигнал)

# ЗАДЕРЖКИ из фаз
lambda_phases = 6.23e-7
c = 299792458
t = phases / (2 * np.pi) * lambda_phases / c
sort_idx = np.argsort(t)
t = t[sort_idx]
signal = second_col[sort_idx]

dt_weights = np.zeros_like(t)
dt_weights[1:-1] = 0.5 * (t[2:] - t[:-2])
dt_weights[0] = t[1] - t[0]
dt_weights[-1] = t[-1] - t[-2]

f_min, f_max = 45e12, 55e12  
f_target = np.linspace(f_min, f_max, 2000)
omega_target = 2 * np.pi * f_target
phase = np.exp(-1j * np.outer(omega_target, t))
spectrum = phase @ (signal * dt_weights)
intensity = np.abs(spectrum)**2 / len(t)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t*1e15, signal, 'b-o', markersize=3)
axes[0].set_xlabel('Задержка t (fs)')
axes[0].set_ylabel('Сигнал (2-й столбец)')
axes[0].set_title('Сигнал по неравномерной сетке')
axes[0].grid(True)
axes[1].plot(f_target*1e-12, intensity, 'r-', linewidth=1.5)
axes[1].set_xlabel('Частота f (ТГц)')
axes[1].set_ylabel('Интенсивность')
axes[1].set_title('Фурье-спектр')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('signal_and_spectrum.png', dpi=150)
plt.show()

print(f"Спектр готов! {len(t)} точек, f: {f_min*1e-12:.1f}-{f_max*1e-12:.1f} ТГц")
