import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
import task_two.hapi as hp
import os


tau = 50e-15
lambda0 = 5e-6
c = 299792458
w0 = 2 * np.pi * c / lambda0

dt_base = 1e-15
T = 10e-12

# *** НОВОЕ: НЕРАВНОМЕРНАЯ СЕТКА ***
t_base = np.arange(-T, T + dt_base, dt_base)  # базовая равномерная
jitter_std = 0.1 * dt_base  # разброс ±10% от dt
jitter = np.random.normal(0, jitter_std, len(t_base))
t = np.sort(t_base + jitter)  # НЕРАВНОМЕРНАЯ! отсортирована

print(f"Сетка: {len(t)} точек, dt среднее={np.mean(np.diff(t) * 1e15):.2f} fs")

# веса для прямого интеграла (трапеции)
dt_weights = np.zeros_like(t)
dt_weights[1:-1] = 0.5 * (t[2:] - t[:-2])
dt_weights[0] = t[1] - t[0]
dt_weights[-1] = t[-1] - t[-2]

# остальное без изменений
df = 1 / (2 * T)
F = 1 / (2 * dt_base)
f = np.arange(-F, F + df, df)

f_min = 5e13
f_max = 7e13
f_min_idx = np.argmin(np.abs(f - f_min))
f_max_idx = np.argmin(np.abs(f - f_max))


temperature = 296.0
p_total = 1.0
p_sat = 0.0276
path_l = 300.0
p_ref = 1.0


hp.db_begin('3')
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

f_cm1 = f / c * 1e-2
delays = np.arange(0, 10000) * 1e-15



t_mask = np.abs(t) < 9 * tau  # ±9×tau — 99.9% энергии гаусса
t_cropped = t[t_mask]
print(f"Обрезана сетка: {len(t)} → {len(t_cropped)} точек ({len(t_cropped) / len(t) * 100:.1f}%)")

# ЧАСТОТНЫЙ интервал (только 5-7 ТГц)
f_slice = f[(f >= f_min) & (f <= f_max)]
omega_slice = 2 * np.pi * f_slice  # только нужные частоты!
print(f"Частоты: {len(f)} → {len(f_slice)}")

dt_weights_crop = np.zeros_like(t_cropped)
dt_weights_crop[1:-1] = 0.5 * (t_cropped[2:] - t_cropped[:-2])
dt_weights_crop[0] = t_cropped[1] - t_cropped[0]
dt_weights_crop[-1] = t_cropped[-1] - t_cropped[-2]


f_slice_idx = np.where((f >= f_min) & (f <= f_max))[0]



def nonuniform_fourier_integral_fast(t_crop, f_t_crop, omega_crop, dt_weights_crop):
    """ОБРЕЗАННАЯ версия — в 10 раз быстрее!"""
    phase = np.exp(-1j * np.outer(omega_crop, t_crop))  # теперь маленький!
    return phase @ (f_t_crop * dt_weights_crop)



def compute_and_plot_for_RH(relative_humidity, frame_id):
    p_h2o = (relative_humidity / 100.0) * p_sat
    k_hitran = k_ref * (p_h2o / p_ref)
    tr_h = np.exp(-k_hitran * path_l)

    interp_T = interp1d(nu_hitran, tr_h, kind='linear', bounds_error=False, fill_value=1.0)
    T_on_f_slice = interp_T(f_slice / c * 1e-2)  #

    res_spec = []
    for delay in delays:
        # *** ИМПУЛЬСЫ ТОЛЬКО НА ОБРЕЗАННОЙ СЕТКЕ ***
        pulse1_crop = np.exp(-t_cropped ** 2 / (2 * tau ** 2)) * np.sin(w0 * t_cropped)
        pulse2_crop = np.exp(-(t_cropped - delay) ** 2 / (2 * tau ** 2)) * np.sin(w0 * (t_cropped - delay))
        pulse_crop = pulse1_crop + pulse2_crop

        # *** ИНТЕГРАЛ НА СРЕЗЕ ЧАСТОТ ***
        spec_integral = nonuniform_fourier_integral_fast(t_cropped, pulse_crop, omega_slice, dt_weights_crop)
        spec_intensity_integral = np.abs(spec_integral) ** 2 / len(t_cropped)

        # для совместимости с кодом (но теперь только срез!)
        spec_slice = spec_intensity_integral * T_on_f_slice
        res_spec.append(spec_slice)

    res_spec = np.array(res_spec)  # теперь (delays, 4000) вместо (delays, 20000)!

    # второе FFT без изменений (по задержке)
    df_delay = 1.0 / (delays[-1] - delays[0] + (delays[1] - delays[0]))
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

    # график
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(f_slice * 1e-12, res_spec[-1])
    axes[0].set_title(f'Спектр (RH={relative_humidity}%)')
    axes[0].set_xlabel('Частота (ТГц)')

    im = axes[1].imshow(res_spec2d, extent=[f_slice[0] * 1e-12, f_slice[-1] * 1e-12,
                                            f_delay[f_min_delay_idx] * 1e-12, f_delay[f_max_delay_idx] * 1e-12],
                        aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title('2D FFT по задержке')
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out_dir = "frames_rh_fast"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"fr_{frame_id:03d}.png"), dpi=100)
    plt.close(fig)



frame_id = 0
for RH in range(0, 50,4 ):
    print(f"RH = {RH}%, неравномерная сетка")
    compute_and_plot_for_RH(RH, frame_id)
    frame_id += 1

print("Готово! ffmpeg -framerate 10 -i frames_rh_nonuniform/fr_*.png animation.mp4")
