import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
import hapi as hp
import os

# ================== 0. ПАРАМЕТРЫ СИГНАЛА ==================
tau = 50e-15
lambda0 = 5e-6
c = 299792458
w0 = 2 * np.pi * c / lambda0

dt = 1e-15
T = 10e-12
t = np.arange(-T, T + dt, dt)

df = 1 / (2 * T)
F = 1 / (2 * dt)
f = np.arange(-F, F + df, df)

f_min = 5e13
f_max = 7e13
f_min_idx = np.argmin(np.abs(f - f_min))
f_max_idx = np.argmin(np.abs(f - f_max))

# ================== 1. ПАРАМЕТРЫ СРЕДЫ (кроме RH) ==================
temperature = 296.0
p_total = 1.0
p_sat = 0.0276   # насыщенный пар при 23C
path_l = 3000.0   # см
p_ref = 1.0
# ================== 2. HITRAN / HAPI (один раз) ==================
hp.db_begin('hitran_data')
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


# частотная сетка в см^-1 под сигнал
f_cm1 = f / c * 1e-2

# задержки для интерферограммы
delays = np.arange(0, 10000) * 1e-15

# ================== ФУНКЦИЯ: строим всё для заданной RH ==================
def compute_and_plot_for_RH(relative_humidity, frame_id):
    # --- парциальные давления ---
    p_h2o = (relative_humidity / 100.0) * p_sat
    k_hitran = k_ref *(p_h2o/ p_ref)

    tr_h = np.exp(-k_hitran * path_l)

    interp_T = interp1d(
        nu_hitran,
        tr_h,
        kind='linear',
        bounds_error=False,
        fill_value=1.0
    )
    T_on_f_grid = interp_T(f_cm1)

    # --- первая FFT: res_spec(delay, f) ---
    res_spec = []
    for delay in delays:
        pulse1 = np.exp(-t ** 2 / (2 * tau ** 2)) * np.sin(w0 * t)
        pulse2 = np.exp(-(t - delay) ** 2 / (2 * tau ** 2)) * np.sin(w0 * (t - delay))
        pulse = pulse1 + pulse2

        spec = fftshift(fft(pulse))
        spec_intensity = np.abs(spec) ** 2

        spec_slice = spec_intensity[f_min_idx:f_max_idx] * T_on_f_grid[f_min_idx:f_max_idx]
        res_spec.append(spec_slice)

    res_spec = np.array(res_spec)   # (len(delays), N_freq)

    # --- второе FFT по задержке ---
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

    res_spec2d = np.array(res_spec2d).T  # (freq_delay, freq)

    # --- рисунок для текущего RH ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1) 1D спектр для последней задержки
    axes[0].plot(f[f_min_idx:f_max_idx] * 1e-12, res_spec[-1])
    axes[0].set_xlabel('Частота (ТГц)')
    axes[0].set_ylabel('Интенсивность')
    axes[0].set_title(f'Спектр (последняя задержка)RH = {relative_humidity:.0f} %')
    axes[0].grid(True)

    # 2) 2D FFT по задержке
    im = axes[1].imshow(
        res_spec2d,
        extent=[f[f_min_idx] * 1e-12,
                f[f_max_idx] * 1e-12,
                f_delay[f_min_delay_idx] * 1e-12,
                f_delay[f_max_delay_idx] * 1e-12],
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )
    axes[1].set_xlabel('Частота (ТГц)')
    axes[1].set_ylabel('Частота по задержке (ТГц)')
    axes[1].set_title(f'2D FFT по задержкеи RH = {relative_humidity:.0f} %')
    fig.colorbar(im, ax=axes[1], label='Интенсивность')

    plt.tight_layout()

    # сохраняем кадр
    out_dir = "frames_rh"
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"frW_{frame_id:03d}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)


# ================== ГЛАВНЫЙ ЦИКЛ ПО RH ==================
frame_id = 0
for RH in range(60, 62, 1):   # 0..100 с шагом 1
    print(f"Computing frame {frame_id}, RH = {RH} %")
    compute_and_plot_for_RH(RH, frame_id)
    frame_id += 1

print("Готово. Теперь собери кадры в GIF/видео, например:")
print("  ffmpeg -framerate 10 -i frames_rh/frame_%03d.png -pix_fmt yuv420p rh_animation.mp4")
# или:
#  magick convert -delay 5 -loop 0 frames_rh/frame_*.p