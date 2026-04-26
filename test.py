import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import task_two.hapi as hp
import os


c = 299792458
tau = 50e-15
lambda0 = 5e-6
w0 = 2 * np.pi * c / lambda0
lambda_phases = 6.23e-7
p_sat = 0.0276
data = np.loadtxt('POS_SCAN_19.47.54.txt')
phases = data[:, 0]
amplitude = data[:, 1] if data.shape[1] > 1 else np.ones_like(phases)

# ЗАДЕРЖКИ ИЗ ФАЗ
t = phases / (2 * np.pi) * lambda_phases / c
sort_idx = np.argsort(t)
t = t[sort_idx]
amplitude = amplitude[sort_idx]

# ВЕСА ТРАПЕЦИЙ
dt_weights = np.zeros_like(t)
dt_weights[1:-1] = 0.5 * (t[2:] - t[:-2])
dt_weights[0] = t[1] - t[0]
dt_weights[-1] = t[-1] - t[-2]

# ОБРЕЗКА
t_mask = np.abs(t) < 9 * tau
t_cropped = t[t_mask]
dt_weights_crop = dt_weights[t_mask]
amplitude_crop = amplitude[t_mask]
print(f"Сетка: {len(t_cropped)} точек")

# delays = РЕАЛЬНЫЕ ЗАДЕРЖКИ
delays = t_cropped.copy()

# ЧАСТОТНАЯ СЕТКА
f_min, f_max = 5e13, 7e13
f_slice = np.linspace(f_min, f_max, 2000)
omega_slice = 2 * np.pi * f_slice

# ================== HAPI ==================
hp.db_begin('hitran_dat')  # исправил hitra_data → hitran_data
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

# ФУНКЦИЯ ФУРЬЕ
def nonuniform_fourier(t_crop, f_t_crop, omega_crop, dt_weights_crop):
    phase = np.exp(-1j * np.outer(omega_crop, t_crop))
    return phase @ (f_t_crop * dt_weights_crop)

# ================== ОСНОВНАЯ ФУНКЦИЯ ==================
def compute_and_plot_for_RH(relative_humidity, frame_id):
    p_h2o = (relative_humidity / 100.0) * p_sat
    k_hitran = k_ref * (p_h2o / 1.0)
    tr_h = np.exp(-k_hitran * 300.0)

    interp_T = interp1d(nu_hitran, tr_h, kind='linear', bounds_error=False, fill_value=1.0)
    T_on_f_slice = interp_T(f_slice / c * 1e-2)

    res_spec = []
    for delay in delays:
        pulse1_crop = amplitude_crop * np.exp(-t_cropped ** 2 / (2 * tau ** 2)) * np.sin(w0 * t_cropped)
        pulse2_crop = amplitude_crop * np.exp(-(t_cropped - delay) ** 2 / (2 * tau ** 2)) * np.sin(w0 * (t_cropped - delay))
        pulse_crop = pulse1_crop + pulse2_crop

        spec_integral = nonuniform_fourier(t_cropped, pulse_crop, omega_slice, dt_weights_crop)
        spec_intensity = np.abs(spec_integral) ** 2 / len(t_cropped)
        spec_slice = spec_intensity * T_on_f_slice
        res_spec.append(spec_slice)

    res_spec = np.array(res_spec)

    # 2D FFT
    df_delay = 1.0 / (delays[-1] - delays[0])
    F_delay = 1.0 / (2 * np.mean(np.diff(delays)))
    f_delay = np.linspace(-F_delay, F_delay, len(delays))
    f_min_delay_idx = np.argmin(np.abs(f_delay - 5e13))
    f_max_delay_idx = np.argmin(np.abs(f_delay - 7e13))

    res_spec2d = []
    for k in range(res_spec.shape[1]):
        spec_delay = fftshift(fft(res_spec[:, k]))
        if f_max_delay_idx < len(spec_delay):
            spec_delay_slice = np.abs(spec_delay[f_min_delay_idx:f_max_delay_idx])
        else:
            spec_delay_slice = np.abs(spec_delay[f_min_delay_idx:])
        res_spec2d.append(spec_delay_slice)
    res_spec2d = np.array(res_spec2d).T

    # ГРАФИКИ
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].plot(f_slice * 1e-12, res_spec[-1])
    axes[0].set_title(f'Спектр (RH={relative_humidity}%)')
    axes[0].set_xlabel('Частота (ТГц)')
    axes[0].grid(True)
    
    im = axes[1].imshow(res_spec2d, extent=[f_slice[0]*1e-12, f_slice[-1]*1e-12,
                                          f_delay[f_min_delay_idx]*1e-12, f_delay[f_max_delay_idx]*1e-12 
                                          if f_max_delay_idx < len(f_delay) else f_delay[-1]*1e-12],
                        aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title('2D FFT по реальным задержкам')
    axes[1].set_xlabel('Частота (ТГц)')
    plt.colorbar(im, ax=axes[1])
    
    axes[2].plot(t_cropped*1e15, amplitude_crop, 'b.-')
    axes[2].set_title('Реальные задержки')
    axes[2].set_xlabel('Задержка (fs)')
    axes[2].grid(True)
    
    
    plt.tight_layout()
    out_dir = "frames_real_delays"
    os.makedirs(out_dir, exist_ok=True)
    plt.show()
    #plt.savefig(os.path.join(out_dir, f"fr_{frame_id:03d}.png"), dpi=100)
    #plt.close(fig)


frame_id = 0
for RH in range(20, 21, 1):
    print(f"RH = {RH}%, реальные задержки")
    compute_and_plot_for_RH(RH, frame_id)
    frame_id += 1


