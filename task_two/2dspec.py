import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# =======================
#   ПАРАМЕТРЫ
# =======================
FILE_NAME = "07_15ps_4800-4-5030.txt"
c = 299792458
lambda_ref = 632.8e-9
mono = np.arange(4800, 5030, 4) / 1000  # мкм

T_AI3 = 500e-13  # Окно вокруг t0
tau_gauss = 7e-14  # Ширина Гауссиана для сглаживания
N_freq = 512
Fmin, Fmax = 5.996e13, 6.311e13  # Hz

def compute_weights(t):
    dt = np.diff(t)
    W = np.zeros_like(t)
    if len(dt) > 0:
        W[1:-1] = (dt[:-1] + dt[1:]) / 2
        W[0] = dt[0] / 2
        W[-1] = dt[-1] / 2
    return W

def gaussian_smooth(t, y, tau):
    """ Реализация Гауссова сглаживания из Matlab референса """
    y_smooth = np.zeros_like(y)
    for i in range(len(t)):
        g = np.exp(-(t - t[i])**2 / tau**2)
        y_smooth[i] = np.sum(y * g) / np.sum(g)
    return y_smooth


DATA = np.loadtxt(FILE_NAME)
N_scans = DATA.shape[1] // 4
N_scans = min(N_scans, len(mono))

freq = np.linspace(Fmin, Fmax, N_freq)
R_final = np.zeros((N_scans, N_freq))

for k in range(N_scans):
    print(f" {k+1}/{N_scans}")
    
   
    Phase  = DATA[:, k*4 + 0]
    Signal = DATA[:, k*4 + 1]
    AI3    = DATA[:, k*4 + 2]
    

    valid = Signal < 900
    Phase, Signal, AI3 = Phase[valid], Signal[valid], AI3[valid]
    
    
    if np.max(AI3[::2]) > np.max(AI3[1::2]):
        idx_open = slice(0, None, 2)
    else:
        idx_open = slice(1, None, 2)
        
    t = (Phase[idx_open] / (2*np.pi)) * (lambda_ref / c)
    S_raw = Signal[idx_open]
    A_raw = AI3[idx_open]
    
    # 5. Удаление дублей
    t, unique_idx = np.unique(t, return_index=True)
    S_raw, A_raw = S_raw[unique_idx], A_raw[unique_idx]
    
    # 6. Сглаживание и выделение быстрой компоненты 
    S_smooth = gaussian_smooth(t, S_raw, tau_gauss)
    A_smooth = gaussian_smooth(t, A_raw, tau_gauss)
    
    S_fast = S_raw - S_smooth
    A_fast = A_raw - A_smooth
    
    # 7. Определение t0 по AI3
    idx_t0 = np.argmax(A_fast)
    t0 = t[idx_t0]
    
    # 8. Выделение окна для фазовой коррекции
    mask_window = (t >= (t0 - T_AI3)) & (t <= (t0 + T_AI3))
    t_direct = t[mask_window]
    A_direct = A_fast[mask_window]
    W_direct = compute_weights(t_direct)
    
    # 11. Оценка фазы (Линейная аппроксимация как в Matlab)
    Spec_phase_check = np.zeros(N_freq, dtype=complex)
    for i, f in enumerate(freq):
        w = 2 * np.pi * f
        Spec_phase_check[i] = np.sum(A_direct * np.exp(1j * w * (t_direct - t0)) * W_direct)
    
    phi_raw = np.unwrap(np.angle(Spec_phase_check))
    coeffs = np.polyfit(freq, phi_raw, 1)
    phase_fit = np.polyval(coeffs, freq)
    
    # 13. Обнуление после t0 (как в Matlab)
    S_fast[idx_t0:] = 0 
    
    # 14. Вычисление спектра блока
    W_open = compute_weights(t)
    norm_level = np.mean(S_raw)
    
    Spec_block = np.zeros(N_freq, dtype=complex)
    for i, f in enumerate(freq):
        w = 2 * np.pi * f
        phi = phase_fit[i]
        # Используем комплексную экспоненту для c + i*s
        Spec_block[i] = np.sum(S_fast * np.exp(1j * (w * (t - t0) - phi)) * W_open)
    
    R_final[k, :] = np.real(Spec_block / norm_level)


freq_excitation = freq / c / 100  # cm^-1
freq_detection = 1e4 / mono[:N_scans]  # cm^-1

R_plot = R_final.copy()
# Раздельная нормировка плюса и минуса
pos_mask = R_plot > 0
neg_mask = R_plot < 0
if np.any(pos_mask): R_plot[pos_mask] /= np.max(R_plot)
if np.any(neg_mask): R_plot[neg_mask] /= np.abs(np.min(R_plot))

plt.figure(figsize=(8, 8))
# В Matlab imagesc отображает X как частоту возбуждения, Y как регистрацию
plt.imshow(-R_plot, extent=[freq_excitation[0], freq_excitation[-1], 
                            freq_detection[-1], freq_detection[0]], 
           aspect='auto', cmap='jet', vmin=-1, vmax=1)

plt.xlabel(' (cm$^{-1}$)')
plt.ylabel(' (cm$^{-1}$)')

plt.colorbar(label='Normalized Signal')

plt.show()