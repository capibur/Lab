import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import tkinter as tk

# --- Константы ---
CONST_C = 299792458.0
CONST_TAU = 50e-15
CONST_LAMBDA0 = 6.3e-7
CONST_W0 = 2 * np.pi * CONST_C / CONST_LAMBDA0

# Параметры по умолчанию
amp_main = 1.0
amp_pre_init = 0.5
abs_center_shift_init = 5e12
abs_width_init = 0.5e12
abs_depth_init = 0.9

t_min_init, t_max_init = -5e-12, 5e-12
nt = 40000 

DELAY_MAX = 5e-12
DELAY_SIZE = 10000 
pulse_delay_init = 1.0 # ps
win_min_init, win_max_init = -5.0, 5.0 # ps

# --- СЕТКИ ---
t = np.linspace(t_min_init, t_max_init, nt)
dt_global = t[1] - t[0]
delays = np.linspace(-DELAY_MAX, DELAY_MAX, DELAY_SIZE)

# --- ФИЗИКА ---
def gaussian_pulse(t_arr, t0=0.0, amp=1.0, tau=CONST_TAU, omega=CONST_W0):
    envelope = amp * np.exp(-((t_arr - t0) ** 2) / (2 * (tau ** 2)))
    return envelope * np.sin(omega * (t_arr - t0))

def apply_doublet_absorption(E_t, t_grid, center_omega, shift, width, depth):
    freq_hz = rfftfreq(len(t_grid), t_grid[1] - t_grid[0])
    omega_grid = freq_hz * 2 * np.pi
    # Создаем две линии поглощения
    dip1 = depth * np.exp(-0.5 * ((omega_grid - (center_omega - shift)) / width) ** 2)
    dip2 = depth * np.exp(-0.5 * ((omega_grid - (center_omega + shift)) / width) ** 2)
    H = (1 - dip1) 
    return irfft(rfft(E_t) * H, n=len(t_grid))

def interferogram_fast(delays_grid, t_grid, E_field):
    dt = t_grid[1] - t_grid[0]
    I_0 = np.sum(E_field**2) * dt
    S = np.abs(rfft(E_field))**2
    AC = np.fft.fftshift(irfft(S, n=len(t_grid)) * dt)
    N = len(t_grid)
    d_arr = (np.arange(N) - N // 2) * dt
    return np.interp(delays_grid, d_arr, 2 * I_0 + 2 * AC)

# --- ГРАФИКА ---
fig, axes = plt.subplots(4, 1, figsize=(9, 11))
plt.subplots_adjust(left=0.1, bottom=0.07, top=0.95, hspace=0.6)

# Начальный расчет поля
E_init = gaussian_pulse(t, 0, amp_main) + gaussian_pulse(t, pulse_delay_init*1e-12, amp_pre_init)
E_example = apply_doublet_absorption(E_init, t, CONST_W0, abs_center_shift_init, abs_width_init, abs_depth_init)
I_full_data = interferogram_fast(delays, t, E_example)

# Инициализация линий графиков
line0, = axes[0].plot(rfftfreq(nt, dt_global)*1e-12, np.abs(rfft(E_example)), color='blue')
line1, = axes[1].plot(t*1e12, E_example, color='red')
line2, = axes[2].plot(delays*1e12, I_full_data, color='purple', lw=0.7)

# Границы окна обрезки
v_min = axes[2].axvline(win_min_init, color='orange', linestyle='--')
v_max = axes[2].axvline(win_max_init, color='orange', linestyle='--')

line3, = axes[3].plot([0], [0], color='green')

for i, title in enumerate(["Спектр источника (физический)", "Поле E(t)", "Интерферограмма", "Восстановленный спектр (после обрезки)"]):
    axes[i].set_title(title)
    axes[i].grid(True)
axes[0].set_xlim(450, 500)
axes[3].set_xlim(450, 500)

# --- ИНТЕРФЕЙС TKINTER ---
root = tk.Tk()
root.title("FTS Simulation Control")

def create_entry(txt, row, val):
    tk.Label(root, text=txt).grid(row=row, column=0, sticky='e', padx=5)
    e = tk.Entry(root)
    e.insert(0, str(val))
    e.grid(row=row, column=1, padx=5, pady=2)
    return e

e_delay = create_entry("Задержка импульса (ps):", 0, pulse_delay_init)
e_amp2  = create_entry("Амплитуда 2-го имп.:", 1, amp_pre_init)
e_shift = create_entry("Разнос дублета (THz):", 2, abs_center_shift_init/1e12)
e_wmin  = create_entry("Окно Min (ps):", 3, win_min_init)
e_wmax  = create_entry("Окно Max (ps):", 4, win_max_init)

def update():
    try:
        # Считываем значения
        p_del = float(e_delay.get()) * 1e-12
        a2 = float(e_amp2.get())
        sh = float(e_shift.get()) * 1e12
        w_m = float(e_wmin.get())
        w_M = float(e_wmax.get())
        
        # 1. Формируем новое поле и применяем поглощение
        E_raw = gaussian_pulse(t, 0, amp_main) + gaussian_pulse(t, p_del, a2)
        E_new = apply_doublet_absorption(E_raw, t, CONST_W0, sh, abs_width_init, abs_depth_init)
        
        # 2. Считаем полную интерферограмму
        I_data = interferogram_fast(delays, t, E_new)
        
        # 3. Обрезаем интерферограмму согласно "Окну"
        mask = (delays*1e12 >= w_m) & (delays*1e12 <= w_M)
        I_cropped = I_data[mask]
        delays_cropped = delays[mask]
        
        # Обновление графиков
        line0.set_ydata(np.abs(rfft(E_new)))
        line1.set_ydata(E_new)
        line2.set_ydata(I_data)
        v_min.set_xdata([w_m])
        v_max.set_xdata([w_M])
        
        if len(I_cropped) > 1:
            # Спектр из обрезанной интерферограммы
            I_res = np.abs(rfft(I_cropped - np.mean(I_cropped)))
            f_res = rfftfreq(len(I_cropped), delays_cropped[1]-delays_cropped[0]) * 1e-12
            line3.set_data(f_res, I_res)
            axes[3].relim()
            axes[3].autoscale_view(scalex=False, scaley=True)
            
        fig.canvas.draw_idle()
    except Exception as ex:
        print(f"Ошибка: {ex}")

tk.Button(root, text="ОБНОВИТЬ", command=update, bg='#adebad', font='Arial 10 bold').grid(row=5, columnspan=2, sticky='we', padx=5, pady=10)

update() # Запуск начальной отрисовки
plt.show()
root.mainloop()