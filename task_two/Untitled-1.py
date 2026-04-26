import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import tkinter as tk

# --- КОНСТАНТЫ ---
C = 299792458.0
TAU = 50e-15
LAMBDA0 = 6.3e-7
W0 = 2 * np.pi * C / LAMBDA0

# --- ГЛОБАЛЬНАЯ ПОСТОЯННАЯ СЕТКА ---
# Делаем сетку один раз. Она большая, чтобы всё влезло.
T_MAX_ABS = 15e-12 
NP_POINTS = 40000 
t_grid = np.linspace(-T_MAX_ABS, T_MAX_ABS, NP_POINTS)
dt = t_grid[1] - t_grid[0]

# Сетка задержек
DELAY_SIZE = 400 # Уменьшил для скорости отклика UI
delays = np.linspace(0.0, 4e-12, DELAY_SIZE)

def get_pulse_with_absorption(t0, amp, width, depth, shift):
    """Генерирует один импульс с поглощением на фиксированной сетке t_grid"""
    # 1. Чистый импульс
    env = amp * np.exp(-((t_grid - t0)**2) / (2 * TAU**2))
    E_pure = env * np.sin(W0 * (t_grid - t0))
    
    # 2. Фильтр в частотной области
    freqs = rfftfreq(len(t_grid), d=dt) * 2 * np.pi
    E_f = rfft(E_pure)
    # Сдвиг поглощения относительно несущей W0
    H = 1 - depth * np.exp(-0.5 * ((freqs - (W0 + shift)) / width)**2)
    
    return np.fft.irfft(E_f * H, n=len(t_grid))

def update_plots():
    try:
        # Считываем параметры из UI
        t_min_mask = float(ent_t_min.get()) * 1e-12
        t_max_mask = float(ent_t_max.get()) * 1e-12
        d_idx = int(ent_d_idx.get())
        
        amp2 = float(ent_amp.get())
        w_abs = float(ent_width.get()) * 1e12
        depth_abs = float(ent_depth.get())
        shift_abs = float(ent_shift.get()) * 1e12

        # Создаем маску
        mask = (t_grid >= t_min_mask) & (t_grid <= t_max_mask)
        
        # Считаем базовые импульсы (без задержки и с задержкой)
        # Для интерферограммы нам нужно пересчитывать импульс 2 для каждой задержки
        # Но для скорости в примере спектра возьмем конкретную задержку d_idx
        
        p1 = get_pulse_with_absorption(0, 1.0, w_abs, depth_abs, -shift_abs)
        
        # Считаем интерферограмму
        I = np.zeros(DELAY_SIZE)
        for i, d in enumerate(delays):
            p2 = get_pulse_with_absorption(d, amp2, w_abs, depth_abs, shift_abs)
            E_total = p1 + p2
            # НАКЛАДЫВАЕМ МАСКУ ТУТ
            E_masked = E_total[mask]
            I[i] = np.sum(np.abs(E_masked)**2) * dt

        # Обновляем график Интерферограммы
        line_int.set_ydata(I)
        ax_int.relim()
        ax_int.autoscale_view()

        # Обновляем график Времени (сигнал при выбранной задержке)
        p2_selected = get_pulse_with_absorption(delays[d_idx], amp2, w_abs, depth_abs, shift_abs)
        E_vis = (p1 + p2_selected)
        line_time.set_ydata(E_vis)
        # Подсвечиваем маску
        rect_mask.set_xy([[t_min_mask*1e12, -2], [t_min_mask*1e12, 2], 
                          [t_max_mask*1e12, 2], [t_max_mask*1e12, -2]])

        # Обновляем Спектр (только того, что в маске!)
        E_for_fft = E_vis * mask
        E_f = np.abs(rfft(E_for_fft))
        f_hz = rfftfreq(len(t_grid), d=dt)
        
        # Перевод в длину волны (мкм)
        valid = f_hz > 1e14 # только оптический диапазон
        lambdas = (C / f_hz[valid]) * 1e6
        line_spec.set_xdata(lambdas)
        line_spec.set_ydata(E_f[valid])
        ax_spec.set_xlim(0.55, 0.75) # Центрируемся на 630нм
        ax_spec.relim()
        ax_spec.autoscale_view(scaley=True)

        # FFT Интерферограммы
        I_fft = np.abs(np.fft.rfft(I - np.mean(I)))
        line_ifft.set_ydata(I_fft)
        ax_ifft.relim()
        ax_ifft.autoscale_view()

        fig.canvas.draw_idle()
        
    except Exception as e:
        print(f"Ошибка: {e}")

# --- ИНТЕРФЕЙС TKINTER ---
root = tk.Toplevel()
root.title("Настройки маски и сканирования")
root.attributes('-topmost', True)

fields = [("Мин t (пс)", -3.0), ("Макс t (пс)", 3.0), ("Индекс задержки (0-399)", 200),
          ("Ампл. предимп.", 0.3), ("Ширина погл. (THz)", 0.8), ("Глубина погл.", 0.8), ("Сдвиг (THz)", 2.5)]
entries = []
for i, (txt, val) in enumerate(fields):
    tk.Label(root, text=txt).grid(row=i, column=0, padx=5, sticky='e')
    e = tk.Entry(root)
    e.insert(0, str(val))
    e.grid(row=i, column=1, padx=5)
    entries.append(e)

ent_t_min, ent_t_max, ent_d_idx, ent_amp, ent_width, ent_depth, ent_shift = entries
tk.Button(root, text="ОБНОВИТЬ ГРАФИКИ", command=update_plots, bg='orange').grid(row=len(fields), columnspan=2, pady=10)

# --- ГРАФИКИ MATPLOTLIB ---
fig, (ax_spec, ax_time, ax_int, ax_ifft) = plt.subplots(4, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.5)

line_spec, = ax_spec.plot([], [], color='red', lw=1.5)
ax_spec.set_title("Спектр (видимый в окне маски)")
ax_spec.set_xlabel("Длина волны (мкм)")

line_time, = ax_time.plot(t_grid * 1e12, np.zeros_like(t_grid), color='blue', alpha=0.5)
rect_mask = ax_time.add_patch(plt.Rectangle((0, -2), 0, 4, color='green', alpha=0.1, label='Маска'))
ax_time.set_title("Полный сигнал и область маски (зеленая)")
ax_time.set_xlim(-6, 6)
ax_time.set_ylim(-1.5, 1.5)

line_int, = ax_int.plot(delays * 1e12, np.zeros(DELAY_SIZE), color='black')
ax_int.set_title("Интерферограмма (интеграл внутри маски)")

freq_ifft = np.fft.rfftfreq(DELAY_SIZE, d=(delays[1]-delays[0])) * 1e-12
line_ifft, = ax_ifft.plot(freq_ifft, np.zeros(len(freq_ifft)), color='purple')
ax_ifft.set_title("FFT Интерферограммы")
ax_ifft.set_xlim(0, 10)

update_plots()
plt.show()