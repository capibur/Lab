import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

filename = "POS_SCAN_19.23.34.txt"
lambda_phases = 6.23e-7
c = 299792458

# Загрузить данные
data = np.loadtxt(filename)
x = data[:, 0]
y = data[:, 1]   # сигнал
x = x/(2*np.pi) * lambda_phases / c

peaks_idx, _ = find_peaks(
    y,
    height=np.max(y) * 0.65,   # хотя бы 10 % от максимума
    distance=100,               # минимум 5 точек между пиками
)

peaks_x = x[peaks_idx]       # координаты пиков по x
peaks_y = y[peaks_idx]       # значения в пиках

print("Пики:")
for i, (px, py) in enumerate(zip(peaks_x, peaks_y)):
    print(f"{i+1:2d}: x = {px:8.4g}, y = {py:8.4g}")
distances = np.diff(peaks_x)  # расстояние между соседними пиками
print("Расстояния между соседними пиками:", distances)
print("Среднее расстояние между пиками:", np.mean(distances))
plt.figure(figsize=(10, 4))
plt.plot(x, y, label="Сигнал")
plt.plot(peaks_x, peaks_y, "ro", label="Пики", markersize=6)
plt.xlabel("x (фаза / delay)")
plt.ylabel("Сигнал")
plt.legend()
plt.grid(True)
plt.show()
print((distances[0]) * c )