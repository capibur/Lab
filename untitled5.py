import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import hapi as hp

c = 3e8
tau = 50e-15
lambda0 = 5e-6
w0 = 2 * np.pi * c / lambda0
lambda_phases = 6.23e-7
p_sat = 0.0276
filename = "POS_SCAN_19.23.34.txt"

w_size = 100
data = np.loadtxt(filename)

phases = data[:,0]
signal = data[:,1]

delay = phases/(2*np.pi) * lambda_phases / c

idx = np.argsort(delay)
delay = delay[idx]
signal = signal[idx]

delay_min = -1e-11   
delay_max = 1e-11  
mask = (delay >= delay_min) & (delay <= delay_max)
delay = delay[mask]
signal = signal[mask]
print(f"Обрезано с {len(data)} до {len(delay)} точек")

# Усреднение фона
ker = np.ones(w_size)/w_size
background = np.convolve(signal, ker, mode='same')
signal = signal - background

plt.figure()
plt.plot(delay, signal)
plt.xlabel("Delay (s)")
plt.ylabel("Signal")
plt.title("Signal after moving average subtraction")
plt.grid()
plt.show()

N = len(delay)
delay_uniform = np.linspace(delay.min(), delay.max(), N)
interp = interp1d(delay, signal, kind='linear')
signal_uniform = interp(delay_uniform)

window = np.hanning(N)
signal_uniform = signal_uniform

spec = np.fft.rfft(signal_uniform)
dt = delay_uniform[1] - delay_uniform[0]
freq = np.fft.rfftfreq(N, dt)
spectrum = np.abs(spec)

plt.figure()
plt.plot(delay_uniform, signal_uniform)
plt.xlim(delay_min, delay_max)

plt.figure()
plt.plot(freq, spectrum, label="Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0.45e14, 0.55e14)
plt.legend()
plt.grid()
plt.show()
