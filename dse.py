import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def analyze_interferometer_data(filename=r'c:\Users\Андрей\Desktop\data_2\01_step8_speed10_CaseOpen_FiltersON.txt'):

    try:
        data = np.loadtxt(filename)
        signal1 = data[:, 0]
        signal2 = data[:, 1]
        signal3 = data[:, 2]
    except Exception as e:
        return

    s1_centered = signal1 - np.mean(signal1)
    s2_centered = signal2 - np.mean(signal2)


    s1_norm = s1_centered / (np.max(s1_centered) - np.min(s1_centered))
    s2_norm = s2_centered / (np.max(s2_centered) - np.min(s2_centered))


    phase_int = np.unwrap(np.arctan2(s2_norm, s1_norm))

    signal3_centered = signal3 - np.mean(signal3)
    analytic_signal = signal3_centered +   hilbert(signal3_centered)
    p_hw = np.angle(analytic_signal)
    phase_3 = np.unwrap(np.angle(analytic_signal))

    phase_diff_raw = phase_3 - phase_int
    phase_diff = np.unwrap(phase_diff_raw)

    k, const = np.polyfit(phase_int, phase_3, 1)
    

    

    phase_diff_corr = phase_3 - (k * phase_int + const)



    step = 100 
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # График 1: Абсолютные набеги фаз
    axs[0].plot(phase_int[::step] - phase_3[::step], label='phase_int', color='blue')
    axs[0].plot(phase_3[::step], label='phase_3 ', color='orange', alpha=0.8)

    axs[0].set_ylabel('Фаза (рад)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # График 2: Разность фаз
    axs[1].stem(p_hw[::step])
    axs[1].set_title(' фаза signal3 ')
    axs[1].set_ylabel('Фаза (рад)')
    axs[1].set_ylim(-np.pi - 0.5, np.pi + 0.5) # Фиксируем масштаб от -pi до pi
    axs[1].grid(True)

    # График 3: Фазовое пространство
    # Здесь тоже можно использовать децимацию, если точек слишком много
    axs[2].plot(phase_int[::step], phase_3[::step], '.', label='Данные', color='gray', markersize=1)
    axs[2].set_title('Фазовое пространство: phase_3 от phase_int')
    axs[2].set_xlabel('phase_int (рад)')
    axs[2].set_ylabel('phase_3 (рад)')
    axs[2].legend(loc='upper left')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":

    analyze_interferometer_data(r'c:\Users\Андрей\Desktop\data_2\01_step8_speed10_CaseOpen_FiltersON.txt')