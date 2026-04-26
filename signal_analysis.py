"""
Программа для анализа данных интерферометрического эксперимента.

Техническое задание (ТЗ):
- Вход: файл с данными (например, data.txt или .txt из папки data_2), содержащий 3 колонки:
  1) signal1 — косинусная квадратурная компонента интерферометра (от гелий-неонового лазера He-Ne).
  2) signal2 — синусная квадратурная компонента интерферометра (от того же лазера).
  3) signal3 — внешний оптический сигнал (от лиол/детектора), зависящий от перемещения подвижки.
- Цель: проверить фазовую согласованность signal3 с интерферометром.
- Обработка:
  1. Восстановить фазу интерферометра: phase_int = unwrap(atan2(signal2, signal1)).
  2. Восстановить фазу signal3: использовать Hilbert-трансформу для аналитического сигнала, phase_3 = unwrap(angle(analytic_signal)).
  3. Вычислить разность фаз: phase_diff = unwrap(phase_3 - phase_int).
  4. Оценить линейный дрейф: аппроксимация phase_3 ≈ k * phase_int + const, где k — коэффициент масштабирования.
  5. Скорректировать разность фаз: phase_diff_corr = unwrap(phase_3 - (k * phase_int + const)).
- Требования: работать только в фазовом пространстве, без использования временной оси, частоты дискретизации или FFT.
- Дополнительно: проверить зависимость от амплитуды signal3 (чувствительности диода).

Используемые библиотеки: numpy, scipy.signal, matplotlib.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def analyze_interferometer_data(filename='data.txt'):
    """
    Основная функция анализа данных интерферометрического эксперимента.

    Параметры:
    - filename: путь к файлу с данными (3 колонки: signal1, signal2, signal3).

    Возвращает: словарь со статистикой (k, const, средние, std).
    """
    # Загрузка данных из файла
    data = np.loadtxt(filename)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError('Ожидается файл с тремя колонками данных.')

    # Извлечение сигналов: signal1 и signal2 — квадратурные компоненты интерферометра (He-Ne лазер)
    signal1 = data[:, 0]  # Косинусная компонента
    signal2 = data[:, 1]  # Синусная компонента
    signal3 = data[:, 2]  # Внешний сигнал от детектора

    # Центрирование сигналов: вычитаем среднее, чтобы убрать постоянное смещение (DC-компоненту)
    # Это важно для корректного вычисления фазы через atan2 и Hilbert
    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)
    signal3 = signal3 - np.mean(signal3)

    # 1. Восстановление фазы интерферометра
    # Интерферометр дает квадратурные сигналы, фаза — atan2(sin, cos), unwrap для непрерывности
    # phase_int пропорциональна перемещению подвижки: phase_int = (4π / λ) * d, где λ — длина волны He-Ne лазера (~632.8 нм)
    phase_int = np.unwrap(np.arctan2(signal2, signal1))

    # 2. Восстановление фазы signal3 через аналитический сигнал (Hilbert-трансформа)
    # Hilbert создает комплексный сигнал, угол которого — мгновенная фаза
    # phase_3 также пропорциональна перемещению, но с другой эффективной длиной волны или оптическим путем
    analytic_signal3 = hilbert(signal3)
    phase_3 = np.unwrap(np.angle(analytic_signal3))

    # 3. Разность фаз: насколько signal3 "опережает" или "отстает" от интерферометра
    # unwrap для корректной разности при больших изменениях фазы
    phase_diff = np.unwrap(phase_3 - phase_int)

    # 4. Оценка линейного дрейфа: аппроксимация phase_3 = k * phase_int + const
    # k показывает, во сколько раз быстрее/медленнее меняется phase_3 относительно phase_int
    # const — фазовый сдвиг
    k, const = np.polyfit(phase_int, phase_3, 1)
    phase_diff_corr = np.unwrap(phase_3 - (k * phase_int + const))

    # 5. Проверка зависимости от амплитуды signal3 (чувствительности диода)
    # Амплитуда из аналитического сигнала: показывает интенсивность
    amplitude_3 = np.abs(analytic_signal3)
    # Корреляция: если |r| > 0.5, амплитуда влияет на фазу (нелинейность диода)
    corr_amp_phase_diff = np.corrcoef(amplitude_3, phase_diff)[0, 1]
    corr_amp_phase_int = np.corrcoef(amplitude_3, phase_int)[0, 1]

    # Статистика для анализа
    stats = {
        'k': k,
        'const': const,
        'phase_diff_mean': np.mean(phase_diff),
        'phase_diff_std': np.std(phase_diff),
        'corr_diff_mean': np.mean(phase_diff_corr),
        'corr_diff_std': np.std(phase_diff_corr),
        'corr_amp_phase_diff': corr_amp_phase_diff,
        'corr_amp_phase_int': corr_amp_phase_int,
    }

    # Вывод результатов
    print('Файл:', filename)
    print('Оценка согласованности phase_3 ≈ k * phase_int + const')
    print(f'  k = {k:.6f} (масштаб фазы signal3 относительно интерферометра)')
    print(f'  const = {const:.6f} (фазовый сдвиг, рад)')
    print(f'phase_diff: mean = {stats["phase_diff_mean"]:.6f}, std = {stats["phase_diff_std"]:.6f}')
    print(f'Скорректированная разность фаз: mean = {stats["corr_diff_mean"]:.6f}, std = {stats["corr_diff_std"]:.6f}')
    print('Проверка зависимости от амплитуды signal3 (чувствительности диода):')
    print(f'  Корреляция амплитуды с phase_diff: {stats["corr_amp_phase_diff"]:.6f}')
    print(f'  Корреляция амплитуды с phase_int: {stats["corr_amp_phase_int"]:.6f}')
    print('  Если корреляция высокая (|r| > 0.5), амплитуда может влиять на дрейф (нелинейность детектора).')

    # Графики
    fig, axs = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

    # График 1: Сравнение фаз
    axs[0].plot(phase_int, color='tab:blue', label='phase_int (интерферометр)')
    axs[0].plot(phase_3, color='tab:orange', label='phase_3 (внешний сигнал)', alpha=0.8)
    axs[0].set_ylabel('Фаза (рад)')
    axs[0].set_title('Сравнение фаз интерферометра и внешнего сигнала')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # График 2: Амплитуда signal3
    axs[1].plot(amplitude_3, color='tab:purple')
    axs[1].set_ylabel('Амплитуда')
    axs[1].set_title('Амплитуда signal3 (из аналитического сигнала)')
    axs[1].grid(True)

    # График 3: Фаза интерферометра
    axs[2].plot(phase_int, color='tab:blue')
    axs[2].set_ylabel('phase_int')
    axs[2].set_title('Развернутая фаза интерферометра (He-Ne лазер)')
    axs[2].grid(True)

    # График 4: Фаза signal3
    axs[3].plot(phase_3, color='tab:orange')
    axs[3].set_ylabel('phase_3')
    axs[3].set_title('Развернутая фаза внешнего сигнала')
    axs[3].grid(True)

    # График 5: Разность фаз
    axs[4].plot(phase_diff, color='tab:green')
    axs[4].set_ylabel('phase_diff')
    axs[4].set_title('Разность фаз signal3 - interferometer')
    axs[4].grid(True)

    # График 6: Скорректированная разность
    axs[5].plot(phase_diff_corr, color='tab:red')
    axs[5].set_ylabel('corrected diff')
    axs[5].set_title('Скорректированная разность фаз после удаления линейного дрейфа')
    axs[5].set_xlabel('Номер сэмпла')
    axs[5].grid(True)

    plt.tight_layout()
    plt.show()

    # Дополнительный scatter plot: амплитуда vs phase_diff
    plt.figure(figsize=(8, 6))
    plt.scatter(amplitude_3, phase_diff, alpha=0.5, color='tab:cyan')
    plt.xlabel('Амплитуда signal3')
    plt.ylabel('phase_diff')
    plt.title('Зависимость phase_diff от амплитуды signal3')
    plt.grid(True)
    plt.show()

    return stats


if __name__ == '__main__':
    # Запуск из командной строки: python signal_analysis.py <путь_к_файлу>
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'data.txt'

    analyze_interferometer_data(input_file)


