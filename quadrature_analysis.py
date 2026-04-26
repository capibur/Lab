import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# =========================
# ФУНКЦИЯ КОРРЕКЦИИ I/Q
# =========================
def correct_iq(I, Q):
    # Убираем DC
    I0 = I - np.mean(I)
    Q0 = Q - np.mean(Q)

    X = np.vstack((I0, Q0))

    # Ковариационная матрица
    cov = np.cov(X)

    # Собственные значения и векторы
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Whitening (делаем круг)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    W = eigvecs @ D_inv_sqrt @ eigvecs.T

    X_corr = W @ X

    return X_corr[0, :], X_corr[1, :]


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
data = np.loadtxt(r'C:\Users\Андрей\Desktop\data_2\01_step8_speed10_CaseOpen_FiltersON.txt')

I = data[:, 0]
Q = data[:, 1]
Ref = data[:, 2]

t = np.arange(len(I))


# =========================
# КОРРЕКЦИЯ I/Q
# =========================
I_corr, Q_corr = correct_iq(I, Q)


# =========================
# ФАЗА ИЗ QD
# =========================
phi_qd = np.arctan2(Q_corr, I_corr)
phi_qd = np.unwrap(phi_qd)


# =========================
# ФАЗА ИЗ ДИОДА (ГИЛЬБЕРТ)
# =========================
analytic_signal = hilbert(Ref)

I_A, R_A = correct_iq(analytic_signal.imag, analytic_signal.real)
phi_ref =  np.arctan2(I_A, R_A)
phi_ref = np.unwrap(phi_ref) 
plt.scatter(t, phi_ref)
plt.axhline(y= np.pi/2, color='r', )
plt.axhline(y=- np.pi/2, color='r',)
plt.title('Фаза из гильберта (до unwrap)')
plt.grid()
plt.show()
 # unwrap с порогом для предотвращения больших скачков


# =========================
# УБИРАЕМ СДВИГ
# =========================
phi_qd -= phi_qd[0]
phi_ref -= phi_ref[0]


# =========================
# РАЗНОСТЬ ФАЗ
# =========================
delta_phi = phi_qd - phi_ref


# =========================
# ЛИНЕЙНАЯ АППРОКСИМАЦИЯ
# =========================
coeffs = np.polyfit(phi_ref, phi_qd, 1)
fit_line = np.polyval(coeffs, phi_ref)

print(f"Наклон (идеал ~1): {coeffs[0]:.6f}")
print(f"Смещение: {coeffs[1]:.6f}")


# =========================
# ГРАФИКИ
# =========================
plt.figure(figsize=(14, 10))

# --- I-Q до/после ---
plt.subplot(2, 3, 1)
plt.scatter(I, Q, s=5)
plt.title('I-Q до коррекции')
plt.axis('equal')
plt.grid()

plt.subplot(2, 3, 2)
plt.scatter(I_corr, Q_corr, s=5)
plt.title('I-Q после коррекции')
plt.axis('equal')
plt.grid()


# --- Фазы ---
plt.subplot(2, 3, 3)
plt.plot(t, phi_qd, label='phi_QD')
plt.plot(t, phi_ref, label='phi_ref')
plt.title('Фазы')
plt.legend()
plt.grid()


# --- Разность фаз ---
plt.subplot(2, 3, 4)
plt.plot(t, delta_phi)
plt.title('Δφ (должна быть константа)')
plt.grid()


# --- Сравнение фаз ---
plt.subplot(2, 3, 5)
plt.scatter(phi_ref, phi_qd, s=5, label='данные')
plt.plot(phi_ref, fit_line, color='black', label='fit')
plt.title('phi_QD vs phi_ref')
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()