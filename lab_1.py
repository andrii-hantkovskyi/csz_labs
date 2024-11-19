import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Дано:
A = 1.5  # Амплітуда
tau = 4e-3  # Тривалість імпульсу в секундах
T_interval = 2.5 * tau  # Інтервал аналізу
N = 128  # Кількість відліків
fs = N / T_interval  # Частота дискретизації

# Створення неперервного сигналу (прямокутний імпульс)
t_continuous = np.linspace(0, T_interval, 1000)  # Часові відліки для неперервного сигналу
s_continuous = np.where((t_continuous >= 0) & (t_continuous <= tau), A, 0)  # Формування імпульсу

# Побудова спектру неперервного сигналу
frequencies_continuous = np.linspace(0, fs, 1000)
spectrum_continuous = A * tau * np.sinc(frequencies_continuous * tau)

# Дискретизація сигналу
t_discrete = np.linspace(0, T_interval, N, endpoint=False)
s_discrete = np.where((t_discrete >= 0) & (t_discrete <= tau), A, 0)

# Амплітудний спектр дискретного сигналу за допомогою ДПФ
spectrum_discrete = fft(s_discrete)
frequencies_discrete = fftfreq(N, 1 / fs)

# Нормалізація спектра дискретного сигналу
spectrum_discrete_magnitude = 2.0 / N * np.abs(spectrum_discrete[:N // 2])

# Побудова графіків
plt.figure(figsize=(12, 8))

# Неперервний сигнал
plt.subplot(3, 1, 1)
plt.plot(t_continuous * 1e3, s_continuous, label="Неперервний сигнал")
plt.stem(t_discrete * 1e3, s_discrete, 'r', markerfmt="ro", basefmt=" ", label="Дискретний сигнал")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid()

# Амплітудний спектр неперервного сигналу
plt.subplot(3, 1, 2)
plt.plot(frequencies_continuous, np.abs(spectrum_continuous), label="Спектр неперервного сигналу")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid()

# Амплітудний спектр дискретного сигналу
plt.subplot(3, 1, 3)
plt.stem(frequencies_discrete[:N // 2], spectrum_discrete_magnitude, 'b', markerfmt="bo", basefmt=" ", label="Спектр дискретного сигналу")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплітуда")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
