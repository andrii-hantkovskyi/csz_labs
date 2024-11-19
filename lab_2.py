import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження зображення
image_path = "image.jpg"
image = cv2.imread(image_path)
channels = cv2.split(image)

# Обчислення гістограм оригінального зображення
hist_original = [np.histogram(channel.ravel(), bins=256, range=(0, 256))[0] for channel in channels]

# Еквалізація гістограми
equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
equalized_image = cv2.merge(equalized_channels)
hist_equalized = [np.histogram(channel.ravel(), bins=256, range=(0, 256))[0] for channel in equalized_channels]

# Перетворення на відтінки сірого для операторів
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Оператор Робертса
kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=int)
kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=int)
roberts_x = cv2.filter2D(gray_image, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(gray_image, -1, kernel_roberts_y)
roberts_combined = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)

# Оператор Превіта
kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewitt_x = cv2.filter2D(gray_image, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(gray_image, -1, kernel_prewitt_y)
prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# Оператор Собела
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)

# Збереження оброблених зображень
cv2.imwrite("original_image.jpg", image)
cv2.imwrite("equalized_image.jpg", equalized_image)
cv2.imwrite("roberts_operator.jpg", roberts_combined)
cv2.imwrite("prewitt_operator.jpg", prewitt_combined)
cv2.imwrite("sobel_operator.jpg", sobel_combined)

# Візуалізація
plt.figure(figsize=(16, 18))

# 1. Оригінальне зображення та його гістограма
plt.subplot(4, 2, 1)
plt.title("Оригінальне зображення")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 2, 2)
plt.title("Гістограма оригінального зображення")
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.bar(range(256), hist_original[i], color=color, alpha=0.7, label=f'{color} channel')
plt.xlim(10,100)
plt.xlabel("Інтенсивність")
plt.ylabel("Кількість пікселів")
plt.legend()
plt.grid(True)

# 2. Еквалізоване зображення та його гістограма
plt.subplot(4, 2, 3)
plt.title("Еквалізоване зображення")
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 2, 4)
plt.title("Гістограма еквалізованого зображення")
for i, color in enumerate(colors):
    plt.bar(range(256), hist_equalized[i], color=color, alpha=0.7, label=f'{color} channel')
plt.xlim(10,100)
plt.xlabel("Інтенсивність")
plt.ylabel("Кількість пікселів")
plt.legend()
plt.grid(True)

# 3. Оператор Робертса
plt.subplot(4, 2, 5)
plt.title("Оператор Робертса")
plt.imshow(roberts_combined, cmap='gray')
plt.axis('off')

# 4. Оператор Превіта
plt.subplot(4, 2, 6)
plt.title("Оператор Превіта")
plt.imshow(prewitt_combined, cmap='gray')
plt.axis('off')

# 5. Оператор Собела
plt.subplot(4, 2, 7)
plt.title("Оператор Собела")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

