import cv2
import numpy as np


img = cv2.imread("text_1.jpg")         # зчитуємо фотографію


Array_grey = []

for Color_Pixel in img:
    list_1 = []
    for grey_pixel in Color_Pixel:
        list_1.append(sum(grey_pixel) // 3)
    Array_grey.append(list_1)

# отримуємо маємо масив із відтінками сірого
grey_img = np.array(Array_grey, dtype=np.uint8)

# масив для інтенсивностей пікселів
img_array = np.array(Array_grey)

# створення гістограми
hist, _ = np.histogram(img_array, bins=256, range=(0, 256))

total_pixels = grey_img.shape[0] * grey_img.shape[1]

# змінних для оптимального порогу та максимального міжгрупового дисперсійного варіанту
optimal_threshold = 0
max_variance = 0

# перебір можливих порогів від 0 до 255
for threshold in range(256):
    # розрахунок ймовірностей належності до класів об'єктів та фону
    pixels_lower = np.sum(hist[:threshold])
    pixels_higher = np.sum(hist[threshold:])
    prob_lower = pixels_lower / total_pixels
    prob_higher = pixels_higher / total_pixels

    # розрахунок середніх значень об'єктів та фону
    mean_lower = np.sum(np.arange(threshold) * hist[:threshold]) / pixels_lower if pixels_lower > 0 else 0
    mean_higher = np.sum(np.arange(threshold, 256) * hist[threshold:]) / pixels_higher if pixels_higher > 0 else 0

    # розрахунок внутрішньогрупової дисперсії
    variance_lower = np.sum(((np.arange(threshold) - mean_lower) ** 2) * hist[:threshold]) / pixels_lower if pixels_lower > 0 else 0
    variance_higher = np.sum(((np.arange(threshold, 256) - mean_higher) ** 2) * hist[threshold:]) / pixels_higher if pixels_higher > 0 else 0

    # розрахунок міжгрупової дисперсії
    interclass_variance = prob_lower * prob_higher * (mean_lower - mean_higher) ** 2

    # оновлення оптимального порогу, якщо знайдено кращий
    if interclass_variance > max_variance:
        max_variance = interclass_variance
        optimal_threshold = threshold

# створення пустого бінарного зображення тієї ж розмірності, що і сіре зображення
binary_img = np.zeros_like(grey_img)

# використання оптимального порогу для бінаризації зображення
for i in range(grey_img.shape[0]):
    for j in range(grey_img.shape[1]):
        pixel_value = grey_img[i, j]
        if pixel_value < optimal_threshold:
            binary_img[i, j] = 255  # піксель менше порогу - білий
        else:
            binary_img[i, j] = 0

# створення пустого зображення тієї ж розмірності, що й оригінальне
color_img = np.zeros_like(img)

# нівелюємо різницю між білим та іншим кольором фону
i = 0
for pixel in binary_img:        # підрахунок білих пікселів
    for x in pixel:
        if x == 255:
            i += 1

if total_pixels - i >= total_pixels // 2:
    a = 255
else:
    a = 0

# ітеруєтеся по кожному пікселю зображення
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if binary_img[y, x] == a:
            color_img[y, x] = img[y, x]  # встановлюємо колір пікселя з оригінального зображення

# збереження кольорового зображення маски(об'єкта)
cv2.imwrite("output_text_1.jpg", color_img)
