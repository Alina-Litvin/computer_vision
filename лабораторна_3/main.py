import numpy as np
import cv2
from numba import njit

image = cv2.imread('photo_1.jpg')


Array_grey = []

for Color_Pixel in image:
    list_1 = []
    for grey_pixel in Color_Pixel:
        list_1.append(sum(grey_pixel) // 3)
    Array_grey.append(list_1)

# отримуємо маємо масив із відтінками сірого
grey_img = np.array(Array_grey, dtype=np.uint8)
@njit()
def erosion(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            min_val = 255
            for m in range(kernel_height):
                for n in range(kernel_width):
                    if i + m < height and j + n < width and kernel[m, n] == 1:
                        min_val = min(min_val, image[i + m, j + n])
            result[i, j] = min_val

    return result


@njit()
def dilation(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            max_val = 0
            for m in range(kernel_height):
                for n in range(kernel_width):
                    if i + m < height and j + n < width and kernel[m, n] == 1:
                        max_val = max(max_val, image[i + m, j + n])
            result[i, j] = max_val

    return result
@njit()
def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)
@njit()
def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)
@njit()
def edges(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)

    return np.clip(result, 0, 255)  # Ensure values are in the valid range

# Визначення розміру ядра (structuring element)

kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]
                   ])

# Застосування морфологічних операцій
result = erosion(grey_img, kernel)
cv2.imwrite("erosion.jpg", result)

result = dilation(grey_img, kernel)
cv2.imwrite("dilation.jpg", result)

result = opening(grey_img, kernel)
cv2.imwrite("opening.jpg", result)

result = closing(grey_img, kernel)
cv2.imwrite("closing.jpg", result)

result = edges(grey_img)
cv2.imwrite("edges.jpg", result)
