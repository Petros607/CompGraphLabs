import cv2
import numpy as np

def get_black_white(src_image_path):
    """
    Производит загрузку и перевод цветного изображения в оттенки серого.
    """
    src_image = cv2.imread(src_image_path)
    if src_image is None:
        raise ValueError("Изображение не может быть загружено")

    height, width, _ = src_image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = src_image[y, x]
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[y, x] = gray

    return gray_image

def brightness_slice(image, lower_threshold, upper_threshold):
    """
    Выполняет преобразование яркостного среза изображения на основе заданных порогов.
    """
    height, width = image.shape
    sliced_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if lower_threshold <= image[y, x] <= upper_threshold:
                sliced_image[y, x] = 255  
            else:
                sliced_image[y, x] = 0  

    return sliced_image

def mask_filter(image, mask, A, B):
    """
    Применяет линейную масочную фильтрацию к изображению.
    """
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            region = image[y - 1:y + 2, x - 1:x + 2]  
            new_gray = np.sum(region * mask) * B + image[y, x] * A
            new_gray = np.clip(new_gray, 0, 255)  
            filtered_image[y, x] = int(new_gray)

    return filtered_image

gray_image = get_black_white('Lab4/img/1.jpeg')
cv2.imshow('Grayscale Image', gray_image)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.imwrite('Lab4/2img2/1.jpeg', gray_image)

lower_threshold = int(input("Введите нижний порог (0-255): "))
upper_threshold = int(input("Введите верхний порог (0-255): "))
if lower_threshold > upper_threshold: lower_threshold, upper_threshold = upper_threshold, lower_threshold
sliced_image = brightness_slice(gray_image, lower_threshold, upper_threshold)
cv2.imshow('Brightness Sliced Image', sliced_image)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.imwrite('Lab4/2img2/2.jpeg', sliced_image)

mask = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]], dtype=np.float64) / 9
A = 0
B = 1 / 9
filtered_image = mask_filter(gray_image, mask, A, B)
cv2.imshow('Filtered image', filtered_image)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.imwrite('Lab4/2img2/3.jpeg', filtered_image)
