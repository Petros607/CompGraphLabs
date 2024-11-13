import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

IMAGE_DIR = 'Lab4/2img2/'

def get_black_white(src_image):
    """
    Производит загрузку и перевод цветного изображения в оттенки серого.
    """
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

def zhang_suen_thinning(image):
    """
    Применяет алгоритм утончения линий Зонга-Суена для бинарного изображения.
    """
    binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    binary_image = binary_image // 255  # превращаем в бинарное 0 и 1

    prev_image = np.zeros_like(binary_image)
    while True:
        marker1 = np.zeros_like(binary_image)
        for y in range(1, binary_image.shape[0] - 1):
            for x in range(1, binary_image.shape[1] - 1):
                P = binary_image[y-1:y+2, x-1:x+2]
                if P[1,1] == 1:
                    conditions = [
                        (P[0,1] == 0), (P[1,0] == 0), (P[1,2] == 0), (P[2,1] == 0),
                        (P[1,1] == 1), (np.sum(P) >= 2 and np.sum(P) <= 6), (np.sum(P[0:2, 0:2]) == 0)
                    ]
                    if all(conditions):
                        marker1[y, x] = 1
        binary_image[marker1 == 1] = 0

        marker2 = np.zeros_like(binary_image)
        for y in range(1, binary_image.shape[0] - 1):
            for x in range(1, binary_image.shape[1] - 1):
                P = binary_image[y-1:y+2, x-1:x+2]
                if P[1,1] == 1:
                    conditions = [
                        (P[0,1] == 0), (P[1,0] == 0), (P[1,2] == 0), (P[2,1] == 0),
                        (P[1,1] == 1), (np.sum(P) >= 2 and np.sum(P) <= 6), (np.sum(P[0:2, 1:3]) == 0)
                    ]
                    if all(conditions):
                        marker2[y, x] = 1
        binary_image[marker2 == 1] = 0

        if np.array_equal(binary_image, prev_image):
            break
        prev_image = binary_image.copy()

    return binary_image * 255

def save_images():
    lower_threshold = int(lower_threshold_entry.get())
    upper_threshold = int(upper_threshold_entry.get())

    if lower_threshold > upper_threshold: 
        lower_threshold, upper_threshold = upper_threshold, lower_threshold

    image_file_path = "1.jpeg" #введение пути к файлу
    src_image_path = os.path.join("Lab4/img/", image_file_path)
    src_image = cv2.imread(src_image_path)
    if src_image is None:
        raise ValueError("Изображение не может быть загружено")
    
    gray_image = get_black_white(src_image)

    sliced_image = brightness_slice(gray_image, lower_threshold, upper_threshold)

    mask = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], dtype=np.float64) / 9
    A = 0
    B = 1 / 9
    filtered_image = mask_filter(gray_image, mask, A, B)

    suen_image = zhang_suen_thinning(gray_image)

    cv2.imwrite(os.path.join(IMAGE_DIR, 'original.jpg'), src_image)
    cv2.imwrite(os.path.join(IMAGE_DIR, 'grayscale.jpg'), gray_image)
    cv2.imwrite(os.path.join(IMAGE_DIR, 'brightness_slice.jpg'), sliced_image)
    cv2.imwrite(os.path.join(IMAGE_DIR, 'filtered.jpg'), filtered_image)
    cv2.imwrite(os.path.join(IMAGE_DIR, 'suen_image.jpg'), suen_image)

def resize_image(img, max_width, max_height):
    """
    Масштабирует изображение до заданных максимальных ширины и высоты, сохраняя пропорции.
    """
    width, height = img.size
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    
    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return img.resize((width, height), Image.Resampling.LANCZOS)


def update_images():
    images = ['original.jpg', 'grayscale.jpg', 'brightness_slice.jpg', 'filtered.jpg', 'suen_image.jpg']
    titles = ['Original', 'Grayscale', 'Brightness Slice', 'Masked Filter', 'Zhang Suen']

    for i, (image_file, title) in enumerate(zip(images, titles)):
        img_path = os.path.join(IMAGE_DIR, image_file)
        img = Image.open(img_path)

        MAX_WIDTH = 300
        MAX_HEIGHT = 300
        img_resized = resize_image(img, MAX_WIDTH, MAX_HEIGHT)

        if image_file == 'original.jpg':
            img_resized = img_resized.convert("RGB")

        img_tk = ImageTk.PhotoImage(img_resized)

        label = labels[i]
        label.config(image=img_tk)
        label.image = img_tk
        label_title[i].config(text=title)

root = tk.Tk()
root.title("Изображения и фильтрация")

frame = ttk.Frame(root)
frame.grid(row=0, column=0, padx=10, pady=10)

lower_threshold_label = ttk.Label(frame, text="Нижний порог:")
lower_threshold_label.grid(row=0, column=0)
lower_threshold_entry = ttk.Entry(frame)
lower_threshold_entry.grid(row=0, column=1)
lower_threshold_entry.insert(0, "0")

upper_threshold_label = ttk.Label(frame, text="Верхний порог:")
upper_threshold_label.grid(row=1, column=0)
upper_threshold_entry = ttk.Entry(frame)
upper_threshold_entry.grid(row=1, column=1)
upper_threshold_entry.insert(0, "255")

update_button = ttk.Button(frame, text="Обновить изображения", command=lambda: [save_images(), update_images()])
update_button.grid(row=2, column=0, columnspan=2)

image_frame = ttk.Frame(root)
image_frame.grid(row=1, column=0, padx=10, pady=10)

labels = []
label_title = []

for i in range(6):
    label_title.append(ttk.Label(image_frame, text=""))
    label_title[i].grid(row=i//3, column=(i%3)*2, padx=5, pady=5)
    labels.append(ttk.Label(image_frame))
    labels[i].grid(row=i//3, column=(i%3)*2+1, padx=5, pady=5)

save_images()
root.mainloop()
