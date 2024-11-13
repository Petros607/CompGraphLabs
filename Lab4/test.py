import cv2
import numpy as np
import time

# def get_zhang_suen(src_image):
#     gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    
#     _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
#     pixels = (binary_image // 255).astype(int)
    
#     height, width = pixels.shape
#     result_image = np.copy(pixels)
    
#     def iterate_step(pixels, step):
#         marker = np.zeros_like(pixels)
        
#         for y in range(1, height - 1):
#             for x in range(1, width - 1):
#                 if pixels[y, x] == 1:
#                     P = [pixels[y-1, x-1], pixels[y-1, x], pixels[y-1, x+1], 
#                          pixels[y, x-1], pixels[y, x+1], 
#                          pixels[y+1, x-1], pixels[y+1, x], pixels[y+1, x+1]]
                    
#                     sum_neighbors = sum(P)
#                     S = sum([P[i] - P[(i+1) % 8] == 1 for i in range(8)])
                    
#                     if step == 1:
#                         condition1 = (2 <= sum_neighbors <= 6)
#                         condition2 = (S == 1)
#                         condition3 = (P[1] * P[3] * P[5] == 0)
#                         condition4 = (P[3] * P[5] * P[7] == 0)
#                     else:
#                         condition1 = (2 <= sum_neighbors <= 6)
#                         condition2 = (S == 1)
#                         condition3 = (P[1] * P[3] * P[7] == 0)
#                         condition4 = (P[1] * P[5] * P[7] == 0)
                    
#                     if condition1 and condition2 and condition3 and condition4:
#                         marker[y, x] = 1
                        
#         pixels[marker == 1] = 0
#         return pixels, marker
    
#     has_changed = True
#     while has_changed:
#         has_changed = False
#         for step in [1, 2]:
#             pixels, marker = iterate_step(pixels, step)
#             if np.any(marker == 1):
#                 has_changed = True
#         print(f"Изменения сделаны {time.time()}")
    
#     result_image = np.uint8(pixels * 255)
    
#     return result_image

def get_zhang_suen(image):
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

src_image = cv2.imread('Lab4/img/5.jpeg')
gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

zhang_suen_result = get_zhang_suen(gray_image)

gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
thinned_image = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)

combined_image = np.hstack((zhang_suen_result, thinned_image))

cv2.imshow('Zhang-Suen vs OpenCV Thinning', combined_image)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Сохраняем результат
# cv2.imwrite('zhang_suen_vs_thinning.jpg', combined_image)
