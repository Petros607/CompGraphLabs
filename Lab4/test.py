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
    # has_changed = True
    # while has_changed:
    #     has_changed = False
    #     for step in [1, 2]:
    #         pixels, marker = iterate_step(pixels, step)
    #         if np.any(marker == 1):
    #             has_changed = True
    #     print(f"Изменения сделаны {time.time()}")
    
    # result_image = np.uint8(pixels * 255)
    
    # return result_image

def get_zhang_suen(src_image):
    height0, width0 = src_image.shape
    height, width = src_image.shape
    # pixels = np.zeros((height, width), dtype=int)
    pixels = np.where(src_image > 127, 1, 0)
    for y in range(height):
        for x in range(width):
            pixels[y, x] = 1 - round(src_image[y, x] / 255)
    height, width = pixels.shape
    has_changed = True
    while has_changed:
        has_changed = False
        marker = np.zeros((height, width), dtype=int)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if pixels[y, x] == 1:
                    P9 = pixels[y - 1, x - 1]
                    P2 = pixels[y - 1, x]
                    P3 = pixels[y - 1, x + 1]
                    P8 = pixels[y, x - 1]
                    P4 = pixels[y, x + 1]
                    P7 = pixels[y + 1, x - 1]
                    P6 = pixels[y + 1, x]
                    P5 = pixels[y + 1, x + 1]
                    
                    sum_neighbors = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    S = 0
                    if (P3 - P2) == 1: S += 1
                    if (P4 - P3) == 1: S += 1
                    if (P5 - P4) == 1: S += 1
                    if (P6 - P5) == 1: S += 1
                    if (P7 - P6) == 1: S += 1
                    if (P8 - P7) == 1: S += 1
                    if (P9 - P8) == 1: S += 1
                    if (P2 - P9) == 1: S += 1
                    
                    if 2 <= sum_neighbors <= 6 and S == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0:
                        marker[y, x] = 1
                        has_changed = True
        height, width = marker.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if marker[y, x] == 1: 
                    pixels[y, x] = 0
                    marker[y, x] = 0
        height, width = pixels.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if pixels[y, x] == 1:
                    P9 = pixels[y - 1, x - 1]
                    P2 = pixels[y - 1, x]
                    P3 = pixels[y - 1, x + 1]
                    P8 = pixels[y, x - 1]
                    P4 = pixels[y, x + 1]
                    P7 = pixels[y + 1, x - 1]
                    P6 = pixels[y + 1, x]
                    P5 = pixels[y + 1, x + 1]
                    
                    sum_neighbors = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    S = 0
                    if (P3 - P2) == 1: S += 1
                    if (P4 - P3) == 1: S += 1
                    if (P5 - P4) == 1: S += 1
                    if (P6 - P5) == 1: S += 1
                    if (P7 - P6) == 1: S += 1
                    if (P8 - P7) == 1: S += 1
                    if (P9 - P8) == 1: S += 1
                    if (P2 - P9) == 1: S += 1
                    
                    if 2 <= sum_neighbors <= 6 and S == 1 and P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0:
                        marker[y, x] = 1
                        has_changed = True
        height, width = marker.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if marker[y, x] == 1: pixels[y, x] = 0
                marker[y, x] = 0
        height, width = pixels.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if pixels[y, x] == 1:
                    P9 = pixels[y - 1, x - 1]
                    P2 = pixels[y - 1, x]
                    P3 = pixels[y - 1, x + 1]
                    P8 = pixels[y, x - 1]
                    P4 = pixels[y, x + 1]
                    P7 = pixels[y + 1, x - 1]
                    P6 = pixels[y + 1, x]
                    P5 = pixels[y + 1, x + 1]
                    
                    if (abs(1 - P9) * P4 * P6 == 1 or abs(1 - P5) * P8 * P2 == 1 or abs(1 - P3) * P6 * P8 == 1 or abs(1 - P7) * P2 * P4 == 1):
                        pixels[y, x] = 0

        result_image = np.zeros((height0, width0, 4), dtype=np.uint8)
        for y in range(height0):
            for x in range(width0):
                color_value = 1 - pixels[y, x]
                result_image[y, x] = [color_value * 255, color_value * 255, color_value * 255, 255]

        return result_image
    

src_image = cv2.imread('Lab4/img/a.jpg')
src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

# gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

zhang_suen_result = get_zhang_suen(src_image)

cv2.imshow('Zhang-Suen vs OpenCV Thinning', zhang_suen_result)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Сохраняем результат
# cv2.imwrite('zhang_suen_vs_thinning.jpg', combined_image)
