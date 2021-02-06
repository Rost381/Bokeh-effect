# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numba import  jit

OUTPUT_DIR = 'output/'

def main_bokeh():
    input_file = input('Image file: ')
    input_img = cv2.imread(input_file)
    if  input_img is  None :
        return print(f'Error loadind file {input_file}')

    kernel_file = input('Kernel file: ')
    kernel_img = cv2.imread(kernel_file)
    if  kernel_img is  None :
        return print(f'Error loadind file {kernel_file}')

    output_img =  input('Output file: ')

    # Размер картинки и ядра
    # The size of the image and kernel
    h_img, w_img = input_img.shape[:2]
    h_kernel, w_kernel = kernel_img.shape[:2]
    if max(h_kernel, w_kernel) >= min(h_img, w_img):
        return print('Image smaller than kernel!')

    # Рамка Zero-border
    # Frame Zero-border
    h_border, w_border = h_kernel//2, w_kernel//2
    border_img = cv2.copyMakeBorder(input_img, h_border, h_border, w_border, w_border, 0)

    # Свертка по всем каналам
    # Convolution across all channels
    out = channel_convolution(border_img, kernel_img, h_img, w_img, h_kernel, w_kernel)

    #Original (Оригинал)
    cv2.imshow('Image', input_img)
    cv2.waitKey(0)

    #Bokeh Effect (Эффект боке)
    cv2.imshow('Image', out)
    cv2.waitKey(0)

    output_name = OUTPUT_DIR + output_img
    try:
        cv2.imwrite(output_name, out)
    except:
        return print('Error writing file.')
    return print(f'Image is successfully saved as file: {output_name}')

# Свертка по всем каналам
# Convolution across all channels
@jit
def channel_convolution(border_img, kernel_original, h_img, w_img, h_kern, w_kern):
    # Нормирование
    # Normalization
    border_img = border_img / 255
    kernel_original = kernel_original / 255
    
    # Массив с размерами основного изображения
    # Array with the dimensions of the main image
    out_img = np.zeros((h_img, w_img,3))

    i, j, k = 0, 0, 0
    for k in range(3): # Chanels (По каналам)
        for i in range(h_img): # Hight (По высоте)
            for j in range(w_img): # Width (По ширине)
                kernel_sum = np.sum(kernel_original[:,:,k]) # Sum (Сумма для текущего)
                out_img[i,j,k]= np.sum(kernel_original[:,:,k] * border_img[i:i + h_kern, j:j + w_kern, k])/kernel_sum
            j+=1
        i+=1
    k +=1

    return  (out_img * 255).astype(np.uint8)# Denormalization (Денормирование)

if __name__ == '__main__':
    main_bokeh()
















