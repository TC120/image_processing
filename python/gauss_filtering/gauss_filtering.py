import cv2
import math
import numpy as np
from tqdm import *
import numba as nb


#   生成双边核
def kernel_generate(ds, a=1.1):
    s = 0
    k = np.zeros((2*ds + 1, 2*ds + 1))
    for x in range(-ds, ds + 1):
        for y in range(-ds, ds + 1):
            k[x + ds, y + ds] = math.exp(-(x * x + y * y) / (2* a*a))
            s += k[x + ds, y + ds]
    # print(k / s)
    return k / s


#   对图像进行边缘镜像处理
def set_border_sym(image_data, ksize):
    h_, w_, c_ = image_data.shape
    border = int((ksize - 1) / 2)
    result_data = np.zeros((h_ + ksize - 1, w_ + ksize - 1, c_))
    for c in range(c_):
        for i in range(h_ + 2 * border):
            if i < border:
                y_ori = border - i
            elif i > h_ + border - 1:
                y_ori = (h_ + border - 1) - (i - (border + h_ - 1)) - border
            else:
                y_ori = i - border
            for j in range(w_ + 2 * border):
                if j < border:
                    x_ori = border - j
                elif j > w_ + border - 1:
                    x_ori = (w_ + border - 1) - (j - (border + w_ - 1)) - border
                else:
                    x_ori = j - border
                result_data[i, j, c] = image_data[y_ori, x_ori, c]
    return result_data.astype("uint8")


def gauss_filtering(image_pad_data, ksize):
    h_, w_, c_ = image_pad_data.shape
    result_data = np.zeros_like(image_pad_data)
    border = int((ksize - 1) / 2)
    #   定义卷积核大小
    k_ = kernel_generate(border, a=1.1)

    #   对图像分图层进行处理
    for c in range(c_):
        #   输入的是填充后的图像，但索引要从原图像素处开始
        for y in range(border, h_ - border):
            for x in range(border, w_ - border):
                #   获得(x, y)处的像素参考框，然后与k逐像素相乘并求和
                reference_matrix = image_pad_data[y-border:y+border + 1, x-border:x+border+1, c]
                sum_ref = 0
                for i in range(ksize):
                    for j in range(ksize):
                        sum_ref += reference_matrix[i, j] * k_[i, j]

                result_data[y, x, c] = sum_ref

    return result_data[border:h_ - border, border:w_ - border, :].astype("uint8")


if __name__ == "__main__":
    ori_image = cv2.imread("../../data/image/lena.png")

    #   cv2自带的均值滤波  cv2.blur(原始图像,核大小)

    cv2_result = cv2.GaussianBlur(ori_image, (5, 5), 1.1)

    sym_border_image = set_border_sym(ori_image, 5)
    result_data_median = gauss_filtering(sym_border_image, 5)

    cv2.imshow("ori_image", ori_image)
    # cv2.imshow("zero_border_image", zero_border_image)
    cv2.imshow("sym_border_image", sym_border_image)
    cv2.imshow("cv2_result", cv2_result)
    cv2.imshow("result_data_gauss", result_data_median)
    cv2.waitKey(0)


