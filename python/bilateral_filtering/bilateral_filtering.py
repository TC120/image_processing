import cv2
import os
import math
import numpy as np


def add_noisy_one_img(src, mean, sigma):
    # 获取图片的高度和宽度
    height, width, channels = src.shape
    gauss = np.random.normal(mean, sigma, (height, width, channels))
    noisy_img = src + gauss
    # noisy_img = noisy_img.astype(np.float32)
    # 将加噪图片的像素值缩放放到cv2接收的uint8范围
    noisy_img1 = np.clip(noisy_img, a_min=0, a_max=255)
    noisy_img1 = noisy_img1.round().astype(np.uint8)

    return noisy_img1


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


def bilateral_filtering(image_pad_data, k_size, sigma_c, sigma_s):
    image_pad_data = image_pad_data.astype("float32")
    h_, w_, c_ = image_pad_data.shape
    result_data = np.zeros_like(image_pad_data)
    border = int((k_size - 1) / 2)
    #   对图像分图层进行处理
    for c in range(c_):
        #   输入的是填充后的图像，但索引要从原图像素处开始
        for y in range(border, h_ - border):
            for x in range(border, w_ - border):
                #   获得(x, y)处的像素参考框，然后与k逐像素相乘并求和
                sum_w = 0
                sum_pixel = 0
                for i in range(-border, border + 1):
                    for j in range(-border, border + 1):
                        ref_y = y + i
                        ref_x = x + j
                        # diff = np.sqrt(i**2 + j**2)
                        dist2_s = math.exp(-((i**2 + j**2) / (2 * sigma_s**2)))
                        p_ref = image_pad_data[ref_y, ref_x, c]
                        p_ori = image_pad_data[y, x, c]
                        dist2_c = math.exp(-(p_ref-p_ori)**2 / (2 * sigma_c**2))
                        sum_w += dist2_s * dist2_c
                        sum_pixel += dist2_s * dist2_c * p_ref
                a = sum_pixel / sum_w
                result_data[y, x, c] = int(round(a))

    return result_data[border:h_ - border, border:w_ - border, :].astype("uint8")


if __name__ == "__main__":
    ori_image = cv2.imread("../../data/image/lena.png")
    noisy_image = add_noisy_one_img(ori_image, 0, 15)
    #   cv2自带的均值滤波  cv2.blur(原始图像,核大小)

    cv2_result = cv2.bilateralFilter(noisy_image, 7, 30, 150)

    sym_border_image = set_border_sym(noisy_image, 7)
    result_data_bil = bilateral_filtering(sym_border_image, 7, 30, 150)

    cv2.imshow("sym_border_image", sym_border_image)
    cv2.imshow("noisy_image", noisy_image)
    cv2.imshow("cv2_result", cv2_result)
    cv2.imshow("result_data_bil", result_data_bil)
    cv2.imwrite("my_bil.png", result_data_bil)
    cv2.waitKey(0)
