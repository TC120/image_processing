import cv2
import numpy as np


#   对图像边界进行0填充
def set_border_zero(image_data, ksize):
    h_, w_, c_ = image_data.shape
    border = int((ksize - 1) / 2)
    result_data = np.zeros((h_ + ksize - 1, w_ + ksize - 1, c_))
    result_data[border:border+h_, border:border+w_, :] = image_data
    return result_data.astype("uint8")


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


def median_filtering(image_pad_data, ksize):
    h_, w_, c_ = image_pad_data.shape
    result_data = np.zeros_like(image_pad_data)
    #   定义卷积核大小
    border = int((ksize - 1) / 2)
    #   对图像分图层进行处理
    for c in range(c_):
        #   输入的是填充后的图像，但索引要从原图像素处开始
        for y in range(border, h_ - border):
            for x in range(border, w_ - border):
                #   获得(x, y)处的像素参考框，然后排序取中值
                reference_matrix = image_pad_data[y-border:y+border + 1, x-border:x+border+1, c]
                #   转为1维矩阵
                reference_matrix = reference_matrix.reshape(-1)
                #   排序
                reference_matrix = np.sort(reference_matrix)
                result_data[y, x, c] = reference_matrix[border]

    return result_data[border:h_ - border, border:w_ - border, :].astype("uint8")


if __name__ == "__main__":
    ori_image = cv2.imread("../../data/image/lena.png")

    #   cv2自带的均值滤波  cv2.blur(原始图像,核大小)

    cv2_result = cv2.medianBlur(ori_image, 5)

    #   手写均值滤波, 这里ksize一般为奇数，便于边缘填充，当ksize = 1 时，算发不起作用
    # zero_border_image = set_border_zero(ori_image, 5)
    sym_border_image = set_border_sym(ori_image, 5)
    result_data_median = median_filtering(sym_border_image, 3)

    cv2.imshow("ori_image", ori_image)
    # cv2.imshow("zero_border_image", zero_border_image)
    cv2.imshow("sym_border_image", sym_border_image)
    cv2.imshow("cv2_result", cv2_result)
    cv2.imshow("result_data_median", result_data_median)
    cv2.waitKey(0)








