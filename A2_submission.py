from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.filters import laplace
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def part1():
    I = io.imread('moon.png')
    h_image, w_image = I.shape

    K = np.array([[ 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])
    K = np.array([[-1, -1, -1], [1, -8, -1], [-1, -1, -1]])

    # K = K/K.sum()
    h_ker, w_ker = K.shape

    hf_ker = np.cast['int']((h_ker - 1.) / 2.)
    wf_ker = np.cast['int']((w_ker - 1.) / 2.)

    J = np.zeros_like(I)
    for i in np.arange(hf_ker, h_image - hf_ker, 1):
        for j in np.arange(wf_ker, w_image - wf_ker, 1):
            for l in np.arange(-hf_ker, hf_ker + 1, 1):
                for m in np.arange(-wf_ker, wf_ker + 1, 1):
                    J[i, j] += I[i + l, j + m] * K[l + hf_ker, m + wf_ker]

    fig = plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.imshow(I, cmap='gray')
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(J.astype('float64'), cmap='gray')
    plt.title("Filtered Image")
    plt.show()


def part2():
    import cv2
    import numpy as np

    img = cv2.imread('noisy.jpg')
    median = cv2.medianBlur(img, 5)
    pup_blur = cv2.GaussianBlur(img, (15, 15), 0)
    compare = np.concatenate((img, median, pup_blur), axis=1)  # side by side comparison

    cv2.imshow('img', compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows



def part3():
    """add your code here"""


def part4():
    """add your code here"""


def part5():
    """add your code here"""


if __name__ == '__main__':
    part1()
    # part2()
    # part3()
    part4()
    part5()