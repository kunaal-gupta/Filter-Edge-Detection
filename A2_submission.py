import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import arange
from scipy import ndimage as ndi
from scipy.signal import convolve2d
from skimage import feature
from skimage import io


def part1_a():
    img = io.imread('moon.png')

    K = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    h, w = K.shape

    I = np.pad(img, pad_width=((h, h), (w, w)), mode='constant')
    I = I.astype('float64')

    H, W = I.shape

    h = np.cast['int']((h - 1.) / 2.)
    w = np.cast['int']((w - 1.) / 2.)

    J = np.zeros_like(I)
    for i in np.arange(h, H - h, 1):
        for j in np.arange(w, W - w, 1):
            for m in np.arange(-h, h + 1, 1):
                for n in np.arange(-w, w + 1, 1):
                    J[i, j] += I[i + m, j + n] * K[m + h, n + w]

    fig = plt.figure(figsize=(6, 3))
    plt.subplot(121)
    plt.imshow(I, cmap='gray')
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(J.astype('float64'), cmap='gray')
    plt.title("Filtered Image")
    plt.show()

def part1_b():
    I = io.imread('moon.png')
    h_image, w_image = I.shape

    K = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])

    K = K / K.sum()
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


def part1_c():
    I = io.imread('moon.png')
    h_image, w_image = I.shape

    K = np.array([[0, 0, 0], [6, 0, 6], [0, 0, 0]])

    K = K / K.sum()
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


def part1():
    part1_a()
    part1_b()
    part1_c()
    part1_c()


def part2():
    img = cv2.imread('noisy.jpg')
    median = cv2.medianBlur(img, 5)
    gaussian = cv2.GaussianBlur(img, (15, 15), 0)
    compare = np.concatenate((img, median, gaussian), axis=1)  # side by side comparison

    plt.subplot(131)
    io.imshow((img).astype(int), cmap='gray')
    plt.title("Original")

    plt.subplot(132)
    io.imshow((median).astype(int), cmap='gray')
    plt.title("Median")

    plt.subplot(133)
    io.imshow((gaussian).astype(int), cmap='gray')
    plt.title("Gaussian")

    plt.show()


def part3():
    I = io.imread('damage_cameraman.png')
    U = io.imread('damage_mask.png')
    h, w = I.shape

    J = I.copy()

    for i in range(7):
        J = cv2.GaussianBlur(J, (15, 15), 0)

        for i in np.arange(0, h, 1):
            for j in np.arange(0, w, 1):
                J[i, j] = np.where(U[i, j] != 0, I[i, j], J[i, j])

    compare = np.concatenate((I, J), axis=1)  # side by side comparison

    plt.subplot(211)
    io.imshow((I).astype(int), cmap='gray')
    plt.title("Damaged Image")

    plt.subplot(212)
    io.imshow((J).astype(int), cmap='gray')
    plt.title("Restored Image")

    plt.show()


def part4():

    I = io.imread("ex2.jpg")

    fig = plt.figure()
    plt.imshow(I, cmap='gray', vmin=0, vmax=255)
    plt.title("Cameraman")
    plt.show()

    gh = 0.5 * np.array([[-1.0, 0.0, 1.0]])
    gv = 0.5 * np.array([[-1.0, 0.0, 1.0]]).T
    print(gh.shape, gv.shape)

    Ix = convolve2d(I, gh, mode='same', boundary='symm')
    Iy = convolve2d(I, gv, mode='same', boundary='symm')
    E = np.sqrt(Ix ** 2 + Iy ** 2)  # Edge strength or Gradient magnitude
    #
    plt.subplot(221)
    io.imshow((I).astype(int), cmap='gray')
    plt.title("Image")

    plt.subplot(223)
    io.imshow((Ix).astype(int), cmap='gray')
    plt.title("Vertical gradient:")

    plt.subplot(222)
    io.imshow((Iy).astype(int), cmap='gray')
    plt.title("Horizontal gradient")

    plt.subplot(224)
    io.imshow((E).astype(int), cmap='gray')
    plt.title("Horizontal gradient")

    plt.show()



def part5():
    CannyEdgeImage = io.imread("ex2.jpg", as_gray=True)
    TargetImage = io.imread("canny_target.jpg", as_gray=True)
    BestParam = [0, 0, 0]

    def EdgeDetection(CannyEdgeImage, TargetImage, BestParam):

        step_length = 2
        best_distance = 10 ** 10

        for low_thress in arange(50, 101, step_length):
            for high_threshold in arange(100, 201, 15):
                for sigma in arange(1, 4, 0.1):
                    canny_edgeImage = feature.canny(CannyEdgeImage, sigma=sigma, high_threshold=high_threshold,
                                                    low_threshold=low_thress)

                    this_distance = scipy.spatial.distance.cosine(canny_edgeImage.ravel(), TargetImage.ravel())

                    if this_distance < best_distance and this_distance != 0:
                        best_distance = this_distance
                        BestParam[0:] = sigma, high_threshold, low_thress
        print('Best Cosine Distance: ', BestParam)

    EdgeDetection(CannyEdgeImage, TargetImage, BestParam)
    FilteredImage = ndi.gaussian_filter(CannyEdgeImage, 4)
    OuputImage = feature.canny(FilteredImage, sigma=1, low_threshold=5)

    plt.subplot(221)
    plt.imshow((CannyEdgeImage).astype(int), cmap='gray')
    plt.title("Original Image")

    plt.subplot(222)
    plt.imshow((FilteredImage).astype(int), cmap='gray')
    plt.title("Gaussian Filter")


    plt.subplot(223)
    plt.imshow((TargetImage).astype(int), cmap='gray')
    plt.title("Target Image")

    plt.subplot(224)
    plt.imshow((OuputImage).astype(int), cmap='gray')
    plt.title("My Image")
    plt.show()


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()
    part5()
