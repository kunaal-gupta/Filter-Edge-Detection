import matplotlib.pyplot as plt
import numpy as np
from skimage import io

I = io.imread('moon.png')
h_image, w_image = I.shape

K = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])
# K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

K = K / K.sum()
h_ker, w_ker = K.shape

h_ker = np.cast['int']((h_ker - 1.) / 2.)
wf_ker = np.cast['int']((w_ker - 1.) / 2.)

J = np.zeros_like(I)
for i in np.arange(h_ker, h_image - h_ker, 1):
    for j in np.arange(wf_ker, w_image - wf_ker, 1):
        for l in np.arange(-h_ker, h_ker + 1, 1):
            for m in np.arange(-wf_ker, wf_ker + 1, 1):
                J[i, j] += I[i + l, j + m] * K[l + h_ker, m + wf_ker]

fig = plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(I, cmap='gray')
plt.title("Original Image")

plt.subplot(122)
plt.imshow(J.astype('float64'), cmap='gray')
plt.title("Filtered Image")
plt.show()