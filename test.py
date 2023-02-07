from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.filters import laplace
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
# I = io.imread('moon.png')
# h_image, w_image = I.shape
#
# K = np.array([[-1, -1, -1], [1, -8, -1], [-1, -1, -1]])
#
# h_ker, w_ker = K.shape
#
# hf_ker = np.cast['int']((h_ker - 1.) / 2.)
# wf_ker = np.cast['int']((w_ker - 1.) / 2.)
#
# J = np.zeros_like(I)
# for i in np.arange(hf_ker, h_image - hf_ker-1, 1):
#     for j in np.arange(wf_ker, w_image - wf_ker-1, 1):
#         for l in np.arange(-hf_ker, hf_ker, 1):
#             for m in np.arange(-wf_ker, wf_ker, 1):
#                 J[i, j] += I[i + l, j + m] * K[l + hf_ker, m + wf_ker]



# I = io.imread('moon.png')
# h_image, w_image = I.shape
#
# K = np.array([[-1, -1, -1], [1, -8, -1], [-1, -1, -1]])
#
# h_ker, w_ker = K.shape
#
# hf_ker = np.cast['int']((h_ker - 1.) / 2.)
# wf_ker = np.cast['int']((w_ker - 1.) / 2.)
#
# J = np.zeros_like(I)
# for i in np.arange(hf_ker, h_image - hf_ker-1, 1):
#     for j in np.arange(wf_ker, w_image - wf_ker-1, 1):
#         for l in np.arange(-hf_ker, hf_ker, 1):
#             for m in np.arange(-wf_ker, wf_ker, 1):
#                 J[i, j] += I[i + l, j + m] * K[l + hf_ker, m + wf_ker]

# fig = plt.figure(figsize=(6, 3))
# plt.subplot(121)
# plt.imshow(I, cmap='gray')
# plt.title("Original Image")
#
# plt.subplot(122)
# plt.imshow(J.astype(int), cmap='gray')
# plt.title("Filtered Image")
# plt.show()

a = np.array([[1,1], [2,2]])
b = np.pad(a, pad_width=2, mode='constant')
print(a)
print(b)