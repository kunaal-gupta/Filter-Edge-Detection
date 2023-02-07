from skimage import data, io, filters
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.filters import laplace
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
img = io.imread('moon.png')

K = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
h, w = K.shape

I = np.pad(img, pad_width=((h, h), (w,w)), mode='constant')
I = I.astype('float64')

H, W = I.shape

h = np.cast['int']((h - 1.) / 2.)
w = np.cast['int']((w - 1.) / 2.)

J = np.zeros_like(I)
for i in np.arange(h, H - h, 1):
    for j in np.arange(w, W - w, 1):
        for m in np.arange(-h, h+1, 1):
            for n in np.arange(-w, w+1, 1):
                J[i, j] += I[i + m, j + n] * K[m + h, n + w]


fig = plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.imshow(I, cmap='gray')
plt.title("Original Image")

plt.subplot(122)
plt.imshow(J.astype('float64'), cmap='gray')
plt.title("Filtered Image")
plt.show()
