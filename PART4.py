from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from matplotlib.pyplot import quiver

I = io.imread("ex2.jpg")

fig = plt.figure()
plt.imshow(I, cmap='gray', vmin=0, vmax=255)
plt.title("Cameraman")
plt.show()

gh = 0.5*np.array([[-1.0,0.0,1.0]])
gv = 0.5*np.array([[-1.0,0.0,1.0]]).T
print(gh.shape,gv.shape)

Ix = convolve2d(I, gh, mode='same', boundary='symm')
Iy = convolve2d(I, gv, mode='same', boundary='symm')


plt.subplot(121)
io.imshow((Ix).astype(int), cmap='gray')
plt.title("X derivative")
plt.subplot(122)
io.imshow((Iy).astype(int), cmap='gray')
plt.title("Y derivative")
plt.show()