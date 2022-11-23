import numpy as np
import cv2
import matplotlib.pyplot as plt


#discrete fourier transform 1d implementation
def dft(img, inv=False):
    img = np.array(img)
    if (len(img.shape) == 1):
        img = img.reshape(len(img), 1)
    m, n = img.shape[0], img.shape[1]
    X = np.zeros((m, n), dtype=np.complex_)
    exp_cons = -2j*np.pi
    if inv:
        exp_cons = -exp_cons
    for u in range(m):
        for v in range(n):
            X[u, v] = 0
            for x in range(m):
                for y in range(n):
                    X[u, v] += img[x, y] * \
                        np.exp(exp_cons * (x*u/m + y*v/n))

    return X/m*n if inv else X

#fast fourier transform 1d implementation
def fft(x):
    x = np.asarray(x, dtype=np.complex_)
    N = x.shape[0]

    if N % 2 > 0:
        print("Powers of 2 allowed only for input size.")
        return
    elif N <= 2:
        return dft(x).flatten()
    else:
        X_even, X_odd, factor = fft(x[::2]), fft(
            x[1::2]), np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N / 2)] * X_odd, X_even + factor[int(N / 2):] * X_odd])

img = cv2.imread('img.jpg', 0)

#fast fourier transform 2d implementation
def fft2(img):
    fourier = [[] for _ in range(img.shape[0])]
    for i in range(img.shape[0]):
        fourier[i] = fft(img[i])
    
    fourier = np.array(fourier)
    for i in range(fourier.shape[1]):
        fourier[:,i] = fft(fourier[:,i])
    
    return fourier

fourier_domain = fft2(img)
plt.imshow(np.real(np.log(np.fft.fftshift(fourier_domain))), cmap='gray')

#comparing with the function from numpy
plt.imshow(np.real(np.log((np.fft.fftshift(np.fft.fft2(img))))), cmap='gray')
