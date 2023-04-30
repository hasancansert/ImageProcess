import cv2
import numpy as np
from matplotlib import pyplot as plt

#This function calculates the dark channel prior of an image.
#It takes the minimum intensity value of the three color channels (R, G, B) and then 
#erodes the result using a rectangular structuring element.
def dark_channel_prior(img, window_size=15):
    min_channel = np.amin(img, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size)))
    return dark_channel
''' This function estimates the atmospheric light (A) in the image.
It selects the brightest pixels in the dark channel and takes 
the average intensity of these pixels in the original image.'''
def atmospheric_light(img, dark_channel, percentile=0.001):
    total_pixels = img.shape[0] * img.shape[1]
    num_pixels = int(total_pixels * percentile)
    flat_dark_channel = dark_channel.ravel()
    indices = np.argpartition(flat_dark_channel, -num_pixels)[-num_pixels:]
    brightest_pixels = img.reshape(total_pixels, 3)[indices]
    A = np.average(brightest_pixels, axis=0)
    return A
''' This function computes the transmission map (t) of the image.
It normalizes the input image by dividing it by the estimated atmospheric light
(A) and then calculates the dark channel prior of the normalized image.'''
def transmission_map(img, A, omega=0.95, window_size=15):
    norm_img = img / A
    t = 1 - omega * dark_channel_prior(norm_img, window_size)
    return t
''' This function implements the guided filter algorithm as described in the paper
 "Guided Image Filtering" by Kaiming He et al.
It takes the guidance image (I), the input image (p), the radius (r), and a regularization parameter (eps).'''
def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q
''' This function refines the transmission map using the guided filter.
It first converts the input image to grayscale and normalizes it, 
then applies the guided filter with the given radius and regularization parameter.'''

def refine_transmission_map(img, t, radius=60, eps=0.0001):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64) / 255
    t = guided_filter(gray, t, radius, eps)
    return t
''' This function recovers the dehazed image from the input image, the estimated atmospheric light, 
and the refined transmission map.It clamps the transmission map to a minimum value (t0) and then calculates the
dehazed image using the formula: J = (I - A) / t_clamped + A.'''

def recover_image(img, A, t, t0=0.1):
    t_clamped = np.maximum(t, t0)
    J = (img - A) / np.expand_dims(t_clamped, axis=-1) + A
    J = np.clip(J, 0, 255)
    return J.astype(np.uint8)

def apply_adaptive_histogram_equalization(img):
    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    # Apply adaptive histogram equalization to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    # Merge the equalized L channel with the original A and B channels
    lab_eq = cv2.merge((l_eq, a, b))
    # Convert the LAB equalized image back to BGR color space
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return img_eq


image = cv2.imread("haze.png")
#Read the input image and calculate the dark channel prior.
dark_channel = dark_channel_prior(image)
#Estimate the atmospheric light using the dark channel prior.
A = atmospheric_light(image, dark_channel)
#Calculate the transmission map using the estimated atmospheric light.
transmission = transmission_map(image, A)
#Refine the transmission map using the guided filter.
refined_transmission = refine_transmission_map(image, transmission)
#Recover the dehazed image using the input image, the estimated atmospheric light, and the refined transmission map.
dehazed_image = recover_image(image, A, refined_transmission)

# Apply adaptive histogram equalization to the dehazed image
dehazed_image_eq = apply_adaptive_histogram_equalization(dehazed_image)

#Display the original and dehazed images using matplotlib.
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
plt.subplot(122), plt.imshow(cv2.cvtColor(dehazed_image_eq, cv2.COLOR_BGR2RGB)), plt.title("Dehazed Image")
plt.show()

cv2.imwrite("dehazed_image.png", dehazed_image_eq)

#Alternatively, display the original and dehazed images using OpenCV's imshow function.

cv2.imshow('OI', image)
cv2.imshow('DI', dehazed_image_eq)
while True:
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == 27:
        cv2.destroyAllWindows()
        break