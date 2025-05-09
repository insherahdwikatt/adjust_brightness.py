import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('watermarked_image.jpeg')


if image is None:
    raise Exception("Could not load image!")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


print("Image dimensions:", gray_image.shape)
print("Mean pixel value:", np.mean(gray_image))
print("Max pixel value:", np.max(gray_image))
print("Min pixel value:", np.min(gray_image))
print("Median:", np.median(gray_image))
print("Standard Deviation:", np.std(gray_image))


c = np.random.uniform(0.4, 2.0)
bright_image = np.clip(gray_image * c, 0, 255).astype(np.uint8)


plt.hist(bright_image.ravel(), 256, [0,256])
plt.title("Histogram of Brightened Image")
plt.show()


corrected_image = cv2.equalizeHist(bright_image)


fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(bright_image.ravel(), 256, [0,256])
axs[0].set_title("Brightened Image Histogram")
axs[1].hist(corrected_image.ravel(), 256, [0,256])
axs[1].set_title("Corrected Image Histogram")
plt.show()


def salt_pepper_noise(img, amount=0.05):
    noisy_img = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)

    # Add Salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 255

    # Add Pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0

    return noisy_img

noisy_image = salt_pepper_noise(corrected_image)

# Reduce noise using mean filter
mean_filtered = cv2.blur(noisy_image, (3, 3))

# Reduce noise using median filter
median_filtered = cv2.medianBlur(noisy_image, 3)

# Display comparison of filtering
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(noisy_image, cmap='gray')
axs[0].set_title('Noisy Image')
axs[0].axis('off')

axs[1].imshow(mean_filtered, cmap='gray')
axs[1].set_title('Mean Filtered')
axs[1].axis('off')

axs[2].imshow(median_filtered, cmap='gray')
axs[2].set_title('Median Filtered')
axs[2].axis('off')

plt.show()