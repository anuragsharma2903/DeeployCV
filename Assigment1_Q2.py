import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function to display images
def display_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load your two images
# Replace 'image1.jpg' and 'image2.jpg' with the paths to your images
img1 = cv2.imread(r'/Users/anuragsharma/Downloads/Photo on 18-12-24 at 6.52pm.jpg', cv2.IMREAD_GRAYSCALE)  # For high-pass filtering
img2 = cv2.imread(r'/Users/anuragsharma/Downloads/Photo on 18-12-24 at 6.53pm.jpg', cv2.IMREAD_GRAYSCALE)  # For low-pass filtering

# Ensure both images are of the same size
img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

# Apply a low-pass filter (Gaussian Blur) to img2
low_pass = cv2.GaussianBlur(img2, (25, 25), 0)

# Apply a high-pass filter to img1
low_freq_img1 = cv2.GaussianBlur(img1, (25, 25), 0)
high_pass = cv2.subtract(img1, low_freq_img1)

# Combine high-pass and low-pass images
hybrid_image = cv2.add(high_pass, low_pass)

# Display the results
imgs = [img1, high_pass, img2, low_pass, hybrid_image]
titles = ['Original Image 1', 'High-Pass Filtered', 'Original Image 2', 'Low-Pass Filtered', 'Hybrid Image']

display_images(imgs, titles)
