import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function 1: Capture an image from the camera
def capture_the_image():
    cap = cv2.VideoCapture(0)  # Try 0, 1, etc., depending on your camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness (experiment with values)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    ret, frame = cap.read()  # Capture a single frame
    cap.release()
    if ret:
        return frame
    else:
        print("Error: Could not capture the image.")
        return None

# Function 2: Grey Scaling
def grayscaling_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function 3: Thresholding (Binary - Black and White)
def threshold_image(image, threshold_value=128):
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

# Function 4: Containing 16 Grey Colors (Dividing 0-255 into 16 regions)
def reducing_greyscale(image, levels=16):
    step = 256 // levels
    reduced = (image // step) * step
    return reduced

# Function 5: Sobel Filter (Edge Detection)
def sobel_filter_edge(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel_combined)

# Function 6: Canny Edge Detection
def canny_edgedetect(image):
    return cv2.Canny(image, 100, 200)

# Function 7: Gaussian Blur (Noise Reduction)
def Gaussian_Blur(image, kernel_size=5):
    kernel = cv2.getGaussianKernel(kernel_size, sigma=-1)
    gaussian = np.outer(kernel, kernel)  # Create a 2D Gaussian kernel
    blurred = cv2.filter2D(image, -1, gaussian)
    return blurred

# Function 8: Sharpen the Blurred Image
def sharpen_image(image):
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    return cv2.filter2D(image, -1, sharpening_kernel)

# Function 9: Convert RGB to BGR
def RGB_to_BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Main function to perform all tasks and display results
def main():
    # Capture the image
    frame = capture_the_image()
    if frame is None:
        return

    # Convert to grayscale
    gray = grayscaling_image(frame)

    # Thresholding
    thresholded = threshold_image(gray)

    # 16 grey levels
    grey_16 = reducing_greyscale(gray, levels=16)

    # Sobel and Canny edge detection
    sobel = sobel_filter_edge(gray)
    canny = canny_edgedetect(gray)

    # Gaussian blur and sharpening
    blurred = Gaussian_Blur(gray)
    sharpened = sharpen_image(blurred)

    # Convert RGB to BGR
    rgb_bgr = RGB_to_BGR(frame)

    # Display all results in a 2x4 grid using Matplotlib
    titles = ['Original Image', 'Grayscale', 'Thresholded (BW)', '16 Grey Colors',
              'Sobel Edge', 'Canny Edge', 'Gaussian Blurred', 'Sharpened Image']
    images = [frame, gray, thresholded, grey_16, sobel, canny, blurred, sharpened]

    plt.figure(figsize=(15, 8))
    for k in range(8):
        plt.subplot(2, 4, k + 1)
        if k == 0:
            plt.imshow(cv2.cvtColor(images[k], cv2.COLOR_BGR2RGB))  # Convert original BGR to RGB
        else:
            plt.imshow(images[k], cmap='gray')
        plt.title(titles[k])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
