from PIL import Image
import numpy as np

def is_flag_of_indonesia_or_poland(image_path):

    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    image = image.resize((400, 300))  # Resize for consistency and performance

    # Convert image to NumPy array
    pixels = np.array(image)

    # Thresholds for detecting red and white pixels
    def is_red(pixel):
        r, g, b = pixel
        return r > 150 and g < 100 and b < 100  # Strong red dominance

    def is_white(pixel):
        r, g, b = pixel
        return r > 200 and g > 200 and b > 200  # Near equal strong RGB values

    # Mask to detect red and white regions
    height, width, _ = pixels.shape
    red_mask = np.zeros((height, width), dtype=bool)
    white_mask = np.zeros((height, width), dtype=bool)

    for i in range(height):
        for j in range(width):
            if is_red(pixels[i, j]):
                red_mask[i, j] = True
            elif is_white(pixels[i, j]):
                white_mask[i, j] = True

    # Combine red and white masks to locate the flag region
    combined_mask = red_mask | white_mask

    # Find the bounding box of the flag
    rows, cols = np.where(combined_mask)
    if len(rows) == 0 or len(cols) == 0:
        return "Neither"  # No red or white detected

    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)

    # Crop the detected flag region
    cropped_flag = image.crop((left, top, right, bottom))
    cropped_flag = cropped_flag.resize((200, 100))  # Standardize the size

    # Convert cropped flag to a NumPy array
    flag_pixels = np.array(cropped_flag)

    # Count red and white pixels in top and bottom halves
    mid = flag_pixels.shape[0] // 2
    top_half = flag_pixels[:mid, :, :]
    bottom_half = flag_pixels[mid:, :, :]

    def count_colors(section):
        red_count = 0
        white_count = 0
        for row in section:
            for pixel in row:
                if is_red(pixel):
                    red_count += 1
                elif is_white(pixel):
                    white_count += 1
        return red_count, white_count

    # Analyze top and bottom halves
    red_top, white_top = count_colors(top_half)
    red_bottom, white_bottom = count_colors(bottom_half)

    # Determine flag type
    if red_top > white_top and white_bottom > red_bottom:
        return "Indonesia"
    elif white_top > red_top and red_bottom > white_bottom:
        return "Poland"
    else:
        return "Neither"

result = is_flag_of_indonesia_or_poland(r'/Users/anuragsharma/Downloads/Unknown')
print(result)
