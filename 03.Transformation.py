import cv2
import numpy as np
import matplotlib.pyplot as plt
# from plantcv import plantcv as pcv
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import sys

def plot_image(image):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Original
    # Gaussian Blur
    
    # Mask
    
    # Roi Objects
    
    # Analyze Objects

    # Pseudolandmark

def rgb2gray_lab(image, channel: int, thresh: float, type: int):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    binary_img = lab_image[:, :, channel]
    _, binary_image = cv2.threshold(binary_img, thresh, 255, type)
    return binary_image

def rgb2gray_hsv(image, channel: int, thresh: float, type: int):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    binary_img = hsv_image[:, :, channel]
    _, binary_image = cv2.threshold(binary_img, thresh, 255, type)
    return binary_image

def create_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([35, 40, 40])
    upper_bound = np.array([85, 255, 255])

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

def _gaussian_blur(image):
    binary_image = rgb2gray_hsv(image, 1, 58, cv2.THRESH_BINARY)
    gaussian_blur_img = cv2.GaussianBlur(binary_image, (5, 5), 0)
    return gaussian_blur_img

def transform_one_image(image):
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Gaussian Blur
    gaussian_blur_img = _gaussian_blur(image) 
    axes[0, 1].imshow(cv2.cvtColor(gaussian_blur_img, cv2.COLOR_RGB2BGR))
    axes[0, 1].set_title("Gaussian Blur")
    axes[0, 1].axis("off")

    # Mask
    mask = create_mask(image)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = [255, 255, 255]
    axes[1, 0].imshow(result)
    axes[1, 0].set_title("Mask")
    axes[1, 0].axis("off")

    # ROI
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), cv2.FILLED)
    axes[1, 1].imshow(cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
    axes[1, 1].set_title("Roi Objects")
    axes[1, 1].axis("off")

    # Analysis Object
    analysis_image = image.copy()

    # Find the largest contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the largest contour
        cv2.drawContours(analysis_image, largest_contour, -1, (255, 0, 255), 5)

        ellipse = cv2.fitEllipse(largest_contour)
        # print(len(ellipse), ellipse)
        center = tuple(map(int, ellipse[0]))
        axes_length = tuple(map(int, ellipse[1]))
        angle = ellipse[2]
        print("center:", center)
        print("axes_length:", axes_length)
        print("angle:", angle, "\n")
        # Draw the center point
        cv2.circle(analysis_image, center, 8, (255, 0, 255), 5)

        # cv2.circle(analysis_image, axes_length, 8, (255, 0, 0), 5)
        # cv2.ellipse(analysis_image, ellipse, (255, 0, 0), 3)

        rmajor = max(axes_length)/2
        # print(rmajor)
        if angle > 90:
            angle -= 90
        else:
            angle += 90

        # Draw the midrib line (mayor axis of the ellipse)
        midrib_start = (
            int(center[0] + np.cos(np.deg2rad(angle)) * rmajor),
            int(center[1] + np.sin(np.deg2rad(angle)) * rmajor)
        )

        midrib_end = (
            int(center[0] + np.cos(np.deg2rad(angle + 180)) * rmajor),
            int(center[1] + np.sin(np.deg2rad(angle + 180)) * rmajor)
        )

        # cv2.circle(analysis_image, midrib_start, 8, (255, 0, 0), 5)
        # print("midrib_start:", midrib_start)
        # cv2.circle(analysis_image, midrib_end, 8, (255, 0, 0), 5)
        # print("midrib_end:", midrib_end)

        cv2.line(analysis_image, midrib_start, midrib_end, (255, 0, 255), 5)

    axes[2, 0].imshow(cv2.cvtColor(analysis_image, cv2.COLOR_RGB2BGR))
    axes[2, 0].set_title("Analyze Object")
    axes[2, 0].axis("off")

    # Pseudolandmark
    pseudolandmark_img = image.copy()

    for contour in contours:
        # print(contour, contour.shape, len(contour))
        step = max(1, len(contour)//10)
        for point in contour[::step]:
            cv2.circle(pseudolandmark_img, tuple(point[0]), 1, (0, 0, 255), 3)
    axes[2, 1].imshow(cv2.cvtColor(pseudolandmark_img, cv2.COLOR_RGB2BGR))
    axes[2, 1].set_title("Pseudolandmark")
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig('test.png')
    # plt.show()

def main():
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [path_to_image]")
    elif len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
        # print(sys.argv[1], image)
        transform_one_image(image)

if __name__ == "__main__":
    main()