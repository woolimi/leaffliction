import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import sys
import os

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

def transform_one_image(image, filepath):
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    axes[0, 0].set_title("Original Image")

    # Gaussian Blur
    gaussian_blur_img = _gaussian_blur(image) 
    axes[0, 1].imshow(cv2.cvtColor(gaussian_blur_img, cv2.COLOR_RGB2BGR))
    axes[0, 1].set_title("Gaussian Blur")

    # Mask
    mask = create_mask(image)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask == 0] = [255, 255, 255]
    axes[1, 0].imshow(result)
    axes[1, 0].set_title("Mask")

    # ROI
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), cv2.FILLED)
    axes[1, 1].imshow(cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
    axes[1, 1].set_title("Roi Objects")

    # Analysis Object
    analysis_image = pcv.analyze.size(img=image, labeled_mask=mask, n_labels=1)

    axes[2, 0].imshow(cv2.cvtColor(analysis_image, cv2.COLOR_RGB2BGR))
    axes[2, 0].set_title("Analyze Object")

    # Pseudolandmark
    output_path = os.path.join(pcv.params.debug_outdir, (str(pcv.params.device) + "_x_axis_pseudolandmarks.png"))
    input_path = os.path.join(pcv.params.debug_outdir, "input_image.png")
    pcv.params.debug = "print"
    pcv.homology.x_axis_pseudolandmarks(img=image, mask=mask)
    pcv.params.debug = None
    img = pcv.readimage(output_path)
    # print(img[0].shape)
    if os.path.exists(input_path) and os.path.isfile(input_path):
        os.remove(input_path)
    if os.path.exists(output_path) and os.path.isfile(output_path):
        os.remove(output_path)

    axes[2, 1].imshow(img[0])
    axes[2, 1].set_title("Pseudolandmark")

    dir_name = os.path.normpath(filepath).split(os.sep)[1]
    filename = os.path.splitext(os.path.basename(filepath))[0]
    image_num = filename.split('(')[-1].split(')')[0]

    plt.tight_layout()
    plt.savefig(f"{dir_name}_{image_num}.png")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [path_to_image]")
    elif len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
        # print(sys.argv[1])
        transform_one_image(image, sys.argv[1])

if __name__ == "__main__":
    main()