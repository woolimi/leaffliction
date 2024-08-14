import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
# import matplotlib
# matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import sys
import os

class Transformation():
    def __init__(self, img):
        self.img = img
        self._create_mask()

    def _create_mask(self):
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([35, 40, 40])
        upper_bound = np.array([85, 255, 255])

        self.mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        return self.mask

    def _grayscale_lab(self, channel: int, thresh: float, type: int):
        lab_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)
        binary_img = lab_image[:, :, channel]
        _, binary_image = cv2.threshold(binary_img, thresh, 255, type)
        return binary_image

    def _grayscale_hsv(self, channel: int, thresh: float, type: int):
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        binary_img = hsv_image[:, :, channel]
        _, binary_image = cv2.threshold(binary_img, thresh, 255, type)
        return binary_image

    def _transform_and_save(self, file_path, transforms, transform_type):
        img = transforms()
        filename, file_extension = os.path.splitext(file_path)
        try:
            pcv.params.debug = "print"
            pcv.print_image(img, f"{filename}_{transform_type}{file_extension}")
            pcv.params.debug = None
        except Exception as e:
            print(e)

    def transform_gaussian_blur(self):
        """
        Gaussian Blur
        """

        binary_image = self._grayscale_hsv(1, 58, cv2.THRESH_BINARY)
        gaussian_blur_img = cv2.GaussianBlur(binary_image, (5, 5), 0)
        return gaussian_blur_img

    def transform_mask(self):
        """
        Mask
        """

        result = cv2.bitwise_and(self.img, self.img, mask=self.mask)
        result[self.mask == 0] = [255, 255, 255]
        return result

    def transform_roi_objects(self):
        """
        Roi Objects
        """

        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        roi_image = self.img.copy()
        cv2.drawContours(roi_image, contours, -1, (0, 255, 0), cv2.FILLED)
        return roi_image

    def transform_analyze_object(self):
        """
        Anaylze Object
        """

        objects, object_hierarchy = pcv.find_objects(self.img, self.mask)
        obj, self.mask = pcv.object_composition(self.img, objects, object_hierarchy)
        analyze_image = pcv.analyze_object(self.img, obj, self.mask)
        return analyze_image

    def transform_pseudolandmarks(self):
        """
        Pseudolandmarks
        """

        objects, object_hierarchy = pcv.find_objects(self.img, self.mask)
        obj, self.mask = pcv.object_composition(self.img, objects, object_hierarchy)

        output_path = os.path.join(pcv.params.debug_outdir, (str(pcv.params.device) + "_y_axis_pseudolandmarks.png"))
        input_path = os.path.join(pcv.params.debug_outdir, "input_image.png")

        pcv.params.debug = "print"
        pcv.y_axis_pseudolandmarks(img=self.img, obj=obj, mask=self.mask)
        pcv.params.debug = None
        img = pcv.readimage(output_path)

        # print(img[0].shape)
        if os.path.exists(input_path) and os.path.isfile(input_path):
            os.remove(input_path)
        if os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)
        return img[0]

    def transform_color_histogram(self):
        """
        Color Histogram
        """

        output_path = os.path.join(pcv.params.debug_outdir, (str(pcv.params.device) + "_analyze_color_hist.png"))
        input_path = os.path.join(pcv.params.debug_outdir, "input_image.png")

        pcv.params.debug = None
        color_channels = pcv.analyze_color(self.img, mask=self.mask, colorspaces="all", label="plant")
        pcv.print_image(color_channels, output_path)
        img = pcv.readimage(output_path)
        # print(output_path)

        if os.path.exists(input_path) and os.path.isfile(input_path):
            os.remove(input_path)
        if os.path.exists(output_path) and os.path.isfile(output_path):
            os.remove(output_path)
        return img[0]

    def plot_transformation_image(self):
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))
        axes[0, 0].set_title("Original Image")

        # Gaussian Blur
        gaussian_blur_img = self.transform_gaussian_blur() 
        axes[0, 1].imshow(cv2.cvtColor(gaussian_blur_img, cv2.COLOR_RGB2BGR))
        axes[0, 1].set_title("Gaussian Blur")

        # Mask
        masked_img = self.transform_mask()
        axes[1, 0].imshow(masked_img)
        axes[1, 0].set_title("Mask")

        # ROI
        roi_image = self.transform_roi_objects()
        axes[1, 1].imshow(cv2.cvtColor(roi_image, cv2.COLOR_RGB2BGR))
        axes[1, 1].set_title("Roi Objects")

        # Analysis Object
        analyze_image = self.transform_analyze_object()
        axes[2, 0].imshow(cv2.cvtColor(analyze_image, cv2.COLOR_RGB2BGR))
        axes[2, 0].set_title("Analyze Object")

        # Pseudolandmark
        img = self.transform_pseudolandmarks()
        axes[2, 1].imshow(img)
        axes[2, 1].set_title("Pseudolandmark")

        plt.tight_layout()
        plt.show()

        # Color histogram
        img = self.transform_color_histogram()

        plt.imshow(img)
        plt.tight_layout()
        plt.show()

    def transform_one_image(self, file_path):
        transforms = [self.transform_gaussian_blur, self.transform_mask, self.transform_roi_objects, self.transform_analyze_object, self.transform_pseudolandmarks]
        for transform in transforms:
            self._transform_and_save(file_path, transform, transform.__name__[10:])

def main(file_path):
    try:
        if (file_path):
            image = cv2.imread(sys.argv[1])
            Transformer = Transformation(image)
            # Transformer.plot_transformation_image()
            Transformer.transform_one_image(sys.argv[1])
        else:
            print(f"Usage: {sys.argv[0]} [path_to_image]")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to display image transformation."
    )

    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        help="Image file path."
    )

    args = parser.parse_args()
    main(args.file_path)