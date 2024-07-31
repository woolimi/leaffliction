import os
import zipfile
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import argparse
import numpy as np
from lib.images import analyze, IMAGE_FOLDER, IMAGE_URL


def download_and_extract_zip(url, extract_to='.'):
    """
    Download and extract a ZIP file from a given URL.
    """
    print("Downloading and extracting images...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)
    print("Download and extraction complete.")




def plot_pie_chart_and_bar_chart(data, title):
    """
    Plot a pie chart and a bar chart side by side
    for the given data with consistent colors.
    """

    # Get a colormap from matplotlib
    colormap = plt.get_cmap("tab20")

    # Sample the colormap to get enough colors
    colors = colormap(np.linspace(0, 1, len(data)))

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Pie chart
    axs[0].pie(data.values(), autopct='%1.1f%%', startangle=140, colors=colors)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    axs[0].axis('equal')

    # Bar chart
    axs[1].bar(data.keys(), data.values(), color=colors)
    axs[1].tick_params(axis='x', labelright=False)
    axs[1].set_axisbelow(True)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Add a main title to the figure at the top left corner
    fig.suptitle(f'{title} class distribution', fontsize=16, ha='left', x=0.1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the combined chart as a PNG file
    title = title.lower()
    plt.savefig(f'{title}_combined_chart.png')
    plt.close()
    print(f'Combined chart saved as {title}_combined_chart.png ✨')


def main(directories):
    """
    Main function to process images and create charts.
    """

    # Ensure the directories exists
    if not os.path.exists(IMAGE_FOLDER):
        download_and_extract_zip(IMAGE_URL, './')
    else:
        print("directories already exists, skip downloading...")

    # Analyze images
    for directory in directories:
        plant_counts = analyze(directory)

        if not plant_counts:
            print(f"No class found in the '{directory}' directories.")
            return

        # Generate charts
        plot_pie_chart_and_bar_chart(plant_counts, os.path.basename(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to analyze plant images and generate charts."
    )

    # Adding argument for the directories
    parser.add_argument(
        'directories',
        type=str,
        help='The directories to store extracted images and save the charts \
        (ex: 01.Distribution apple)',
        nargs='+',
    )

    # Parsing the arguments
    args = parser.parse_args()

    # URL of the ZIP file containing images    
    main(args.directories)
