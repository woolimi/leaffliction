import os
import zipfile
import matplotlib.pyplot as plt
from collections import Counter
import requests
from io import BytesIO
import argparse

def download_and_extract_zip(url, extract_to='.'):
    """
    Download and extract a ZIP file from a given URL.
    """
    print("Downloading and extracting images...")
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)
    print("Download and extraction complete.")

def analyze_images(directory):
    """
    Analyze images in the directory and return a dictionary of counts per plant type.
    """
    plant_counts = Counter()

    # Traverse through the directory and count the number of images in each subdirectory
    for root, dirs, files in os.walk(directory):
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            num_images = len([file for file in os.listdir(dir_path) if file.endswith(('.png', '.jpg', '.jpeg'))])
            plant_counts[dirname] = num_images

    return plant_counts

def plot_pie_chart(data, title):
    """
    Plot a pie chart for the given data.
    """
    plt.figure(figsize=(8, 8))
    plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart of {title}')
    plt.axis('equal')
    plt.savefig(f'{title}_pie_chart.png')
    plt.close()
    print(f'Pie chart saved as {title}_pie_chart.png')

def plot_bar_chart(data, title):
    """
    Plot a bar chart for the given data.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Bar Chart of {title}')
    plt.xlabel('Plant Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{title}_bar_chart.png')
    plt.close()
    print(f'Bar chart saved as {title}_bar_chart.png')

def main(directory, url):
    """
    Main function to process images and create charts.
    """
    download_and_extract_zip(url, directory)
    
    # Analyze images
    plant_counts = analyze_images(directory)
    
    # Generate charts
    plot_pie_chart(plant_counts, os.path.basename(directory))
    plot_bar_chart(plant_counts, os.path.basename(directory))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to analyze plant images and generate charts.")

    # Adding argument for the directory
    parser.add_argument(
        '--directory', 
        type=str,
        default='images', 
        help='The directory to store extracted images and save the charts (default: images)'
    )

    # Parsing the arguments
    args = parser.parse_args()

    # Ensure the directory exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # URL of the ZIP file containing images
    url = "https://cdn.intra.42.fr/document/document/17547/leaves.zip"
    main(args.directory, url)
