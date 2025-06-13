import numpy as np
import pandas as pd
import os
import requests

# get all images from product_image_urls.csv from https://github.com/Crossing-Minds/shopping-queries-image-dataset

# print the headers of the csv file
def get_header(cvs_file):
    df = pd.read_csv(cvs_file)
    print(df.columns.tolist())

def get_image_urls(cvs_file):
    df = pd.read_csv(cvs_file)
    image_urls = df['image_url'].tolist()
    # number of images
    print(f'Number of images: {len(image_urls)}')

def download_images(cvs_file, num_selected, output_dir='data'):
    df = pd.read_csv(cvs_file)
    num_images = len(df)
    print(f'Number of images: {num_images}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Select a random sample of images
    selected_indices = np.random.choice(num_images, num_selected, replace=False)
    selected_urls = df['image_url'].iloc[selected_indices].tolist()
    print(f'Selected {num_selected} images from {num_images} total images.')
    for i, url in enumerate(selected_urls):
        print('url', url)
        try:
            # Download the image
            response = requests.get(url)
            if response.status_code == 200:
                image_data = response.content
                with open(f'{output_dir}/{i}.jpg', 'wb') as f:
                    f.write(image_data)
            else:
                print(f'Failed to download image {url}')
        except Exception as e:
            print(f'Error downloading image {url}: {e}')

    
download_images('product_image_urls.csv', 400)
