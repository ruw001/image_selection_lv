import numpy as np
import pandas as pd
import os
import requests
import ollama
import shutil
import base64
from PIL import Image
import io
import time
# get all images from product_image_urls.csv from https://github.com/Crossing-Minds/shopping-queries-image-dataset

# print the headers of the csv file
def get_header(cvs_file):
    df = pd.read_csv(cvs_file)
    print(df.columns.tolist())

def get_image_urls(cvs_file):
    df = pd.read_csv(cvs_file)
    image_urls = df['image_url'].tolist()
    # remove nan values
    image_urls = [url for url in image_urls if pd.notna(url)]
    # number of images
    print(f'Number of images: {len(image_urls)}')

# get images with multiple salient objects using vision language model
def get_images_with_multiple_salient_objects(csv_file):
    df = pd.read_csv(csv_file)
    image_urls = df['image_url'].tolist()
    image_urls = [url for url in image_urls if pd.notna(url)]
    # remove duplicate image urls
    image_urls = list(set(image_urls))
    print(f'Number of images: {len(image_urls)}')
    # shuffle the image urls
    np.random.shuffle(image_urls)
    cnt = 0
    cat_control = {
        'clothes': 0,
        'foods': 0,
        'household': 0,
        'electronics': 0,
        'health': 0,
        'entertainment': 0,
        'other': 0
    }
    cat_limit = 400
    os.makedirs('images_with_multiple_salient_objects', exist_ok=True)
    # get images already in the selected folder
    existing_images = os.listdir('images_with_multiple_salient_objects_selected')
    existing_images = ['_'.join(img.split('_')[1:]) for img in existing_images if 'jpg' in img]

    # take a rest after every 100 images
    rest_interval = 1000
    for idx, url in enumerate(image_urls):
        # if all categories have reached the limit, stop
        if all(cat_control[cat] >= cat_limit for cat in cat_control):
            print(f'All categories have reached the limit, stopping')
            break
        print(f'Processing image {idx} / {len(image_urls)}')
        # download the image and save it to a temporary file temp.xxx
        print(f'Downloading image {url}')
        img_name = url.split('/')[-1]
        if img_name in existing_images:
            print(f'Image {img_name} already exists, skipping...')
            continue
        response = requests.get(url)
        # save the image as a Image object
        image_data = response.content
        im = Image.open(io.BytesIO(image_data))
        # if the image aspect ratio (width / height) is less than 1, ignore it
        if im.width / im.height < 1:
            print(f'Image aspect ratio is less than 1, ignoring it: {im.width} / {im.height}')
            continue
        prompt = f"Does the attached image include more than one salient objects (e.g., person, animal, car, bag, glasses, phone, etc.)? \n\n" \
            + "Your response should be two lines: on the first line, respond only 'true' or 'false'. Do not include any other text in this line." \
            + "On the second line, respond one word (choosing from 'clothes', 'foods', 'household', 'electronics', 'health', 'entertainment', 'other') indicating which category the main product on the image is about. "
        
        resp = ollama.generate(
            model="qwen2.5vl:latest",
            prompt=prompt,
            images=[image_data],  # Pass the raw image data instead
        )
        response = resp['response'].lower()
        multi_obj, cat = response.split('\n')
        
        if multi_obj == 'true':
            print(f'This image has multiple salient objects -> about to save!!!')
            if cat not in cat_control:
                print(f'Invalid category: {cat}, skipping...')
                continue
            if cat_control[cat] >= cat_limit:
                print(f'Reached the limit for category {cat}, skipping...')
                continue
            cat_control[cat] += 1
            # move the image to the images_with_multiple_salient_objects folder
            im.save(f'images_with_multiple_salient_objects/{cnt:03d}_{img_name}')
            print(f'Saved image {cnt:03d}_{img_name}')
            cnt += 1
        else:
            print(f'this image does not have multiple salient objects')
        if idx % rest_interval == 0:
            print(f'Taking a rest after {idx} images')
            time.sleep(2)


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

def add_source_prefix(img_dir):
    for img_path in os.listdir(img_dir):
        new_img_path = 'ecommerce_' + img_path
        new_img_path = os.path.join(img_dir, new_img_path)
        os.rename(os.path.join(img_dir, img_path), new_img_path)
    return
    
# download_images('product_image_urls.csv', 400)
# get_images_with_multiple_salient_objects('product_image_urls.csv')
add_source_prefix('images_with_multiple_salient_objects')