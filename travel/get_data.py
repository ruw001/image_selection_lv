import numpy as np
import pandas as pd
import os
import requests
import ollama
import re
import shutil

# get all images from photos.csv000 from https://github.com/Crossing-Minds/shopping-queries-image-dataset

# print the headers of the csv file
def get_header(cvs_file):
    df = pd.read_csv(cvs_file, sep='\t')
    print(df.columns.tolist())

def get_images(cvs_file):
    df = pd.read_csv(cvs_file, sep='\t')
    # sort the rowsbased on the word count of column ai_description, show full description
    df = df.sort_values(by='ai_description', key=lambda x: x.str.split().str.len(), ascending=False)
    # get images with aspect ratio >= 1
    df = df[df['photo_aspect_ratio'] >= 1]

    print(f"total images: {len(df)}")

    # get the top 1000 images
    df = df.head(1000)

    outpath = 'images'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for index, row in df.iterrows():
        desc = row['ai_description']
        # use agent to determine whether the description involves at least 2 salient objects, return a boolean
        print(f"prompting agent with desc: {desc}")
        resp = ollama.generate(
            model="deepseek-r1:latest",
            prompt=f"Does the following description for an image explicitly mention at least 2 (>= 2) salient objects (e.g., people, animals, objects, etc) on the foreground? \n \"{desc}\" \n\n "
            "Note that mountains, rivers, very small objects, etc. are not salient objects.\n\n"
            "Return only 'true' or 'false'. Do not include any other text in your response. Here is an example: \n"
            "For a description 'selective focus photography of short-coated white and brown dog on fallen brown leaves during daytime', the dog is a salient object, but the leaves are not. There is only one salient object. You should return 'false'."

            # prompt=f"Does the following description for an image involve at least 2 people or animals on the foreground? {desc} \n\n "
            # "Return only 'true' or 'false'. Do not include any other text in your response."

            # prompt=f"Does the following description for an image involve at least 2 buildings (or other man-made structures) on the foreground? {desc} \n\n "
            # "Return only 'true' or 'false'. Do not include any other text in your response."
        )
        # print(resp)
        name = row['photo_id']
        if resp['response'].lower() == "true":
            print(f">>>> {name} is a good image")
        else:
            print(f"<<<< {name} is a bad image")
            continue
        
        print(f"saving image {name}...")
        url = row['photo_image_url']
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{outpath}/{name}.jpg', 'wb') as f:
                f.write(response.content)
        else:
            print(f"!!! Failed to download image from {url}")
        
def extract_pixabay_link(filepath):
    folders = os.listdir(filepath)
    for fd in folders:
        if not os.path.isdir(os.path.join(filepath, fd)):
            continue
        imgs = os.listdir(os.path.join(filepath, fd))
        # Process images in each folder
        # check if an image is a pattern xxx-{idx}_yyy.jpg, if so, get the idx, xxx can be anything not just digits, yyy is a number
        print(f"processing {fd}...")
        for img in imgs:
            if img.endswith(('.jpg', '.jpeg', '.png')) and '-' in img:
                # get the idx from the image name
                idx = img.split('-')[-1].split('_')[0]
                print(f"idx: {idx}")
                # get the link from the image name
                link = f"https://pixabay.com/photos/id-{idx}"
                print(f"link: {link}")

def copy_images_to_finalized():
    """Copy all images from travel_new subfolders to image_finalized directory"""
    source_dir = 'travel_new'
    target_dir = 'image_finalized'
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Get all subfolders in travel_new
    subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    
    total_copied = 0
    
    for subfolder in subfolders:
        print(f"Processing subfolder: {subfolder}")
        subfolder_path = os.path.join(source_dir, subfolder)
        
        # Get all image files in the subfolder
        image_files = [f for f in os.listdir(subfolder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        for image_file in image_files:
            source_path = os.path.join(subfolder_path, image_file)
            target_path = os.path.join(target_dir, f'travel_{total_copied:03d}_' + image_file)
            
            try:
                shutil.copy2(source_path, target_path)
                print(f"Copied: {image_file}")
                total_copied += 1
            except Exception as e:
                print(f"Error copying {image_file}: {e}")
    
    print(f"Total images copied: {total_copied}")

# get_header('unsplash-research-dataset-lite-latest/photos.csv000')
# get_images('photos.csv000')
# extract_pixabay_link('travel_new')
copy_images_to_finalized()