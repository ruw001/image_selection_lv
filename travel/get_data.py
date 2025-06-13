import numpy as np
import pandas as pd
import os
import requests
import ollama

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

    df = df.head(1000)
    outpath = 'images_building'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for index, row in df.iterrows():
        desc = row['ai_description']
        # use agent to determine whether the description involves at least 2 salient objects, return a boolean
        print(f"prompting agent with desc: {desc}")
        resp = ollama.generate(
            model="llama3.2:latest",
            # prompt=f"Does the following description for an image involve at least 2 salient objects on the foreground? {desc} \n\n "
            # "Note that mountains, rivers, small objects, etc. are not salient objects.\n\n"
            # "Return only 'true' or 'false'. Do not include any other text in your response. Here is an example: \n"
            # "For example, for a description 'selective focus photography of short-coated white and brown dog on fallen brown leaves during daytime', the dog is a salient object, but the leaves are not. There is only one salient object. You should return 'false'.\n\n"

            # prompt=f"Does the following description for an image involve at least 2 people or animals on the foreground? {desc} \n\n "
            # "Return only 'true' or 'false'. Do not include any other text in your response."

            prompt=f"Does the following description for an image involve at least 2 buildings (or other man-made structures) on the foreground? {desc} \n\n "
            "Return only 'true' or 'false'. Do not include any other text in your response."
        )
        print(resp)
        name = row['photo_id']
        if resp['response'].lower() == "true":
            print(f">>>> {name} is a good image")
        else:
            print(f">>>> {name} is a bad image")
            continue
        
        print(f"saving image {name}...")
        url = row['photo_image_url']
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{outpath}/{name}.jpg', 'wb') as f:
                f.write(response.content)
        else:
            print(f"!!! Failed to download image from {url}")
        

# get_header('unsplash-research-dataset-lite-latest/photos.csv000')
get_images('unsplash-research-dataset-lite-latest/photos.csv000')