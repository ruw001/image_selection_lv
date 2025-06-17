import os
import numpy as np
import pandas as pd
import requests
import json
from collections import Counter
import ollama

'''
{
    "section": "Health", 
    "headline": "F.D.A. Plans to Ban Most E-Cigarette Flavors but Menthol", 
    "abstract": "The tobacco and vaping industries and conservative allies intensively lobbied against a ban on popular flavored e-cigarettes.", 
    "caption": "A new study by the National Institute on Drug Abuse found that teenagers preferred mint and mango Juul flavors. Menthol was the least popular.", 
    "image_url": "https://static01.nyt.com/images/2019/11/06/science/05FDA-FLAVORS/05FDA-FLAVORS-facebookJumbo.jpg?year=2020&h=550&w=1050&s=ad27f3a70c71de51e7605bbbc258e54782257f2ebb657a8df4f0a833b1b42809&k=ZQJBKqZ0VN", 
    "article_url": "https://www.nytimes.com/2019/12/31/health/e-cigarettes-flavor-ban-trump.html", 
    "image_id": "42d25485-0e48-50bf-8d16-948"
}

'''

def get_data_info(news_path):
    news = json.load(open(news_path, 'r'))
    section_names = Counter()
    for article in news:
        section_names[article['section']] += 1
    total = len(news)
    for section_name, count in section_names.items():
        print(f"{section_name}: {count}")

def get_eligible_images(news_path, num_limit=600):
    news = json.load(open(news_path, 'r'))
    ignore_list = [
        'Well', 'Books'
    ]
    section_names = Counter()
    new_counter = Counter()
    for article in news:
        section_names[article['section']] += 1
    total_count = len(news)
    for section_name, count in section_names.items():
        print(f"{section_name}: {count}")
    if not os.path.exists('images'):
        os.makedirs('images')
    # sort the news by length of caption
    news.sort(key=lambda x: len(x['caption'].split(' ')), reverse=True)
    for i in range(len(news)):
        article = news[i]
        if i % 100 == 0:
            print(f"Processed {i} images")
        if article['section'] in ignore_list:
            continue
        if article['image_url'] is None:
            continue
        if article['image_url'] == '':
            continue
        section = article['section']
        if new_counter[section] > num_limit * section_names[section] / total_count:
            continue
        img_id = article['image_id']
        url = article['image_url']
        caption = article['caption']
        print(f"Processing image {i}/{total_count}, {img_id}...")
        # determine if the caption includes multiple salient objects using an agent
        resp = ollama.generate(
            model="deepseek-r1:latest",
            prompt=f"Does the following caption for an image explicitly mention at least 2 (>= 2) salient objects (e.g., people, animals, objects, etc) on the foreground? \n \"{caption}\" \n\n "
            "Return only 'true' or 'false'. Do not include any other text in your response."
        )
        if resp['response'].lower() == 'true':
            print(f">>> Image {img_id} is eligible")
        elif resp['response'].lower() == 'false':
            print(f">>> Image {img_id} discarded")
            continue
        new_counter[section] += 1
        # download the image
        resp = requests.get(url)
        with open(f"images/{section}_{img_id}.jpg", "wb") as f:
            f.write(resp.content)
        print(f"Image {img_id} downloaded")


def rename_images(news_img_path):
    # get all the images in the folder
    images = os.listdir(news_img_path)
    cnt = 1
    print(f"Renaming {len(images)} images...")
    for image in images:
        # get the image id
        new_image = f"{cnt:03d}_" + image
        # rename the file
        os.rename(os.path.join(news_img_path, image), os.path.join(news_img_path, new_image))
        cnt += 1
        print(f"Renamed {cnt} images")

def gen_img_fname_list(news_img_path):
    # get all the images in the folder
    fnames = [f for f in os.listdir(news_img_path) if '.jpg' in f]
    # sort based on the number in the filename
    fnames.sort(key=lambda x: int(x.split('_')[0]))
    # write to a txt file
    with open('images_news/img_fname_list.txt', 'w') as f:
        for fname in fnames:
            f.write(fname + '\n')
    

if __name__ == '__main__':
    # get_data_info('N24News/news/nytimes_dataset.json')
    # get_eligible_images('N24News/news/nytimes_dataset.json')
    # rename_images('images_news/')
    gen_img_fname_list('images_news/')