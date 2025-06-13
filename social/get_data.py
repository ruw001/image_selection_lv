import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_genome.local as vg
import json
import tqdm
import urllib.request

# read visual_genome/data/objects.json 
# obj_data = json.load(open('visual_genome/data/objects.json'))
# list of dicts. image_id, image_url, objects: list of dicts, each dict include object_id, names or synsets: list of one string, h, w, x, y

def get_img_meta_data(img_meta_data_path):
    img_id_url_mapping = dict()
    img_meta_data = json.load(open(img_meta_data_path))
    print(len(img_meta_data))
    for img in img_meta_data:
        img_id_url_mapping[img['image_id']] = img['url']
    return img_id_url_mapping

    # return img_meta_data

def get_all_possible_obj(obj_data_path, img_meta_data_path):
    all_possible_obj = dict()
    all_possible_synsets = dict()
    img_id_url_mapping = get_img_meta_data(img_meta_data_path)
    obj_data = json.load(open(obj_data_path))
    print(obj_data[0])
    print(len(obj_data))
    # return
    for img in tqdm.tqdm(obj_data):
        img_id = img['image_id']
        try:
            img_url = img['image_url']
        except:
            print('image_url not found for image_id: ', img_id)
            continue

        objects = img['objects']
        for obj in objects:
            obj_id = obj['object_id']
            assert len(obj['names']) == 1, "\neach object should have one name\n"
            obj_name = obj['names'][0]
            synsets = ' '.join(obj['synsets'])
            all_possible_obj[obj_name] = img_url if 'stanford' in img_url else img_id_url_mapping[img_id] # latest image url
            all_possible_synsets[synsets] = img_url if 'stanford' in img_url else img_id_url_mapping[img_id]
    all_possible_obj = [(obj, url) for obj, url in all_possible_obj.items()]
    all_possible_obj.sort(key=lambda x: x[0])
    all_possible_synsets = [(synset, url) for synset, url in all_possible_synsets.items()]
    all_possible_synsets.sort(key=lambda x: x[0])
    with open('visual_genome/data/all_possible_obj.txt', 'w') as f:
        for obj in all_possible_obj:
            f.write(obj[0] + '\t' + str(obj[1]) + '\n')
    with open('visual_genome/data/all_possible_synsets.txt', 'w') as f:
        for synset in all_possible_synsets:
            f.write(synset[0] + '\t' + str(synset[1]) + '\n')
    return all_possible_obj, all_possible_synsets

def get_eligible_imgs(obj_data_path, img_meta_data_path, num_limit=400):
    lowest_dim = 600
    obj_data = json.load(open(obj_data_path))
    img_meta_data = json.load(open(img_meta_data_path))
    img_id_info = dict()
    for img in img_meta_data:
        img_id_info[img['image_id']] = img
    eligible_imgs = []
    for img in tqdm.tqdm(obj_data):
        img_id = img['image_id']
        img_w = img_id_info[img_id]['width']
        img_h = img_id_info[img_id]['height']
        if img_w < lowest_dim or img_h < lowest_dim:
            continue
        img_url = img_id_info[img_id]['url']
        objects = img['objects']
        unique_objs = set()
        obj_cnt = 0
        include_person = False
        for obj in objects:
            # obj_id = obj['object_id']
            for syn in obj['synsets']:
                if syn.startswith('person.') or syn.startswith('man.') or syn.startswith('woman.'):
                    include_person = True
                unique_objs.add(syn)
            obj_cnt += 1
        if not include_person:
            continue
        eligible_imgs.append(
            {
                'img_id': img_id,
                'img_url': img_url,
                'obj_cnt': obj_cnt,
                'unique_objs': unique_objs
            }
        )
    eligible_imgs.sort(key=lambda x: (x['obj_cnt'], len(x['unique_objs'])), reverse=True)
    with open('visual_genome/data/eligible_imgs.tsv', 'w') as f:
        for img in eligible_imgs[:num_limit]:
            f.write(str(img['img_id']) + '\t' + str(img['img_url']) + '\t' + str(img['obj_cnt']) + '\t' + str(img['unique_objs']) + '\n')
    return eligible_imgs

def download_imgs(eligible_imgs_path, img_dir):
    eligible_imgs = pd.read_csv(eligible_imgs_path, sep='\t')
    # get the first 2 columns
    eligible_imgs = eligible_imgs.iloc[:, :2]
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for img_id, img_url in tqdm.tqdm(eligible_imgs.values):
        img_path = os.path.join(img_dir, str(img_id) + '.jpg')
        if not os.path.exists(img_path):
            urllib.request.urlretrieve(img_url, img_path)
    return

# all_possible_obj = get_all_possible_obj('visual_genome/data/objects.json', 'visual_genome/data/image_data.json')

# get_img_meta_data('visual_genome/data/image_data.json')

# get_eligible_imgs('visual_genome/data/objects.json', 'visual_genome/data/image_data.json')

download_imgs('visual_genome/data/eligible_imgs.tsv', 'images_eligible')