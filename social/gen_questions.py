import os 
import pandas as pd
import numpy as np
import random
import json
import re
import time
import matplotlib.pyplot as plt
from collections import Counter

def rename_imgs(img_dir):
    img_fnames = os.listdir(img_dir)
    img_fnames = [img_fname for img_fname in img_fnames if img_fname != '.DS_Store']
    img_fnames_add = [img_fname for img_fname in img_fnames if 'ADD' in img_fname]
    img_fnames_ori = [img_fname for img_fname in img_fnames if 'ADD' not in img_fname]
    # change file names of images in img_fnames_ori so that 0,1 -> 0, 2,3 -> 1,
    img_fnames_mapping = {}
    for i in range(len(img_fnames_ori)):
        fname = img_fnames_ori[i]
        handle, ext = fname.split('.')
        S, C, CD, CN, img_id = handle.split('_')
        S = int(S[1:])
        C = int(C[1:])
        CD = int(CD[2:])
        CN = int(CN[2:])
        S = 0 if S < 2 else 1
        C = 0 if C < 2 else 1
        CD = 0 if CD < 2 else 1
        CN = 0 if CN < 2 else 1
        img_fnames_mapping[fname] = f'S{S}_C{C}_CD{CD}_CN{CN}_{img_id}.{ext}'
    
    for img_fname in img_fnames_add:
        img_fnames_mapping[img_fname] = '_'.join(img_fname.split('_')[1:]) 

    # rename images
    for i, img_fname in enumerate(sorted(img_fnames_mapping.keys())):
        os.rename(os.path.join(img_dir, img_fname), os.path.join(img_dir, f'{i:03d}_{img_fnames_mapping[img_fname]}'))

def gen_img_distribution_via_stats(img_dir, stats_path):
    img_fnames = [f for f in os.listdir(img_dir) if f != '.DS_Store']
    with open(stats_path, 'r') as f:
        lines = f.readlines()
    img_info = {}
    for line in lines[1:]:
        line = line.strip()
        if line == '':
            continue
        line = line.split('\t')
        img_fname = line[0]
        crowdedness = float(line[1])
        obj_cnt = int(line[2])
        size = float(line[3])
        contrast = float(line[4])
        img_info[img_fname] = {'crowdedness': crowdedness, 'contrast': contrast, 'size': size, 'obj_cnt': obj_cnt}

    bins = Counter()
    img_ids = []
    for img_fn in img_fnames:
        idx, S, C, CD, CN, img_id = img_fn.split('.')[0].split('_')
        S = int(S[1:])
        C = int(C[1:])
        CD = int(CD[2:])
        CN = int(CN[2:])
        bins[f'{S}_{C}_{CD}_{CN}'] += 1
        img_ids.append(img_id)
        
    cds = []
    cns = []
    sizes = []
    contrasts = []
    for img_id in img_ids:
        if img_id not in img_info:
            print(f'img_id {img_id} not found in stats')
            continue
        cds.append(img_info[img_id]['crowdedness'])
        cns.append(img_info[img_id]['obj_cnt'])
        sizes.append(img_info[img_id]['size'])
        contrasts.append(img_info[img_id]['contrast'])
    
    # plot the distribution of cds, cns, sizes, and contrasts in 4 subplots, with individual titles
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.hist(cds, bins=10, alpha=0.5, label='crowdedness')
    plt.title('crowdedness')
    plt.subplot(2, 2, 2)
    plt.hist(cns, bins=10, alpha=0.5, label='obj_cnt')
    plt.title('obj_cnt')
    plt.subplot(2, 2, 3)
    plt.hist(sizes, bins=10, alpha=0.5, label='size')
    plt.title('size')
    plt.subplot(2, 2, 4)
    plt.hist(contrasts, bins=10, alpha=0.5, label='contrast')
    plt.legend()
    # plt.show()
    plt.savefig('img_distribution_finalized.png')

    # plot the distribution of bins
    plt.figure(figsize=(10, 10))
    plt.bar(bins.keys(), bins.values())
    plt.xticks(rotation=90)
    plt.title('bin distribution')
    # add legend Size_Contrast_Crowdedness_ObjectCount
    plt.legend(['Size_Contrast_Crowdedness_ObjectCount'])
    plt.show()
    plt.savefig('bin_distribution_finalized.png')


def gen_questions(img_dir):
    pass
    





# rename_imgs('finalized')
gen_img_distribution_via_stats('finalized', 'all_eligible_imgs_stats.tsv')