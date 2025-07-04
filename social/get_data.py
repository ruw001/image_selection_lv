import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual_genome.local as vg
import json
import tqdm
import urllib.request
import random
from PIL import Image
from io import BytesIO
import cv2
import heapq
import collections
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

def get_all_img_stats(obj_data_path, img_meta_data_path, start_idx=0, rewrite=False):
    lowest_dim = 600
    obj_data = json.load(open(obj_data_path))
    img_meta_data = json.load(open(img_meta_data_path))
    img_id_info = dict()

    print('getting img meta data...')
    for img in tqdm.tqdm(img_meta_data):
        img_id_info[img['image_id']] = img

    dst_dir = 'all_eligible_imgs_stats.tsv'

    if rewrite:
        if os.path.exists(dst_dir):
            os.remove(dst_dir)
        with open(dst_dir, 'w') as f:
            f.write('img_id\tcrowdedness\tobj_cnt\tobj_size\tcontrast\n')
    # 'objects': [{'synsets': ['tree.n.01'], 'h': 557, 'object_id': 1058549, 'merged_object_ids': [], 'names': ['trees'], 'w': 799, 'y': 0, 'x': 0}, {'synsets': ['sidewalk.n.01'], 'h': 290, 'object_id': 1058534, 'merged_object_ids': [5046], 'names': ['sidewalk'], 'w': 722, 'y': 308, 'x': 78},...]
    print('getting img stats...')
    for img in tqdm.tqdm(obj_data):
        img_id = img['image_id']
        if img_id < start_idx:
            continue
        img_w = img_id_info[img_id]['width']
        img_h = img_id_info[img_id]['height']
        if img_w < lowest_dim or img_h < lowest_dim:
            continue
        if img_w < img_h: # only use landscape images
            continue
        objects = img['objects']
        obj_cnt = len(objects)
        # calculate the crowdedness of the image, the crowdedness measures how close the objects are to each other
        obj_positions = []
        obj_sizes = []
        for obj in objects:
            obj_id = obj['object_id']
            obj_synsets = obj['synsets']
            obj_h = obj['h']
            obj_w = obj['w']
            obj_x = obj['x']
            obj_y = obj['y']
            obj_area = obj_h * obj_w
            obj_center_x = obj_x + obj_w / 2
            obj_center_y = obj_y + obj_h / 2
            obj_positions.append([obj_center_x, obj_center_y])
            obj_size_ratio = obj_area / (img_w * img_h)
            obj_sizes.append(obj_size_ratio)
        # calculate the distance between each object
        obj_distances = []
        obj_positions = np.array(obj_positions)
        for i in range(len(obj_positions)):
            for j in range(i+1, len(obj_positions)):
                obj_distance = np.linalg.norm(obj_positions[i] - obj_positions[j])
                obj_distances.append(obj_distance)
        # calculate the crowdedness, normalized by the image diagonal
        crowdedness = np.mean(obj_distances) / np.sqrt(img_w**2 + img_h**2)
        obj_size_avg = np.mean(obj_sizes)

        # calculate overall contrast of the image
        # get the image via url and convert it to grayscale using opencv
        img_url = img_id_info[img_id]['url']
        img_data = urllib.request.urlopen(img_url).read()
        img = np.array(Image.open(BytesIO(img_data)))
        # cv2.imwrite('temp_rgb_reversed.jpg', img) # rgb, but opencv reverses it
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite('temp_gray_dir.jpg', img) # gray
        # calculate the contrast of the image
        contrast = img.std()
        # img_stats[img_id] = {
        #     'crowdedness': crowdedness,
        #     'obj_cnt': obj_cnt,
        #     'obj_size': obj_size_avg,
        #     'contrast': contrast
        # }
        # write one line to the file
        with open(dst_dir, 'a') as f:
            f.write(str(img_id) + '\t' + str(crowdedness) + '\t' + str(obj_cnt) + '\t' + str(obj_size_avg) + '\t' + str(contrast) + '\n')
    print('all done!!!')
    return

def which_bin(stats, bins, ranges_size, ranges_contrast, ranges_crowdedness, ranges_cnt):
    # Check if stats values fall within any bin ranges
    # Note: Ranges are left exclusive (>) and right inclusive (<=)
    for bin in bins:
        if stats['obj_size'] > ranges_size[bin[0]][0] and stats['obj_size'] <= ranges_size[bin[0]][1] and \
            stats['contrast'] > ranges_contrast[bin[1]][0] and stats['contrast'] <= ranges_contrast[bin[1]][1] and \
                stats['crowdedness'] > ranges_crowdedness[bin[2]][0] and stats['crowdedness'] <= ranges_crowdedness[bin[2]][1] and \
                    stats['obj_cnt'] > ranges_cnt[bin[3]][0] and stats['obj_cnt'] <= ranges_cnt[bin[3]][1]:
            return bin
    return None

def get_eligible_imgs_from_stats(stats_path):
    # read the stats file
    stats = pd.read_csv(stats_path, sep='\t')
    # remove rows with nan values
    stats = stats.dropna()
    # remove rows with obj_cnt < 2
    stats = stats[stats['obj_cnt'] >= 2]
    # print basic description of the stats
    print(stats.describe())

    # get the img_id, crowdedness, obj_cnt, obj_size, contrast
    # img_id = stats['img_id']
    # crowdedness = stats['crowdedness']
    # obj_cnt = stats['obj_cnt']
    # obj_size = stats['obj_size']
    # contrast = stats['contrast']

    # generate histogram plot with title 'hist_xxx.png' and finegrained bins, saved to 'hists' folder
    # os.makedirs('hists', exist_ok=True)
    # plt.hist(crowdedness, bins=100)
    # plt.title('crowdedness')
    # plt.savefig('hists/hist_crowdedness.png')
    # plt.close()
    # plt.hist(obj_cnt, bins=max(obj_cnt))
    # plt.title('obj_cnt')
    # plt.savefig('hists/hist_obj_cnt.png')
    # plt.close()
    # plt.hist(obj_size, bins=100)
    # plt.title('obj_size')
    # plt.savefig('hists/hist_obj_size.png')
    # plt.close()
    # plt.hist(contrast, bins=100)
    # plt.title('contrast')
    # plt.savefig('hists/hist_contrast.png')
    # plt.close()
    
    # divide each range (min_max) into num_bins bins evenly
    num_bins = 4
    size_min_max = [0, 0.4]
    ranges_size = [[size_min_max[0] + i*(size_min_max[1] - size_min_max[0])/num_bins, size_min_max[0] + (i+1)*(size_min_max[1] - size_min_max[0])/num_bins] for i in range(num_bins)]
    contrast_min_max = [20, 100]
    ranges_contrast = [[contrast_min_max[0] + i*(contrast_min_max[1] - contrast_min_max[0])/num_bins, contrast_min_max[0] + (i+1)*(contrast_min_max[1] - contrast_min_max[0])/num_bins] for i in range(num_bins)]
    crowdedness_min_max = [0.1, 0.4]
    ranges_crowdedness = [[crowdedness_min_max[0] + i*(crowdedness_min_max[1] - crowdedness_min_max[0])/num_bins, crowdedness_min_max[0] + (i+1)*(crowdedness_min_max[1] - crowdedness_min_max[0])/num_bins] for i in range(num_bins)]
    cnt_min_max = [0, 40]
    ranges_cnt = [[cnt_min_max[0] + i*(cnt_min_max[1] - cnt_min_max[0])/num_bins, cnt_min_max[0] + (i+1)*(cnt_min_max[1] - cnt_min_max[0])/num_bins] for i in range(num_bins)]

    # 4 x 4 x 4 x 4 = 256 bins, each bin is a tuple of (size_range, contrast_range, crowdedness_range, cnt_range), left exclusive, right inclusive
    bins = {}
    for si in range(len(ranges_size)):
        for ct in range(len(ranges_contrast)):
            for cd in range(len(ranges_crowdedness)):
                for cnt in range(len(ranges_cnt)):
                    bins[(si, ct, cd, cnt)] = []
    
    # for each row in stats, find the bin it belongs to
    for i in range(len(stats)):
        bin = which_bin(stats.iloc[i], bins, ranges_size, ranges_contrast, ranges_crowdedness, ranges_cnt)
        if bin is None:
            # print('bin is None for img_id: ', stats.iloc[i]['img_id'])
            continue
        bins[bin].append(stats['img_id'].iloc[i])
    
    print('stats for each bin:')
    # plot bar plot for each bin with title 'bar_xxx.png' and save. x axis is the bin index, y axis is the number of images in the bin
    plt.bar(np.arange(len(bins)), [len(bins[bin]) for bin in bins])
    plt.savefig('bar_all.png')
    plt.close()

    # get an example image from each bin
    img_id_info = get_img_meta_data('visual_genome/data/image_data.json')
    os.makedirs('example_imgs', exist_ok=True)
    for bin in tqdm.tqdm(bins):
        if len(bins[bin]) == 0:
            print('bin is empty: ', bin)
            continue
        # in each bin, choose top n images that are the most central
        # Find image closest to center of bin ranges
        bin_center_size = (ranges_size[bin[0]][0] + ranges_size[bin[0]][1]) / 2
        bin_center_contrast = (ranges_contrast[bin[1]][0] + ranges_contrast[bin[1]][1]) / 2
        bin_center_crowdedness = (ranges_crowdedness[bin[2]][0] + ranges_crowdedness[bin[2]][1]) / 2
        bin_center_cnt = (ranges_cnt[bin[3]][0] + ranges_cnt[bin[3]][1]) / 2
        
        n_sample = 6
        # create a priority queue to store the images and their distances
        priority_queue = []
        for img_id_candidate in bins[bin]:
            img_stats = stats[stats['img_id'] == img_id_candidate].iloc[0]
            dist = (
                ((img_stats['obj_size'] - bin_center_size) / (size_min_max[1] - size_min_max[0]))**2 +
                ((img_stats['contrast'] - bin_center_contrast) / (contrast_min_max[1] - contrast_min_max[0]))**2 + 
                ((img_stats['crowdedness'] - bin_center_crowdedness) / (crowdedness_min_max[1] - crowdedness_min_max[0]))**2 +
                ((img_stats['obj_cnt'] - bin_center_cnt) / (cnt_min_max[1] - cnt_min_max[0]))**2
            )**0.5
            heapq.heappush(priority_queue, (dist, img_id_candidate))
        # get top n images
        top_n_imgs = heapq.nsmallest(n_sample, priority_queue)
        for img_tuple in top_n_imgs:
            img_id = img_tuple[1]
            img_url = img_id_info[img_id]
            img_data = urllib.request.urlopen(img_url).read()
            img = Image.open(BytesIO(img_data))
            img.save('example_imgs/' + f'S{bin[0]}_C{bin[1]}_CD{bin[2]}_CN{bin[3]}_{img_id}.jpg')
        
    return
    
def check_img_stats(img_id, obj_data_path, img_meta_data_path):
    obj_data = json.load(open(obj_data_path))
    img_meta_data = get_img_meta_data(img_meta_data_path)
    os.makedirs('checking_imgs', exist_ok=True)
    for img in obj_data:
        if img['image_id'] == img_id:
            objects = img['objects']
            # download image using url 
            img_url = img_meta_data[img_id]
            img_data = urllib.request.urlopen(img_url).read()
            img_data = np.array(Image.open(BytesIO(img_data)))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            print(img_data.shape, img_data.dtype)
            for idx, obj in enumerate(objects):
                obj_id = obj['object_id']
                obj_synsets = obj['synsets']
                print(idx, obj_id, obj_synsets)
                obj_h = obj['h']
                obj_w = obj['w']
                obj_x = obj['x']
                obj_y = obj['y']
                # draw bounding box on the image, with index shown within the box
                cv2.rectangle(img_data, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (0, 0, 255), 2)
                # add index to the image
                cv2.putText(img_data, str(idx), (obj_x, obj_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite('checking_imgs/' + f'{img_id}.jpg', img_data)
            break

def get_avg_num_img_per_bin(img_path):
    img_fnames = os.listdir(img_path)
    num_img_per_bin = collections.Counter()
    # add all bins to the counter, 0, 1 
    for S in range(2):
        for C in range(2):
            for CD in range(2):
                for CN in range(2):
                    num_img_per_bin[(S, C, CD, CN)] = 0
    
    # using simplified binning: 2x2x2x2 = 16 bins
    for img_fname in img_fnames:
        if img_fname == '.DS_Store':
            continue
        if 'ADD' in img_fname:
            continue
        S, C, CD, CN, _ = img_fname.split('.')[0].split('_')
        S, C, CD, CN = int(S[1:]), int(C[1:]), int(CD[2:]), int(CN[2:])
        S = 0 if S < 2 else 1
        C = 0 if C < 2 else 1
        CD = 0 if CD < 2 else 1
        CN = 0 if CN < 2 else 1
        bin = (S, C, CD, CN)
        num_img_per_bin[bin] += 1
    print('stats for bin counter:')
    # how many unique bins?
    print('number of unique bins: ', len(num_img_per_bin))
    print('mean: ', np.mean(list(num_img_per_bin.values())))
    print('median: ', np.median(list(num_img_per_bin.values())))
    print('max: ', np.max(list(num_img_per_bin.values())))
    print('min: ', np.min(list(num_img_per_bin.values())))
    # how many bins with less than 5 images?
    print('number of bins with less than 5 images: ', sum(1 for bin in num_img_per_bin if num_img_per_bin[bin] < 5))
    # how many bins with less than 3 images?
    print('number of bins with less than 3 images: ', sum(1 for bin in num_img_per_bin if num_img_per_bin[bin] < 3))
    # how many bins with less than 2 images?
    print('number of bins with less than 2 images: ', sum(1 for bin in num_img_per_bin if num_img_per_bin[bin] < 2))
    
    # set a total number goal of 400 images, generate an algorithm to select images from each bin to reach the goal while making the number of images in each bin as balanced as possible
    # Calculate target number of images per bin for balanced distribution
    total_goal = 400
    num_bins = len(num_img_per_bin)
    target_per_bin = total_goal // num_bins

    # Track how many more images needed for each bin
    additional_needed = {}
    for bin in num_img_per_bin:
        current = num_img_per_bin[bin]
        if current < target_per_bin:
            additional_needed[bin] = target_per_bin - current
        else:
            additional_needed[bin] = 0

    # Distribute any remaining images after initial allocation
    remaining = total_goal - (target_per_bin * num_bins)
    if remaining > 0:
        # Sort bins by current count ascending to prioritize under-filled bins
        sorted_bins = sorted(num_img_per_bin.items(), key=lambda x: x[1])
        for i in range(remaining):
            bin = sorted_bins[i % len(sorted_bins)][0]
            additional_needed[bin] += 1

    # Print allocation plan, and return a dict of bin and the number of images needed
    print("\nAllocation plan to reach 400 total images:")
    print("Bin (S,C,CD,CN) | Current | Target | Additional Needed")
    print("-" * 50)
    for bin in sorted(additional_needed.keys()):
        current = num_img_per_bin[bin]
        needed = additional_needed[bin]
        target = current + needed
        print(f"{bin} | {current:^7d} | {target:^6d} | {needed:^17d}")
    print(f"\nTotal additional images needed: {sum(additional_needed.values())}")
    return additional_needed

# final step: add more images
def add_new_images_based_on_stats(stats_path, bin_img_needed: dict, first_rnd_path):
    stats = pd.read_csv(stats_path, sep='\t')
    # remove rows with nan values
    stats = stats.dropna()
    # remove rows with obj_cnt < 2
    stats = stats[stats['obj_cnt'] >= 2]

    existing_imgs = os.listdir(first_rnd_path)
    existing_imgs = [img_fname.split('.')[0].split('_')[4] if 'ADD' not in img_fname else img_fname.split('.')[0].split('_')[5] for img_fname in existing_imgs if img_fname != '.DS_Store']

    existing_ad_imgs = os.listdir('additional_imgs') if os.path.exists('additional_imgs') else []
    existing_ad_imgs = [img_fname.split('.')[0].split('_')[5] for img_fname in existing_ad_imgs if img_fname != '.DS_Store']

    # get the binning scheme for the first round
    num_bins = 2 # per category
    size_min_max = [0, 0.4]
    ranges_size = [[size_min_max[0] + i*(size_min_max[1] - size_min_max[0])/num_bins, size_min_max[0] + (i+1)*(size_min_max[1] - size_min_max[0])/num_bins] for i in range(num_bins)]
    contrast_min_max = [20, 100]
    ranges_contrast = [[contrast_min_max[0] + i*(contrast_min_max[1] - contrast_min_max[0])/num_bins, contrast_min_max[0] + (i+1)*(contrast_min_max[1] - contrast_min_max[0])/num_bins] for i in range(num_bins)]
    crowdedness_min_max = [0.1, 0.4]
    ranges_crowdedness = [[crowdedness_min_max[0] + i*(crowdedness_min_max[1] - crowdedness_min_max[0])/num_bins, crowdedness_min_max[0] + (i+1)*(crowdedness_min_max[1] - crowdedness_min_max[0])/num_bins] for i in range(num_bins)]
    cnt_min_max = [0, 40]
    ranges_cnt = [[cnt_min_max[0] + i*(cnt_min_max[1] - cnt_min_max[0])/num_bins, cnt_min_max[0] + (i+1)*(cnt_min_max[1] - cnt_min_max[0])/num_bins] for i in range(num_bins)]

    # 4 x 4 x 4 x 4 = 256 bins, each bin is a tuple of (size_range, contrast_range, crowdedness_range, cnt_range), left exclusive, right inclusive
    bins = {}
    for si in range(len(ranges_size)):
        for ct in range(len(ranges_contrast)):
            for cd in range(len(ranges_crowdedness)):
                for cnt in range(len(ranges_cnt)):
                    bins[(si, ct, cd, cnt)] = []
    
    # for each row in stats, find the bin it belongs to
    for i in range(len(stats)):
        bin = which_bin(stats.iloc[i], bins, ranges_size, ranges_contrast, ranges_crowdedness, ranges_cnt)
        if bin is None:
            # print('bin is None for img_id: ', stats.iloc[i]['img_id'])
            continue
        bins[bin].append(stats['img_id'].iloc[i])

    # get an example image from each bin
    img_id_info = get_img_meta_data('visual_genome/data/image_data.json')
    os.makedirs('additional_imgs', exist_ok=True)

    for bin in bin_img_needed:
        if len(bins[bin]) == 0:
            print('bin is empty: ', bin)
            continue # skip
        # sample 10 * bin_img_needed[bin] images
        # select images that are not in existing_imgs and existing_ad_imgs and then sample from the remaining images
        new_images = [str(img_id) for img_id in bins[bin] if str(img_id) not in existing_imgs and str(img_id) not in existing_ad_imgs]
        sampled_imgs = random.sample(new_images, 10 * bin_img_needed[bin] if 10 * bin_img_needed[bin] < len(new_images) else len(new_images))
        print(f'sampling {len(sampled_imgs)} images for bin {bin}')
        for img_id in sampled_imgs:
            img_id = int(img_id)
            if img_id not in img_id_info:
                print('img_id not in img_id_info: ', img_id)
                continue
            img_url = img_id_info[img_id]
            img_data = urllib.request.urlopen(img_url).read()
            img = Image.open(BytesIO(img_data))
            img.save('additional_imgs/' + f'ADD_S{bin[0]}_C{bin[1]}_CD{bin[2]}_CN{bin[3]}_{img_id}.jpg')
    return

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
        if img_w < img_h: # only use landscape images
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
    # sample 400 images by choosing 50 images from each obj_cnt level
    eligible_imgs = sorted(eligible_imgs, key=lambda x: (x['obj_cnt'], len(x['unique_objs'])), reverse=True)
    min_obj_cnt = 2
    max_obj_cnt = eligible_imgs[0]['obj_cnt']
    obj_cnt_range = max_obj_cnt - min_obj_cnt + 1
    # generate 5 bins with lo and hi based on obj cnt range
    bin_size = obj_cnt_range // 5
    bins = [(min_obj_cnt + i * bin_size, min_obj_cnt + (i+1) * bin_size) for i in range(5)] # left inclusive, right exclusive
    # divide eligible_imgs into 5 groups based on bins
    eligible_imgs_groups = [[] for _ in range(5)]
    for img in eligible_imgs:
        if img['obj_cnt'] < min_obj_cnt: # single object images are not included
            continue
        for i in range(5):
            if img['obj_cnt'] >= bins[i][0] and img['obj_cnt'] < bins[i][1]:
                eligible_imgs_groups[i].append(img)
                break
    # sample 80 images from each group
    sampled_imgs = []
    for group in eligible_imgs_groups:
        print(len(group))
        if len(group) < 80:
            sampled_imgs.extend(group)
        else:
            sampled_imgs.extend(random.sample(group, 80))
    # sample 400 images by choosing 50 images from each obj_cnt level
    with open('visual_genome/data/eligible_imgs_061725.tsv', 'w') as f:
        for img in sampled_imgs:
            f.write(str(img['img_id']) + '\t' + str(img['img_url']) + '\t' + str(img['obj_cnt']) + '\t' + str(img['unique_objs']) + '\n')
    return sampled_imgs
    

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

def rename_imgs(img_dir):
    img_dir = 'images_eligible_061725'
    cnt = 1
    for img_path in os.listdir(img_dir):
        new_img_path = f'{cnt:03d}_' + img_path
        new_img_path = os.path.join(img_dir, new_img_path)
        os.rename(os.path.join(img_dir, img_path), new_img_path)
        cnt += 1
    return

def get_img_fname_list(img_dir):
    img_fname_list = os.listdir(img_dir)
    img_fname_list.sort(key=lambda x: int(x.split('_')[0]))
    with open('visual_genome/data/img_fname_list_061725.txt', 'w') as f:
        for img_fname in img_fname_list:
            f.write(img_fname + '\n')
    return

# all_possible_obj = get_all_possible_obj('visual_genome/data/objects.json', 'visual_genome/data/image_data.json')

# get_img_meta_data('visual_genome/data/image_data.json')

# get_eligible_imgs('visual_genome/data/objects.json', 'visual_genome/data/image_data.json')
# get_all_img_stats('visual_genome/data/objects.json', 'visual_genome/data/image_data.json', start_idx=1593266, rewrite=False)
get_eligible_imgs_from_stats('all_eligible_imgs_stats.tsv')
# check_img_stats(2578, 'visual_genome/data/objects.json', 'visual_genome/data/image_data.json')
# img_needed = get_avg_num_img_per_bin('first_round')
# add_new_images_based_on_stats('all_eligible_imgs_stats.tsv', img_needed, 'first_round')


# download_imgs('visual_genome/data/eligible_imgs_061725.tsv', 'images_eligible_061725')
# rename_imgs('images_eligible_061725')
# get_img_fname_list('images_eligible_061725')