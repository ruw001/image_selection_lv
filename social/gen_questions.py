import os 
import pandas as pd
import numpy as np
import random
import json
import re
import time
import matplotlib.pyplot as plt
from collections import Counter
import tqdm
import base64
from openai import OpenAI
from PIL import Image
import io
import pandas as pd
import shutil

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
    # Initialize OpenAI client
    client = OpenAI()

    complete_output = []
    
    imgs = [f for f in os.listdir(img_dir) if f != '.DS_Store']
    # sort by source, and then by index
    imgs.sort(key=lambda x: x.split('_')[0])
    for idx, img_filename in tqdm.tqdm(enumerate(imgs)):
        img_path = os.path.join(img_dir, img_filename)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            with io.BytesIO() as buffered:
                img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # print('processing image: ', img_path)
        
        question_gen_prompt = 'We define the following three types of attention in an image: \n' +\
        '1) focused attention: the user\'s attention is focused on a specific feature of an object in the image;\n' +\
        '2) alternating attention: the user\'s attention switches back and forth among multiple objects in the image;\n' +\
        '3) divided attention: the user\'s attention is divided between different locations in space, between multiple objects in the image.\n\n' +\
        'Your task is to generate a list of questions that can induce the above three types of attention for the image. At least three questions should be generated for each type of attention. \n' +\
        'Question to induce focused attention can be about identifying the identity, details (e.g., color, facial expression, clothing, etc.), and activity (e.g., body language) of a single object. \n' +\
        'Question to induce alternating attention can be about examining the relationship (e.g., position, orientation, etc.), interaction (e.g., behavior between two people or objects, etc), or feature differences (e.g., color, texture, size, etc.) between multiple objects. \n' +\
        'Question to induce divided attention can be about searching for a specific object, traversing over multiple objects without sustained focus on any specific objects (e.g., counting), or exploring the scene to get the emotion, atmosphere, event, or purpose of the image. \n\n' +\
        'For example, for a image where on the left is a woman in yellow rowing a yellow boat, and on the right are a couple rowing a red boat, \n' +\
        'possible questions to induce focused attention can be "What is the color of the woman\'s boat?", "What is the color of the couple\'s boat?", "What is the hair style of the woman?", etc. \n' +\
        'possible questions to induce alternating attention can be "What is the relationship between the woman and the couple?", "What is the relationship between the woman and the boat?", "Is the woman rowing towards the same direction as the couple?", etc. \n' +\
        'possible questions to induce divided attention can be "what event does this image describe?", "what is the emotion of the image?", "what is the atmosphere of the image?", "Where is the person in yellow in the image?", etc. \n' + \
        'Now please generate at least three questions for each type of attention for the image attached. The output should follow the exact format (only the content in the curly braces should be replaced): \n\n' +\
        '{overall image description} \n' +\
        '1) focused attention: \n' +\
        'a) {question 1} \n' +\
        'b) {question 2} \n' +\
        'c) {question 3} \n' +\
        '2) alternating attention: \n' +\
        'a) {question 1} \n' +\
        'b) {question 2} \n' +\
        'c) {question 3} \n' +\
        '3) divided attention: \n' +\
        'a) {question 1} \n' +\
        'b) {question 2} \n' +\
        'c) {question 3} \n' +\
        'The questions should be concise, clear, and easy to understand by a person with average visual ability. The target objects involved in the questions should be salient objects in the image and should be unambiguous to locate by the person answering the question. \n' +\
        'The questions should be based on the definition of the three types of attention, but please be creative and feel free to expand based on the definitions. Please DO NOT hallucinate and DO NOT asking about objects or features that are absent in the image. \n' + \
        'The questions should make sense and represent visual tasks people would do in their daily life. \n' + \
        'For example, a question like "What is the relationship between the person and the tree in the garden?" is NOT a good question when the person barely interacts with the tree, and this is not a visual task people would usually do in their daily life. \n' + \
        'another example is "What is the difference between the fountain and the bench on the image?" is NOT a good question because the objects being compared are too different and not the same type of objects. \n'

        # For example, for focused attention questions, asking the color of an object is good, but asking about the size of an object is not, since it is difficult to tell the exact number from the image. \n' +\
        # 'The questions can be open ended but the answer should be clear and unambiguous. \n' + \
        try:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": question_gen_prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low"
                        }
                    ]
                }
            ]
            
            response = client.responses.create(
                model="gpt-4.1",
                input=message,
                temperature=0.7
            )
            resp = response.output_text
            # print(resp)
            complete_output.append({
            'img_filename': img_filename,
            'questions': resp
        })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    with open('all_imgs_final_questions_social.txt', 'w') as f:
        for item in complete_output:
            f.write(item['img_filename'] + '\n\n')
            f.write(item['questions'] + '\n\n')
    





# rename_imgs('finalized')
# gen_img_distribution_via_stats('finalized', 'all_eligible_imgs_stats.tsv')
gen_questions('finalized')