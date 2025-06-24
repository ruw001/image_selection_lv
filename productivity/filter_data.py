import os
import numpy as np
import shutil
from PIL import Image

def filter_data(data_path):
    # get all the files in the data_path
    files = os.listdir(data_path)
    os.makedirs('filtered', exist_ok=True)
    # shuffle the files
    np.random.shuffle(files)

    # select 80 images from the files
    files = files[:150]

    for file in files:
        # copy the file to the new folder
        shutil.copy(os.path.join(data_path, file), os.path.join('filtered', file))
    
def add_source_prefix(img_dir):
    for img_path in os.listdir(img_dir):
        # replace '{' and '}' with ''
        new_img_path = img_path.replace('{', '').replace('}', '')
        new_img_path = new_img_path.split('_')[1] if '_' in new_img_path else new_img_path
        new_img_path = 'productivity_' + new_img_path
        new_img_path = os.path.join(img_dir, new_img_path)
        os.rename(os.path.join(img_dir, img_path), new_img_path)
    return

def add_id_to_images(img_dir):
    cnt = 0
    imgs = os.listdir(img_dir)
    for img_path in imgs:
        new_img_path = img_path.split('_')[1]
        new_img_path = f'productivity_{cnt:03d}_' + new_img_path
        new_img_path = os.path.join(img_dir, new_img_path)
        os.rename(os.path.join(img_dir, img_path), new_img_path)
        cnt += 1

if __name__ == '__main__':
    # files = filter_data('first_round_images')
    # add_source_prefix('filtered_finalized')
    add_id_to_images('filtered_finalized')