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
    


if __name__ == '__main__':
    files = filter_data('first_round_images')