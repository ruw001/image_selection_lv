import os
import numpy as np

def num_images_simulation(overlap_rate, num_participants, num_images):
    image_bins = np.zeros(num_participants).astype(int)
    p_selected = int(overlap_rate * num_participants)
    for i in range(num_images):
        # generate p_selected random numbers between 0 and num_participants-1, no repeat
        random_numbers = np.random.choice(num_participants, p_selected, replace=False)
        # set the image bin of the selected participants to +1
        image_bins[random_numbers] += 1
    # average number of images per participant
    avg_images_per_participant = np.mean(image_bins)
    # std of the number of images per participant
    std_images_per_participant = np.std(image_bins)
    return avg_images_per_participant, std_images_per_participant

def num_images_simulation2(num_img_each_participant, num_participants, num_images):
    image_bins = np.zeros(num_images).astype(int)
    for i in range(num_participants):
        # generate num_img_each_participant random numbers between 0 and num_participants-1, no repeat
        random_numbers = np.random.choice(num_images, num_img_each_participant, replace=False)
        # set the image bin of the selected participants to +1
        image_bins[random_numbers] += 1
    # average overlap rate per image
    average_overlap_rate = np.mean(image_bins) / num_participants
    return average_overlap_rate

# run simulation for different overlap rates and image numbers
for overlap_rate in [0.1, 0.3, 0.5]:
    for num_images in [100, 200, 300, 400, 500, 1000, 2000]:
        avg_images_per_participant, std_images_per_participant = num_images_simulation(overlap_rate, 20, num_images)
        print(f"Overlap rate: {overlap_rate}, Number of images: {num_images}, #img per participant: {avg_images_per_participant}, std img per participant: {std_images_per_participant}")

# run simulation for different number of images per participant and image numbers
for num_img_each_participant in [20, 30, 40]:
    for num_images in [100, 200, 300, 400, 500, 1000, 2000]:
        average_overlap_rate = num_images_simulation2(num_img_each_participant, 20, num_images)
        print(f"Number of images per participant: {num_img_each_participant}, Number of images: {num_images}, Average overlap rate: {average_overlap_rate}")




