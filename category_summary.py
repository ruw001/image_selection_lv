import pandas as pd
import openai
import os
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_category_summary(img_dir):
    client = openai.OpenAI()

    imgs = [f for f in os.listdir(img_dir) if f != '.DS_Store']
    imgs.sort()
    results = []
    prompt = "Your task is to calculate the number of salient objects in the image, " + \
        "and the number of object categories in the image.\n" + \
        "The object categories of interest are: \n" + \
        "1) Person, 2) Accessory (e.g., hat, glasses, bag, etc.), 3) Transportation (e.g., car, bus, plane, etc.), " + \
        "4) Outdoor Object (e.g., tree, building, street signs, etc.), 5) Animal (e.g., dog, cat, bird, etc.), 6) Sports Equipment (e.g., ball, bat, etc.), " + \
        "7) Kitchenware (e.g., fork, knife, pan, etc.), 8) Food (e.g., fruit, cake, pizza, etc.), 9) Furniture (e.g., chair, desk, sofa, etc.), " + \
        "10) Electronics (e.g., phone, laptop, TV, etc.), 11) Appliances (e.g., washer, dryer, oven, etc.), 12) Indoor Object (e.g., hammer, screwdriver, book, etc.)\n\n" + \
        "Your response should be two lines. On the first line, please answer the number of salient objects in the image, and the number of object categories in the image, separated by a comma.\n" + \
        "On the second line, please give a brief description of the image, including the number of salient objects and the number of object categories.\n\n" + \
        "Please return the results in the following format (only text in curly braces can be replaced): \n\n" + \
        "{number_of_salient_objects},{number_of_object_categories}\n" + \
        "Description: {description}\n\n"
    
    for idx, img_fn in enumerate(imgs):
        img_path = os.path.join(img_dir, img_fn)
        print(f"{idx}/{len(imgs)}: {img_path}")
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            with io.BytesIO() as buffered:
                img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        try:
            message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt
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
                temperature=0.0
            )
            resp = response.output_text
            print(resp)
            results.append({
                'img_filename': img_fn,
                'stats': resp.split('\n')[0]
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    with open('category_summary.txt', 'w') as f:
        for result in results:
            f.write(f"{result['img_filename']},{result['stats']}\n")

def plot_distribution(summary_path):
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    stats = [line.split(',') for line in lines]
    num_obj_no_productivity = [int(stat[1]) for stat in stats if 'productivity' not in stat[0]]
    num_cat_no_productivity = [int(stat[2]) for stat in stats if 'productivity' not in stat[0]]
    num_obj_productivity = [int(stat[1]) for stat in stats if 'productivity' in stat[0]]
    num_cat_productivity = [int(stat[2]) for stat in stats if 'productivity' in stat[0]]

    num_obj_no_productivity = np.array(num_obj_no_productivity)
    num_cat_no_productivity = np.array(num_cat_no_productivity)
    num_obj_productivity = np.array(num_obj_productivity)
    num_cat_productivity = np.array(num_cat_productivity)

    # plot the distribution of the number of objects and categories
    # make each bin to include exactly 1 unit, and make titles
    plt.hist(num_obj_no_productivity, bins=range(1, num_obj_no_productivity.max() + 1))
    # make ticks to be 0, 5, 10, 15, 20, 25,...
    plt.xticks(range(0, num_obj_no_productivity.max() + 1, 5))
    plt.title('Distribution of Number of Objects (No Productivity)')
    plt.xlabel('Number of Objects')
    plt.ylabel('Frequency')
    plt.show()
    plt.hist(num_cat_no_productivity, bins=range(1, num_cat_no_productivity.max() + 1))
    plt.title('Distribution of Number of Categories (No Productivity)')
    plt.xlabel('Number of Categories')
    plt.ylabel('Frequency')
    plt.show()
    plt.hist(num_obj_productivity, bins=range(1, num_obj_productivity.max() + 1))
    plt.xticks(range(0, num_obj_productivity.max() + 1, 5))
    plt.title('Distribution of Number of Objects (Productivity)')
    plt.xlabel('Number of Objects')
    plt.ylabel('Frequency')
    plt.show()
    plt.hist(num_cat_productivity, bins=range(1, num_cat_productivity.max() + 1))
    plt.title('Distribution of Number of Categories (Productivity)')
    plt.xlabel('Number of Categories')
    plt.ylabel('Frequency')
    plt.show()

    

# get_category_summary('all_imgs_final')
plot_distribution('category_summary.txt')