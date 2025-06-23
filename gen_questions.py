import os
import json
import random
import tqdm
import base64
import ollama
from PIL import Image
import io

def gen_questions(img_dir):
    imgs = os.listdir(img_dir)
    for idx, img_filename in enumerate(imgs):
        img_path = os.path.join(img_dir, img_filename)
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            with io.BytesIO() as buffered:
                img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print('processing image: ', img_path)
        
        question_gen_prompt = 'Your task is to generate a list of questions for the image. \n' +\
        'we categorized information in an image into the following three types based on salient objects in an image (object can be a person): \n' + \
        '1) Within-object information (level 1): including object identity, details (e.g., color, facial expression, clothing, etc.), and activity (e.g., body language) of the object \n' + \
        '2) Cross-object information (level 2): including identifying interaction or relationship between objects (e.g., the relationship/behavior between two people), and identifying difference between multiple objects; \n' + \
        '3) Overall interpretation (level 3): including the atmosphere of the image, the event the image describes, emotion the image conveys, and the overall meaning/purpose of the image. \n' + \
        'The questions should be generated based on the information in the image, and the wording should be concise, clear, and easy to understand. You need to generate at least 2 questions for each type of information (at least 6 questions in total). Following the definition of the three types of information, but also be creative and feel free to expand based on the definitions: \n\n' +\
        'For example, for a image where on the left is a woman in yellow rowing a yellow boat, and on the right are a couple rowing a red boat, the questions should be generated as follows: \n' +\
        '1) Within-object information (level 1): \n' +\
        'a) What is the color of the woman\'s boat? \n' +\
        'b) What is the hairstyle of the woman on the left? \n' +\
        '2) Cross-object information (level 2): \n' +\
        'a) What is the relationship between the woman and the couple? \n' +\
        'b) How many boats are there in the image? \n' +\
        '3) Overall interpretation (level 3): \n' +\
        'a) What is the atmosphere of the image? \n' +\
        'b) What activity does the image describe? \n\n' +\
        'Now please generate at least 6 questions for the image attached following the same format. \n'

        resp = ollama.generate(
            model='qwen2.5vl:latest',
            prompt=question_gen_prompt,
            images=[img_b64],
        )
        print(resp['response'])

if __name__ == '__main__':
    gen_questions('ecommerce/images_with_multiple_salient_objects_selected')