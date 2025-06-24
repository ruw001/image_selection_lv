import os
import json
import random
import tqdm
import base64
from openai import OpenAI
from PIL import Image
import io

def gen_questions(img_dir):
    # Initialize OpenAI client
    client = OpenAI()

    complete_output = []
    
    imgs = [f for f in os.listdir(img_dir) if f != '.DS_Store']
    # sort by source, and then by index
    imgs.sort(key=lambda x: (x.split('_')[0], int(x.split('_')[1])))
    for idx, img_filename in tqdm.tqdm(enumerate(imgs[:3])):
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
        'Question to induce alternating attention can be about examining the relationship, interaction, or feature differences between multiple objects. \n' +\
        'Question to induce divided attention can be about searching for a specific object, traverse over multiple objects without sustained focus on any specific objects (e.g., counting), or explore the scene without to get the emotion, atmosphere, event, or purpose of the image. \n\n' +\
        'For example, for a image where on the left is a woman in yellow rowing a yellow boat, and on the right are a couple rowing a red boat, \n' +\
        'one possible question to induce focused attention can be "What is the color of the woman\'s boat?" \n' +\
        'One possible question to induce alternating attention can be "What is the relationship between the woman and the couple?" \n' +\
        'One possible question to induce divided attention can be "what event does this image describe?" \n' + \
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
        'The questions should be concise, clear, and easy to understand by a person with average visual ability. The target object involved in the questions should be the salient object in the image. \n' +\
        'The questions should be in English. \n' +\
        'The questions should be based on the definition of the three types of attention, but please be creative and feel free to expand based on the definitions. But please DO NOT hallucinate. \n' + \
        'The questions should make sense and represent visual tasks people would do in their daily life. \n' + \
        'For example, a question like "What is the relationship between the person and the grass in the garden?" is NOT a good question because person and grasss seem to be irrelevant to each other, and this is not a visual task people would usually do in their daily life. \n' + \
        'another example is "What is the difference between the fountain and the bench on the image?" is NOT a good question because the objects being compared are too different and not the same type of objects. \n'

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
    
    with open('all_imgs_final_questions.txt', 'w') as f:
        for item in complete_output:
            f.write(item['img_filename'] + '\n\n')
            f.write(item['questions'] + '\n\n')

if __name__ == '__main__':
    gen_questions('all_imgs_final')