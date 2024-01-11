#instructBLIP.py : Python 3.10.12 ,  GPL-3.0 license 
#use instruct to create custom question based captions for images
#0.1 - initial setup
#0.2 - add fp16, os walk, add image_dir argument
#0.3 - add load list of prompts from text file
#0.4 - add iter per unit time

import os
import time
import argparse
import requests
from pathlib import Path

import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


#argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, default="images", help="path to image directory")
parser.add_argument("--prompts", type=str, default="prompts.txt", help="path to image directory")
args = parser.parse_args()


#initiate variables & lists
image_ext = [".jpg", ".png", ".jpeg", ".bmp", ".jfif", ".webp", ".tif"]
image_dir = Path(args.img_dir)
#fp16: torch_dtype=torch.float16, device_map="auto"
#8bit: load_in_8bit=True, device_map="auto"
prompt_file = args.prompts
prompts = [] #list of prompts

#load list of prompts
with open(prompt_file, "r") as f:
    prompts = f.readlines() #each line to an item in list
    prompts = [line.strip() for line in prompts]  # Remove newlines
    print("Prompt list file opened: ' + prompt_file")
    print("Prompts:")
    for prompt in prompts:
        print(prompt)



processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


#os.walk to collect images
captions_processed = 0 #initial count
start_time = time.time() #start timer
for root, dirs, files in os.walk(image_dir):
    for file in files:
        #check if image
        ext = os.path.splitext(file)[1]
        if ext.lower() in image_ext:
            #collect needed names, dirs, & paths
            basename = os.path.splitext(file)[0]
            image_path = Path(root, file)
            image_dir = Path(root)
            image = Image.open(image_path).convert("RGB")
            caption_file = Path(image_dir, basename + ".txt")

            #iterate through prompt list
            for prompt in prompts:
                text = prompt

                #execute instructBLIP model
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

                outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=256,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                )
                
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                #append output to caption_file
                with open(caption_file, "a") as f:
                    if prompt == prompt[0]:
                        f.write({generated_text})
                    else:
                        f.write(f", {generated_text}")
                
                #print to terminal
                print(f"\nimage: {file}")
                print(f"caption: {generated_text}")





