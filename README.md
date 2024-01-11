# caption_models
Repo for image-to-text model caption projects

GPL-3.0 license 

Tested with Ubuntu 22.04, Python 3.10.12

Setup:  
git clone into directory  
cd into directory  
setup venv:  
	python3 -m venv venv  
	source venv/bin/activate  
	pip install -U -r requirements.txt  

## instructBLIP

run:  
python instructBLIP_#.#.py --img_dir PATH/TO/IMAGE_DIR --prompts PATH/TO/prompts.txt --model model_name

replace #.# with script version number

prompts.txt:  
Designed for using multiple prompts per image.  
Create a list of prompts in a text file, example is prompts.txt  
Per prompt, the generated caption is ", " appended to image_name.txt file  

additional aruguments:  
--overwrite : overwrites extisting caption .txt files  

example .sh file: instructBLIP.sh
 
Available models are:  
  Salesforce/instructblip-vicuna-7b  
  Salesforce/instructblip-vicuna-13b  
  Salesforce/instructblip-flan-t5-xl  
  Salesforce/instructblip-flan-t5-xxl  
  

