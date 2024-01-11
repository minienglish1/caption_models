export CUDA_VISIBLE_DEVICES=0
source venv/bin/activate
python instructBLIP_0.4.py --img_dir /mnt/storage/caption/test --prompts /mnt/storage/caption/prompts.txt --model Salesforce/instructblip-flan-t5-xl
