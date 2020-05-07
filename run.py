import os


stage1 = 'python -u ./source_code/train.py'

stage2 = "python -u ./source_code/train.py --batch_size=16 --lr=5e-5 --image_size=500\
	--weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 \
	--model_path='./source_code/checkpoint/best_model_400.pth'"

stage3 = "python -u ./source_code/train.py --batch_size=16 --lr=1e-6 --image_size=500\
	--weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 --cutmix \
	--label_smooth --model_path='./source_code/checkpoint/best_model_500.pth'"

print('*********Stage1***********')
os.system(stage1)
print('*********Stage2***********')
os.system(stage2)
print('*********Stage3***********')
os.system(stage3)
print('train finished')




