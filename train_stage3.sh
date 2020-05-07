nohup python -u train.py --batch_size=16 --lr=1e-6 --image_size=500 \
	--weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 --cutmix \
	--label_smooth --model_path='./checkpoint/best_model_400.pth'> ./log/train_stage3.log 2>&1 &
tail -f ./log/train_stage3.log
