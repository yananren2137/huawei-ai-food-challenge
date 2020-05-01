nohup python -u train.py --batch_size=16 --lr=5e-5 --image_size=500 \
	--weight_decay=1e-4 --resize_scale=0.6 --erasing_prob=0.3 \
	--model_path='./checkpoint/best_model_400.pth'> ./log/train_stage2.log 2>&1 &
tail -f ./log/train_stage2.log
