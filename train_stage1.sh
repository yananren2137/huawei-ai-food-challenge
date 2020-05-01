nohup python -u train.py > ./log/train_stage1.log 2>&1 &
tail -f ./log/train_stage1.log