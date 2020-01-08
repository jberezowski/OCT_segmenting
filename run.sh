time python ./train_script.py unet 200 labels_onehot dice | tee training_logs/unet-training-1.txt
time python ./train_script.py renet 200 labels_onehot dice | tee training_logs/renet-training-1.txt
