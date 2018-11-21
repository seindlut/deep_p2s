#! /bin/sh

export LD_LIBRARY_PATH="/home_directory/cudnn/lib64:$LD_LIBRARY_PATH"

/home_directory/anaconda2/bin/python ./sketchrnn_cnn_train.py --dataset quickdraw_shoe --load_pretrain False --use_jade True
