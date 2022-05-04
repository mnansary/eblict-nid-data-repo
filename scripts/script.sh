#!/bin/sh
save_path="/home/ansary/WORK/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
front_path="${save_path}front/"
back_path="${save_path}back/"
#------------------------------------------card------------------------------------------------------
#python card.py $src_path $save_path "back" --num_data 5
python card.py $src_path $save_path "front" --num_data 5 
#-------------------------------------------yolo-----------------------------------------------------
# python datagen_yolo.py $src_path $back_path $save_path "back" --num_data 50000
# python datagen_yolo.py $src_path $front_path $save_path "front" --num_data 5
#-------------------------------------------dbnet-----------------------------------------------------
echo succeded