#!/bin/sh
save_path="/home/ansary/WORK/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
#------------------------------------------card------------------------------------------------------
python card.py $src_path $save_path --num_data 1500
#-------------------------------------------yolo-----------------------------------------------------
python yolo.py $src_path $card_path $save_path
echo succeded