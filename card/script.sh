#!/bin/sh
save_path="/backup2/NID/data_repo/data/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
#------------------------------------------card------------------------------------------------------
#python card.py $src_path $save_path --num_data 1500
python yolo.py $src_path $card_path $save_path 