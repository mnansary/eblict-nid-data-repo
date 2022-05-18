#!/bin/sh
save_path="/backup2/NID/data_repo/"
backs_dir="/backup2/NID/data/styles/"
bn_fonts="/backup2/NID/data/fonts/bangla/"
bn_std="/backup2/NID/data/fonts/bangla/Bangla.ttf"
en_fonts="/backup2/NID/data/fonts/english/"
en_std="/backup2/NID/data/fonts/english/English.ttf"
save_dir="/backup2/NID/data/datasets/"

#save_path="/home/ansary/WORK/Work/APSIS/datasets/NID/"
#-----------------------------------------------------------------------------------------------
src_path="${save_path}source/"
card_path="${save_path}cards/"
#------------------------------------------card------------------------------------------------------
python card.py $src_path $save_path --num_data 1500
#-------------------------------------------yolo-----------------------------------------------------
python yolo_loc.py $src_path $card_path $save_path 

#-----------------------------------synthetic------------------------------------------
#python lang.py "bangla" "bn" $save_dir $bn_fonts $backs_dir $bn_std --num_samples 1000000
#python lang.py "english" "en" $save_dir $en_fonts $backs_dir $en_std --num_samples 100000
#-----------------------------------synthetic------------------------------------------
echo succeded
