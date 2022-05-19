#!/bin/sh
backs_dir="/backup2/NID/data/styles/"
bn_fonts="/backup2/NID/data/fonts/bangla/"
bn_std="/backup2/NID/data/fonts/bangla/Bangla.ttf"
en_fonts="/backup2/NID/data/fonts/english/"
en_std="/backup2/NID/data/fonts/english/English.ttf"
save_dir="/backup2/NID/data/datasets/"

#-----------------------------------synthetic------------------------------------------
python lang.py "bangla" "bnrs" $save_dir $bn_fonts $backs_dir $bn_std --num_samples 1000000
python lang.py "english" "enrs" $save_dir $en_fonts $backs_dir $en_std --num_samples 200000
#-----------------------------------synthetic------------------------------------------
echo succeded
