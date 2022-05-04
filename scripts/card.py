# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
from dataLib.data import Data
from dataLib.utils import *
from tqdm.auto import tqdm
import os
import cv2
import random
import pandas as pd
import json
import matplotlib.pyplot as plt


def main(args):
    data_dir=args.data_dir
    save_dir=args.save_dir
    face    =args.face
    use_smart_only=args.use_smart_only
    save_dir=create_dir(save_dir,face)
    img_dir =create_dir(save_dir,"images")
    mask_dir=create_dir(save_dir,"masks")
    anon_dir=create_dir(save_dir,"anons")
    data_csv =os.path.join(save_dir,"data.csv")
    n_data=int(args.num_data)
    src=Data(data_dir)
    LOG_INFO(save_dir)

    dicts=[]

    for card_type in ["nid","smart"]:
        if card_type=="nid":
            aug=Modifier(use_brightness=False)
            if use_smart_only:
                continue
        else:
            aug=Modifier()
        for i in tqdm(range(n_data)):
            #try:
            if face=="front":
                image,mask,data=src.createCardFront(card_type)
            else:
                image,mask,data=src.createCardBack(card_type)
            image=enhanceImage(image)
            image=aug.noise(image)
            # save
            cv2.imwrite(os.path.join(img_dir,f"{card_type}_{i}.png"),image)
            cv2.imwrite(os.path.join(mask_dir,f"{card_type}_{i}.png"),mask)
            data["file"]=f"{card_type}_{i}.png"
            with open(os.path.join(anon_dir,f"{card_type}_{i}.json"), 'w') as fp:
                json.dump(data, fp,sort_keys=True, indent=4,ensure_ascii=False)
            dicts.append(data)    
            # except Exception as e:
            #     pass
    df=pd.DataFrame(dicts)
    df.to_csv(data_csv,index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Data Creation Script")
    parser.add_argument("data_dir", help="Path to source data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("face", help="front or back")
    parser.add_argument("--num_data",required=False,default=100000,help ="number of data to create : default=100000")
    parser.add_argument("--use_smart_only",type=str2bool,required=False,default=False,help ="generate smart only")
    args = parser.parse_args()
    main(args)