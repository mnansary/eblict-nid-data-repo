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
from locLib.data import Data
from locLib.utils import *
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
    save_dir=create_dir(save_dir,"cards")
    img_dir =create_dir(save_dir,"images")
    mask_dir=create_dir(save_dir,"masks")
    anon_dir=create_dir(save_dir,"anons")
    n_data=int(args.num_data)
    src=Data(data_dir)
    LOG_INFO(save_dir)

    for face in ["front","back"]:
        for card_type in ["nid","smart"]:
            for i in tqdm(range(n_data)):
                try:
                    if face=="front":
                        image,mask,data=src.createCardFront(card_type)
                    else:
                        image,mask,data=src.createCardBack(card_type)
                    if random_exec(weights=[0.4,0.6]):
                        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                        if random_exec(weights=[0.5,0.5]):
                            _,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            thcheck=cv2.merge((thresh,thresh,thresh))
                            cmask=np.copy(mask)
                            cmask[mask>0]=255
                            cmask=cmask.astype("uint8")
                            if check_visibility(thcheck,cmask):
                                image=np.copy(thresh)
                                
                    
                    # save
                    cv2.imwrite(os.path.join(img_dir,f"{card_type}_{face}_{i}.png"),image)
                    cv2.imwrite(os.path.join(mask_dir,f"{card_type}_{face}_{i}.png"),mask)
                    data["file"]=f"{card_type}_{face}_{i}.png"
                    with open(os.path.join(anon_dir,f"{card_type}_{face}_{i}.json"), 'w') as fp:
                        json.dump(data, fp,sort_keys=True, indent=4,ensure_ascii=False)
                except Exception as e:
                    print(e)
    
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Data Creation Script")
    parser.add_argument("data_dir", help="Path to source data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--num_data",required=False,default=100000,help ="number of data to create : default=100000")
    args = parser.parse_args()
    main(args)