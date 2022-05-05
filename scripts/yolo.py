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
from dataLib.locator import render_data,data_classes,marking
from dataLib.utils import *
from dataLib.config import aug

from tqdm.auto import tqdm
from glob import glob
import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tqdm.pandas()

def main(args):
    #-----------------
    card_dir=args.card_dir
    src_dir =args.src_dir
    save_dir=args.save_dir
    save_dir=create_dir(save_dir,"yolo")
    img_dir =create_dir(save_dir,"images")
    data_csv =os.path.join(save_dir,"data.csv")
    data_dim=int(args.data_dim)
    
    src=Data(src_dir)
    LOG_INFO(save_dir)
    backgen=src.backgroundGenerator()
    # data division
    card_img_dir =os.path.join(card_dir,"images")
    img_paths=[img_path for img_path in tqdm(glob(os.path.join(card_img_dir,"*.*")))]
    
    df=pd.DataFrame({"file":img_paths})
    df["fname"]=df["file"].progress_apply(lambda x: os.path.basename(x))
    df["card_type"]=df["fname"].progress_apply(lambda x: x.split("_")[0])
    df["face"]=df["fname"].progress_apply(lambda x: x.split("_")[1])
    df=df[["file","card_type","face"]]
    df=df.sample(frac=1)
    

    dicts=[]
    LOG_INFO(data_classes)
    
    for idx in tqdm(range(len(df))):
        img_path =df.iloc[idx,0]
        card_type=df.iloc[idx,1]
        face     =df.iloc[idx,2]
        img,mask=render_data(backgen,img_path,face,aug)
        img,_   =padDetectionImage(img,pad_value=0)
        mask,_  =padDetectionImage(mask,gray=True)
        # save
        img=cv2.resize(img,(data_dim,data_dim))
        mask=cv2.resize(mask,(data_dim,data_dim),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)

        # saving
        cv2.imwrite(os.path.join(img_dir,f"{card_type}_{face}_{idx}.jpg"),img)
        vals=np.unique(mask)
        # yolo data
        for k,v in marking.items():
            try:
                if v in vals:
                    imdx=np.where(mask==v)
                    y,y_max,x,x_max = np.min(imdx[0]), np.max(imdx[0]), np.min(imdx[1]), np.max(imdx[1])
                    h=y_max-y
                    w=x_max-x
                    x_center=x+w//2
                    y_center=y+h//2
                    _classes=data_classes.index(k)
                    image_id=f"{card_type}_{face}_{idx}"
                    dicts.append({"image_id":image_id,
                                "x":x,
                                "y":y,
                                "w":w,
                                "h":h,
                                "x_center":x_center,
                                "y_center":y_center,
                                "classes":_classes
                                })
            except Exception as e:
                continue        
                
                
   
    df=pd.DataFrame(dicts)
    df.to_csv(data_csv,index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Synthetic NID/Smartcard Segmentation Data Creation Script")
    parser.add_argument("src_dir", help="Path to source data")
    parser.add_argument("card_dir", help="Path to cards data")
    parser.add_argument("save_dir", help="Path to save the processed data")
    parser.add_argument("--data_dim",required=False,default=512,help="dimension of data to save the images")
    
    args = parser.parse_args()
    main(args)