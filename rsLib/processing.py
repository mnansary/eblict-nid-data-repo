# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import pandas as pd 
import cv2
import math
import numpy as np
from tqdm import tqdm
from .utils import *
tqdm.pandas()
from indicparser import graphemeParser
GP=graphemeParser("bangla")
vd       =    ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
cd       =    ['ঁ']
#--------------------
# helpers
#--------------------
def reset(df):
    # sort df
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True) 
    return df 


#---------------------------------------------------------------
def processMasks(df,img_dim,factor=32):
    img_height,img_width=img_dim
    masks=[]
    for idx in tqdm(range(len(df))):
        try:
            imask   =   df.iloc[idx,-1]
            # mask
            imask=math.ceil((imask/img_width)*(img_width//factor))
            mask=np.zeros((img_height//factor,img_width//factor))
            mask[:,:imask]=1
            mask=mask.flatten().tolist()
            mask=[int(i) for i in mask]
            masks.append(mask)
            
        except Exception as e:
            masks.append(None)
            LOG_INFO(e)
    df["mask"]=masks
    df=reset(df)
    return df

#---------------------------------------------------------------
def get_label(text,vocab,max_len):
    try:
        _label=[]
        text=str(text)
        graphemes=GP.process(text)
        for grapheme in graphemes:
            _vd=None
            _cd=None
            for v in vd:
                if v in grapheme:
                    _vd=v
                    grapheme=grapheme.replace(v,'')
                    break
            for c in cd:
                if c in grapheme:
                    _cd=v
                    grapheme=grapheme.replace(c,'')
                    break 
            _label.append(grapheme)
            if _vd is not None:
                _label.append(_vd)
            if _cd is not None:
                _label.append(_cd)
        _label=["start"]+_label+["end"]
        for p in range(max_len - len(_label)):
            _label.append("pad")
        label=[]
        for v in _label:
            label.append(vocab.index(v))
        return label
    except Exception as e:
        print(e,text)
        return None
#------------------------------------------------
def processData(csv,vocab,max_len,img_dim):
    LOG_INFO(csv)
    df=pd.read_csv(csv)
    # masks
    df=processMasks(df,img_dim)
    df.to_csv(csv,index=False)
    # labels
    df["label"]=df.word.progress_apply(lambda x:get_label(x,vocab,max_len))
    df=reset(df)
    # save data
    cols=["filepath","mask","label"]
    df=df[cols]
    df.to_csv(csv,index=False)
    return df