#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import math
import pandas as pd 
import tensorflow as tf
import numpy as np 
from ast import literal_eval
from tqdm.auto import tqdm
from .utils import *
tqdm.pandas()
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
cols=["filepath","mask","label"]
eval_cols=cols[1:]
    
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def toTfrecord(df,rnum,rec_path,iden):
    tfrecord_name=f'id_{iden}_rec_{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            # base
            img_path=df.iloc[idx,0]
            # img
            with(open(img_path,'rb')) as fid:
                image_png_bytes=fid.read()
            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes)}

            for cidx,col in enumerate(cols):
                if col in eval_cols:
                    data[col]=_int64_list_feature(df.iloc[idx,cidx]) 

            
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  

def createRecords(data,save_path,tf_size,iden):
    if type(data)==str:
        data=pd.read_csv(data)
        for col in eval_cols:
            data[col]=data[col].progress_apply(lambda x: literal_eval(x))
    
    LOG_INFO(f"Creating TFRECORDS No folds:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path,iden)

    
    