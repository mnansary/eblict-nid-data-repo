#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

from multiprocessing import Pool
from glob import glob

import os 
import json
import pandas as pd 


from rsLib.utils import *
from rsLib.processing import processData
from rsLib.store import createRecords
from rsLib.vocab import vocabs
#--------------------
# main
#--------------------
def generate(csv):
    img_height  =   64
    img_width   =   512
    iden        =   os.path.basename(csv).split(".")[0]
    seq_max_len =   40
    tf_size     =   1024
    vocab_iden  =   "all"
    vocab=vocabs[vocab_iden]
    img_dim=(img_height,img_width)
    data_dir=os.path.dirname(csv)
    # processing
    df=processData(csv,vocab,seq_max_len,img_dim)
    # storing
    save_path=create_dir(data_dir,iden)
    LOG_INFO(save_path)
    createRecords(df,save_path,tf_size,iden)

def save_config():
    img_height  =   64
    img_width   =   512
    seq_max_len =   40
    tf_size     =   1024
    vocab_iden  =   "all"
    vocab=vocabs[vocab_iden]
    config_json  =   "../config.json"
    config={"vocab":vocab,
            "pos_max":seq_max_len,
            "img_height":img_height,
            "img_width" :img_width,
            "tf_size":tf_size,
            "vocab_iden":vocab_iden,
            "zip_iden":"multi"}

    with open(config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)




if __name__=="__main__":
    data_dir="/backup2/NID/data/datasets/proc/tfr"
    div=2
    #-------------------------------------------------------------------------
    csvs=[csv for csv in glob(os.path.join(data_dir,"*.csv"))]
    print("csvs:",len(csvs))
    for i in range(0,len(csvs),div):
        _csvs=csvs[i:i+div]
        with Pool(processes=len(_csvs)) as pool:
            pool.map(generate,_csvs) 
    LOG_INFO("Processing done")
    save_config() 
