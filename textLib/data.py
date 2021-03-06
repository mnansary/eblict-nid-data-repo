# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import resource
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont
import math

from .style import get_background, get_blended_data, get_foreground 
from .augmentation import augment
from .config import text_conf,augment_conf
from .utils import create_dir,LOG_INFO,random_exec,post_process_word_image,padAllAround,correctPadding

tqdm.pandas()

#--------------------
# helpers
#--------------------
def createFontImage(font,text):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
        return:
            non-pad-corrected raw binary target
    '''
    
    # draw text
    image = PIL.Image.new(mode='L', size=font.getsize(text))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=text, fill=255, font=font)
    # clear extra white space
    img=np.array(image)
    idx=np.where(img>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    img=img[y_min:y_max,x_min:x_max]
    return img    
    
def createRandomDictionary(language,num_samples):
    '''
        creates a randomized dictionary
        args:
            language        :       language class that holds graphemes,numbers and puncts 
            num_samples     :       number of data to be created if no dictionary is provided       
        returns:
            a dictionary dataframe with "word" and "graphemes"
    '''
    word=[]
    graphemes=[]
    for _ in tqdm(range(num_samples)):
        len_word=random.choices(population=text_conf.lens,weights=text_conf.weights,k=1)[0]
        _graphemes=[]
        _space_added=False
        for _ in range(len_word):
            if text_conf.include_space:
                # space
                if random_exec(weights=text_conf.space_weights,match=1) and not _space_added:
                    num_space=random.randint(0,text_conf.max_space)
                    if num_space>0:
                        for _ in range(num_space):
                            _graphemes.append(" ")
                    _space_added=True
            _ctype=random.choices(population=["g","n","p"],weights=text_conf.comp_weights,k=1)[0]
            if _ctype=="g":    
                _graphemes.append(random.choice(language.dict_graphemes))
            elif _ctype=="n":    
                _graphemes.append(random.choice(language.numbers))
            else:
                _graphemes.append(random.choice(language.punctuations))        
        graphemes.append(_graphemes)
        word.append("".join(_graphemes))
    df=pd.DataFrame({"word":word,"graphemes":graphemes})
    return df 


def mask_negation(mask):
    flat_mask=mask.flatten()
    idx=np.where(flat_mask>0)[0]
    neg_size=random.randint(10,augment_conf.mask_negation_prec)/100
    idx=np.random.choice(idx, size=int(idx.size*neg_size), replace=False)
    flat_mask[idx]=0
    mask=flat_mask.reshape(mask.shape)
    return mask


def create_word_mask(word,font):
    # image
    img=createFontImage(font,word)
    img=post_process_word_image(img,augment_conf)
    img=np.squeeze(img)
    # extend image
    if random_exec(weights=augment_conf.mask_negation_weights):
        mask=mask_negation(img)
    else:
        mask=np.copy(img)
    return mask

def create_single_image(word,font,resources):
    mask=create_word_mask(word,font)
    # image
    back=get_background(mask,resources.backs)
    fore=get_foreground(mask)
    image=get_blended_data(back,fore,mask)
    image=augment(image,base=True)
    img_height=random.randint(text_conf.min_font_text_dim,text_conf.max_font_text_dim)
    h,w,_=image.shape
    w_new=int(img_height* w/h) 
    image=cv2.resize(image,(w_new,img_height))
    return image
    



#--------------------
# ops
#--------------------
def createSyntheticData(iden,
                        save_dir,
                        language,
                        img_dim,
                        resources,
                        num_samples=100000,
                        comp_dim=64,
                        fname_offset=0):
    '''
        creates: 
            * handwriten word image
            * fontspace word image
            * a dataframe/csv that holds word level groundtruth
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    LOG_INFO(save_dir)
    # save_paths
    class save:    
        image=create_dir(save_dir,"image")
        #mask =create_dir(save_dir,"mask")
        #std  =create_dir(save_dir,"std")
        csv=os.path.join(save_dir,"data.csv")
        txt=os.path.join(save_dir,"data.txt")
    
    dictionary=createRandomDictionary(language,num_samples)
    # dataframe vars
    filepaths=[]
    words=[]
    mvals=[]
    fiden=0+fname_offset
    #def_font=PIL.ImageFont.truetype(resources.def_font,comp_dim)
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            fname=f"{fiden}.png"
            word=dictionary.iloc[idx,0]
            font=PIL.ImageFont.truetype(random.choice(resources.fonts),comp_dim)
            image=create_single_image(word,font,resources)
            #-----------------------------------------------------------------------
            image,mval=correctPadding(image,img_dim,ptype="left")
            mvals.append(mval)
            cv2.imwrite(os.path.join(save.image,fname),image)
            filepaths.append(os.path.join(save.image,fname))
            words.append(word)
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{fiden}.png#,#{word}#,#{mval}#\n")
        except Exception as e:
           LOG_INFO(e)
    
    df=pd.DataFrame({"filepath":filepaths,"word":words,"mask":mvals})
    df.to_csv(os.path.join(save.csv),index=False)