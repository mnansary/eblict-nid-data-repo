# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont
import math
from matplotlib import pyplot as plt
from .style import get_background, get_blended_data, get_foreground 
from .augmentation import get_augmented_data
from .config import text_conf,augment_conf
from .utils import create_dir,LOG_INFO, post_process_images,random_exec,post_process_word_image,padAllAround,correctPadding
from .craft  import get_maps_from_masked_images
tqdm.pandas()
#--------------------
# helpers
#--------------------
def createFontData(font,comps):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
        return:
            word mask,char-region mask
    '''
    text="".join(comps)
    tsize=font.getsize(text)
    comp_str=''
    imgs=[]
    for comp in comps:
        comp_str+=comp
        # draw
        image = PIL.Image.new(mode='L', size=tsize)
        draw = PIL.ImageDraw.Draw(image)
        draw.text(xy=(0, 0), text=comp_str, fill=1, font=font)
        _img=np.array(image)
        _img[_img>0]=1
        imgs.append(_img)
        
    
    # clear extra white space
    img=sum(imgs)
    idx=np.where(img>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    img=img[y_min:y_max,x_min:x_max]
    # offset
    vals=list(np.unique(img))
    vals=sorted(vals,reverse=True)
    vals=vals[:-1]
    
    iden=2
    c_img=np.zeros(img.shape)
    for v in vals:
        c_img[img==v]=iden
        iden+=1 
    
    # text mask
    image = PIL.Image.new(mode='L', size=tsize)
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=text, fill=255, font=font)
    wmask=np.array(image)
    idx=np.where(wmask>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    wmask=wmask[y_min:y_max,x_min:x_max]
    
    hmap,lmap=get_maps_from_masked_images(c_img)

    return wmask,hmap,lmap    
    
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


def create_masks(comps,font,comp_dim):
    # image
    img,hmap,lmap=createFontData(font,comps)
    #resize to std height
    h,w=img.shape
    w_new=int(comp_dim* w/h) 
    images=[img,hmap,lmap]
    for idx,img in enumerate(images):
        img=cv2.resize(img,(w_new,comp_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        images[idx]=img
    
    images=post_process_images(images,augment_conf)
    
    for idx,img in enumerate(images):
        img=np.squeeze(img)
        images[idx]=img
    
    # extend image
    if random_exec(weights=text_conf.ext_weights):
        img=images[0]
        hi,wi=img.shape
        ptype=random.choice(["tb","lr",None])
        pdim=math.ceil(0.01*wi*random.randint(text_conf.min_pad_perc,text_conf.max_pad_perc))
        for idx,img in enumerate(images):
            img=padAllAround(img,pdim,0,pad_single=ptype)
            images[idx]=img
        
    # resize to std height
    img=images[0]
    h,w=img.shape
    w_new=int(comp_dim* w/h) 
    for idx,img in enumerate(images):
        img=cv2.resize(img,(w_new,comp_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        images[idx]=img
    
    img=images[0]
    if random_exec(weights=augment_conf.mask_negation_weights):
        mask=mask_negation(img)
    else:
        mask=np.copy(img)
    img,hmap,lmap=images
    return mask,img,hmap,lmap 



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
        wmask=create_dir(save_dir,"mask")
        form=create_dir(save_dir,"form")
        hmap =create_dir(save_dir,"hmap")
        lmap =create_dir(save_dir,"lmap")
        csv=os.path.join(save_dir,"data.csv")
        txt=os.path.join(save_dir,"data.txt")
    
    dictionary=createRandomDictionary(language,num_samples)
    # dataframe vars
    filepaths=[]
    words=[]
    fiden=0+fname_offset
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            # word mask
            fsize=random.randint(text_conf.min_font_text_dim,text_conf.max_font_text_dim)
            font=PIL.ImageFont.truetype(random.choice(resources.fonts),fsize)
            mask,wmask,hmap,lmap=create_masks(comps,font,comp_dim)
            # image
            back=get_background(mask,resources.backs)
            fore=get_foreground(mask,resources.backs)
            image=get_blended_data(back,fore,mask)
            image=get_augmented_data(image)
            #-----------------------------------------------------------------------
            # save
            fname=f"{fiden}.png"
            ## image
            image,_=correctPadding(image,img_dim,ptype="left")
            cv2.imwrite(os.path.join(save.image,fname),image)
            
            wmask,_=correctPadding(wmask,img_dim,ptype="left",pvalue=0)
            cv2.imwrite(os.path.join(save.wmask,fname),wmask)

            form=255-wmask
            form=cv2.merge((form,form,form))
            form,_=correctPadding(form,img_dim,ptype="left")
            cv2.imwrite(os.path.join(save.form,fname),form)

            hmap,_=correctPadding(hmap,img_dim,ptype="left",pvalue=0)
            cv2.imwrite(os.path.join(save.hmap,fname),hmap)

            lmap,_=correctPadding(lmap,img_dim,ptype="left",pvalue=0)
            cv2.imwrite(os.path.join(save.lmap,fname),lmap)

            #-----------------------------------------------------------------------

            filepaths.append(os.path.join(save.image,fname))
            words.append("".join(comps))
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{fiden}.png#,#{''.join(comps)}#\n")
        except Exception as e:
           LOG_INFO(e)
    
    df=pd.DataFrame({"filepath":filepaths,"word":words})
    df.to_csv(os.path.join(save.csv),index=False)
