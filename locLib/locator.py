# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from .utils import *
from .config import marking
from indicparser import graphemeParser
gp=graphemeParser("bangla")

#---------------------------------------------------------------------------------------

def get_warped_image(img,mask,src,xwarp,ywarp,warp_vec):
    height,width,_=img.shape
 
    # construct dict warp
    x1,y1=src[0]
    x2,y2=src[1]
    x3,y3=src[2]
    x4,y4=src[3]
    # construct destination
    dx=int(width*xwarp)
    dy=int(height*ywarp)
    # const
    if warp_vec=="p1":
        dst= [[dx,dy], [x2,y2],[x3,y3],[x4,y4]]
    elif warp_vec=="p2":
        dst=[[x1,y1],[x2-dx,dy],[x3,y3],[x4,y4]]
    elif warp_vec=="p3":
        dst= [[x1,y1],[x2,y2],[x3-dx,y3-dy],[x4,y4]]
    else:
        dst= [[x1,y1],[x2,y2],[x3,y3],[dx,y4-dy]]
    M   = cv2.getPerspectiveTransform(np.float32(src),np.float32(dst))
    img = cv2.warpPerspective(img, M, (width,height))
    mask= cv2.warpPerspective(mask, M, (width,height),flags=cv2.INTER_NEAREST)
    return img,mask,dst
#---------------------------------------------------------------------------------------

def augment_img_base(img_path,mask_path,face,config):
    
    img=cv2.imread(img_path)
    height,width,d=img.shape
    warp_types=["p1","p2","p3","p4"]
    
    mask=cv2.imread(mask_path,0)
    mask[mask==0]=marking[face]
    curr_coord=[[0,0], 
                [width-1,0], 
                [width-1,height-1], 
                [0,height-1]]
    
    if random_exec(weights=[0.6,0.4],match=0):    
        w1type=warp_types[random.choice([0,2])]
        w2type=warp_types[random.choice([1,3])]
        # warping calculation
        xwarp=random.randint(0,config.max_warp_perc)/100
        ywarp=random.randint(0,config.max_warp_perc)/100
        
        img,mask,curr_coord=get_warped_image(img,mask,curr_coord,xwarp,ywarp,w1type)
        img,mask,curr_coord=get_warped_image(img,mask,curr_coord,xwarp,ywarp,w2type)


    if random_exec(weights=[0.3,0.7],match=0): 
        # plane rotation
        angle=random.randint(-config.max_rotation,config.max_rotation)
        img,M =rotate_image(img,angle)
        mask,_=rotate_image(mask,angle)
        curr_coord=get_image_coords(curr_coord,M)
        
    # scope rotation
    if config.use_scope_rotation:
        if random_exec():
            flip_op=random.choice([-90,180,90])                  
            img,M=rotate_image(img,flip_op)
            mask,_=rotate_image(mask,flip_op)
            curr_coord=get_image_coords(curr_coord,M)
            
   
    return img,mask,curr_coord

#---------------------------------------------------------------------------------------
def pad_image_mask(img,mask,coord,config):
    '''
        pads data 
    '''
    h,w,d=img.shape
    coord=np.array(coord)
    # change vars
    w_pad_left=int(w*( random.randint(2,config.max_pad_perc) /100))
    h_pad_top =int(h*( random.randint(2,config.max_pad_perc) /100))
    # correct co-ordinates lr
    coord[:,0]+=w_pad_left
    # image left right
    left_pad=np.zeros((h,w_pad_left,d))
    w_pad_right=int(w*( random.randint(2,config.max_pad_perc) /100))
    right_pad=np.zeros((h,w_pad_right,d))
    img=np.concatenate([left_pad,img,right_pad],axis=1)
    mask=np.concatenate([cv2.cvtColor(left_pad.astype("uint8"),cv2.COLOR_BGR2GRAY) ,
                        mask,
                        cv2.cvtColor(right_pad.astype("uint8"),cv2.COLOR_BGR2GRAY)],axis=1)
    # correct co-ordinates lr
    coord[:,1]+=h_pad_top
    # image top bottom
    h,w,d=img.shape
    top_pad=np.zeros((h_pad_top,w,d))
    h_pad_bot=int(h*( random.randint(2,config.max_pad_perc) /100))
    bot_pad=np.zeros((h_pad_bot,w,d))
    img=np.concatenate([top_pad,img,bot_pad],axis=0)
    mask=np.concatenate([cv2.cvtColor(top_pad.astype("uint8"),cv2.COLOR_BGR2GRAY),
                        mask,
                        cv2.cvtColor(bot_pad.astype("uint8"),cv2.COLOR_BGR2GRAY)],axis=0)
    return img,mask,coord
    

def render_data(backgen,img_path,mask_path,face,config):
    # base augment
    img,mask,coord=augment_img_base(img_path,mask_path,face,config)    
        
    # background
    if random_exec(weights=[0.7,0.3],match=0):
        # pad
        img,mask,coord=pad_image_mask(img,mask,coord,config)
        back=next(backgen)
        h,w,d=img.shape
        back=cv2.resize(back,(w,h))
    else:
        back=np.copy(img)
        coord=np.array(coord)

    back[mask>0]=img[mask>0]
    
    return back,mask