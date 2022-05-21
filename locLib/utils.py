#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance
import argparse
from math import pi
#---------------------------------------------------------------
# common utils
#---------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def randColor(depth,single=False):
    '''
        generates random color
    '''
    if single:
        d=random.randint(0,depth)
        return (d,d,d)
    else:
        r=random.randint(0,depth)
        g=random.randint(0,depth)
        b=random.randint(0,depth)
        return (r,g,b)


def random_exec(poplutation=[0,1],weights=[0.7,0.3],match=0):
    return random.choices(population=poplutation,weights=weights,k=1)[0]==match
#---------------------------------------------------------------

def padToFixedHeight(img,h_max):
    '''
        pads an image to fixed height 
    '''
    # shape
    h,w=img.shape
    if h<h_max:    
        # pad heights
        pad_height_bot =h_max-h                
        pad_bot =np.zeros((pad_height_bot,w))
        # pad
        img =np.concatenate([img,pad_bot],axis=0)
    elif h>h_max:
        w_new=int(h_max*w/h)
        img = cv2.resize(img, (w_new,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    img = cv2.resize(img, (w,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img

def padToFixedHeightWidth(img,h_max,w_max):
    '''
        pads an image to fixed height and width
    '''
    # shape
    h,w=img.shape
    if w<w_max:
        # pad widths
        pad_width =(w_max-w)        
        pad =np.zeros((h,pad_width))
        # pad
        img =np.concatenate([img,pad],axis=1)
    elif w>w_max: # reduce height
        h_new=int(w_max*h/w)
        img = cv2.resize(img, (w_max,h_new), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # shape
    h,w=img.shape
    if h<h_max:    
        # pad heights
        pad_height_top =1+(h_max-h)//2
        pad_height_bot =1+h_max-h-pad_height_top
                
        pad_top =np.zeros((pad_height_top,w))
        pad_bot =np.zeros((pad_height_bot,w))
        # pad
        img =np.concatenate([pad_top,img,pad_bot],axis=0)
    elif h>h_max:
        w_new=int(h_max*w/h)
        img = cv2.resize(img, (w_new,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    img = cv2.resize(img, (w_max,h_max), fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img

def enhanceImage(img):
    '''
        enhances an image based on contrast
    '''
    img=Image.fromarray(img)
    if random_exec(weights=[0.5,0.5]):
        # color
        factor=random.randint(1,5)
        col_enhancer = ImageEnhance.Color(img)
        img= col_enhancer.enhance(factor)
    if random_exec(weights=[0.5,0.5]):
        # contrast
        factor=random.randint(1,5)
        con_enhancer = ImageEnhance.Contrast(img)
        img= con_enhancer.enhance(factor)
    img=np.array(img)
    return img

#---------------------------------------------------------------
def padDetectionImage(img,gray=False,pad_value=255):
    cfg={}
    if gray:
        h,w=img.shape
    else:
        h,w,d=img.shape
    if h>w:
        # pad widths
        pad_width =h-w
        # pads
        if gray:
            pad =np.zeros((h,pad_width))
        else:    
            pad =np.ones((h,pad_width,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=1)
        # cfg
        cfg["pad"]="width"
        cfg["dim"]=w
    
    elif w>h:
        # pad height
        pad_height =w-h
        # pads
        if gray:
            pad=np.zeros((pad_height,w))
        else:
            pad =np.ones((pad_height,w,d))*pad_value
        # pad
        img =np.concatenate([img,pad],axis=0)
        # cfg
        cfg["pad"]="height"
        cfg["dim"]=h
    else:
        cfg=None
    if not gray:
        img=img.astype("uint8")
    return img,cfg




#--------------------
# augment data
#--------------------
def rotate_image(mat, angle):
    """
        Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    return rotated_mat,rotation_mat

def get_image_coords(curr_coord,M):
    '''
        returns rotated co-ords
        args:
            curr_coord  : list of co-ords
            M           : rotation matrix
    '''
    curr_coord=np.float32(curr_coord)
    # co-ord change
    new_coord=[]
    curr_coord=np.concatenate([curr_coord,np.ones((4,1))],axis=1)
    for c in curr_coord:
        dot=np.dot(M,c)
        new_coord.append([int(i) for i in dot])
    return new_coord

def check_visibility(image, mask):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1