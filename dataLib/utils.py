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

#----------------------------------------
# noise utils
#----------------------------------------
class Modifier:
    def __init__(self,
                blur_kernel_size_max=6,
                blur_kernel_size_min=3,
                bi_filter_dim_min=7,
                bi_filter_dim_max=12,
                bi_filter_sigma_max=80,
                bi_filter_sigma_min=70,
                use_gaussblur=True,
                use_brightness=True,
                use_bifilter=True,
                use_gaussnoise=False,
                use_medianblur=False):

        self.blur_kernel_size_max   =   blur_kernel_size_max
        self.blur_kernel_size_min   =   blur_kernel_size_min
        self.bi_filter_dim_min      =   bi_filter_dim_min
        self.bi_filter_dim_max      =   bi_filter_dim_max
        self.bi_filter_sigma_min    =   bi_filter_sigma_min
        self.bi_filter_sigma_max    =   bi_filter_sigma_max
        self.use_brightness         =   use_brightness
        self.use_bifilter           =   use_bifilter
        self.use_gaussnoise         =   use_gaussnoise
        self.use_gaussblur          =   use_gaussblur
        self.use_medianblur         =   use_medianblur
        
    def __initParams(self):
        self.blur_kernel_size=random.randrange(self.blur_kernel_size_min,
                                               self.blur_kernel_size_max, 
                                               2)
        self.bi_filter_dim   =random.randrange(self.bi_filter_dim_min,
                                               self.bi_filter_dim_max, 
                                               2)
        self.bi_filter_sigma =random.randint(self.bi_filter_sigma_min,
                                             self.bi_filter_sigma_max)
        self.ops             =   [  self.__blur]
        if self.use_medianblur:
            self.ops.append(self.__medianBlur)
        if self.use_gaussblur:
            self.ops.append(self.__gaussBlur)
        if self.use_gaussnoise:
            self.ops.append(self.__gaussNoise)
        if self.use_bifilter:
            self.ops.append(self.__biFilter)
        if self.use_brightness:
            self.ops.append(self.__addBrightness)


    def __blur(self,img):
        return cv2.blur(img,
                        (self.blur_kernel_size,
                        self.blur_kernel_size),
                         0)
    def __gaussBlur(self,img):
        return cv2.GaussianBlur(img,
                                (self.blur_kernel_size,
                                self.blur_kernel_size),
                                0) 
    def __medianBlur(self,img):
        return  cv2.medianBlur(img,
                               self.blur_kernel_size)
    def __biFilter(self,img):
        return cv2.bilateralFilter(img,
                                   self.bi_filter_dim,
                                   self.bi_filter_sigma,
                                   self.bi_filter_sigma)

    def __gaussNoise(self,image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        image = image+gauss
        return image.astype("uint8")
    
    def __addBrightness(self,image):    
        ## Conversion to HLSmask
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)     
        image_HLS = np.array(image_HLS, dtype = np.float64)
        ## generates value between 0.5 and 1.5       
        random_brightness_coefficient = np.random.uniform()+0.5  
        ## scale pixel values up or down for channel 1(Lightness) 
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
        ##Sets all values above 255 to 255    
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255     
        image_HLS = np.array(image_HLS, dtype = np.uint8)    
        ## Conversion to RGB
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)     
        return image_RGB
    
    def noise(self,img):
        self.__initParams()
        img=img.astype("uint8")
        idx = random.choice(range(len(self.ops)))
        img = self.ops.pop(idx)(img)
        return img