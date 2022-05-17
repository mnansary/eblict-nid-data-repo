#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .config import text_conf
#---------------------------------------------------------------
def gaussian_heatmap(size=512, distanceRatio=2):
    '''
        creates a gaussian heatmap
        This is a fixed operation to create heatmaps
    '''
    # distrivute values
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    # create a value mesh grid
    x, y = np.meshgrid(v, v)
    # spreading heatmap
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')
#----------------------------------------------------------------
#----------------------------------------------------------------------------
def get_maps(cbox,heat_map,link_map,prev,idx):
    '''
        creates heat_map and link_map:
        args:
            cbox             : charecter bbox[ cxmin,cymin,cxmax,cymax]
            ghmap : the original heatmap to fit
            heat_map         : image charecter heatmap
            link_map         : link_map of the word
            prev             : list of list of previous charecter center lines
            idx              : index of current charecter
    '''
    ghmap=gaussian_heatmap(size=512, distanceRatio=text_conf.ghmap_dr)
    glmap=gaussian_heatmap(size=512, distanceRatio=text_conf.glmap_dr)
    
    src = np.array([[0, 0], 
                    [ghmap.shape[1], 0], 
                    [ghmap.shape[1],ghmap.shape[0]],
                    [0,ghmap.shape[0]]]).astype('float32')

    
    #--------------------
    # heat map
    #-------------------
    cxmin,cymin,cxmax,cymax=cbox
    # char points
    cx1,cx2,cx3,cx4=cxmin,cxmax,cxmax,cxmin
    cy1,cy2,cy3,cy4=cymax,cymax,cymin,cymin
    heat_points = np.array([[cx1,cy1], 
                            [cx2,cy2], 
                            [cx3,cy3], 
                            [cx4,cy4]]).astype('float32')
    M_heat = cv2.getPerspectiveTransform(src=src,dst=heat_points)
    heat_map+=cv2.warpPerspective(ghmap,M_heat, dsize=(heat_map.shape[1],heat_map.shape[0]),flags=cv2.INTER_NEAREST).astype('float32')

    #-------------------------------
    # link map
    #-------------------------------
    lx2=cx1+(cx2-cx1)/2
    lx3=lx2
    y_shift=(cy4-cy1)/4
    ly2=cy1+(y_shift+y_shift//2)
    ly3=cy4-(y_shift+y_shift//2)
    if prev is not None:
        prev[idx]=[lx2,lx3,ly2,ly3]
        if idx>0:
            lx1,lx4,ly1,ly4=prev[idx-1]
            link_points = np.array([[lx1,ly1], [lx2,ly2], [lx3,ly3], [lx4,ly4]]).astype('float32')
            M_link = cv2.getPerspectiveTransform(src=src,dst=link_points)
            link_map+=cv2.warpPerspective(glmap,M_link, dsize=(link_map.shape[1],link_map.shape[0]),flags=cv2.INTER_NEAREST).astype('float32')

    return heat_map,link_map,prev

def get_maps_from_masked_images(img):
    '''
        args:
            img     :   marked image
        returns:
            img     :       word image
            hmap    :       heat map of the image
            lmap    :       link map of the image
    '''
    # link mask
    lmap=np.zeros(img.shape)
    # heat mask
    hmap=np.zeros(img.shape)

    vals=[v for v in np.unique(img) if v>0]
    num_char=len(vals)
    # maps
    if num_char>1:
        prev=[[] for _ in range(num_char)]
    else:
        prev=None
    
    for cidx,v in enumerate(vals):
        if v>0:
            idx = np.where(img==v)
            y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
            hmap,lmap,prev=get_maps([x_min,y_min,x_max,y_max],hmap,lmap,prev,cidx)
                        
    lmap=lmap.astype("uint8")
    hmap=hmap.astype("uint8")
    return hmap,lmap
