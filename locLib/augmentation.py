#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import imgaug.augmenters as iaa
import random
import numpy as np
import cv2 
from .utils import random_exec

def additiveGaussian(img):
    scale=np.random.uniform(8,32)
    per_channel=np.random.rand() < 0.5
    aug = iaa.AdditiveGaussianNoise(scale=scale, per_channel=per_channel)
    img = aug(image=img)
    return img

def coarseDropout(img):
    p =np.random.uniform(0.05, 0.25)
    per_channel=np.random.rand() < 0.5
    size_px = None
    size_percent = None
    aug = iaa.CoarseDropout(p=p, size_px=size_px, size_percent=size_percent, per_channel=per_channel)
    img = aug(image=img)
    return img
    
def elasticDistortion(img):
    alpha = np.random.uniform(10,15)
    sigma = np.random.uniform(3,3)
    aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest")
    img = aug(image=img)
    return img

def gaussianBlur(img):
    sigma = np.random.randint(1, 3)
    aug = iaa.GaussianBlur(sigma=sigma)
    img = aug(image=img)
    return img

def medianBlur(img):
    k = np.random.choice([1, 3])
    aug = iaa.MedianBlur(k=k)
    img = aug(image=img)
    return img

def motionBlur(img):
    k = np.random.choice([3,5,7])
    angle = np.random.uniform(0,360)
    aug = iaa.MotionBlur(k=k, angle=angle)
    img = aug(image=img)
    return img


BLURS=[iaa.AverageBlur(),iaa.GaussianBlur(),iaa.MedianBlur(),
       iaa.BilateralBlur(),iaa.MotionBlur(),iaa.MeanShiftBlur()]
CONTRASTS=[iaa.GammaContrast(),
        iaa.SigmoidContrast(),
        iaa.LogContrast(),
        iaa.LinearContrast(),
        iaa.AllChannelsHistogramEqualization(),
        iaa.HistogramEqualization(),
        iaa.AllChannelsCLAHE(),
        iaa.CLAHE()]
COLORS=[iaa.WithBrightnessChannels(),
        iaa.MultiplyAndAddToBrightness(),
        iaa.MultiplyBrightness(),
        iaa.AddToBrightness(),
        iaa.WithHueAndSaturation(),
        iaa.MultiplyHueAndSaturation(),
        iaa.MultiplyHue(),
        iaa.MultiplySaturation(),
        iaa.RemoveSaturation(),
        iaa.AddToHueAndSaturation(),
        iaa.AddToHue(),
        iaa.AddToSaturation(),
        iaa.Grayscale(),
        iaa.ChangeColorTemperature(),
        iaa.UniformColorQuantization(),
        iaa.Posterize()]



AUGMENTATIONS=[additiveGaussian,
              elasticDistortion,
              gaussianBlur,
              medianBlur,
              motionBlur]


def aug_template(img):
    if random_exec():
        op=random.choice(COLORS)
        img=op(image=img)
    if random_exec():    
        op=random.choice(CONTRASTS)
        img=op(image=img)
    if random_exec():
        op=random.choice(BLURS)
        img=op(image=img)
    return img

def get_augmented_data(img):
    op=random.choice(AUGMENTATIONS)
    return op(img)

