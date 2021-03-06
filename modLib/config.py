# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''

class augment_conf:
    rotation_angle_max    = 5
    rotation_exec_weights = [0.3,0.7]   
    warping_len_max_perc  = 20
    warping_exec_weights  = [0.3,0.7]
    mask_negation_prec    = 30

class text_conf:
    lens                  = [1,2,3,4,5,6,7,8,9,10]
    weights               = [0.05,0.05,0.1,0.15,0.15,0.15,0.15,0.1,0.05,0.05]
    include_space         = False
    max_space             = 3
    space_weights         = [0.8,0.2]
    max_font_text_dim     = 72
    min_font_text_dim     = 24
    comp_weights          = [0.75,0.2,0.05]
    