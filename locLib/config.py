
# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
class name:
    max_words  = 4
    min_words  = 2
    min_len    = 1
    max_len    = 5
    total      = 60
    seps        = ["",".",","]
    sep_weights = [0.6,0.2,0.2]
    frag_weights= [0.8,0.2]

class aug:
    max_rotation  = 5
    max_warp_perc = 20 
    max_pad_perc  = 50
    use_scope_rotation=False

#--------------------
# mask data
#--------------------
marking={}    
marking["bname"]    =1  
marking["ename"]    =2
marking["fname"]    =3
marking["mname"]    =4
marking["dob"]      =5
marking["nid"]      =6
marking["addr"]     =7
marking["front"]    =8
marking["back"]     =9
marking["sign"]     =10
data_classes=[_class for _class in marking.keys()]