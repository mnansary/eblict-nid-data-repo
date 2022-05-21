
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
marking["sign"]     =1
marking["bname"]    =2  
marking["ename"]    =3
marking["fname"]    =4
marking["mname"]    =5
marking["dob"]      =6
marking["nid"]      =7
marking["front"]    =8
marking["addr"]     =9
marking["back"]     =10
data_classes=[_class for _class in marking.keys()]
