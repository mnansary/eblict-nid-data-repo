
  
# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#-----------------------
# CARD CLASS
#-----------------------
# bboxes: [xmin,ymin,xmax,ymax]
class card:
    height      =   614
    width       =   1024
            
    class smart:
        class front:
            face        =   [57, 182, 319, 481]
            sign        =   [57, 494, 319, 591]
            text        ={
                            "bname"     :   {"location":[327, 187, 777, 245],"font_size":48,"lang":"bn","font":"bold"},
                            "ename"     :   {"location":[327, 270, 777, 318],"font_size":32,"lang":"en","font":"bold"},
                            "fname"     :   {"location":[327, 342, 777, 403],"font_size":45,"lang":"bn","font":"reg"},
                            "mname"     :   {"location":[327, 410, 777, 485],"font_size":45,"lang":"bn","font":"reg"},
                            "dob"       :   {"location":[480, 485, 777, 530],"font_size":36,"lang":"en","font":"reg"},
                            "nid"       :   {"location":[480, 520, 777, 590],"font_size":42,"lang":"en","font":"bold"}
                        }
        class back:
            text        ={
                            "addr"        :   {"location":[40, 185, 637, 278],"font_size":60,"lang":"bn","font":"reg"}
                        }
            
    class nid:
        class front:
            face        =   [26, 219, 252, 451]
            sign        =   [26, 461, 252, 574]
            text        =   {
                            "bname"       :   {"location":[410, 218, 1011, 280],"font_size":56,"lang":"bn","font":"bold"},
                            "ename"       :   {"location":[410, 282, 1011, 322],"font_size":36,"lang":"en","font":"bold"},
                            "fname"       :   {"location":[410, 325, 1011, 390],"font_size":52,"lang":"bn","font":"reg"},
                            "mname"       :   {"location":[410, 390, 1011, 460],"font_size":52,"lang":"bn","font":"reg"},
                            "dob"         :   {"location":[545, 475, 1011, 515],"font_size":42,"lang":"en","font":"reg"},
                            "nid"         :   {"location":[455, 515, 1011, 600],"font_size":60,"lang":"en","font":"bold"}
                            }
        class back:
            text        =  {
                            "addr"        :   {"location":[35, 135,  1000, 250],"font_size":46,"lang":"bn","font":"reg"}
                           }
            