# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import PIL
import PIL.ImageFont,PIL.Image,PIL.ImageDraw
import random
import json 
import calendar
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from indicparser import graphemeParser
import textwrap


from .card import card
from .config import name,aug
from .utils import padToFixedHeight,padToFixedHeightWidth,randColor, random_exec

gp=graphemeParser("bangla")

#--------------------
# source
#--------------------
class Data(object):
    def __init__(self,src_dir):
        '''
            initilizes all sources under dataset
            args:
                src_dir     :       location of source folder 
        '''
        self.src_dir          =     src_dir
        self.res_dir          =     os.path.join(self.src_dir,"resources")
        self.gpo_df           =     pd.read_csv(os.path.join(self.res_dir,"gpo.csv"))
        self.corps            =     ["ঢাকা","চট্টগ্রাম","খুলনা","সিলেট","রাজশাহী","ময়মনসিংহ","বরিশাল","রংপুর","নারায়ণগঞ্জ","গাজীপুর","কুমিল্লা"]
        #-----------------------
        # data folders
        #-----------------------
        class noise:
            signs     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","signature","*.*")))]
            faces     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","faces","*.*")))]
            backs     = [img_path for img_path in tqdm(glob(os.path.join(self.src_dir,"noise","background","*.*")))]
        
        card.smart.front.template =os.path.join(self.res_dir,"smart_front.png")
        card.smart.back.template  =os.path.join(self.res_dir,"smart_back.png")
        card.nid.front.template   =os.path.join(self.res_dir,"nid_front.png")
        card.nid.back.template    =os.path.join(self.res_dir,"nid_back.png")
        
        
        self.noise  =   noise
        self.card   =   card 
        self.name   =   name  
        self.aug    =   aug 
        
        # extend text
        #----------------------------
        # dictionary json
        #----------------------------
        with open(os.path.join(self.res_dir,"dict.json")) as f:
            json_data       =   json.load(f)
            self.bangla     =   json_data["bangla"]
            self.english    =   json_data["english"]
        
        
        ## smart card font
        self.initTextFonts(self.card.smart.front.text)
        ## smart card back
        self.initTextFonts(self.card.smart.back.text)
        ## nid card font
        self.initTextFonts(self.card.nid.front.text)
        ## nid catd back
        self.initTextFonts(self.card.nid.back.text)
    
    #----------------------------
    # back ground
    #----------------------------
    
    
    def backgroundGenerator(self,dim=(1024,1024),_type=None):
        '''
        generates random background
        args:
            ds   : dataset object
            dim  : the dimension for background
        '''
        # collect image paths
        _paths=self.noise.backs
        while True:
            if _type is None:
                _type=random.choice(["single","double","comb"])
            if _type=="single":
                img=cv2.imread(random.choice(_paths))
                yield img
            elif _type=="double":
                imgs=[]
                img_paths= random.sample(_paths, 2)
                for img_path in img_paths:
                    img=cv2.imread(img_path)
                    h,w,d=img.shape
                    img=cv2.resize(img,dim)
                    imgs.append(img)
                # randomly concat
                img=np.concatenate(imgs,axis=random.choice([0,1]))
                img=cv2.resize(img,(w,h))
                yield img
            else:
                imgs=[]
                img_paths= random.sample(_paths, 4)
                for img_path in img_paths:
                    img=cv2.imread(img_path)
                    h,w,d=img.shape
                    img=cv2.resize(img,dim)
                    imgs.append(img)
                seg1=imgs[:2]
                seg2=imgs[2:]
                seg1=np.concatenate(seg1,axis=0)
                seg2=np.concatenate(seg2,axis=0)
                img=np.concatenate([seg1,seg2],axis=1)
                img=cv2.resize(img,(w,h))
                yield img
    #----------------------------
    # font setup
    #----------------------------
        
    def __getDataFont(self,attr):
        '''
            different size font initialization
        '''
        if attr["lang"]=="bn":
            font_path=os.path.join(self.res_dir,f"bangla_{attr['font']}.ttf")
        else:
            font_path=os.path.join(self.res_dir,f"english_{attr['font']}.ttf")
        font = PIL.ImageFont.truetype(font_path, size=attr["font_size"])
        return font
    

    def initTextFonts(self,text):
        '''
            initializes text fonts
        '''
        for name,attr in text.items():
            text[name]["font"]=self.__getDataFont(attr)

    #----------------------------
    # text construction
    #----------------------------
    def __createNameData(self,vocabs,language):
        num_word=random.randint(self.name.min_words,self.name.max_words)
        words=[]
        # fragment
        if random.choices(population=[0,1],weights=self.name.frag_weights,k=1)[0]==1:
            use_fragment=True
        else:
            use_fragment=False
        
        for _ in range(num_word):
            num_vocab=random.randint(self.name.min_len,self.name.max_len)
            word="".join([random.choice(vocabs) for _ in range(num_vocab)])
            
            # check invalid bangla starting:
            if language=="bangla" and word[0] in ["ঁ","ং","ঃ"]:
                if num_vocab>1:
                    word=word[1:]
                elif num_vocab==1:
                    word=''
            # blank word filter
            if word=='':
                continue

            # handling . and ,  (seps)       
            if num_vocab==1 and not use_fragment:
                if language=="english":word+="."
                else:word+=random.choice([".",","])
            elif num_vocab==2 and language=="bangla" and not use_fragment: 
                word+=random.choices(population=self.name.seps,weights=self.name.sep_weights,k=1)[0]
            words.append(word)
        # check length
        name=" ".join(words)
        while len(name)>self.name.total:
            words=words[:-1]
            name=" ".join(words)
        
        if use_fragment:
            # bracket last word
            if random.choices(population=[0,1],weights=[0.5,0.5],k=1)[0]==1:
                words[-1]="("+words[-1]+")"
            # hyphenate 3 words
            else:
                if len(words)>3:
                    connect=[]
                    for _ in range(3):
                        idx=random.choice([i for i in range(len(words))])
                        connect.append(words[idx])
                        words[idx]=None
                        words=[word for word in words if word is not None]
                    name="-".join(connect)
                    return name
        name=" ".join(words)
        return name
    
    def __createBnName(self,mod_id):
        '''
            creates bangla name
        '''
        mods    = ["মোঃ ","মোছাঃ "]
        name        =   ''
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            name+=mods[mod_id]
        #
        name+=self.__createNameData(self.bangla["graphemes"],"bangla")
        return name

    def __createEnName(self,mod_id,type):
        '''
            creates English name
        '''
        mods    = ["MD. ","MRS. "]
        name        =   ''
        # use starting
        if random.choice([1,0])==1:
            if mod_id is None:
                mod_id=random.choice([0,1])
            name+=mods[mod_id]
              
        # determine case
        if type=="smart":
            vocabs=self.english["uppercase"]
        else:
            if random.choice([1,0])==1:
                vocabs=self.english["uppercase"]
            else:
                vocabs=self.english["uppercase"]+self.english["lowercase"]

        name=self.__createNameData(vocabs,"english")
        return name
    #------------------------------------------------------------------------------------------------
    def __getNumber(self,len):
        num=''
        for _ in range(len):
            num+=random.choice(self.english["numbers"])
        return num 
    def __createDOB(self,type):
        '''
            creates a date of birth
        '''
        months=list(calendar.month_abbr)[1:]
        start=self.__getNumber(2)
        if type=="smart":
            if start[0]=='0':
                start=start[1:]

        return start+' '+random.choice(months)+' '+self.__getNumber(4) 

    def __createNID(self,type):
        if type=="smart":
            return self.__getNumber(3)+' '+self.__getNumber(3)+' '+self.__getNumber(4)
        else:
            return self.__getNumber(random.choice([10,13,17]))
    #------------------------------------------------------------------------------------------------
    def __createTextData(self,vocabs,language,num_word,min_len,max_len,total):
        words=[]
        
        for _ in range(num_word):
            num_vocab=random.randint(min_len,max_len)
            word="".join([random.choice(vocabs) for _ in range(num_vocab)])
            
            # check invalid bangla starting:
            if language=="bangla" and word[0] in ["ঁ","ং","ঃ"]:
                if num_vocab>1:
                    word=word[1:]
                elif num_vocab==1:
                    word=''
            # blank word filter
            if word=='':
                continue

            words.append(word)
        # check length
        text=" ".join(words)
        while len(text)>total:
            words=words[:-1]
            text=" ".join(words)
        return text
    #------------------------------------------------------------------------------------------------
    
    def __createHolding(self):
        if random.choices(population=[0,1],weights=[0.5,0.5],k=1)[0]==1:
            return self.__createTextData(self.bangla["numbers"],"bangla",1,2,4,20)
        else:
            return self.__createTextData(self.bangla["graphemes"],"bangla",random.randint(1,3),2,4,30)
    
    def __createRoad(self):
        return self.__createTextData(self.bangla["numbers"]+self.bangla["graphemes"],"bangla",random.randint(1,6),2,6,40)
    
    def __createGPO(self):
        text="ডাকঘর: "
        idx=random.randint(0,len(self.gpo_df)-1)
        code=self.gpo_df.iloc[idx,0]
        district=self.gpo_df.iloc[idx,1]
        thana=self.gpo_df.iloc[idx,2]
        detais=self.gpo_df.iloc[idx,3].split(" ")[0]
        text+=thana+'-'+code+","+detais+','
        if district in self.corps:
            text+=f"{district} সিটি কর্পোরেশন,{district}"
        else:
            text+=f"{district} পৌরসভা ,{district}"
        
        return text,code
    #------------------------------------------------------------------------------------------------
    
    def createTextCardBack(self):
        holding=self.__createHolding()
        road=self.__createRoad()
        gpo,code=self.__createGPO()
        text="বাসা/হোল্ডিং: "+holding+", "+"গ্রাম/রাস্তা: "+road+", "+gpo
        return text



    def createTextCardFront(self,type):
        return {"bname":self.__createBnName(mod_id=None),
                "ename":self.__createEnName(mod_id=None,type=type),
                "fname":self.__createBnName(mod_id=0),
                "mname":self.__createBnName(mod_id=1),
                "dob"  :self.__createDOB(type),
                "nid"  :self.__createNID(type)}

    #------------------------------------------------------------------------------------------------
    def createCardBack(self,type):
        '''
            creates an image of card back side data
        '''
        template_label={}        
        iden=128
        if type=="smart":
            card_back=self.card.smart.back  
            text_add="        "  
            text_width=60
            xadd=0
        else:
            card_back=self.card.nid.back
            if random_exec(weights=[0.5,0.5]):
                text_add="       "
                xadd=0
            else:
                text_add=""
                xadd=80
            
                
            text_width=100
        template =cv2.imread(card_back.template)
        # mask
        h_t,w_t,d=template.shape
        template_mask=np.zeros((h_t,w_t))
        # text data processing
        text=self.createTextCardBack()
        text=text_add+text
        font=card_back.text["addr"]["font"]
        # res
        mask=np.zeros((h_t,w_t))
        # height width
        x1,y1,x2,y2=card_back.text["addr"]["location"]
        x1=x1+xadd
        width_loc=x2-x1
        height_loc=y2-y1
        # divide lines
        lines = textwrap.wrap(text,width=text_width)

        line_images=[]
        word_labels=[]
            
        for line in lines:
            # comps
            comps=gp.process(line)
            w_text,h_text=font.getsize(line)
            comp_str=''
            images=[]
            label={}
            comb_label={}
            for idx,comp in enumerate(comps):
                comp_str+=comp
                if comp not in [' ',',']: 
                    # data
                    image   =   PIL.Image.new(mode='L', size=(w_text,h_text))
                    draw    =   PIL.ImageDraw.Draw(image)
                    draw.text(xy=(0,0),text=comp_str, fill=1, font=font)
                    image   =   np.array(image)
                    images.append(image)
                    label[iden]=comp
                    comb_label[iden]=comp
                    iden+=1

                else:
                    if bool(label):
                        word_labels.append(label)
                        label={}
                if idx==len(comps)-1 and comp!=' ':
                    if bool(label):
                        word_labels.append(label)
                        label={}

                
            img=sum(images)
            # offset
            vals=list(np.unique(img))
            vals=sorted(vals,reverse=True)
            vals=vals[:-1]
            image=np.zeros(img.shape)
            for lv,l in zip(vals,comb_label.keys()):
                image[img==lv]=l
            line_images.append(image)
        # combine images
        max_w=0
        for image in line_images:
            h,w=image.shape
            max_w=max(max_w,w)
        padded=[]
        for image in line_images:
            h,w=image.shape
            if w<max_w:
                image=np.concatenate([image,np.zeros((h,max_w-w))],axis=1)
            padded.append(image)
        image=np.concatenate(padded,axis=0)
        # crop to size
        tidx    =   np.where(image>0)
        y_min,y_max,x_min,x_max = np.min(tidx[0]), np.max(tidx[0]), np.min(tidx[1]), np.max(tidx[1])
        image=image[y_min:y_max,x_min:x_max]

        h,w=image.shape
        h_new=int((h/w)*width_loc)
        image=cv2.resize(image,(width_loc,h_new),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
        # pad
        image=padToFixedHeight(image,height_loc)
        mask[y1:y2,x1:x2]=image
        template_mask[y1:y2,x1:x2]=image
        template[mask>0]=randColor(100,single=True)
        template_label={"addr":word_labels}
        return template,template_mask,template_label
            
    def createCardFront(self,type):
        '''
            creates an image of card front side data
        '''        
        template_label={}
        iden=2
        if type=="smart":
            card_front=self.card.smart.front
            info_color=randColor(100,single=True)
        else:
            card_front=self.card.nid.front
            info_color=(0,0,255)
        template =cv2.imread(card_front.template)
        # fill signs and images
        sign=cv2.imread(random.choice(self.noise.signs),0)
        face=cv2.imread(random.choice(self.noise.faces))
        
        # mask
        h_t,w_t,d=template.shape
        template_mask=np.zeros((h_t,w_t))
        

        # place face
        x1,y1,x2,y2=card_front.face
        h=y2-y1
        w=x2-x1
        template[y1:y2,x1:x2]=cv2.resize(face,(w,h))
        
        # place sign class 1
        x1,y1,x2,y2=card_front.sign
        h=y2-y1
        w=x2-x1
        mask=np.ones(template.shape[:-1])*255
        mask[y1:y2,x1:x2]=cv2.resize(sign,(w,h),fx=0,fy=0,interpolation=cv2.INTER_NEAREST)
        mask[mask!=0]=255
        template[mask==0]=(0,0,0)
        template_mask[mask==0]=1
        # text data
        info_keys=["nid","dob"]
        
        # text data processing
        text=self.createTextCardFront(type)
        
        # word-level mask
        for k,v in text.items():
            # res
            font=card_front.text[k]["font"]
            mask=np.zeros((h_t,w_t))
            # height width
            x1,y1,x2,y2=card_front.text[k]["location"]
            width_loc=x2-x1
            height_loc=y2-y1
            
            # comps
            comps=gp.process(v)
            w_text,h_text=font.getsize(v)
            comp_str=''
            images=[]
            label={}
            word_labels=[]
            comb_label={}
            for idx,comp in enumerate(comps):
                comp_str+=comp
                if comp!=' ': 
                    # data
                    image   =   PIL.Image.new(mode='L', size=(w_text,h_text))
                    draw    =   PIL.ImageDraw.Draw(image)
                    draw.text(xy=(0,0),text=comp_str, fill=1, font=font)
                    image   =   np.array(image)
                    images.append(image)
                    label[iden]=comp
                    comb_label[iden]=comp
                    iden+=1

                else:
                    word_labels.append(label)
                    label={}
                if idx==len(comps)-1 and comp!=' ':
                    word_labels.append(label)
                    label={}
                
            img=sum(images)
            # offset
            vals=list(np.unique(img))
            vals=sorted(vals,reverse=True)
            vals=vals[:-1]
            image=np.zeros(img.shape)
            for lv,l in zip(vals,comb_label.keys()):
                image[img==lv]=l
            
            # crop to size
            tidx    =   np.where(image>0)
            y_min,y_max,x_min,x_max = np.min(tidx[0]), np.max(tidx[0]), np.min(tidx[1]), np.max(tidx[1])
            image=image[y_min:y_max,x_min:x_max]
            # pad
            image=padToFixedHeightWidth(image,height_loc,width_loc)
            mask[y1:y2,x1:x2]=image
            template_mask[y1:y2,x1:x2]=image
            if k in info_keys:
                template[mask>0]=info_color
            else:
                template[mask>0]=randColor(100,single=True)
            template_label[k]=word_labels
            
        return template,template_mask,template_label