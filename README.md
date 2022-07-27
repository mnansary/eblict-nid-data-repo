# eblict-nid-data-repo
dataset creation repository for eblict nid work

```python
Version: 0.0.1     
Authors: Md. Nazmuddoha Ansary,
         Md. Mobassir Hossain 
```

**LOCAL ENVIRONMENT**  

```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
```

# Environment Setup
>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)

**python requirements**

* **pip requirements**: ```pip install -r requirements.txt``` 

> Its better to use a virtual environment 
> OR use conda-

* **conda**: use environment.yml: ```conda env create -f environment.yml```

# Data creation
- git clone this repo
- change woriking directory to **data**: ```cd data```
- **modify** and run script.sh: ```./script.sh```

# Modifying script.sh
* In the ```script.sh``` essentially you have to identify
    * **save_path**: Where the generated YOLO data will be saved under a folder named YOLO
    * **card_path**: Where the generated synthetic cards will be stored
    * **src_path**: a folder that holds 
        * **noise**  folder : which is available [here](https://www.kaggle.com/datasets/nazmuddhohaansary/nid-noisebgfacesign) 
        * **resources** folder: which is available [here](https://drive.google.com/drive/folders/1Ag_vi8nRaFbdVUpUaHlXPVVKAIsdynQA?usp=sharing)

        **Essential Folder Structre For src_path**

        ```python
        ├── noise
        │   ├── background
        │   ├── faces
        │   └── signature
        └── resources
        ```
        **Essential Folder Structre For resources**

        ```python
        ├── bangla_bold.ttf
        ├── bangla_reg.ttf
        ├── dict.json
        ├── english_bold.ttf
        ├── english_reg.ttf
        ├── gpo.csv
        ├── nid_back_mask.png
        ├── nid_back.png
        ├── nid_front_mask.png
        ├── nid_front.png
        ├── smart_back_mask.png
        ├── smart_back.png
        ├── smart_front_mask.png
        └── smart_front.png
        ```

* example ```script.sh``` 

```bash
#!/bin/sh
save_path="/backup2/NID/data_repo/data/"
src_path="/backup2/NID/data_repo/data/source/"
card_path="/backup2/NID/data_repo/data/cards/"
#------------------------------------------card------------------------------------------------------
python card.py $src_path $save_path --num_data 1500
python yolo.py $src_path $card_path $save_path 
```
* used dataset available [here](https://www.kaggle.com/datasets/nazmuddhohaansary/eblict-nid-yolo)