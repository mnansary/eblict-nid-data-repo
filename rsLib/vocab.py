# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------


en_num_vocab=["pad","start","end","0","1","2","3","4","5","6","7","8","9"]
bn_num_vocab=["pad","start","end","০","১","২","৩","৪","৫","৬","৭","৮","৯"]
en_vocab    =["pad","start","end","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",
            "0","1","2","3","4","5","6","7","8","9",
            ":",";","<","=",">","?","@",
            "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
            "[","\\","]","^","_","`",
            "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
            "{","|","}","~","।"]


bn_vocab    =['pad','start','end','sep','!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';',
                '<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','।','ঁ','ং','ঃ','অ','অ্য','আ','ই',
                'ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ক','ক্ক','ক্ট','ক্ট্র','ক্ত','ক্ত্র','ক্ব',
                'ক্ম','ক্য','ক্র','ক্ল','ক্ষ','ক্ষ্ণ','ক্ষ্ব','ক্ষ্ম','ক্ষ্ম্য','ক্ষ্য','ক্স','খ','খ্য','খ্র','গ','গ্ণ','গ্ধ','গ্ধ্য','গ্ন','গ্ন্য','গ্ব','গ্ম','গ্য','গ্র','গ্র্য','গ্ল','ঘ','ঘ্ন','ঘ্য','ঘ্র',
                'ঙ','ঙ্ক','ঙ্ক্ত','ঙ্ক্য','ঙ্ক্ষ','ঙ্খ','ঙ্খ্য','ঙ্গ','ঙ্গ্য','ঙ্ঘ','ঙ্ঘ্য','ঙ্ঘ্র','ঙ্ম','চ','চ্চ','চ্ছ','চ্ছ্ব','চ্ছ্র','চ্ঞ','চ্ব','চ্য','ছ','জ','জ্জ','জ্জ্ব','জ্ঝ','জ্ঞ','জ্ব','জ্য','জ্র',
                'ঝ','ঞ','ঞ্চ','ঞ্ছ','ঞ্জ','ঞ্ঝ','ট','ট্ট','ট্ব','ট্ম','ট্য','ট্র','ঠ','ড','ড্ড','ড্ব','ড্য','ড্র','ঢ','ঢ্য','ঢ্র','ণ','ণ্ট','ণ্ঠ','ণ্ঠ্য','ণ্ড','ণ্ড্য','ণ্ড্র','ণ্ঢ','ণ্ণ',
                'ণ্ব','ণ্ম','ণ্য','ত','ত্ত','ত্ত্ব','ত্ত্য','ত্থ','ত্ন','ত্ব','ত্ম','ত্ম্য','ত্য','ত্র','ত্র্য','থ','থ্ব','থ্য','থ্র','দ','দ্গ','দ্ঘ','দ্দ','দ্দ্ব','দ্ধ','দ্ব','দ্ভ','দ্ভ্র','দ্ম','দ্য',
                'দ্র','দ্র্য','ধ','ধ্ন','ধ্ব','ধ্ম','ধ্য','ধ্র','ন','ন্ট','ন্ট্র','ন্ঠ','ন্ড','ন্ড্ব','ন্ড্র','ন্ত','ন্ত্ব','ন্ত্য','ন্ত্র','ন্ত্র্য','ন্থ','ন্থ্র','ন্দ','ন্দ্ব','ন্দ্য','ন্দ্র','ন্ধ','ন্ধ্য','ন্ধ্র','ন্ন',
                'ন্ব','ন্ম','ন্য','ন্স','প','প্ট','প্ত','প্ন','প্প','প্য','প্র','প্র্য','প্ল','প্স','ফ','ফ্র','ফ্ল','ব','ব্ব','ব্য','ব্র','ভ','ভ্ব','ভ্য','ভ্র','ম','ম্ন','ম্প','ম্প্র','ম্ফ',
                'ম্ব','ম্ব্র','ম্ভ','ম্ভ্র','ম্ম','ম্য','ম্র','ম্ল','য','য্য','র','র্ক','র্ক্য','র্খ','র্গ','র্গ্য','র্গ্র','র্ঘ','র্ঘ্য','র্চ','র্চ্য','র্ছ','র্জ','র্জ্ঞ','র্জ্য','র্ঝ','র্ট','র্ড','র্ঢ্য','র্ণ',
                'র্ণ্য','র্ত','র্ত্ম','র্ত্য','র্ত্র','র্থ','র্থ্য','র্দ','র্দ্ব','র্দ্র','র্ধ','র্ধ্ব','র্ন','র্প','র্ফ','র্ব','র্ব্য','র্ভ','র্ম','র্ম্য','র্য','র্ল','র্শ','র্শ্ব','র্শ্য','র্ষ','র্ষ্য','র্স','র্হ','র্হ্য',
                'ল','ল্ক','ল্ক্য','ল্গ','ল্ট','ল্ড','ল্প','ল্ফ','ল্ব','ল্ভ','ল্ম','ল্য','ল্ল','শ','শ্চ','শ্ছ','শ্ন','শ্ব','শ্ম','শ্য','শ্র','শ্ল','ষ','ষ্ক','ষ্ক্র','ষ্ট','ষ্ট্য','ষ্ট্র','ষ্ঠ','ষ্ঠ্য',
                'ষ্ণ','ষ্প','ষ্প্র','ষ্ফ','ষ্ব','ষ্ম','ষ্য','স','স্ক','স্ক্র','স্খ','স্ট','স্ট্র','স্ত','স্ত্ব','স্ত্য','স্ত্র','স্থ','স্থ্য','স্ন','স্প','স্প্র','স্প্ল','স্ফ','স্ব','স্ম','স্য','স্র','স্ল','হ',
                'হ্ণ','হ্ন','হ্ব','হ্ম','হ্য','হ্র','হ্ল','া','ি','ী','ু','ূ','ৃ','ে','ৈ','ো','ৌ','ৎ','ড়','ঢ়','য়','০','১','২','৩','৪','৫','৬','৭','৮',
                '৯']


all_vocab    =['pad','start','end','sep','!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';',
                '<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',
                'Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w',
                'x','y','z','{','|','}','~','।','ঁ','ং','ঃ','অ','অ্য','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ক','ক্ক','ক্ট','ক্ট্র','ক্ত','ক্ত্র','ক্ব',
                'ক্ম','ক্য','ক্র','ক্ল','ক্ষ','ক্ষ্ণ','ক্ষ্ব','ক্ষ্ম','ক্ষ্ম্য','ক্ষ্য','ক্স','খ','খ্য','খ্র','গ','গ্ণ','গ্ধ','গ্ধ্য','গ্ন','গ্ন্য','গ্ব','গ্ম','গ্য','গ্র','গ্র্য','গ্ল','ঘ','ঘ্ন','ঘ্য','ঘ্র',
                'ঙ','ঙ্ক','ঙ্ক্ত','ঙ্ক্য','ঙ্ক্ষ','ঙ্খ','ঙ্খ্য','ঙ্গ','ঙ্গ্য','ঙ্ঘ','ঙ্ঘ্য','ঙ্ঘ্র','ঙ্ম','চ','চ্চ','চ্ছ','চ্ছ্ব','চ্ছ্র','চ্ঞ','চ্ব','চ্য','ছ','জ','জ্জ','জ্জ্ব','জ্ঝ','জ্ঞ','জ্ব','জ্য','জ্র',
                'ঝ','ঞ','ঞ্চ','ঞ্ছ','ঞ্জ','ঞ্ঝ','ট','ট্ট','ট্ব','ট্ম','ট্য','ট্র','ঠ','ড','ড্ড','ড্ব','ড্য','ড্র','ঢ','ঢ্য','ঢ্র','ণ','ণ্ট','ণ্ঠ','ণ্ঠ্য','ণ্ড','ণ্ড্য','ণ্ড্র','ণ্ঢ','ণ্ণ',
                'ণ্ব','ণ্ম','ণ্য','ত','ত্ত','ত্ত্ব','ত্ত্য','ত্থ','ত্ন','ত্ব','ত্ম','ত্ম্য','ত্য','ত্র','ত্র্য','থ','থ্ব','থ্য','থ্র','দ','দ্গ','দ্ঘ','দ্দ','দ্দ্ব','দ্ধ','দ্ব','দ্ভ','দ্ভ্র','দ্ম','দ্য',
                'দ্র','দ্র্য','ধ','ধ্ন','ধ্ব','ধ্ম','ধ্য','ধ্র','ন','ন্ট','ন্ট্র','ন্ঠ','ন্ড','ন্ড্ব','ন্ড্র','ন্ত','ন্ত্ব','ন্ত্য','ন্ত্র','ন্ত্র্য','ন্থ','ন্থ্র','ন্দ','ন্দ্ব','ন্দ্য','ন্দ্র','ন্ধ','ন্ধ্য','ন্ধ্র','ন্ন',
                'ন্ব','ন্ম','ন্য','ন্স','প','প্ট','প্ত','প্ন','প্প','প্য','প্র','প্র্য','প্ল','প্স','ফ','ফ্র','ফ্ল','ব','ব্ব','ব্য','ব্র','ভ','ভ্ব','ভ্য','ভ্র','ম','ম্ন','ম্প','ম্প্র','ম্ফ',
                'ম্ব','ম্ব্র','ম্ভ','ম্ভ্র','ম্ম','ম্য','ম্র','ম্ল','য','য্য','র','র্ক','র্ক্য','র্খ','র্গ','র্গ্য','র্গ্র','র্ঘ','র্ঘ্য','র্চ','র্চ্য','র্ছ','র্জ','র্জ্ঞ','র্জ্য','র্ঝ','র্ট','র্ড','র্ঢ্য','র্ণ',
                'র্ণ্য','র্ত','র্ত্ম','র্ত্য','র্ত্র','র্থ','র্থ্য','র্দ','র্দ্ব','র্দ্র','র্ধ','র্ধ্ব','র্ন','র্প','র্ফ','র্ব','র্ব্য','র্ভ','র্ম','র্ম্য','র্য','র্ল','র্শ','র্শ্ব','র্শ্য','র্ষ','র্ষ্য','র্স','র্হ','র্হ্য',
                'ল','ল্ক','ল্ক্য','ল্গ','ল্ট','ল্ড','ল্প','ল্ফ','ল্ব','ল্ভ','ল্ম','ল্য','ল্ল','শ','শ্চ','শ্ছ','শ্ন','শ্ব','শ্ম','শ্য','শ্র','শ্ল','ষ','ষ্ক','ষ্ক্র','ষ্ট','ষ্ট্য','ষ্ট্র','ষ্ঠ','ষ্ঠ্য',
                'ষ্ণ','ষ্প','ষ্প্র','ষ্ফ','ষ্ব','ষ্ম','ষ্য','স','স্ক','স্ক্র','স্খ','স্ট','স্ট্র','স্ত','স্ত্ব','স্ত্য','স্ত্র','স্থ','স্থ্য','স্ন','স্প','স্প্র','স্প্ল','স্ফ','স্ব','স্ম','স্য','স্র','স্ল','হ',
                'হ্ণ','হ্ন','হ্ব','হ্ম','হ্য','হ্র','হ্ল','া','ি','ী','ু','ূ','ৃ','ে','ৈ','ো','ৌ','ৎ','ড়','ঢ়','য়','০','১','২','৩','৪','৫','৬','৭','৮',
                '৯']


vocabs={"english_numbers":en_num_vocab,
        "bangla_numbers":bn_num_vocab,
        "english_all":en_vocab,
        "bangla_all":bn_vocab,
        "all":all_vocab}