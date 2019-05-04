import os
import re, nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from collections import Counter

 
stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
text = ['.mmf']#, '.txt', 'MXP']

files = os.listdir(os.curdir)
c = 0
clean_flag = True

actual_nwords = modified_nwords = actual_nchar = modified_nchar = 0


meat_list = ['Ham', 'ham', 'pork', 'Pork', 'Bacon', 'bacon', 'lard', "Lard",  'pepperoni', 'Pepperoni', 'skillet', 'Skillet']
water_list = ['Wine', 'wine', 'beer', 'Beer' , 'Rum', 'rum','Vodka', 'vodka', 'Whiskey', 'whiskey'] 


def char_count(strn):
    s = 0
    counter = Counter(strn)
    for ch in counter:
        if ch!=' ':s+=counter[ch]
    return s, counter
def get_nwords_nchars(strn):
    nchar, counter = char_count(strn)
    nwrd = 0
    for w in strn.strip().split():
        if len(w)>1: nwrd+=1
    return nwrd, nchar

def get_halal(token):
    if token in meat_list: token = 'beef'
    if token in water_list: token = 'water'
    token = token.replace('DC4','')
    return token

def clean_token(t, clean=clean_flag):
    
    t = t.replace('MMMMM',"")
    t = t.replace('mmmmm',"")
    if clean==True:
        t=t.replace('<','')
        t=t.replace('>','')
        t=t.replace('=','')
        t =t.replace("...","")
        t =t.replace("..","")
        t=t.replace('+','')
    return t


def text_preprocess(text, ing = False, title = False, clean=clean_flag):
    # if clean==False and ing==False and title==False: 
    #     modified_text = " ".join(w.lower()  for w in text.split())
    #     return modified_text
    # remove all non-alphabet characters
    if title==False and clean==True :modified_text = re.sub(r'[^a-z.A-Z]', ' ', text)
    else: modified_text = re.sub(r'[^a-z:.A-Z]', ' ', text)
    # remove all non-ascii characters
    if clean==True :modified_text = "".join(ch for ch in modified_text if ord(ch) < 128)
    # convert to lowercase
    modified_text = modified_text.lower() 
    tokens =[word for sent in nltk.sent_tokenize(modified_text) for word in nltk.word_tokenize(sent)]
    # print tokens, 'after nltk tokenizer'
    allowed_tokens = []
    for token in tokens:
        # words must contain 1 letters
        # if re.search('[a-zA-Z]{2,}', token):
        if ing==True:
            if token not in stopWords and len(token)>2: allowed_tokens.append(get_halal(token))   
        else:allowed_tokens.append(get_halal(token))  
    tokens = ""
    for t in allowed_tokens:
        if '.' in t:
            tokens += " "+t.split('.')[0]+" "+ t.split('.')[1]
        if ing ==True and token in ['c', 't', 'lbs']: continue
        if len(tokens)==0 and t=="." : continue
        if len(tokens)>0 and tokens[-1] in stopWords and t=="." and clean==True : continue
        if len(tokens)>0 and tokens[-1]=='.' and t=="." and clean==True : continue
        tokens += " "+t #stemmer.stem(t) 

    return tokens

# for file in files:
#     if fext in file:
#         for nf in no_data_files:
#             if nf in file: continue
#         with open (file, 'r') as f:
#             for line in f:
#                 if dext in line:
#                     for word in line.split():
#                         if 'href=' in word:
#                             link =  word.split('href=')[1].split('>')[0].replace('"','')
#                             os.system('wget '+link)
#                             c+=1


# files = os.listdir(os.curdir)
#
# for file in files:
#     if dext in file:
#         # os.system('mv '+file+' ./data')
#         os.system('unzip '+file)



# ###mmf parser, format
#line, MMMMM----- Recipe .... MMMMM, ---------- Recipe ... -----, ---------- Pro-Exchange ... -----,
# ------------- Recipe (garlic.mmf),
#
#Title block
#
#Ingredient block
#
# * then blank line or if ---------------------------------- then another ingredient block (Canada.mmf)
#
# Corpus all lines if contains not ** or '==2or more', 'From', ' and (replace 'DC4', '')
#Recipe: sometimes, line has 'colon'
#
# -----
#sometimes
#---------- Recipe via Meal-Master or the following
# source name block (sometimes no mark block bellow)
#
# Mark 0 = 250 block/ source name block is last
#
# ----------------------------------------------------------------------------- sometimes
############

structure_id = 0
start = ['MMMMM----- Recipe', '---------- Recipe', '---------- Pro-Exchange', '------------- Recipe']
exclude_list = ['Source', 'Posted', 'Recipe', 'Croeso', 'Anthony', 'Cardiff', 'British', 'Mrs', 'From', '----------',\
                'Copyright', '(C)', 'USENET', 'Difficulty:', 'Time:', 'Precision:', 'SOURCE:' ]


def add_ingredient(items):
    # print 'got: ', items
    new_item=''
    if '/' in items: items = items.replace('/', '\t')
    items.replace('\t', ',')
    itesm = ''.join([i for i in items if not i.isdigit()])
    for item in items.split():

        # print 'item: ', item
        if len(item)>2 and '(' not in item and ')' not in item:
            new_item+=item+' '
        else:
            new_item += ', '
    # print  new_item.strip(',')
    # exit()
    return new_item.replace(' , ', ' ').strip(',').strip().replace(' , ', ', ').strip(', ')


start_block = 0
title_block = 0
ingredient_block = 0
recipe_block = 0
block_change = 0
flag = 0


titles = []
recipies =[]
ingredients = []
items = -1


def set_start_block():
    global start_block, title_block, ingredient_block, recipe_block, block_change, flag
    start_block = 1
    title_block = 0
    ingredient_block = 0
    recipe_block = 0
    block_change = 1
    flag = 0
def set_title_block():
    global start_block, title_block, ingredient_block, recipe_block, block_change, titles
    start_block = 0
    title_block = 1
    ingredient_block = 0
    recipe_block = 0
    block_change = 1
    titles.append([])
    # print ' in title block'
    # print_block_status()
def set_ingredient_block():
    global start_block, title_block, ingredient_block, recipe_block, block_change, ingredients
    start_block = 0
    title_block = 0
    ingredient_block = 1
    recipe_block = 0
    block_change = 1
    ingredients.append([])

def set_recipe_block():
    global start_block, title_block, ingredient_block, recipe_block, block_change, recipies
    start_block = 0
    title_block = 0
    ingredient_block = 0
    recipe_block = 1
    block_change = 1
    recipies.append([])

def clear_all_blocks():
    global start_block, title_block, ingredient_block, recipe_block, block_change, flag
    start_block = 0
    title_block = 0
    ingredient_block = 0
    recipe_block = 0
    block_change = 1
    flag = 0

def print_block_status():
    print 'start_block = ', start_block, ' title_block = ',  title_block, ' ingredient_block = ', ingredient_block, \
        ' recipe_block = ', recipe_block, ' block_change = ', block_change
def print_titles():
    print 'titles: '
    global titles
    print titles
def print_ingedients():
    print 'ingredients: '
    global ingredients
    print ingredients
def print_recipies():
    print 'recipies: '
    global recipies
    print recipies
def print_all_data():
    global titles, ingredients, recipies
    # print_titles()
    # print_ingedients()
    # print_recipies()
    print len(titles), len(ingredients), len(recipies)

def remove_last_instance():
    global titles, ingredients, recipies
    del(titles[-1])
    del(ingredients[-1])
    del(recipies[-1])
    print_all_data()


for file in files:
    for t in text:
        
        if file.endswith(t):
            print file
            with open(file, 'r') as f:
                clear_all_blocks()
                for line in f:
                    block_change = 0
                    line = line.strip()

                    # print 'reading line: ', line,  line in ['\n', '\r\n']
                    for sw in  start:
                        if sw in line:
                            # print 'start new item', line
                            items = items+1
                            set_start_block()
                            break


                    if block_change==1: continue
                    if start_block==1:
                        if line=='':
                            set_title_block()
                            titles[-1].append('File: '+ file)
                        else:
                            print 'check new format', line, file
                            # exit()
                        continue


                    if title_block == 1:
                        if line!='':
                            titles[-1].append(text_preprocess(line.strip(), title=True))
                        else:
                            set_ingredient_block()
                        continue

                    if ingredient_block==1:
                        if line!='':
                            ingredients[-1].append((text_preprocess(line.strip(), ing = True)))
                        else:
                            ingredient_block=-1 #ingredient_block may get change or may not depending on following lone
                        continue

                    if ingredient_block==-1:
                        # if '----------------------------------' not in line:
                        if '------------------------------' not in line:
                            set_recipe_block()
                            if '-----' not in line:
                                # recipies[-1].append(text_preprocess(line.strip()))
                                nwrds, nchar = get_nwords_nchars(line.strip())
                                actual_nchar+=nchar
                                actual_nwords+=nwrds
                                m_text=text_preprocess(line.strip())
                                nwrds, nchar = get_nwords_nchars(m_text)
                                modified_nchar+=nchar
                                modified_nwords+=nwrds
                                # print ('ori text: ', line.strip(), ' m text: ', m_text, 'actual_nchar', actual_nchar, 'modified_nchar', modified_nchar, 'actual_nwords', actual_nwords, 'modified_nwords', modified_nwords)
                                recipies[-1].append(m_text)
                        else:
                            ingredient_block=1 #ingredient block continues
                        continue

                    if recipe_block==1:
                        if line!='' and '-----' not in line and '**' not in line and '==' not in line:
                            # recipies[-1].append(text_preprocess(line.strip()))
                            nwrds, nchar = get_nwords_nchars(line.strip())
                            actual_nchar+=nchar
                            actual_nwords+=nwrds
                            m_text=text_preprocess(line.strip())
                            nwrds, nchar = get_nwords_nchars(m_text)
                            modified_nchar+=nchar
                            modified_nwords+=nwrds
                            recipies[-1].append(m_text)
                        else:
                            recipe_block=-1
                        continue
                    if recipe_block == -1:
                        for w in exclude_list:
                            if w in line:
                                clear_all_blocks()
                                # recipe_block=1 
                                break
                        if block_change==1: continue
                        else:
                            recipe_block=1 #recipe block continues
                            # if line!='' or '------' not in line:
                            if line!='' and '-----' not in line and '**' not in line and '==' not in line:
                                # recipies[-1].append(text_preprocess(line.strip()))
                                nwrds, nchar = get_nwords_nchars(line.strip())
                                actual_nchar+=nchar
                                actual_nwords+=nwrds
                                m_text=text_preprocess(line.strip())
                                nwrds, nchar = get_nwords_nchars(m_text)
                                modified_nchar+=nchar
                                modified_nwords+=nwrds
                                recipies[-1].append(m_text)
                                
                        continue
                    if len(titles) != len(recipies): 
                        print 'mismatch: ',
                        exit()
            bb = 1
    # if bb==1: break
# print_all_data()


count=0



assert len(titles) == len(ingredients)
assert len(titles) == len(recipies)
# with open('corpus/all_recipies.txt', 'wb') as rcpf,\
#         open('corpus/all_titles.txt', 'wb') as ttlf, \
#                 open("corpus/all_ingredients.txt", 'wb') as igf,\
#                 open('corpus/train_rcp.txt', 'wb') as trrcpf, \
#                 open('corpus/valid_rcp.txt','wb') as vrcpf,\
#                 open('corpus/test_rcp.txt','wb') as tsrcpf,\
#                 open('corpus/train_ing.txt', 'wb') as trigf,\
#                 open('corpus/valid_ing.txt','wb') as vigf,\
#                 open('corpus/test_ing.txt','wb') as tsigf,\
#                 open('corpus/train_ttl.txt', 'wb') as trttlf,\
#                 open('corpus/valid_ttl.txt','wb') as vttlf,\
#                 open('corpus/test_ttl.txt','wb') as tsttlf:
with open('corpus/uncleaned_all_recipies.txt', 'wb') as rcpf,\
        open('corpus/uncleaned_all_titles.txt', 'wb') as ttlf, \
                open("corpus/uncleaned_all_ingredients.txt", 'wb') as igf,\
                open('corpus/uncleaned_train_rcp.txt', 'wb') as trrcpf, \
                open('corpus/uncleaned_valid_rcp.txt','wb') as vrcpf,\
                open('corpus/uncleaned_test_rcp.txt','wb') as tsrcpf,\
                open('corpus/uncleaned_train_ing.txt', 'wb') as trigf,\
                open('corpus/uncleaned_valid_ing.txt','wb') as vigf,\
                open('corpus/tuncleaned_est_ing.txt','wb') as tsigf,\
                open('corpus/uncleaned_train_ttl.txt', 'wb') as trttlf,\
                open('corpus/uncleaned_valid_ttl.txt','wb') as vttlf,\
                open('corpus/uncleaned_test_ttl.txt','wb') as tsttlf:



    train_val_ids, test_ids = train_test_split( range(len(titles)), random_state=1111, train_size = 0.8, test_size  = 0.2 )
    train_ids, val_ids = train_test_split(train_val_ids, random_state=1111, train_size = 0.8, test_size  = 0.2 )



    for i in train_ids:
        title_b = titles[i]
        ingredient_b = ingredients[i]
        recipie_b = recipies[i]

        title =''
        for t in title_b:
            # print t
            title = title+t+", "
        title = title.rstrip(", ")

        ings = ''
        for t in ingredient_b:
            ings = ings + t.replace(";,", ",").replace("; ", " ") + ", "
        ings = ings.strip(", ")

        rcp = ''
        for t in recipie_b:
            if '-----' in t:continue
            t = clean_token(t)
            rcp = rcp + t.rstrip(" ").lstrip(" ") + " "
        rcp = rcp.rstrip(" ").replace("  "," ")

        trttlf.write(title+"\n")
        trigf.write(ings+"\n")
        trrcpf.write(rcp+"\n")


        ttlf.write(title+"\n")
        igf.write(ings+"\n")
        rcpf.write(rcp+"\n")
        count +=1
    print 'count after train: ', count


    for i in val_ids:
        title_b = titles[i]
        ingredient_b = ingredients[i]
        recipie_b = recipies[i]

        title =''
        for t in title_b:
            # print t
            title = title+t+", "
        title = title.rstrip(", ")

        ings = ''
        for t in ingredient_b:
            ings = ings + t.replace(";,", ",").replace("; ", " ") + ", "
        ings = ings.strip(", ")

        rcp = ''
        for t in recipie_b:
            if '-----' in t:continue
            t = clean_token(t)
            rcp = rcp + t.rstrip(" ").lstrip(" ") + " "
        rcp = rcp.rstrip(" ").replace("  "," ")

        vttlf.write(title+"\n")
        vigf.write(ings+"\n")
        vrcpf.write(rcp+"\n")


        ttlf.write(title+"\n")
        igf.write(ings+"\n")
        rcpf.write(rcp+"\n")
        count +=1
    print 'count after val: ', count

    for i in test_ids:
        title_b = titles[i]
        ingredient_b = ingredients[i]
        recipie_b = recipies[i]

        title =''
        for t in title_b:
            # print t
            title = title+t+", "
        title = title.rstrip(", ")

        ings = ''
        for t in ingredient_b:
            ings = ings + t.replace(";,", ",").replace("; ", " ") + ", "
        ings = ings.strip(", ")

        rcp = ''
        for t in recipie_b:
            if '-----' in t:continue
            t = clean_token(t)
            rcp = rcp + t.rstrip(" ").lstrip(" ") + " "
        rcp = rcp.rstrip(" ").replace("  "," ")

        tsttlf.write(title+"\n")
        tsigf.write(ings+"\n")
        tsrcpf.write(rcp+"\n")


        ttlf.write(title+"\n")
        igf.write(ings+"\n")
        rcpf.write(rcp+"\n")
        count +=1
    print 'count after test: ', count

print ( 'actual_nchar', actual_nchar, 'modified_nchar', modified_nchar, 'actual_nwords', actual_nwords, 'modified_nwords', modified_nwords)
                                
print ' total: ', count, len(titles), ' train size: ', len(train_ids), ' val size: ', len(val_ids), ' test size: ', len(test_ids)
import re
import string

frequency = {}
document_text = open('corpus/all_recipies.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)

for word in match_pattern:
    count_ = frequency.get(word, 0)
    frequency[word] = count_ + 1

# frequency_list = frequency.keys()

# for words in frequency_list:
#     print words, frequency[words]

# i = 0
# for key, value in sorted(frequency.iteritems(), key=lambda (k,v): (v,k),reverse=True):
#     print "%s: %s" % (key, value)
#     i+=1
#     if (i>100):break
size = sum([frequency[words] for words in frequency])
print 'vocab: ', len(frequency), ' #token: ',size , '#instances/corpus size: ', count, \
    'avg instance size: ', size/count







