import os
import re, nltk
from nltk.stem.snowball import SnowballStemmer
# no_data_files = ['pork', 'bacon', 'nyc_cb', 'ChampChili.zip', 'bbqsauces.zip','lowcarbexport.zip']
# fext = 'zip'
# dext = '.zip'
text = ['.mmf']#, '.txt', 'MXP']

files = os.listdir(os.curdir)
c = 0

def text_preprocesser(text):
    # remove all non-alphabet characters
    modified_text = re.sub(r'[^a-zA-Z]', ' ', text)
    # remove all non-ascii characters
    modified_text = "".join(ch for ch in modified_text if ord(ch) < 128)
    # convert to lowercase
    modified_text = modified_text.lower() 
    tokens =[word for sent in nltk.sent_tokenize(modified_text) for word in nltk.word_tokenize(sent)]
    allowed_tokens = []
    for token in tokens:
        # words must contain 2 letters
        if re.search('[a-zA-Z]{2,}', token):
            allowed_tokens.append(token)     
    tokens = [stemmer.stem(t) for t in allowed_tokens]
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
                    line = line.strip().replace('DC4','')

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
                            titles[-1].append(line.strip())
                        else:
                            set_ingredient_block()
                        continue

                    if ingredient_block==1:
                        if line!='':
                            ingredients[-1].append(add_ingredient(line.strip()))
                        else:
                            ingredient_block=-1 #ingredient_block may get change or may not depending on following lone
                        continue

                    if ingredient_block==-1:
                        # if '----------------------------------' not in line:
                        if '------------------------------' not in line:
                            set_recipe_block()
                            if '-----' not in line:
                                recipies[-1].append(line.strip())
                        else:
                            ingredient_block=1 #ingredient block continues
                        continue

                    if recipe_block==1:
                        if line!='' and '-----' not in line and '**' not in line and '==' not in line:
                            recipies[-1].append(line.strip())
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
                        
                                recipies[-1].append(line.strip())
                        continue
                    if len(titles) != len(recipies): 
                        print 'mismatch: ',
                        exit()
print_all_data()
not_list = ['Ham', 'ham', 'pork', 'Pork', 'Bacon', 'bacon', 'Beer', 'beer', 'Wine', 'wine', 'lard', "Lard", 'Vodka', 'vodka', 'Whiskey', 'whiskey', 'pepperoni', 'Pepperoni', 'skillet', 'Skillet']
count=0



assert len(titles) == len(ingredients)
assert len(titles) == len(recipies)
with open('all_recipies.txt', 'wb') as rcpf,\
        open('all_titles.txt', 'wb') as ttlf, \
                open("all_ingredients.txt", 'wb') as igf:

    for i in range(len(titles)):
        title_b = titles[i]
        ingredient_b = ingredients[i]
        recipie_b = recipies[i]

        title =''
        for t in title_b:
            title = title+t+", "
        title = title.rstrip(", ")

        ings = ''
        for t in ingredient_b:
            ings = ings + t.replace(";,", ",").replace("; ", " ") + ", "
        ings = ings.strip(", ")

        rcp = ''
        for t in recipie_b:
            if '-----' in t:continue
            t = t.replace('MMMMM',"")
            t=t.replace('<','')
            t=t.replace('>','')
            t=t.replace('=','')
            t=t.replace('+','')
            rcp = rcp + t.rstrip(" ").lstrip(" ") + " "
        rcp = rcp.rstrip(" ").replace("  "," ")

        haram_flag = -1

        for nw in not_list:
            if nw in title:
                haram_flag=1
                break
            if nw in ings:
                haram_flag=1
                break
            if nw in rcp:
                haram_flag=1
                break
        if haram_flag==1: continue
        ttlf.write(title+"\n")
        igf.write(ings+"\n")
        rcpf.write(rcp+"\n")
        count +=1
print ' total: ', count
import re
import string

frequency = {}
document_text = open('all_recipies.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)

for word in match_pattern:
    count_ = frequency.get(word, 0)
    frequency[word] = count_ + 1

# frequency_list = frequency.keys()

# for words in frequency_list:
#     print words, frequency[words]

i = 0
for key, value in sorted(frequency.iteritems(), key=lambda (k,v): (v,k),reverse=True):
    print "%s: %s" % (key, value)
    i+=1
    if (i>100):break
size = sum([frequency[words] for words in frequency])
print 'vocab: ', len(frequency), ' #token: ',size , '#instances/corpus size: ', count, \
    'avg instance size: ', size/count







