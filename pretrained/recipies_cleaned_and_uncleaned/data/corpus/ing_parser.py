import os
import nltk
import inflect
p = inflect.engine()


def clean_word(word):
	word =word.lower()
	word = word.replace('+','').replace('bacon', 'beef').replace("{", '').replace("}", "").replace("'", '').replace('+', '').replace('_', '').replace('-', '')
	word = word.replace("(", '').replace(")", "").replace("=", '').replace('*', '')
	word =word.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "").replace("'", "").replace('(',"").replace(")", "")  
	return word

### define Superingredients:
def get_ing(file):
	with open(file, 'r') as f:
		fr = []
		for line in f:
			line = nltk.word_tokenize(line.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "").replace("'", "").replace('(',"").replace(")", "")  )
			for frt in line:
				if frt !=",":
					frt = frt.lower()
					if p.singular_noun(frt)==False: fr.append(p.plural_noun(frt))
					else: fr.append(p.singular_noun(frt))
					fr.append(frt)
	return fr
superingredients = {}
covered={}
# superingredients['dairy'] = ['buttermilk', 'margarine', 'butter', 'butteroil' 'cheese', 'cottage', 'milk', 'ricotta', 'sour', 'cream', 'yogurt', 'flavored', 'plain']
superingredients ['fruits'] = get_ing('superingredients/fruits.txt')
superingredients ['grains'] = get_ing('superingredients/grains.txt')
superingredients ['sides'] = get_ing('superingredients/sides.txt')
superingredients ['proteins'] = get_ing('superingredients/proteins.txt')
superingredients ['seasonings'] = get_ing('superingredients/seasonings.txt')
superingredients ['vegetables'] = get_ing('superingredients/vegetables.txt')
superingredients ['drinks'] = get_ing('superingredients/drinks.txt')
superingredients ['dairy'] = get_ing('superingredients/dairy.txt')
for si, iss in superingredients.items():
	for i in iss:
		covered[i] = 1


print superingredients, len(superingredients), len(covered)

#### load one pass ings:
recorded_ings = []
with open('ing_dic.txt', 'r') as f:
	for line in f:
		tokens = nltk.word_tokenize(line)
		for token in tokens:
			token = clean_word(token)
			if token not in ['1', "'", '', ':',',']:
				recorded_ings.append(token)




# print recorded_ings




### parse ing


files = os.listdir(os.curdir)
ing_dic = {}
ing_freq = {it:0 for it in recorded_ings}
for file in files:
    if file.startswith('all_ingredients.txt'):
    	ings =[]
        with open(file, 'r') as f:
        	i = 0
        	for line in f:
       		#tokens = nltk.word_tokenize(line.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "") )
        		tokens = nltk.word_tokenize(line)
        		tagged = nltk.pos_tag(tokens)
        		nouns = [word for word,pos in tagged \
        		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        		downcased = [clean_word(x) for x in nouns if x.lower()!='ff' and len(x)<100]
        		for x in downcased: 
        			# ing_dic[x] = 1
        			if x in recorded_ings:
        				ing_freq[x] = ing_freq[x]+1
        			else: 
        				ing_freq[x] = 1
        				recorded_ings.append(x)
        		ings.append(downcased)
        		i +=1
        		# if i>10:break
        # print ings

        # with open('all_ing_s.txt', 'wb') as f:
        # 	for ing in ings:
        # 		f.write(' '.join(ing)+"\n")
        # print ing_dic, len(ing_dic)

# for key, value in sorted(ing_freq.iteritems(), key=lambda (k,v): (v,k), reverse=True):
# 	if value>0: print key, value

ncov = 0
sum_freqs = 0
not_covered = {}
for ing in ing_freq:
	sum_freqs += ing_freq[ing]
	if ing in covered:
		# ncov+=1
		ncov+=ing_freq[ing]
	else:
		not_covered[ing] = ing_freq[ing]
for key, value in sorted(not_covered.iteritems(), key=lambda (k,v): (v,k), reverse=True):
	if value>0: print key, value

print 'without dbpedia total covered: ', ncov, ' of ', sum_freqs, ' ratio: ', ncov*1.0/sum_freqs, 'not covered items: ', len(not_covered)

with open('img_dic.txt', 'wb') as tf:
	for t in recorded_ings:
		tf.write(t+"\n")
		


