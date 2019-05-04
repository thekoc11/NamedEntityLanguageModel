import os
import nltk



files = os.listdir(os.curdir)
for file in files:
    if file.startswith('all_ingredients.txt'):
    	ings =[]
    	ing_dic = {}
        with open(file, 'r') as f:
        	i = 0
        	for line in f:
        		tokens = nltk.word_tokenize(line.replace("patti","").replace("vdrj67a", "").replace('[',"").replace("]", "") )
        		tagged = nltk.pos_tag(tokens)
        		nouns = [word for word,pos in tagged \
        		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
        		downcased = [x.lower() for x in nouns if x.lower()!='ff']
        		for x in downcased: ing_dic[x] = 1
        		ings.append(downcased)
        		i +=1
        		# if i>10:break
        # print ings

        with open('all_ing_s.txt', 'wb') as f:
        	for ing in ings:
        		f.write(' '.join(ing)+"\n")
        print ing_dic, len(ing_dic)
