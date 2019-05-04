import os
no_data_files = ['pork', 'bacon', 'nyc_cb', 'ChampChili.zip', 'bbqsauces.zip','lowcarbexport.zip']
fext = 'zip'
dext = '.zip'
text = ['.mmf', '.txt']

files = os.listdir(os.curdir)
c = 0



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

# for file in files:
#     if dext in file:
#         # os.system('mv '+file+' ./data')
#         os.system('unzip '+file)


for file in files:
    for t in text:
        if t in file:
        	command = 'mv '+file+' ./data'
        	print command
        	# exit(0)
        	os.system(command)


