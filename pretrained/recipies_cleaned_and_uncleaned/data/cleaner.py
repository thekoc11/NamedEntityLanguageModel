import os

files = os.listdir(os.curdir)
for file in files:
    if file.startswith('all'):
        with open(file, 'r') as f:
            # lines = [line.replace("=",'').replace("*",'').replace("-",'').replace("MMMMM", '')  for line in f]
            # print lines
            lines = f.read().replace("=",'').replace("*",'').replace("-",'').replace("MM", '').replace("\t", ' ').replace("~", ' ').replace(".", "")
        with open(file, 'wb') as f:
            f.write(lines)
        # break
