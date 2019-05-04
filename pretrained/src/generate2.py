# ###############################################################################
# # Language Modeling on Penn Tree Bank
# #
# # This file generates new sentences sampled from the language model
# #
# ###############################################################################

# import argparse

# import torch
# from torch.autograd import Variable

# import data

# parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# # Model parameters.
# parser.add_argument('--data', type=str, default='./data/penn',
#                     help='location of the data corpus')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (LSTM, QRNN)')
# parser.add_argument('--checkpoint', type=str, default='./model.pt',
#                     help='model checkpoint to use')
# parser.add_argument('--outf', type=str, default='generated.txt',
#                     help='output file for generated text')
# parser.add_argument('--words', type=int, default='1000',
#                     help='number of words to generate')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA')
# parser.add_argument('--temperature', type=float, default=1.0,
#                     help='temperature - higher will increase diversity')
# parser.add_argument('--log-interval', type=int, default=100,
#                     help='reporting interval')
# args = parser.parse_args()

# # Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed(args.seed)

# if args.temperature < 1e-3:
#     parser.error("--temperature has to be greater or equal 1e-3")

# with open(args.checkpoint, 'rb') as f:
#     model = torch.load(f)
# model.eval()
# if args.model == 'QRNN':
#     model.reset()

# if args.cuda:
#     model.cuda()
# else:
#     model.cpu()

# corpus = data.Corpus(args.data)
# ntokens = len(corpus.dictionary)
# hidden = model.init_hidden(1)
# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
# if args.cuda:
#     input.data = input.data.cuda()
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model as model_file
import model_ori_with_type 
import data2 as data_ori_type
import nltk
import inflect
p = inflect.engine()
from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--debug', type=int, default=-1,
                    help='location of the data corpus')
parser.add_argument('--data', type=str, default='../data/recipe_ori',
                    help='location of the data corpus')
parser.add_argument('--data_type', type=str, default='../data/recipe_type',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='RCP_or_type.pt', #RCP_LSTM_ori_with_type
                    help='path to save the final model')
parser.add_argument('--save_type', type=str,  default='RCP_type_LSTM_one_vocab.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')

# parser.add_argument('--data', type=str, default='./data/penn',
#                     help='location of the data corpus')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (LSTM, QRNN)')
# parser.add_argument('--checkpoint', type=str, default='./model.pt',
#                     help='model checkpoint to use')

parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
# parser.add_argument('--log-interval', type=int, default=100,
#                     help='reporting interval')
args = parser.parse_args()





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
base_path = '../recipies/data/corpus/'
# superingredients['dairy'] = ['buttermilk', 'margarine', 'butter', 'butteroil' 'cheese', 'cottage', 'milk', 'ricotta', 'sour', 'cream', 'yogurt', 'flavored', 'plain']
superingredients ['fruits'] = get_ing(base_path+'superingredients/fruits.txt')
superingredients ['grains'] = get_ing(base_path+'superingredients/grains.txt')
superingredients ['sides'] = get_ing(base_path+'superingredients/sides.txt')
superingredients ['proteins'] = get_ing(base_path+'superingredients/proteins.txt')
superingredients ['seasonings'] = get_ing(base_path+'superingredients/seasonings.txt')
superingredients ['vegetables'] = get_ing(base_path+'superingredients/vegetables.txt')
superingredients ['drinks'] = get_ing(base_path+'superingredients/drinks.txt')
superingredients ['dairy'] = get_ing(base_path+'superingredients/dairy.txt')

# print(superingredients)
# for s in superingredients:
#     print s





mcq_wrd = ['chicken','bread', 'apple', 'milk', 'salt', 'tomato'] #ch=6134, bread=3553, apple = 16, milk=4359, salt=10576, tomato=3965
#mcq_ids = [192, 398, 1437, 41, 70, 740]

# record = {corpus.dictionary.word2idx['chicken'] : [], corpus.dictionary.word2idx['bread'] : [], corpus.dictionary.word2idx['apple'] : [], corpus.dictionary.word2idx['milk'] : [], corpus.dictionary.word2idx['salt'] : [], corpus.dictionary.word2idx['tomato'] : []}
record = {192:[], 398:[], 1437:[], 41:[], 70:[], 740:[] }
mcq_result = {192:[], 398:[], 1437:[], 41:[], 70:[], 740:[] }
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data_ori_type.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

train_data_type = batchify(corpus.train_type, args.batch_size, args)
val_data_type = batchify(corpus.valid_type, eval_batch_size, args)
test_data_type = batchify(corpus.test_type, test_batch_size, args)



corpus2 = data.Corpus(args.data_type)

train_data2 = batchify(corpus2.train, args.batch_size, args)
val_data2 = batchify(corpus2.valid, eval_batch_size, args)
test_data2 = batchify(corpus2.test, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_ori_with_type.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)


ntokens2 = len(corpus2.dictionary)
model2 = model_file.RNNModel(args.model, ntokens2, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model2.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model2.parameters())
print('Model total parameters:', total_params)



criterion = nn.CrossEntropyLoss()

print ntokens, ntokens2

###############################################################################
# Testing code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def evaluate2(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model2.eval()
    if args.model == 'QRNN': model2.reset()
    total_loss = 0
    ntokens2 = len(corpus2.dictionary)
    hidden = model2.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model2(data, hidden)
        output_flat = output.view(-1, ntokens2)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)



# def evaluate_both(data_source, data_source_type, data_source2, batch_size=10):
#     # Turn on evaluation mode which disables dropout.
#     model2.eval()
#     model.eval()
#     if args.model == 'QRNN': 
#         model2.reset()
#         model.reset()
#     total_loss = 0
#     total_loss2 = 0
#     total_loss_cb = 0
#     # total_loss_cb2 = 0
#     # total_loss_cb3 = 0

#     ntokens2 = len(corpus2.dictionary)
#     ntokens = len(corpus.dictionary)
#     hidden = model.init_hidden(batch_size)
#     hidden2 = model2.init_hidden(batch_size)
#     m = nn.Softmax()
#     mcq_ids = [corpus.dictionary.word2idx[w] for w in mcq_wrd]


#     for batch,i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
#         data, targets = get_batch(data_source, i, args, evaluation=True)
#         data2, targets2 = get_batch(data_source2, i, args, evaluation=True)

#         data_type, targets_type = get_batch(data_source_type, i, args, evaluation=True)

        

#         if(batch_size==1):
#             hidden = model.init_hidden(batch_size)
#             hidden2 = model2.init_hidden(batch_size)

#         output2, hidden2 = model2(data2, hidden2)
#         output, hidden = model(data, data_type, hidden)

#         output_flat2 = output2.view(-1, ntokens2)
#         output_flat = output.view(-1, ntokens)

        
#         # print ('data.data:' ,data.data, [corpus.dictionary.idx2word[i[0]] for i in data.data])
#         # print ([corpus.dictionary.idx2word[i.data[0]] for i in targets])
#         candidates = set([corpus.dictionary.idx2word[i.data[0]] for i in targets])
#         candidates_ids = set([i.data[0] for i in targets])

#         candidates_type = set([corpus2.dictionary.idx2word[i.data[0]] for i in targets2])
#         candidates_ids_type = set([i.data[0] for i in targets2])
#         numwords = output_flat.size()[0]
#         symbol_table = get_symbol_table(targets, targets2)



#         output_flat_cb= output_flat.clone()
#         sums = []
#         for idxx in range(numwords):
#             for pos in candidates_ids: #for all candidates

#                 tp = symbol_table[pos]
#                 var_prob = output_flat_cb.data[idxx][pos]
#                 type_prob = output_flat2.data[idxx][tp]
#                 new_prob1 = 2*var_prob #just to scale values, emperical
                
#                 if corpus.dictionary.idx2word[pos]!=corpus2.dictionary.idx2word[tp]: new_prob1 = (var_prob + type_prob) #/ 2
#                 output_flat_cb.data[idxx][pos] = new_prob1
                
#         total_loss += len(data) * criterion(output_flat, targets).data
#         total_loss2 += len(data2) * criterion(output_flat2, targets2).data
#         total_loss_cb += len(data) * criterion(output_flat_cb, targets).data
        
        
#         #########
#         temp_output = output_flat_cb.clone()
#         print("our model") 
#         # or 
#         # temp_output = output_flat.clone()
#         # print("awd-st baseline")
#         #########


#         val, keys_t = temp_output.data.max(1)

#         for i in range(len(targets.data)): 
#             w= targets.data[i]

#             voilated = 0
#             base = temp_output.data[i][w]
#             if w in mcq_ids: 
#                 r = 0
#                 r2 = 0
#                 pred = keys_t[i]
#                 if pred==w: r=1
#                 for idd in mcq_ids:
#                     if idd!=w:
#                         if base<temp_output.data[i][idd]:
#                             voilated=1
#                             break
                     
#                 record[w].append(r)
#                 if voilated==0: r2 = 1
#                 mcq_result[w].append(r2)


            
            


#         # print (' soccer: ', len(data) * criterion(output_flat, targets).data), ' my: ',  len(data) * criterion(output_flat_cb, targets).data
#         if(batch%500==0): 
#             print(' only ingred not avg')
#             print ("done batch ", batch, ' of ', len(data_source)/ eval_batch_size)
#             test_loss_cb = total_loss_cb[0] / len(data_source)
#             test_loss = total_loss[0] / len(data_source)
#             test_loss2 = total_loss2[0] / len(data_source)
#             p = (100*batch)/(33000)
#             print('=' * 160)
#             print('| after: {:5.2f}% | test var loss {:5.2f} | test var ppl {:8.2f} | test type loss {:5.2f} | test type ppl {:8.2f} | test cb loss {:5.2f} | test cb ppl {:8.2f}'.format(
#                 p, test_loss, math.exp(test_loss), test_loss2, math.exp(test_loss2),  test_loss_cb, math.exp(test_loss_cb) ))
#             print('=' * 160)

#         hidden = repackage_hidden(hidden)
#         hidden2 = repackage_hidden(hidden2)

#         print ('target: ', [corpus.dictionary.idx2word[i] for i in targets.data], ' pred: ', [corpus.dictionary.idx2word[i] for i in keys_t])
        


#         # for idd in record: 
#         #     if len(record[idd]) >0:
#         #         print (corpus.dictionary.idx2word[idd], ' acc: ', sum(record[idd]), '/', len(record[idd]), sum(record[idd])*100.0/len(record[idd])  )
#         #         print (corpus.dictionary.idx2word[idd], ' mcq acc: ', sum(mcq_result[idd]), len(mcq_result[idd]), sum(mcq_result[idd])*100.0/len(mcq_result[idd]))



#     return total_loss[0] / len(data_source), total_loss2[0] / len(data_source2), total_loss_cb[0] / len(data_source)

id_map ={}
i = 0
for s in superingredients:
    for w in superingredients[s]:
        # print (w, s)
        if w in corpus.dictionary.word2idx: id_map.update({corpus.dictionary.word2idx[w]:corpus2.dictionary.word2idx[s]})
# print (id_map)



# Load the best saved model.
with open(args.save, 'rb') as f:
    # model = torch.load(f)
    model.load_state_dict(torch.load(f))
with open(args.save_type, 'rb') as f:
    # model = torch.load(f)
    model2.load_state_dict(torch.load(f))


model2.eval()
model.eval()
batch_size = 1


ntokens2 = len(corpus2.dictionary)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(batch_size)
hidden2 = model2.init_hidden(batch_size)


data = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
data2 = data #Variable(input1.data.clone()) #Variable(torch.rand(1, 1).mul(ntokens2).long(), volatile=True)
data_type = data#Variable(input1.data.clone())


m = nn.Softmax()


# if args.cuda:
#     data.data = data.data.cuda()
#     data2.data = data.data.cuda()
#     data_type = data.data.cuda()

data, targets = get_batch(test_data, i, args, seq_len = 1, evaluation=True)
data2, targets2 = get_batch(test_data_type, i, args, seq_len = 1, evaluation=True)
data_type, targets_type = get_batch(test_data2, i, args, seq_len =1, evaluation=True)

numwords = 1

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        # output, hidden = model(input, hidden)
        output2, hidden2 = model2(data2, hidden2)
        output, hidden = model(data, data_type, hidden)

        output_flat2 = output2.view(-1, ntokens2)
        output_flat = output.view(-1, ntokens)

        # numwords = output_flat.size()[0]
        # symbol_table = get_symbol_table(targets, targets2)

        output_flat_cb= output_flat.clone()
        print(output_flat_cb.size(), output_flat_cb.data)
        # exit()
        sums = []
        for idxx in range(numwords):
            for pos in id_map: #for all candidates

                tp = id_map[pos]
                var_prob = output_flat_cb.data
                type_prob = output_flat2.data
                new_prob1 = 2*var_prob #just to scale values, emperical
                
                if corpus.dictionary.idx2word[pos]!=corpus2.dictionary.idx2word[tp]: new_prob1 = (var_prob + type_prob) #/ 2
                output_flat_cb.data[pos] = new_prob1


        output = output_flat_cb
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        data.data.fill_(word_idx)

        if word_idx in id_map: 
            data2.data.fill_(id_map[word_idx])
            data_type.data.fill_(id_map[word_idx])
        else: 
            data2.data.fill_(word_idx)
            data_type.data.fill_(word_idx)


        

        word = corpus.dictionary.idx2word[word_idx]


        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
