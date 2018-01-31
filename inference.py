import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model as model_file

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
parser.add_argument('--save', type=str,  default='RCP_ori_LSTM.pt',
                    help='path to save the final model')
parser.add_argument('--save_type', type=str,  default='RCP_type_LSTM.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)



corpus2 = data.Corpus(args.data_type)

train_data2 = batchify(corpus2.train, args.batch_size, args)
val_data2 = batchify(corpus2.valid, eval_batch_size, args)
test_data2 = batchify(corpus2.test, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_file.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
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


def get_symbol_table(data, types):
	print('in sym table')

	# print ({i[0]: types.data[0] for i in data.data})
	# data = data.data
	# types = types.data
	print(data, " \n types: ", types)
	id_map ={}
	i = 0
	# for pos, tp in zip(data.data, types.data):
	# 	id_map.update({pos[0]:tp[0]})
	for pos, tp in zip(data, types):
		id_map.update({pos.data[0]:tp.data[0]})

	print ('symbol table:::')
	print (id_map)
	return id_map


def evaluate_both(data_source, data_source2, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model2.eval()
    model.eval()
    if args.model == 'QRNN': 
    	model2.reset()
    	model.reset()
    total_loss = 0
    total_loss2 = 0
    total_loss_cb = 0


    ntokens2 = len(corpus2.dictionary)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    hidden2 = model2.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        data2, targets2 = get_batch(data_source2, i, args, evaluation=True)


        output2, hidden2 = model2(data2, hidden2)
        output, hidden = model(data, hidden)

        output_flat2 = output2.view(-1, ntokens2)
        output_flat = output.view(-1, ntokens)

     #    m = nn.Softmax()
    	# output_flat_f = m(output_flat)
    	# output_flat_tf = m(output_flat2)
    	print ('data.data:' ,data.data, [corpus.dictionary.idx2word[i[0]] for i in data.data])
    	print ('targets.data:' ,targets)
    	print ([corpus.dictionary.idx2word[i.data[0]] for i in targets])
    	candidates = set([corpus.dictionary.idx2word[i.data[0]] for i in targets])
    	candidates_ids = set([i.data[0] for i in targets])

    	candidates_type = set([corpus2.dictionary.idx2word[i.data[0]] for i in targets2])
    	candidates_ids_type = set([i.data[0] for i in targets2])

    	print ('data2.data:' ,data2.data, [corpus2.dictionary.idx2word[i[0]] for i in data2.data])
    	numwords = output_flat.size()[0]
    	print ('numwords to predict: ', numwords)
    	# symbol_table = get_symbol_table(data, data2)
    	symbol_table = get_symbol_table(targets, targets2)
    	output_flat_cb= output_flat.clone()

    	print ('two words: ',corpus.dictionary.idx2word[data.data[0][0]], corpus.dictionary.idx2word[data.data[1][0]])
    	print( ' target: ', corpus.dictionary.idx2word[targets[0].data[0]], corpus.dictionary.idx2word[targets[1].data[0]])
    	print ('total # word to pred: ', len(data), len(targets), numwords)
    	for idxx in range(numwords):
    		print (idxx, "'th prediction: ", 'word: ', corpus.dictionary.idx2word[data.data[idxx][0]], 'target: ', corpus.dictionary.idx2word[targets[idxx].data[0]])
    		i = 0
    		print (candidates, len(candidates))

    		for pos in targets: #for all candidates
    			
    			pos = pos.data[0]
    			print  ('cand id : ', pos)
    			print ('word: ', corpus.dictionary.idx2word[pos])

    			# print  ('cand type id by loop from targets2: ', tp)
    			# print ('word: ', corpus.dictionary.idx2word[tp])
    			tp = symbol_table[pos]
    			print  ('cand type id by symbol table: ', tp)
    			print ('type word: ', corpus2.dictionary.idx2word[tp])
    			
    			print 'prob for candidate : ', corpus.dictionary.idx2word[pos], ' is: ', output_flat_cb.data[idxx][pos], ' map to type: ', corpus2.dictionary.idx2word[tp], ' with prob: ', output_flat2.data[idxx][tp] 
    			output_flat_cb.data[idxx][pos] += output_flat2.data[idxx][tp]
    		# 	i +=1

# 
        total_loss += len(data) * criterion(output_flat, targets).data
        total_loss2 += len(data2) * criterion(output_flat2, targets2).data
        total_loss_cb += len(data2) * criterion(output_flat_cb, targets).data

        # print (' soccer: ', len(data) * criterion(output_flat, targets).data), ' my: ',  len(data) * criterion(output_flat_f, targets).data


        hidden = repackage_hidden(hidden)
        hidden2 = repackage_hidden(hidden2)

    return total_loss[0] / len(data_source), total_loss2[0] / len(data_source2), total_loss_cb[0] / len(data_source)



def infer(valid_data_trimed, valid_type_data_trimed, forward_model,  forward_type_model, var_dict, type_dict, criterion, eval_batch_size):
        # Turn on evaluation mode which disables dropout.
        forward_model.eval()
       
        forward_type_model.eval()
        

        total_mean_loss_f = 0
        total_mean_loss_tf = 0
        total_mean_loss_cb = 0
        

        ntokens = len(var_dict)
        type_ntokens = len(type_dict)


        for batch, i in enumerate  (range(0, valid_data_trimed.size(0) - 1, args.bptt)):
        	data_f, targets_f = get_batch(valid_data_trimed, i, args, evaluation=True)
        	data_tf, targets_tf = get_batch(valid_type_data_trimed, i, args, evaluation=True)
        	#mask = data.ne(dictionary.padding_id)
        	hidden_f = forward_model.init_hidden(eval_batch_size)
        	hidden_tf = forward_type_model.init_hidden(eval_batch_size) #for each sentence need to initialize

        	output_f, hidden_f = forward_model(data_f, hidden_f)
        	output_tf, hidden_tf = forward_type_model(data_tf, hidden_tf)

        	output_flat_f = output_f.view(-1, ntokens)
        	output_flat_tf = output_tf.view(-1, type_ntokens) # (batch x seq) x ntokens 

        	m = nn.Softmax()
        	output_flat_f = m(output_flat_f)
        	output_flat_tf = m(output_flat_tf)


        	numwords = output_flat_f.size()[0]
        	symbol_table = get_symbol_table(data_f, data_tf)
        	# print(symbol_table)
        	output_flat_cb= output_flat_f.clone()#torch.FloatTensor(output_flat_b.size()).zero_()#copy[:]
        	# print(output_flat_cb[0])
        	# make_symbol_table()

        	for idxx in range(numwords):
        		for pos in set(data_f.data[0]): 
        			tp = symbol_table[pos]
        			output_flat_cb.data[idxx][pos] += output_flat_tf.data[idxx][tp]

        	loss_f = criterion(output_flat_f, targets_f)
        	loss_tf = criterion(output_flat_tf, targets_tf)
        	loss_cb =  nn.functional.nll_loss(torch.log(m(output_flat_cb)), targets_f, size_average=True)


        	total_mean_loss_f += len(data_f) * loss_f.data
        	total_mean_loss_cb += len(data_f) * loss_cb.data
        	total_mean_loss_tf += len(data_f) * loss_tf.data


        	if(batch%500==0): print ("done batch ", batch, ' of ', len(valid_data_trimed)/ eval_batch_size)

        batch +=1 #starts counting from 0 hence total num batch (after finishing) = batch + 1

        avg_loss_f = total_mean_loss_f[0]/valid_data_trimed.size(0)
        avg_loss_cb = total_mean_loss_cb[0]/valid_data_trimed.size(0)
        avg_loss_tf = total_mean_loss_tf[0]/valid_data_trimed.size(0)



        ppl_f = math.exp(avg_loss_f)
        ppl_cb = math.exp(avg_loss_cb)
        ppl_tf = math.exp(avg_loss_tf)



             
        print('Var model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'forward', avg_loss_f, ppl_f))
        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'combined ', avg_loss_cb, ppl_cb))
        print('Type model direction {} avg loss: {:.2f}  ppl: {:.2f} '.format(  'forward', avg_loss_tf, ppl_tf))
        

           
           










# Load the best saved model.
with open(args.save, 'rb') as f:
    # model = torch.load(f)
    model.load_state_dict(torch.load(f))
with open(args.save_type, 'rb') as f:
    # model = torch.load(f)
    model2.load_state_dict(torch.load(f))

# Run on test data.
# test_loss = evaluate(test_data[:args.debug], test_batch_size)
# print('=' * 89)
# print('| End of var training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)



# test_loss = evaluate2(test_data2[:args.debug], test_batch_size)
# print('=' * 89)
# print('| End of type training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)

test_loss, test_loss2, test_loss_cb = evaluate_both(test_data[:args.debug], test_data2[:args.debug], test_batch_size)
print('=' * 189)
print('| End of training | test var loss {:5.2f} | test var ppl {:8.2f} | test type loss {:5.2f} | test type ppl {:8.2f} | test cb loss {:5.2f} | test cb ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss), test_loss2, math.exp(test_loss2),  test_loss_cb, math.exp(test_loss_cb) ))
print('=' * 189)



# infer(test_data[:args.debug], test_data2[:args.debug], model, model2, corpus.dictionary, corpus2.dictionary, criterion, test_batch_size)
     

