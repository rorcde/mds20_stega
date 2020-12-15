import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator, TabularDataset

import spacy

import numpy as np

import random
import math
import time

class RNNStegaDetector(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, n_layers,
              embedding_length, pad_idx, dropout, bidirectional=True):
		super(RNNStegaDetector, self).__init__()

		"""
		Arguments
		---------
		hidden_size : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		pad_idx : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""

		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
				
		self.embedding = nn.Embedding(vocab_size, embedding_length, padding_idx = pad_idx)
		  
				
		self.rnn = nn.LSTM(embedding_length, 
                           hidden_size, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
		
		self.fc = nn.Linear(2*hidden_size, output_size)
		self.dropout = nn.Dropout(dropout)
				
	def forward(self, text):      
		#text shape = (sent len, batch size)
		embedded = self.dropout(self.embedding(text))
		    
		#pack sequence
		# packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
		    
		output, (hidden, cell) = self.rnn(embedded)
		    
		#unpack sequence
		# output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
		    
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
  
		logits = self.fc(hidden)
		        
		return logits
