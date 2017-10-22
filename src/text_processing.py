'''
This file will include all general text processing functions.
'''
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

'''
Take in a string and split into sentences; strip all non-ascii characters and lower-case all text; then tokenize into words.

@args
text - string of text
@return
tokenized_sents - list of lists of words (i.e. [['this', 'is', 'first', 'sentence'], ['this', is', 'second', 'sentence']])
'''
def process_text(text):
  sents = sent_tokenize(text)
  tokenized_sents = [word_tokenize(sent.encode('ascii', 'ignore').lower()) for sent in sents]
  tokenized_sents = [sent for sent in tokenized_sents if sent != []]
  
  return tokenized_sents
