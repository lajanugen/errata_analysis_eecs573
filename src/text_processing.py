'''
This file will include all general text processing functions.
'''
import csv
import re
import string
import sys

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
  text = re.sub(r'[^\x00-\x7f]',r'', text)
  sents = sent_tokenize(text)
  tokenized_sents = [word_tokenize(sent.encode('ascii', 'ignore').lower()) for sent in sents]
  tokenized_sents = [sent for sent in tokenized_sents if sent != []]
  
  return tokenized_sents

'''
Take in a file and extract the text from the specified fields.

@args
filename - filename of the tsv/csv file
fields - list of the names of the columns that text should be extracted from
delimiter - delimiter character of the file

@return
tokenized_sents - list of lists of words (i.e. [['this', 'is', 'first', 'sentence'], ['this', is', 'second', 'sentence']])
'''
def extract_text(filename, fields=['Details'], delimiter='\t'):
  all_text = []
  with open(filename, 'rb') as infile:
    reader = csv.DictReader(infile, delimiter=delimiter) 
    for row in reader:
      all_text.extend([row[field] for field in fields if field in row])
  
  tokenized_sents = []
  for text in all_text:
    tokenized_sents.extend(process_text(text)) 
       
  return tokenized_sents

if __name__ == '__main__':
  filename = sys.argv[1]
  print extract_text(filename, ['Details'])[:10]
