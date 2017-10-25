import argparse
import matplotlib.pyplot as plt
import numpy as np
import string

from collections import defaultdict, Counter
#from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from proc import BOW
from text_processing import extract_text

'''
@args
sentences - list of list of words ([['first', 'sentence'], ['this', 'is', 'second', 'sentence']])
n - specifies how many top words to return (aka, return top n most frequent words)

@return
top_words - list of n pairs of words and counts that were the most frequent (ignoring stopwords)
'''
def get_most_frequent_words(sentences, n=50):
  top_words = []
  counter = Counter()
  stop_words = set(stopwords.words('english')).union(set(string.punctuation))
  for sentence in sentences:
    for word in sentence:
      if word not in stop_words:
        counter[word] += 1 

  top_words = counter.most_common(n) 
  return top_words

def get_all_unique_words(sentences):
  words = set()
  for sentence in sentences:
    for word in sentence:
      words.add(word)   
  return list(words)

'''
@args
sentences - list of lists of words, where each inner list is a sentence
embedding_model - BOW model
n - number of top words to use
'''
def visualize_top_words(sentences, embedding_model, n=50):
  top_words = get_most_frequent_words(sentences, n)
  top_words = [pair[0] for pair in top_words]
  words_ = top_words 
  word_embeddings_ = embedding_model.encode(words_)
 
  words = []
  word_embeddings = []
  for i in range(len(words_)):
    if not (word_embeddings_[i] == [0.0]*300).all():
      words.append(words_[i])
      word_embeddings.append(word_embeddings_[i]) 

  tsne = TSNE(n_components=2)
  tsne_embeddings = tsne.fit_transform(word_embeddings)

  x_vals = [pair[0] for pair in tsne_embeddings]
  y_vals = [pair[1] for pair in tsne_embeddings]
  fig, ax = plt.subplots()

  for i in range(len(words)):
    ax.annotate(words[i], (x_vals[i], y_vals[i]))
    ax.scatter([x_vals[i]], [y_vals[i]], color='k', marker='o')
    
  plt.show()

def visualize_sentences(sentences):
  pass 

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('-encoder', default='bow')
  args.add_argument('-word_embeddings_path')
  args.add_argument('-filename', help='Filename of the errata file.')
  opts = args.parse_args()

  #if opts.encoder == 'skip':
  #  model = skipthoughts.load_model()
  #  encoder = skipthoughts.Encoder(model)
  #elif opts.encoder == 'bow':
  print 'Loading embeddings'
  encoder = BOW(opts.word_embeddings_path)

  sentences = extract_text(opts.filename)


  visualize_top_words(sentences, encoder, 500)
   
