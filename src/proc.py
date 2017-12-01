import argparse
import numpy as np
#import tensorflow as tf

from text_processing import extract_text
#import skipthoughts # NOTE: commenting out for now because I don't have it installed


import errata

class BOW():

  def __init__(self, word_embeddings_path):

    with open(word_embeddings_path) as f:
      lines = f.readlines()
    word_embeddings = {}
    for line in lines:
      line = line.strip().split()
      word_embeddings[line[0]] = np.array(map(float, line[1:]))

    print('Done loading embeddings')

    self.word_embeddings = word_embeddings
    self.vocabulary = set(self.word_embeddings.keys())

  def encode(self, sentences):
    embeddings = []
    for sent in sentences:
      sent = sent.split()
      bow = np.sum([self.word_embeddings[s] if s in self.vocabulary else [0.0]*300 for s in sent], axis=0)
      embeddings.append(bow)
      
    return embeddings

'''
Normalize vectors

@args
2D numpy array

@return
Array of row normalized vectors
'''
def normalize(embeddings):
  embeddings = [v/np.linalg.norm(v) for v in embeddings]
  return embeddings

'''
Identify nearest neighbor sentences in embedding space

@args
encoder - Encoder object
query - List of query sentences
candidates - List of candidate sentences from which to choose NNs
N - Number of nearest neighbors to choose

@return
Dictionary with sentence:[it's nearest neighbors] as (key,value) pairs
'''
def nearest_neighbors(encoder, query, candidates, N):
  query_embs = encoder.encode(query)
  candidate_embs = encoder.encode(candidates)
  
  query_embs = normalize(query_embs)
  candidate_embs = normalize(candidate_embs)

  neg_cos_dist = -np.matmul(query_embs, np.transpose(candidate_embs))

  cos_dist_sort_idx = np.argsort(neg_cos_dist, axis=1)

  nn_sentences = {}
  for i in range(len(query)):
    query_sent = query[i]
    nn_sents = [candidates[j] for j in cos_dist_sort_idx[i][:N]]
    nn_sentences[query_sent] = nn_sents

  return nn_sentences

'''
Identify nearest neighbor *errata* in embedding space, where the distance is computed using the text in the specified field.

@args
encoder - Encoder object
query - List of query sentences (strings)
candidate_errata - List of candidate errors from which to choose nearest neighbors
N - Number of nearest neighbors to choose

@return
nn_errata - Dictionary with {sentence:[its nearest neighbors as *errata* (not strings) ]} as (key, value) pairs
'''
def nearest_neighbors_errata(encoder, query, candidate_errata, N, field='effect'): 
  query_embs = encoder.encode(query)
  candidate_text = [candidates.get_field(field) for candidate in candidate_errata]
  candidate_embs = encoder.encode(candidate_text)
  
  query_embs = normalize(query_embs)
  candidate_embs = normalize(candidate_embs)
  
  neg_cos_dist = -np.matmul(query_embs, np.transpose(candidate_embs))
  
  cos_dist_sort_indices = np.argsort(neg_cos_dist, axis=1)

  nn_errata = {}
  for i in range(len(query)):
    query_sent = query[i]
    curr_nn_errata = [candidate_errata[j] for j in cos_dist_sort_indices[i][:N]]
    nn_errata[query_sent] = curr_nn_errata
  
  return nn_errata
  

'''
Embed a given list of sentences

@args
encoder - Encoder object
sentences - List of sentences to embed

@return
Numpy array of cncoded representations of input sentences
'''
def encode(encoder, sentences):

  embeddings = encoder.encode(sentences)

  return embeddings

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('-encoder', default='skip')
  args.add_argument('-word_embeddings_path', default='glove.840B.300d.txt')
  args.add_argument('-filename', default='CortexA9.txt')

  opts = args.parse_args()

  if opts.encoder == 'skip':
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
  elif opts.encoder == 'bow':
    #encoder = BOW()
    encoder = BOW(opts.word_embeddings_path)
  
  # TODO - parse in the data into Error objects and pass into nearest neighbor function

  sentences = extract_text(opts.filename)

  sentences = sentences

  sentences = [' '.join(sent) for sent in sentences]

  embeddings = encode(encoder, sentences)

  #nns = nearest_neighbors(encoder, sentences[:2], sentences[2:], 5)
  nns = nearest_neighbors(encoder, sentences, sentences, 5)

  nn_file = open('nns.txt', 'w')
  for k, v in nns.iteritems():
    nn_file.write('Query: ' + k + '\n')
    nn_file.write('NNs\n')
    for s in v:
      nn_file.write(s + '\n')
    nn_file.write('-------\n')
    #print('Query: ' + k)
    #print('NNs')
    #for s in v:
    #  print(s)
    #print('-------')

