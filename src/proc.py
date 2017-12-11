import argparse
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from text_processing import extract_text, extract_errata, process_text

import skipthoughts 

import errata
import text_processing

class BOW():

  def __init__(self, word_embeddings_path):

    with open(word_embeddings_path) as f:
      lines = f.readlines()
    word_embeddings = {}
    for line in lines:
      line = line.strip().split()
      word_embeddings[line[0]] = np.array(map(float, line[1:]))
      self.size = int(word_embeddings[line[0]].size)

    print('Done loading embeddings')

    self.word_embeddings = word_embeddings
    self.vocabulary = set(self.word_embeddings.keys())
    

  def encode(self, sentences):
    embeddings = []
    for sent in sentences:
      sent = sent.split()
      bow = np.sum([self.word_embeddings[s] if s in self.vocabulary else [0.0]*self.size for s in sent], axis=0)
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
field - field to compare with for determining distance
encoder - Encoder object
query - List of query sentences (strings)
candidate_errata - List of candidate errors from which to choose nearest neighbors; each error is an Error class instance
N - Number of nearest neighbors to choose

@return
nn_errata - Dictionary with {sentence:[its nearest neighbors as *errata* (not strings) ]} as (key, value) pairs
'''
def nearest_neighbors_errata(encoder, queries, candidate_errata, N, field='Failure'): 
  processed_queries = [' '.join(process_text(text, sent_tokenize=False)) for text in queries]
  filter_query_indices = [i for i in range(len(processed_queries)) if processed_queries[i] != '']
  queries = [queries[ind] for ind in filter_query_indices]
  processed_queries = [processed_queries[ind] for ind in filter_query_indices]

  query_embs = encoder.encode(processed_queries)

  candidate_text = [candidate.get_field(field) for candidate in candidate_errata]
  candidate_text = [' '.join(process_text(text, sent_tokenize=False)) for text in candidate_text]
  filter_indices = []
  for i in range(len(candidate_text)):
    if candidate_text[i] == '' or candidate_text[i] == ' ':
        filter_indices.append(i)
  
  filter_indices = set(filter_indices)
  candidate_text_filtered = []
  candidate_errata_filtered = []
  for i in range(len(candidate_text)):
    if i not in filter_indices:
      candidate_text_filtered.append(candidate_text[i]) 
      candidate_errata_filtered.append(candidate_errata[i])
  candidate_text = candidate_text_filtered
  candidate_errata = candidate_errata_filtered

  candidate_embs = encoder.encode(candidate_text)
  
  query_embs = normalize(query_embs)
  candidate_embs = normalize(candidate_embs)
  
  neg_cos_dist = -np.matmul(query_embs, np.transpose(candidate_embs))
  
  cos_dist_sort_indices = np.argsort(neg_cos_dist, axis=1)

  nn_errata = {}
  for i in range(len(queries)):
    query_sent = queries[i]
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

'''
Compute recall@k.

@args
field - field that the query comes from, and the one that should be matched from the errata to see if the correct one was 
'''
def recall_at_k(nn_errata, k, field='Failure'):
  # check if the query is in the top 10 of the nearest neighbors
  num_relevant_found = 0
  for query in nn_errata:
    print 'query: ', query
    nns = nn_errata[query]
    for j in range(len(nns)):
      if j >= k:
        break
      errata = nns[j]
      errata_field = errata.get_field(field) 
      print 'field from nns: ', errata_field
      if errata_field == query:
        num_relevant_found += 1
        continue

  print 'num_relevant_found: ', num_relevant_found
  # for each query, there is a single matching error 
  recall_at_k = (1.0*num_relevant_found)/(1.0*len(nn_errata))
  print 'recall@%i: %.2f' % (k, recall_at_k)
  return recall_at_k

'''
@args
query_errata - list of Errata instances
nn_errata - dictionary of string queries mapped to lists of Errata instances representing the nearest neighbors found
k - top k to consider (will consider the highest BLEU score achieved from the top k)
field - field of data that we are computing BLEU score with
'''
def bleu_score(query_errata, nn_errata, k=5, field='Workaround'):
  bleu_score = 0
  cc = SmoothingFunction()
  for ground_truth_error in query_errata: 
    # the failure is used as the query
    ground_truth_failure = ground_truth_error.get_field('Failure')
    if ' '.join(process_text(ground_truth_failure, sent_tokenize=False)) == '':
      continue 
    ground_truth_field = process_text(ground_truth_error.get_field(field), sent_tokenize=False) 
    
    nns = nn_errata[ground_truth_failure]
    best_bleu = 0
    for i in range(len(nns)):
      if i == k:
        break
      nn = nns[i]
      # compute bleu score between the desired fields
      curr_field = process_text(nn.get_field(field), sent_tokenize=False)
      curr_bleu_score = sentence_bleu([ground_truth_field], curr_field, smoothing_function=cc.method3)
      if curr_bleu_score > best_bleu:
        best_bleu = curr_bleu_score 
    bleu_score += best_bleu 
    
  print 'total bleu: ', bleu_score
  avg_bleu = (1.0*bleu_score)/len(query_errata)
  print 'avg bleu: ', avg_bleu 
  return avg_bleu

# ---------------
def get_nn_sentences(args, encoder):
  sentences = extract_text(args.filename)

  sentences = sentences

  sentences = [' '.join(sent) for sent in sentences]

  #embeddings = encode(encoder, sentences)

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

def get_nn_errata(args, encoder, output=False):
  errors = extract_errata(args.filename)
  print 'num total unique errors: ', len(errors)
  # split into train and test 
  errors_train, errors_test = train_test_split(errors, test_size=0.10, random_state=25)

  failure_text = [error.get_field('Failure') for error in errors_test]
  #failure_text = [process_text(error.get_field('Failure'), sent_tokenize=False) for error in errors]
  #failure_text = [' '.join(text) for text in failure_text] 

  nns = nearest_neighbors_errata(encoder, failure_text, errors_train, 5, field='Failure')
  
  bleu = bleu_score(errors_test, nns, field='Workaround')
  #recall_at_k = evaluate_nn_errata(nns, 10, field='Failure')

  if output:
    nn_file = open('nns.txt', 'w')
    for query, neighbors in nns.iteritems():
      nn_file.write('Query: ' + query + '\n')
      nn_file.write('NNs\n')
      for neighbor in neighbors:
        possible_description = neighbor.get_field('Details')
        failure = neighbor.get_field('Failure')
        workaround = neighbor.get_field('Workaround')
        nn_file.write('Failure: \n')
        nn_file.write(failure)
        nn_file.write('\n')
        #nn_file.write('Details: \n')
        #nn_file.write(possible_description) 
        #nn_file.write('\n')
        nn_file.write('Workaround: \n')
        nn_file.write(workaround)
        nn_file.write('\n\n')
      nn_file.write('-------------\n')
  return nns

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('--encoder', default='skip')
  args.add_argument('--word_embeddings_path', default='glove.840B.300d.txt')
  args.add_argument('--filename', default='CortexA9.txt')

  opts = args.parse_args()

  if opts.encoder == 'skip':
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
  elif opts.encoder == 'bow':
    #encoder = BOW()
    print 'Loading word embeddings into BOW encoder...'
    encoder = BOW(opts.word_embeddings_path)
    print 'Done loading word embeddings.'

  # get_nn_sentences(args, encoder)
  nns = get_nn_errata(opts, encoder, output=True)

  
  
