import tensorflow as tf
import numpy as np
from utils import vocab_mapping
from utils import mask_loss

FLAGS = tf.flags.FLAGS

def get_fin(outputs, lens):
  inds = [tf.range(FLAGS.batch_size), tf.cast(lens, tf.int32) - 1]
  out_inds = tf.cast(tf.transpose(tf.stack(inds)), tf.int32)
  gate_enc = tf.gather_nd(outputs, out_inds)
  return gate_enc


class TextCNN(object):
  """ Text classification CNN
  """

  def __init__(
      self, num_classes, vocab_size, sen_batch, labels,
      dropout_keep_prob, embedding_size, filter_sizes, num_filters,
      l2_reg_lambda=0.0, reuse=False, mask_zeros=False, mask_batch=None,
      batch_size=None, embedding_input=False, vocab_tf=False, vocab_tf_file=None,
      multiclass=True, load_embeddings=True):

    #if not batch_size:
    #  batch_size = FLAGS.batch_size

    self.input_x = sen_batch
    self.input_y = labels

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    # Embedding layer

    word_vectors = slim.get_variables_by_name('cnn_sty/word_embedding')

    with tf.variable_scope('cnn_sty', reuse=reuse):

      if embedding_input:
        self.embedded_chars = self.input_x
      else:
        if word_vectors:
          word_vectors = word_vectors[0]
        elif load_embeddings and FLAGS.load_embeddings:
          with open(FLAGS.embeddings_path, 'r') as f:
            word_embs = np.load(f)
          print(word_embs.shape, vocab_size)
          assert word_embs.shape[0] == vocab_size
          word_vectors = tf.get_variable(
              name='word_embedding', shape=word_embs.shape,
              initializer=tf.constant_initializer(word_embs))
          print('Embeddings loaded')
        else:
          # Add UNKNOWN, EOS symbols
          word_vectors = tf.get_variable(
              name='word_embedding',
              shape=[vocab_size, embedding_size],
              initializer=tf.random_uniform_initializer(
                  minval=-FLAGS.uniform_init_scale,
                  maxval=FLAGS.uniform_init_scale)
          )
	  
        if vocab_tf:
          vocab_tf = vocab_mapping(vocab_tf_file, FLAGS.vocab_file)
          word_vectors = tf.gather(word_vectors, vocab_tf)

        self.W = word_vectors
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

      if mask_zeros:
        mask_exp = tf.expand_dims(tf.cast(mask_batch, tf.float32), -1)
        mask_tile = tf.tile(mask_exp, [1, 1, embedding_size])
        self.embedded_chars *= mask_tile

      self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      #switch_matrix_add = []
      #ind = [[p, q] for p in range(batch_size) for q in range(num_filters)]
      #ind = tf.constant(np.array(ind))
      for _, filter_size in enumerate(filter_sizes):
        with tf.variable_scope('conv-maxpool-%s' % filter_size, reuse=reuse):
          # Convolution Layer
          filter_shape = [filter_size, embedding_size, 1, num_filters]
          W = tf.get_variable(name='W',
                              shape=filter_shape,
                              initializer=tf.truncated_normal_initializer(
                                  mean=0.0, stddev=0.1))
          b = tf.get_variable(name='b',
                              shape=[num_filters],
                              initializer=tf.constant_initializer(0.1))
          conv = tf.nn.conv2d(
              self.embedded_chars_expanded,
              W,
              strides=[1, 1, 1, 1],
              padding='VALID',
              name='conv')
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
          # Maxpooling over the outputs
          #pooled = tf.nn.max_pool(
          #    h,
          #    ksize=[1, sequence_length - filter_size + 1, 1, 1],
          #    strides=[1, 1, 1, 1],
          #    padding='VALID',
          #    name='pool')
          pooled = tf.reduce_max(h, axis=1)
          pooled_outputs.append(pooled)


      # Combine all the pooled features
      num_filters_total = num_filters * len(filter_sizes)
      #self.h_pool = tf.concat(pooled_outputs, 3)
      self.h_pool = tf.concat(pooled_outputs, 2)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

      # Add dropout
      self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

      # Final (unnormalized) scores and predictions
      with tf.variable_scope('output', reuse=reuse):
        W = tf.get_variable(
            'W',
            shape=[num_filters_total, num_classes if multiclass else 1],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                            shape=[num_classes if multiclass else 1],
                            initializer=tf.constant_initializer(0.1))
        l2_loss += tf.nn.l2_loss(W)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
        self.scores = tf.squeeze(self.scores)
        if multiclass:
          self.predictions = tf.argmax(self.scores, 1, name='predictions')
        else:
          self.predictions = tf.cast(tf.greater(self.scores, 0.0), tf.int64)

    if multiclass:
      self.scores_probs = tf.nn.softmax(self.scores)
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.scores, labels=self.input_y)
    else:
      self.scores_probs = tf.nn.sigmoid(self.scores)
      labels = tf.cast(self.input_y, tf.float32)
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.scores, labels=labels)

    self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # Accuracy
    correct_predictions = tf.equal(
        self.predictions, tf.cast(self.input_y, tf.int64))
    self.correct_predictions = tf.cast(correct_predictions, 'float')
    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
    self.accuracy = tf.Print(self.accuracy, [self.accuracy])

class TextRNN(object):
  """ Text classification CNN
  """
  def __init__(
      self, num_classes, sen_batch, labels, 
      dropout_keep_prob, embedding_size, batch_size,
      l2_reg_lambda=0.0, reuse=False, mask_zeros=False, 
      mask_batch=None, multiclass=None, loss_fn='xent',
      dirn='uni', src='real', labels_mask=None):

    if not multiclass:
      multiclass = FLAGS.multiclass

    # Placeholders for input, output and dropout
    self.embedded_chars = sen_batch
    self.input_y = labels

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    if mask_zeros:
      mask_exp = tf.expand_dims(tf.cast(mask_batch, tf.float32), -1)
      mask_tile = tf.tile(mask_exp, [1, 1, embedding_size])
      self.embedded_chars *= mask_tile

    if dirn == 'uni':
      cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size, reuse=reuse)
    else:
      cell_fw = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size//2, reuse=reuse)
      cell_bw = tf.contrib.rnn.GRUCell(num_units=FLAGS.lstm_size//2, reuse=reuse)

    zero_state = tf.zeros((batch_size, embedding_size))
    seq_len = tf.reduce_sum(tf.cast(mask_batch, tf.int64), axis=1)

    if dirn == 'uni':
      cnt_enc_states, _ = tf.nn.dynamic_rnn(
          cell=cell, dtype=tf.float32, inputs=self.embedded_chars,
          initial_state=zero_state, sequence_length=seq_len)
      cnt_enc = get_fin(cnt_enc_states, seq_len)
    else:
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32, 
	  inputs=self.embedded_chars, sequence_length=seq_len)
      cnt_enc = tf.concat(states, 1)

    self.cnt_enc = cnt_enc
