from nltk.tokenize import word_tokenize
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

src = []
tgt = []

words = []

def write_data(data, fname):
  with open(fname + '.src', 'w') as f:
    with open(fname + '.tgt', 'w') as g:
      for dat in data:
        f.write(dat[0] + "\n")
        g.write(dat[1] + "\n")

with open('intel.tsv', 'r') as f:
  line = f.readline()
  line = f.readline()
  ct = 0
  while line:
    #print(line)
    line = line.strip().split("\t")
    src_sent = ' '.join(word_tokenize(line[2]))
    tgt_sent = ' '.join(word_tokenize(line[0]))
    src.append(src_sent)
    tgt.append(tgt_sent)
    words.extend(src_sent.split())
    words.extend(tgt_sent.split())
    line = f.readline()

words = set(words)

data = zip(src, tgt)

N = len(data)
train = data[:int(0.9*N)]
dev = data[int(0.9*N):int(0.95*N)]
test = data[int(0.95*N):]

write_data(train, '/home/llajan/errata/nmt_data/train')
write_data(dev, '/home/llajan/errata/nmt_data/dev')
write_data(test, '/home/llajan/errata/nmt_data/test')

with open('/home/llajan/errata/nmt_data/vocab.src', 'w') as f:
  for word in words:
    f.write(word + "\n")
with open('/home/llajan/errata/nmt_data/vocab.tgt', 'w') as f:
  for word in words:
    f.write(word + "\n")
