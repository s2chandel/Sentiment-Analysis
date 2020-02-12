import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from tensorflow.keras import preprocessing
# import tf_sentencepiece
import transformer
from bert_embedding import BertEmbedding
import torch
import warnings

data = pd.read_csv('train.tsv',sep='\t')

sentences =  data['Phrase']
# tokenizer = preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(sentences) 
# tokenized_lines = tokenizer.texts_to_sequences(phrases)
# tokenized_lines = np.array(tokenized_lines)


# loading pretrained bert

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
## model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = sentences.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# padding tokenized_lines
max_len = 0
for i in tokenized:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_lines])


# shape of tokens

np.array(padded).shape

# ignoring the added padding before input
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)


features = last_hidden_states[0][:,0,:].numpy()

import pickle
pickle.dump(emb,open("WordEmbeddings/embbeddings.pkl","wb"))







# bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')

# word_embeddings = bert(attention_mask)


# import tensorflow as tf 
# print(tf.version.VERSION)





