# -*- coding: utf-8 -*-

import os
import pickle
import spacy
import numpy as np

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = '/home/suhang/ABSA/glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
    
class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                try:
                    entity, attribute = lines[i + 1].lower().strip().split()
                except:
                    entity = lines[i + 1].lower().strip()
                    attribute = ''
                text += text_raw + " " + entity + " " + attribute + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + '.graph_entity', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        fin = open(fname + '.graph_attribute', 'rb')
        idx2gragh_a = pickle.load(fin)
        fin.close()

        all_data = []
        graph_id = 0

        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            polarity = lines[i+2].strip()
            text_indices = tokenizer.text_to_sequence(text)
            try:
                polarity = int(polarity)+1
            except:
                print(text)
            dependency_graph = idx2gragh[graph_id]
            aspect_graph = idx2gragh_a[graph_id]

            data = {
                'text_indices': text_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'aspect_graph': aspect_graph,
            }

            all_data.append(data)
            graph_id += 1
        return all_data

    def __init__(self, dataset='15_rest', knowledge_base='conceptnet',embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            '14': {
                'train': './dataset/{}/14_train.raw.tokenized'.format(knowledge_base),
                'test': './dataset/{}/14_test.raw.tokenized'.format(knowledge_base)
            },
            '15_rest': {
                'train': './dataset/{}/15_rest_train.raw.tokenized'.format(knowledge_base),
                'test': './dataset/{}/15_rest_test.raw.tokenized'.format(knowledge_base)
            },
            '16_rest': {
                'train': './dataset/{}/16_rest_train.raw.tokenized'.format(knowledge_base),
                'test': './dataset/{}/16_rest_test.raw.tokenized'.format(knowledge_base)
            },
            '15_lap': {
                'train': './dataset/{}/15_lap_train.raw.tokenized'.format(knowledge_base),
                'test': './dataset/{}/15_lap_test.raw.tokenized'.format(knowledge_base)
            },
            '16_lap': {
                'train': './dataset/{}/16_lap_train.raw.tokenized'.format(knowledge_base),
                'test': './dataset/{}/16_lap_test.raw.tokenized'.format(knowledge_base)
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
