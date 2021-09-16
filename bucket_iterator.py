# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        print(len(data),batch_size,num_batch)
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_polarity = []
        batch_entity_graph = []
        batch_attribute_graph = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])

        for item in batch_data:
            text_indices,polarity, entity_graph, attribute_graph = \
                item['text_indices'], \
                item['polarity'], item['entity_graph'], item['attribute_graph']
            text_padding = [0] * (max_len - len(text_indices))
            batch_text_indices.append(text_indices + text_padding)
            batch_polarity.append(polarity)
            batch_entity_graph.append(numpy.pad(entity_graph, \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))
            batch_attribute_graph.append(numpy.pad(attribute_graph,
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant'))

        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'entity_graph': torch.tensor(batch_entity_graph), \
                'attribute_graph': torch.tensor(batch_attribute_graph),
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
