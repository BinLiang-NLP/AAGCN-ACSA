# -*- coding: utf-8 -*-

import os
import spacy
import csv
import pickle
import numpy as np
import re
import math
import argparse

def generate_graph(opt):
    path = opt.knowledge_base + '/' + opt.dataset
    if os.path.exists(path + '_train.raw.graph_entity') and \
       os.path.exists(path + '_test.raw.graph_entity'):
        print('The praph already exits.')
        return
    else:
        if 'lap' in opt.dataset:
            filename_train = path + '_train.raw'
            fin = open(filename_train, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_train = fin.readlines()
            fin.close()
            lines_train_tokenized = tokenize(lines_train)
            file_train = open(filename_train + '.tokenized', 'w')
            file_train.writelines(lines_train_tokenized)
            file_train.close()
            generate_graph_lap(lines_train, lines_train_tokenized, opt, 'train')

            filename_test = path + '_test.raw'
            fin = open(filename_test, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_test = fin.readlines()
            fin.close()
            lines_test_tokenized = tokenize(lines_test)
            file_test = open(filename_test + '.tokenized', 'w')
            file_test.writelines(lines_test_tokenized)
            file_test.close()
            generate_graph_lap(lines_test, lines_test_tokenized, opt, 'test')
        elif '15' in opt.dataset or '16' in opt.dataset:
            filename_train = path + '_train.raw'
            fin = open(filename_train, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_train = fin.readlines()
            fin.close()
            lines_train_tokenized = tokenize(lines_train)
            file_train = open(filename_train + '.tokenized', 'w')
            file_train.writelines(lines_train_tokenized)
            file_train.close()
            generate_graph_1516(lines_train, lines_train_tokenized, opt, 'train')

            filename_test = path + '_test.raw'
            fin = open(filename_test, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_test = fin.readlines()
            fin.close()
            lines_test_tokenized = tokenize(lines_test)
            file_test = open(filename_test + '.tokenized', 'w')
            file_test.writelines(lines_test_tokenized)
            file_test.close()
            generate_graph_1516(lines_test, lines_test_tokenized, opt, 'test')
        else:
            filename_train = path + '_train.raw'
            fin = open(filename_train, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_train = fin.readlines()
            fin.close()
            lines_train_tokenized = tokenize(lines_train)
            file_train = open(filename_train + '.tokenized', 'w')
            file_train.writelines(lines_train_tokenized)
            file_train.close()
            generate_graph_14(lines_train, lines_train_tokenized, opt, 'train')

            filename_test = path + '_test.raw'
            fin = open(filename_test, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines_test = fin.readlines()
            fin.close()
            lines_test_tokenized = tokenize(lines_test)
            file_test = open(filename_test + '.tokenized', 'w')
            file_test.writelines(lines_test_tokenized)
            file_test.close()
            generate_graph_14(lines_test, lines_test_tokenized, opt, 'test')
    print('The graph have been successfully generated.')

def tokenize(lines):
    newlines = []
    nlp = spacy.load('en_core_web_sm')
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        document = nlp(text)
        newtext = ""
        for token in document:
            newtext += token.text + ' '
        newtext += '\n'
        newlines.append(newtext)
        newlines.append(lines[i + 1])
        newlines.append(lines[i + 2])
    return newlines

def generate_graph_lap(lines, newlines, opt, type):
    nlp = spacy.load('en_core_web_sm')
    csv.field_size_limit(500 * 1024 * 1024)
    if opt.knowledge_base == 'senticnet':
        steps = 5
    else:
        steps = 3

    file_entity_name = opt.knowledge_base +'/seed_words_list_entity_lap.csv'
    entity_seed_words = {}
    with open(file_entity_name, 'r') as f:
        reader = csv.reader(f)
        entities = list(reader)
        for entity in entities[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(entity[s + 1]))
            entity_seed_words[entity[0].lower()] = seed_words
    f.close()
    word2idx = {}
    idx = 0
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word in set(words):
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    word_word_matrix = np.zeros((len(word2idx), len(word2idx)), dtype=int)
    word_sentence_matrix = np.zeros((int(len(newlines) / 3), len(word2idx)), dtype=int)
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word1 in set(words):
            word_sentence_matrix[int(i / 3)][word2idx[word1]] += 1
            for word2 in set(words):
                if word1 != word2:
                    word_word_matrix[word2idx[word1]][word2idx[word2]] += 1

    fout = open(opt.knowledge_base+ '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_entity', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        entity, attribute = lines[i + 1].lower().strip().split()
        steps_list = []
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        for token in document:
            if token.text == entity:
                step = 0
            else:
                step = find_step(token.text, entity_seed_words[entity])
            steps_list.append(step)
            if step != -1:
                weight = 1 / (step + 1)
                for token2 in document:
                    if token2 != token:
                        if token in token2.children or token2 in token.children:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += (pmi + 1) * weight
                                graph[token2.i][token.i] += (pmi + 1) * weight
                            else:
                                graph[token.i][token2.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                                graph[token2.i][token.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                        else:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += pmi * weight
                                graph[token2.i][token.i] += pmi * weight
                            else:
                                graph[token.i][token2.i] += 1 / abs(token.i - token2.i) * weight
                                graph[token2.i][token.i] += 1 / abs(token.i - token2.i) * weight
                    else:
                        graph[token.i][token.i] += 1 * weight
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

    file_attribute_name = opt.knowledge_base + '/seed_words_list_attribute_lap.csv'
    attribute_seed_words = {}
    with open(file_attribute_name, 'r') as f:
        reader = csv.reader(f)
        attributes = list(reader)
        for attribute in attributes[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(attribute[s + 1]))
            attribute_seed_words[attribute[0].lower()] = seed_words
    f.close()

    fout = open(opt.knowledge_base + '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_attribute', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        entity, attribute = lines[i + 1].lower().strip().split()
        steps_list = []
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        for token in document:
            if token.text == attribute:
                step = 0
            else:
                step = find_step(token.text, attribute_seed_words[attribute])
            steps_list.append(step)
            if step != -1:
                weight = 1 / (step + 1)
                for token2 in document:
                    if token2 != token:
                        if token in token2.children or token2 in token.children:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += (pmi + 1) * weight
                                graph[token2.i][token.i] += (pmi + 1) * weight
                            else:
                                graph[token.i][token2.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                                graph[token2.i][token.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                        else:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += pmi * weight
                                graph[token2.i][token.i] += pmi * weight
                            else:
                                graph[token.i][token2.i] += 1 / abs(token.i - token2.i) * weight
                                graph[token2.i][token.i] += 1 / abs(token.i - token2.i) * weight
                    else:
                        graph[token.i][token.i] += 1 * weight
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

def find_step(target, seedwords):
    for i in range(len(seedwords)):
        if target in seedwords[i]:
            return i+1
    return -1

def calculate_pmi(w1, w2, word2idx, word_word_matrix, word_sentence_matrix):
    if w1 == w2:
        return 0
    elif hasALetter(w1) and hasALetter(w2):
        num_pair = word_word_matrix[word2idx[w1]][word2idx[w2]]
        num_w1 = sum(word_sentence_matrix[:,word2idx[w1]])
        num_w2 = sum(word_sentence_matrix[:,word2idx[w2]])
        pmi = math.log(num_pair * len(word_sentence_matrix) / (num_w1 * num_w2))
        return pmi
    else:
        return 0

def hasALetter(word):
    res = re.search(r'[a-z]+', word, re.I)
    return bool(res)

def generate_graph_1516(lines, newlines, opt, type):
    nlp = spacy.load('en_core_web_sm')
    csv.field_size_limit(500 * 1024 * 1024)
    if opt.knowledge_base == 'senticnet':
        steps = 5
    else:
        steps = 3

    file_entity_name = opt.knowledge_base +'/seed_words_list_entity_rest.csv'
    entity_seed_words = {}
    with open(file_entity_name, 'r') as f:
        reader = csv.reader(f)
        entities = list(reader)
        for entity in entities[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(entity[s + 1]))
            entity_seed_words[entity[0].lower()] = seed_words
    f.close()
    word2idx = {}
    idx = 0
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word in set(words):
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    word_word_matrix = np.zeros((len(word2idx), len(word2idx)), dtype=int)
    word_sentence_matrix = np.zeros((int(len(newlines) / 3), len(word2idx)), dtype=int)
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word1 in set(words):
            word_sentence_matrix[int(i / 3)][word2idx[word1]] += 1
            for word2 in set(words):
                if word1 != word2:
                    word_word_matrix[word2idx[word1]][word2idx[word2]] += 1

    fout = open(opt.knowledge_base+ '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_entity', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        entity, attribute = lines[i + 1].lower().strip().split()
        steps_list = []
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        for token in document:
            if token.text == entity:
                step = 0
            else:
                step = find_step(token.text, entity_seed_words[entity])
            steps_list.append(step)
            if step != -1:
                weight = 1 / (step + 1)
                for token2 in document:
                    if token2 != token:
                        if token in token2.children or token2 in token.children:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += (pmi + 1) * weight
                                graph[token2.i][token.i] += (pmi + 1) * weight
                            else:
                                graph[token.i][token2.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                                graph[token2.i][token.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                        else:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += pmi * weight
                                graph[token2.i][token.i] += pmi * weight
                            else:
                                graph[token.i][token2.i] += 1 / abs(token.i - token2.i) * weight
                                graph[token2.i][token.i] += 1 / abs(token.i - token2.i) * weight
                    else:
                        graph[token.i][token.i] += 1 * weight
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

    file_attribute_name = opt.knowledge_base + '/seed_words_list_attribute_rest.csv'
    attribute_seed_words = {}
    with open(file_attribute_name, 'r') as f:
        reader = csv.reader(f)
        attributes = list(reader)
        for attribute in attributes[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(attribute[s + 1]))
            attribute_seed_words[attribute[0].lower()] = seed_words
    f.close()

    fout = open(opt.knowledge_base + '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_attribute', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        entity, attribute = lines[i + 1].lower().strip().split()
        steps_list = []
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        for token in document:
            if token.text == attribute:
                step = 0
            else:
                step = find_step(token.text, attribute_seed_words[attribute])
            steps_list.append(step)
            if step != -1:
                weight = 1 / (step + 1)
                for token2 in document:
                    if token2 != token:
                        if token in token2.children or token2 in token.children:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += (pmi + 1) * weight
                                graph[token2.i][token.i] += (pmi + 1) * weight
                            else:
                                graph[token.i][token2.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                                graph[token2.i][token.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                        else:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += pmi * weight
                                graph[token2.i][token.i] += pmi * weight
                            else:
                                graph[token.i][token2.i] += 1 / abs(token.i - token2.i) * weight
                                graph[token2.i][token.i] += 1 / abs(token.i - token2.i) * weight
                    else:
                        graph[token.i][token.i] += 1 * weight
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

def generate_graph_14(lines, newlines, opt, type):
    nlp = spacy.load('en_core_web_sm')
    csv.field_size_limit(500 * 1024 * 1024)
    if opt.knowledge_base == 'senticnet':
        steps = 5
    else:
        steps = 3

    file_entity_name = opt.knowledge_base +'/seed_words_list_entity_rest.csv'
    entity_seed_words = {}
    with open(file_entity_name, 'r') as f:
        reader = csv.reader(f)
        entities = list(reader)
        for entity in entities[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(entity[s + 1]))
            entity_seed_words[entity[0].lower()] = seed_words
    f.close()
    word2idx = {}
    idx = 0
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word in set(words):
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    word_word_matrix = np.zeros((len(word2idx), len(word2idx)), dtype=int)
    word_sentence_matrix = np.zeros((int(len(newlines) / 3), len(word2idx)), dtype=int)
    for i in range(0, len(newlines), 3):
        text = newlines[i].lower().strip()
        words = text.split()
        for word1 in set(words):
            word_sentence_matrix[int(i / 3)][word2idx[word1]] += 1
            for word2 in set(words):
                if word1 != word2:
                    word_word_matrix[word2idx[word1]][word2idx[word2]] += 1

    fout = open(opt.knowledge_base+ '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_entity', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        entity = lines[i + 1].lower().strip()
        steps_list = []
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        for token in document:
            if token.text == entity:
                step = 0
            else:
                step = find_step(token.text, entity_seed_words[entity])
            steps_list.append(step)
            if step != -1:
                weight = 1 / (step + 1)
                for token2 in document:
                    if token2 != token:
                        if token in token2.children or token2 in token.children:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += (pmi + 1) * weight
                                graph[token2.i][token.i] += (pmi + 1) * weight
                            else:
                                graph[token.i][token2.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                                graph[token2.i][token.i] += (1 / (abs(token.i - token2.i) + 1) + 1) * weight
                        else:
                            pmi = calculate_pmi(token.text, token2.text, word2idx, word_word_matrix,
                                                word_sentence_matrix)
                            if pmi > 0:
                                graph[token.i][token2.i] += pmi * weight
                                graph[token2.i][token.i] += pmi * weight
                            else:
                                graph[token.i][token2.i] += 1 / abs(token.i - token2.i) * weight
                                graph[token2.i][token.i] += 1 / abs(token.i - token2.i) * weight
                    else:
                        graph[token.i][token.i] += 1 * weight
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

    file_attribute_name = opt.knowledge_base + '/seed_words_list_attribute_rest.csv'
    attribute_seed_words = {}
    with open(file_attribute_name, 'r') as f:
        reader = csv.reader(f)
        attributes = list(reader)
        for attribute in attributes[1:]:
            seed_words = []
            for s in range(steps):
                seed_words.append(eval(attribute[s + 1]))
            attribute_seed_words[attribute[0].lower()] = seed_words
    f.close()

    fout = open(opt.knowledge_base + '/' + opt.dataset + '_' + type + '.raw.tokenized.graph_attribute', 'wb')
    graph_idx = 0
    idx2graph = {}
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        document = nlp(text)
        seq_len = len(document)
        graph = np.zeros((seq_len, seq_len)).astype('float32')
        idx2graph[graph_idx] = graph
        graph_idx += 1
    pickle.dump(idx2graph, fout)
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='15_rest', type=str,
                        help='14, 15_rest, 16_rest, 15_lap, 16_lap, mams')
    parser.add_argument('--knowledge_base', default='conceptnet', type=str, help='conceptnet, senticnet')
    opt = parser.parse_args()

    generate_graph(opt)