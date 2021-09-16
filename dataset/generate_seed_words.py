# -*- coding: utf-8 -*-

import os
import csv
import copy
import argparse



def generate_seed_words(opt):
    if 'lap' in opt.dataset:
        if os.path.exists(opt.knowledge_base + '/seed_words_list_entity_lap.csv') and \
           os.path.exists(opt.knowledge_base + '/seed_words_list_attribute_lap.csv'):
            print('Seed words list already exits.')
            return
        elif opt.knowledge_base == 'senticnet':
            etrans = {'laptop': 'laptop', 'battery': 'battery', 'graphics' : 'graphic', 'cpu' : 'cpu',
                      'hard_disc' : 'disc', 'os' : 'system', 'support' : 'support', 'company' : 'company',
                      'display' : 'display', 'mouse' : 'mouse', 'software' : 'software', 'keyboard' : 'keyboard',
                      'optical_drives' : 'drive', 'warranty' : 'warrant', 'multimedia_devices' : 'multimedia',
                      'ports' : 'port', 'power_supply' : 'power', 'hardware' : 'hardware', 'shipping' : 'shipping',
                      'memory' : 'memory', 'motherboard' : 'motherboard', 'fans_cooling' : 'fan'}

            atrans = {'general': 'general', 'operation_performance': 'performance', 'design_features': 'design',
                      'usability': 'usability', 'portability': 'portability', 'price': 'price', 'quality': 'quality',
                      'miscellaneous': 'miscellany', 'connectivity': 'connectivity'}
            depth = 5
            with open('senticnet/dict.csv', 'r') as f:
                reader = csv.reader(f)
                sentic = list(reader)
            f.close()
            process_trans_sentic(etrans, depth, sentic, 'lap', 'entity')
            process_trans_sentic(atrans, depth, sentic, 'lap', 'attribute')

        else:
            etrans = {'laptop': 'laptop', 'battery': 'battery', 'graphics': 'graphics', 'cpu': 'cpu',
                      'hard_disc': 'hard_disc', 'os': 'operating_system', 'support': 'support', 'company': 'company',
                      'display': 'display', 'mouse': 'mouse', 'software': 'software', 'keyboard': 'keyboard',
                      'optical_drives': 'optical_drives', 'warranty': 'warranty', 'multimedia_devices': 'multimedia',
                      'ports': 'port', 'power_supply': 'power_supply', 'hardware': 'hardware', 'shipping': 'shipping',
                      'memory': 'memory', 'motherboard': 'motherboard', 'fans_cooling': 'fan'}

            atrans = {'general': 'general', 'operation_performance': 'performance', 'design_features': 'design',
                      'usability': 'usability', 'portability': 'portability', 'price': 'price', 'quality': 'quality',
                      'miscellaneous': 'miscellaneous', 'connectivity': 'connectivity'}

            depth = 3
            with open('conceptnet/dict.csv', 'r') as f:
                reader = csv.reader(f)
                concept = list(reader)
            f.close()
            process_trans_concept(etrans, depth, concept, 'lap', 'entity')
            process_trans_concept(atrans, depth, concept, 'lap', 'attribute')

    else:
        if os.path.exists(opt.knowledge_base + '/seed_words_list_entity_rest.csv') and \
           os.path.exists(opt.knowledge_base + '/seed_words_list_attribute_rest.csv'):
            print('Seed words list already exits.')
            return
        elif opt.knowledge_base == 'senticnet':
            etrans = {'ambience': 'ambience', 'food': 'food', 'service': 'service', 'restaurant': 'restaurant',
                      'location': 'location', 'drinks': 'drink', 'miscellaneous': 'miscellany', 'prices': 'price',
                      'staff': 'staff', 'menu': 'menu', 'place': 'place', 'anecdotes': 'anecdote', 'price': 'price',
                      'anecdotes/miscellaneous' : 'miscellany'}

            atrans = {'quality': 'quality', 'general': 'general', 'style_options': 'style',
                      'miscellaneous': 'miscellany', 'prices': 'price'}

            depth = 5
            with open('senticnet/dict.csv', 'r') as f:
                reader = csv.reader(f)
                sentic = list(reader)
            f.close()
            process_trans_sentic(etrans, depth, sentic, 'rest', 'entity')
            process_trans_sentic(atrans, depth, sentic, 'rest', 'attribute')

        else:
            etrans = {'ambience': 'ambience', 'food': 'food', 'service': 'service', 'restaurant': 'restaurant',
                      'location': 'location', 'drinks': 'drinks', 'miscellaneous': 'miscellaneous', 'prices': 'prices',
                      'staff': 'staff', 'menu': 'menu', 'place': 'place', 'anecdotes': 'anecdote', 'price': 'price',
                      'anecdotes/miscellaneous': 'miscellaneous'}

            atrans = {'quality': 'quality', 'general': 'general', 'style_options': 'style',
                      'miscellaneous': 'miscellaneous', 'prices': 'prices'}

            depth = 3
            with open('conceptnet/dict.csv', 'r') as f:
                reader = csv.reader(f)
                concept = list(reader)
            f.close()
            process_trans_concept(etrans, depth, concept, 'rest', 'entity')
            process_trans_concept(atrans, depth, concept, 'rest', 'attribute')
    print('Seed words list has been successfully generated.')

def process_trans_sentic(trans, depth, dict, dataset, category):
    words = []
    for s in dict:
        words.append(s[0])
    words_each_dataset = []
    for ea, target in trans.items():
        target = [target]
        chosen = copy.deepcopy(target)
        words_each_entity = []
        words_each_entity.append(target)
        for d in range(depth):
            words_each_depth = []
            for t in words_each_entity[d]:
                for index in range(1, len(dict)):
                    concept = words[index]
                    candidates = dict[index][-5:]
                    if concept == t:
                        for c in candidates:
                            if '_' not in c:
                                if c not in chosen:
                                    words_each_depth.append(c)
                                    chosen.append(c)
                            else:
                                candidatelist = c.strip().split("_")
                                for ca in candidatelist:
                                    if ca not in chosen:
                                        words_each_depth.append(ca)
                                        chosen.append(ca)
            words_each_entity.append(words_each_depth)
        words_each_dataset.append([ea] + words_each_entity[1:])
    fa = open('senticnet/seed_words_list_{0}_{1}.csv'.format(category, dataset), 'w', encoding='utf-8', newline='')
    writer = csv.writer(fa)
    title = [category]
    for d in range(depth):
        title.append('distance_' + str(d + 1))
    writer.writerow(title)
    for row in words_each_dataset:
        writer.writerow(row)
    fa.close()

def process_trans_concept(trans, depth, dict, dataset, category):
    words = []
    for s in dict:
        words.append(s[0])
    words_each_dataset = []
    for ea, target in trans.items():
        target = [target]
        chosen = copy.deepcopy(target)
        words_each_entity = []
        words_each_entity.append(target)
        for d in range(depth):
            words_each_depth = []
            for t in words_each_entity[d]:
                for c in dict[1:]:
                    if c[0] == t:
                        related_word = c[1]
                        if related_word not in chosen:
                            words_each_depth.append(related_word)
                            chosen.append(related_word)
            words_each_entity.append(words_each_depth)
        words_each_dataset.append([ea] + words_each_entity[1:])
    fa = open('conceptnet/seed_words_list_{0}_{1}.csv'.format(category, dataset), 'w', encoding='utf-8', newline='')
    writer = csv.writer(fa)
    title = [category]
    for d in range(depth):
        title.append('distance_' + str(d + 1))
    writer.writerow(title)
    for row in words_each_dataset:
        writer.writerow(row)
    fa.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='15_rest', type=str,
                        help='14, 15_rest, 16_rest, 15_lap, 16_lap, mams')
    parser.add_argument('--knowledge_base', default='conceptnet', type=str, help='conceptnet, senticnet')
    opt = parser.parse_args()

    generate_seed_words(opt)