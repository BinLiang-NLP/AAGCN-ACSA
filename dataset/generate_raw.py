# -*- coding: utf-8 -*-

import os
from xml.etree.ElementTree import parse
import argparse


def generate_raw(opt):
    path = opt.knowledge_base + '/' + opt.dataset
    if os.path.exists(path + '_train.raw') and os.path.exists(path + '_test.raw'):
        print('The raw file already exits.')
        return
    else:
        process_xml(path, 'train')
        process_xml(path, 'test')
    print('The raw file has been successfully generated.')


def process_xml(path, type):
    path_xml = path + '_' + type + '.xml'
    path_raw = path + '_' + type + '.raw'
    file = open(path_raw, 'w')
    polarity_dict = {'negative': '-1', 'neutral': '0', 'positive': '1'}
    data = []
    if '15' in path or '16' in path:
        tree = parse(path_xml)
        reviews = tree.getroot()
        for review in reviews:
            for sentences in review:
                for sentence in sentences:
                    text = sentence.find('text')
                    if text is None:
                        continue
                    text = text.text.lower()
                    aspectCategories = sentence.find('Opinions')
                    if aspectCategories is None:
                        continue
                    for aspectCategory in aspectCategories:
                        piece = []
                        category = aspectCategory.get('category')
                        entity, attribute = category.lower().split('#')
                        polarity = aspectCategory.get('polarity')
                        if polarity == 'conflict':
                            continue
                        piece.append(text + '\n')
                        polarity = polarity_dict[polarity]
                        piece.append(entity + ' ' + attribute + '\n')
                        piece.append(polarity + '\n')
                        if piece not in data:
                            data.append(piece)

    else:
        tree = parse(path_xml)
        sentences = tree.getroot()
        for sentence in sentences:
            text = sentence.find('text')
            if text is None:
                continue
            text = text.text.lower()
            aspectCategories = sentence.find('Opinions')
            if aspectCategories is None:
                continue
            for aspectCategory in aspectCategories:
                piece = []
                category = aspectCategory.get('category')
                polarity = aspectCategory.get('polarity')
                if polarity == 'conflict':
                    continue
                piece.append(text + '\n')
                polarity = polarity_dict[polarity]
                piece.append(category + '\n')
                piece.append(polarity + '\n')
                if piece not in data:
                    data.append(piece)

    flag_conflict = [0 for i in range(len(data))]
    for num1 in range(len(data) - 1):
        for num2 in range(num1 + 1, len(data)):
            if data[num1][0] == data[num2][0] and data[num1][1] == data[num2][1] and data[num1][2] != data[num2][2]:
                flag_conflict[num1] = 1
                flag_conflict[num2] = 1

    for num in range(len(data)):
        if flag_conflict[num] == 0:
            file.writelines(data[num])
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='15_rest', type=str,
                        help='14, 15_rest, 16_rest, 15_lap, 16_lap, mams')
    parser.add_argument('--knowledge_base', default='conceptnet', type=str, help='conceptnet, senticnet')
    opt = parser.parse_args()

    generate_raw(opt)
