# -*- coding: utf-8 -*-

import os
import csv
import re
import argparse

def generate_dict(opt):
    # generate dictionary for seed words (conceptnet)
    if os.path.exists(opt.knowledge_base + '/dict.csv'):
        print('The dictionary already exits.')
        return
    else:
        banlist = ['Antonym', 'DistinctFrom', 'NotCapableOf', 'NotDesires', 'NotHasProperty'] # delete invalid relation
        filename = 'path_of_your_conceptnetcsv' #https://www.conceptnet.io/
        fa = open('conceptnet/dict.csv', 'w', encoding='utf-8', newline='')
        writer = csv.writer(fa)
        title = ['start', 'end', 'realtion', 'weight']
        writer.writerow(title)
        with open(filename) as f:
            for line in f:
                parts = line.strip().split('\t')
                reg = re.compile('/a/\[/r/(?P<relation>.*)/,/c/en/(?P<start>.*),/c/en/(?P<end>.*).*')
                regmatch = reg.match(parts[0])
                if regmatch != None:
                    linebits = regmatch.groupdict()
                    weight = re.match(r'.*"weight":(?P<weight>.*)\}', parts[-1]).groupdict()
                    if linebits['relation'] not in banlist:
                        start = delete_attr(linebits['start'])
                        end = delete_attr(linebits['end'])
                        row = [start, end, linebits['relation'], weight['weight']]
                        writer.writerow(row)
    print('The dictionary has been successfully generated.')

def delete_attr(word):
    pos = word.find('/')
    if pos != -1:
        return word[:pos]
    return word


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='15_rest', type=str,
                        help='14, 15_rest, 16_rest, 15_lap, 16_lap, mams')
    parser.add_argument('--knowledge_base', default='senticnet', type=str, help='conceptnet, senticnet')
    opt = parser.parse_args()

    generate_dict(opt)
