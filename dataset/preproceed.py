# -*- coding: utf-8 -*-

import argparse

from generate_dict import generate_dict
from generate_raw import generate_raw
from generate_seed_words import generate_seed_words
from generate_graph import generate_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='15_rest', type=str,
                        help='14, 15_rest, 16_rest, 15_lap, 16_lap')
    parser.add_argument('--knowledge_base', default='senticnet', type=str, help='conceptnet, senticnet')
    opt = parser.parse_args()
    generate_dict(opt)
    generate_raw(opt)
    generate_seed_words(opt)
    generate_graph(opt)