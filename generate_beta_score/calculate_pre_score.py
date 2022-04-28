#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: calculate_score.py
# @Version: 
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# @For: 
# @Created Time: Mon Jan 11 20:10:55 2021
# ------------------

from scipy.stats import beta

def get_score(path, seed_num, hop):
    word_num = 2898
    #word_num = word_num / pow(10, len(str(word_num))-1)
    #seed_num = seed_num / pow(10, len(str(word_num))-1)
    frac_w = seed_num / word_num
    fp = open(path, 'r')
    word_score_dic = {}
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, all_count, succeed_count = line.split('\t')
        all_count = int(all_count)
        succeed_count = int(succeed_count)
        #all_count = all_count / pow(10, len(str(all_count))-1)
        #succeed_count = succeed_count / pow(10, len(str(all_count))-1)
        # print(all_count, succeed_count)
        if all_count == 0:
            frac = 1
        else:
            frac = (all_count-succeed_count) / (all_count)
        score = 1 - beta.cdf(frac, succeed_count+1, all_count-succeed_count+1)
        word_score_dic[word] = score
    fp.close()
    return word_score_dic


def save_score(path, score_dic):
    fp = open(path, 'w')
    score_dic = sorted(score_dic.items(), key=lambda a: -a[1])
    for word, score in score_dic:
        string = word + '\t' + str(score) + '\n'
        fp.write(string)
    fp.close()


def main():
    path1 = './hop_seed/service_1_link.txt'
    path2 = './hop_seed/service_2_link.txt'
    path3 = './hop_seed/service_3_link.txt'
    w_path1 = './hop_seed/service_seed_score1.txt'
    w_path2 = './hop_seed/service_seed_score2.txt'
    w_path3 = './hop_seed/service_seed_score3.txt'

    c1 = 12
    c2 = 294 + c1
    c3 = 1400 + c1 + c2

    word_score_dic1 = get_score(path1, c1, 1)
    word_score_dic2 = get_score(path2, c2, 2)
    word_score_dic3 = get_score(path3, c3, 3)

    save_score(w_path1, word_score_dic1)
    save_score(w_path2, word_score_dic2)
    save_score(w_path3, word_score_dic3)

if __name__ == '__main__':
    main()
