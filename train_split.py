# -*- ecoding: utf-8 -*-
# @ModuleName: train_split
# @Function: 
# @Author: Yufan-tyf
# @Time: 2022/3/2 8:05 PM
import copy
import json
import argparse
import sys
import os
import shutil

from sklearn.model_selection import train_test_split


def with_toks_split():
    parser = argparse.ArgumentParser()

    parser.add_argument("-C", "--configuration", default='./sel5', help="Configuration (*.json).")

    args = parser.parse_args()

    # train_file = '../data/CSgSQL/train.json'

    with open(os.path.join(args.configuration, 'train.json')) as f:
        train_data = json.load(f)

    x_train, x_test = train_test_split(train_data, test_size=0.20, shuffle=True, random_state=2)

    test_gold = []

    for index, x_dict in enumerate(x_test):
        i = index + 1

        gold_dict = dict()
        gold_dict['question_id'] = 'qid' + ("%06d" % i)
        gold_dict['sql'] = copy.copy(x_dict['sql'])
        gold_dict['query'] = copy.copy(x_dict['query'])
        test_gold.append(gold_dict)

        del x_dict['sql']
        del x_dict['query']
        x_dict['question_id'] = 'qid' + ("%06d" % i)
        del x_dict['query_toks']
        del x_dict['query_toks_no_value']
        del x_dict['question_toks']

    for index, x_dict in enumerate(x_train):
        i = index + 1
        x_dict['question_id'] = 'qid' + ("%06d" % i)

    os.remove(os.path.join(args.configuration, 'train.json'))

    with open(os.path.join(args.configuration, 'train.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(x_train, indent=4, separators=(',', ': '), ensure_ascii=False))

    with open(os.path.join(args.configuration, 'test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(x_test, indent=4, separators=(',', ': '), ensure_ascii=False))

    with open(os.path.join(args.configuration, 'test_gold.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_gold, indent=4, separators=(',', ': '), ensure_ascii=False))


def no_toks_split():
    parser = argparse.ArgumentParser()

    parser.add_argument("-C", "--configuration", default='./data/CSgSQL-div', help="Configuration (*.json).")

    args = parser.parse_args()

    # train_file = '../data/CSgSQL/train.json'

    with open(os.path.join(args.configuration, 'train.json')) as f:
        train_data = json.load(f)

    x_train, x_test = train_test_split(train_data, test_size=0.20, shuffle=True, random_state=2)


    x_test_gold = copy.deepcopy(x_test)

    for index, x_dict in enumerate(x_test):
        i = index + 1
        x_dict['sql'] = ""
        x_dict['query'] = ""
        x_dict['question_id'] = 'qid' + ("%06d" % i)

    for index, x_dict in enumerate(x_train):
        i = index + 1
        x_dict['question_id'] = 'qid' + ("%06d" % i)

    os.remove(os.path.join(args.configuration, 'train.json'))

    with open(os.path.join(args.configuration, 'train.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(x_train, indent=4, separators=(',', ': '), ensure_ascii=False))

    with open(os.path.join(args.configuration, 'test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(x_test, indent=4, separators=(',', ': '), ensure_ascii=False))

    with open(os.path.join(args.configuration, 'test_gold.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(x_test_gold, indent=4, separators=(',', ': '), ensure_ascii=False))


if __name__ == '__main__':
    no_toks_split()

