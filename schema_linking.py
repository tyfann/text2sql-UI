#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import traceback
import logging
import json
from collections import defaultdict
import re

from text2sql.dataproc.dusql_dataset_v2 import load_tables

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(asctime)s %(filename)s'
           ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
    datefmt='%m-%d %H:%M:%S',
    filename=None,
    filemode='a')

g_date_patt = re.compile(
    r'(([0-9]{2})[0-9]{2}年)?[0-9]{1,2}月[0-9]{2}日|([0-9]{2})[0-9]{2}年[0-9]{1,2}月')


def get_char_list(sentence):
    def is_ascii(s):
        """check if s is English album or number
        Args:
            s (str): NULL
        Returns: bool
        """
        return ord(s) < 128

    if len(sentence) == 0:
        return []

    lst_result = [sentence[0]]
    last_is_ascii = lst_result[-1].isalnum()
    for char in sentence[1:]:
        if char == ' ':
            last_is_ascii = False
            continue
        elif char == '-':
            last_is_ascii = False
            lst_result.append(char)
            continue

        if is_ascii(char) and last_is_ascii:
            lst_result[-1] += char
            continue

        if is_ascii(char):
            last_is_ascii = True
        else:
            last_is_ascii = False

        lst_result.append(char)

    return tuple(lst_result)


def _format_date_cell(old_cell):
    new_cell = old_cell.rstrip('月日')
    new_cell = new_cell.replace('年', '-')
    new_cell = new_cell.replace('月', '-')
    return new_cell


def _build(cells):
    dct_index = defaultdict(set)
    for cell in set(cells):
        if type(cell) is not str:
            continue
        cell = cell.strip()
        if re.match(g_date_patt, cell):
            cell = _format_date_cell(cell)
        cell_chars = get_char_list(cell.lower())
        dct_index[cell.lower()].add((cell, len(cell_chars)))
        for pos in range(len(cell_chars) - 1):
            bigram = cell_chars[pos:pos + 2]
            ####tri_gram = cell_chars[pos: pos + 3]
            ####four_gram = cell_chars[pos: pos + 4]
            dct_index[bigram].add((cell, len(cell_chars) - 1))
            ####dct_index[tri_gram].add((cell, len(cell_chars) - 2))
            ####dct_index[four_gram].add(cell)
    return dct_index


def build_cell_index(db_dict):
    for db in db_dict.values():
        column_cells = []
        for column in db.columns:
            cell_index = _build(column.cells)
            column_cells.append(cell_index)
        db.column_cells_index = column_cells


def search_values(query, db):
    lst_match_values = []
    for column, cell_index in zip(db.columns, db.column_cells_index):
        if column.id == 0:
            lst_match_values.append([])
            continue
        col_id = column.id

        candi_cnt = defaultdict(float)
        query_chars = get_char_list(query.lower())
        appear_set = set()
        for pos in range(len(query_chars)):
            unigram = query_chars[pos]
            if len(
                    unigram
            ) > 2 and unigram not in appear_set and unigram in cell_index:
                for cell, base in cell_index[unigram]:
                    candi_cnt[cell] += 1.0 / base
            if pos == len(query_chars) - 1:
                break

            bigram = query_chars[pos:pos + 2]
            if bigram not in cell_index:
                continue
            if bigram in appear_set:
                continue
            appear_set.add(bigram)
            for cell, base in cell_index[bigram]:
                candi_cnt[cell] += 1.0 / base

        lst_match_values.append(
            list(sorted(
                candi_cnt.items(), key=lambda x: x[1], reverse=True))[:10])

    return lst_match_values


if __name__ == "__main__":
    try:
        db_schema = 'data/db_schema.json'
        db_content = 'data/db_content.json'
        dct_db, _ = load_tables(db_schema, db_content)
        build_cell_index(dct_db)

        lst_output = []

        question_id = f'qid{1:06d}'
        question = '查一下电网中统计类型为调度口径抽蓄的上报电量值的最大值是多少'
        db_id = 'AI_SEARCH_1'
        db = dct_db[db_id]

        match_values = search_values(question, db)
        lst_output.append({
            "question_id": question_id,
            "question": question,
            "db_id": db_id,
            "match_values": match_values
        })

        with open('data/match_values_valid.json', 'w') as f:
            f.write(json.dumps(lst_output, indent=2, ensure_ascii=False))

        # json.dump(lst_output, args.output, indent=2, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc()
        # logging.critical(traceback.format_exc())
        exit(-1)
