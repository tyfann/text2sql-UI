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
import logging
from pathlib import Path
from functools import partial
import random
import os

import numpy as np

import paddle

import text2sql
from text2sql import global_config
import pickle

from text2sql import dataproc
from text2sql import launch
from text2sql.grammars.dusql_v2 import DuSQLLanguageV2

ModelClass = None
GrammarClass = None
DataLoaderClass = None
DatasetClass = None
g_input_encoder = None
g_label_encoder = None


def preprocess(config):
    dataset_config = {
        'db_file': config.data.db,
        'input_encoder': g_input_encoder,
        'label_encoder': g_label_encoder,
        'is_cached': False,
    }

    save_dir = 'data/preproc'

    dataset = DatasetClass(name='valid', data_file=config.data.valid_set, **dataset_config)

    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / f'valid.pkl', 'wb') as ofs:
        pickle.dump([dataset._examples, dataset._qid2index], ofs)

    # dataset.save(save_dir, save_db=False)


def load_model(config):
    if config.model.init_model_params is None:
        raise RuntimeError(
            "config.init_model_params should be a valid model path")

    model = ModelClass(config.model, g_label_encoder)
    state_dict = paddle.load(config.model.init_model_params)
    model.set_state_dict(state_dict)
    logging.info("loading model successfully!")

    return model


def inference(config, model):
    # if config.model.init_model_params is None:
    #     raise RuntimeError(
    #         "config.init_model_params should be a valid model path")

    dataset_config = {
        'db_file': config.data.db,
        'input_encoder': g_input_encoder,
        'label_encoder': g_label_encoder,
        'is_cached': True
    }
    valid_set = DatasetClass(
        name='valid', data_file=config.data.valid_set, **dataset_config)
    test_reader = DataLoaderClass(config, valid_set, batch_size=1, shuffle=False)

    # model = ModelClass(config.model, g_label_encoder)
    # logging.info("loading model param from %s", config.model.init_model_params)
    # state_dict = paddle.load(config.model.init_model_params)
    # model.set_state_dict(state_dict)

    logging.info("start of inference...")
    pred_query = launch.infer.inference(
        model,
        test_reader,
        beam_size=config.general.beam_size)
    logging.info("end of inference...")
    return pred_query


def init_env(config):
    log_level = logging.INFO if not config.general.is_debug else logging.DEBUG
    formater = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)03d * %(message)s')
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logger.handlers[0]
    handler.setLevel(log_level)
    handler.setFormatter(formater)

    seed = config.train.random_seed
    if seed is not None:
        random.seed(seed)
        paddle.seed(seed)
        np.random.seed(seed)

    global ModelClass
    global GrammarClass
    global DatasetClass
    global DataLoaderClass
    global g_input_encoder
    global g_label_encoder

    if config.model.grammar_type == 'dusql_v2':
        GrammarClass = DuSQLLanguageV2
    else:
        raise ValueError('grammar type is not supported: %s' %
                         (config.model.grammar_type))
    g_label_encoder = dataproc.SQLPreproc(
        config.data.grammar,
        GrammarClass,
        predict_value=config.model.predict_value,
        is_cached=config.general.mode != 'preproc')

    assert config.model.model_name == 'seq2tree_v2', 'only seq2tree_v2 is supported'
    g_input_encoder = dataproc.ErnieInputEncoderV2(config.model)
    ModelClass = lambda x1, x2: text2sql.models.EncDecModel(x1, x2, 'v2')
    DatasetClass = dataproc.DuSQLDatasetV2
    DataLoaderClass = partial(
        dataproc.DataLoader,
        collate_fn=dataproc.dataloader.collate_batch_data_v2)


def _set_proc_name(config, tag_base):
    # set process name on local machine
    if config.general.is_cloud:
        return
    if tag_base.startswith('train'):
        tag_base = 'train'
    import setproctitle
    setproctitle.setproctitle(tag_base + '_' + config.data.output.rstrip('/')
                              .split('/')[-1])


if __name__ == "__main__":
    # config = global_config.gen_config(['--mode', 'infer', '--is-cached', 'false'])
    config = global_config.gen_config(
        ['--mode', 'infer', '--init-model-param', 'model/model.pdparams', '--valid-set', 'data/preproc/valid.pkl',
         '--data-root', 'data/preproc'])
    init_env(config)

    run_mode = config.general.mode
    if run_mode == 'preproc':
        preprocess(config)
        sys.exit(0)

    _set_proc_name(config, run_mode)
    if run_mode == 'infer':
        inference(config)
