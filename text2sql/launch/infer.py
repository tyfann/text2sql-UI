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
import tqdm

import numpy as np
import paddle
from paddle import nn

from text2sql.models import beam_search
from text2sql.models import sql_beam_search


def inference(model,
              data,
              beam_size=1,
              output_history=True):
    model.eval()
    with paddle.no_grad():
        for i, (inputs, labels) in enumerate(tqdm.tqdm(data())):
            decoded = _infer_one(model, inputs, beam_size, output_history)
            if len(decoded) == 0:
                pred_query = 'select *'
            else:
                pred_query = decoded[0]['pred_query']
        return pred_query


def _infer_one(model,
               inputs,
               beam_size,
               output_history=False):
    """inference one example
    """
    # TODO: from_cond should be true from non-bert model
    beams = sql_beam_search.beam_search_with_heuristics(
        model, inputs, beam_size=beam_size, max_steps=1000, from_cond=False)
    decoded = []
    for beam in beams:
        model_output, inferred_code = beam.inference_state.finalize()

        decoded.append({
            'pred_query': inferred_code,
            'model_output': model_output,
            'score': beam.score,
            **
            ({
                'choice_history': beam.choice_history,
                'score_history': beam.score_history,
            } if output_history else {})
        })
    return decoded


if __name__ == "__main__":
    """run some simple test cases"""
    pass
