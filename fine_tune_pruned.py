# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification, DapBertForSequenceClassificationSearch, DepolyableBert, RandomAlphaMaskModel
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

from arch_helper import store_arch_summary
csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

oncloud = True
try:
    import moxing as mox
except:
    oncloud = False


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None, test_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id
        self.test_id = test_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if not test:
                label = line[0]
                id = None
            else:
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched", test=True)
        
    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if not test:
                label = line[-1]
                id = None
            else:
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")
        
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched", test=True)


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if not test:
                text_a = line[3]
                label = line[1]
                id = None
            else:
                text_a = line[1]
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, id=id))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if not test:
                text_a = line[0]
                label = line[1]
                id = None
            else:
                text_a = line[1]
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, id=id))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if not test:
                label = line[-1]
                id = None
            else:
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                if not test:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
                    id = None
                else:
                    text_a = line[1]
                    text_b = line[2]
                    label = None
                    id = line[0]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if not test:
                label = line[-1]
                id = None
            else:
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", test=True)

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if not test:
                label = line[-1]
                id = None
            else:
                label = None
                id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, id=id))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, test=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if not test:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label_id = None

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        if test:
            test_id = int(example.id)
        else:
            test_id = None
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length,
                          test_id=test_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds), "acc": simple_accuracy(preds, labels)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features, test=False):
    if not test:
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if not test:
        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_label_ids, all_seq_lengths)
        return tensor_data, all_label_ids
    else:
        all_test_ids = torch.tensor([f.test_id for f in features], dtype=torch.long)
        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_seq_lengths, all_test_ids)
        return tensor_data, None

def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _,= model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    # logger.info("preds %s", list(preds))
    # logger.info("eval_labels %s", list(eval_labels.numpy()))

    result = compute_metrics(task_name, preds, eval_labels.numpy())
    # print(list(preds))
    # print(list(eval_labels.numpy()))
    # print(result['mcc'])
    # exit()
    result['eval_loss'] = eval_loss

    return result

def do_inference(model, task_name, test_dataloader, device, output_mode, num_labels, output_dir, labels):
    for batch_ in tqdm(test_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, seq_lengths, test_id = batch_
            start = time.time()
            logits, _, _,= model(input_ids, segment_ids, input_mask)
            end = time.time()
            with open("original_inference_time.txt", "a") as file:
                file.write("Time elasped for a batch: %.3f\n"%(end - start))
            return

def do_test(model, task_name, test_dataloader, device, output_mode, num_labels, output_dir, labels):
    preds = []
    ids = []
    for batch_ in tqdm(test_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, seq_lengths, test_id = batch_
            logits, _, _,= model(input_ids, segment_ids, input_mask)
        ids += list(test_id.detach().cpu().numpy())

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)


    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    with open(os.path.join(output_dir, "%s.tsv"%task_name), "w") as f:
        f.write("id\tlabel\n")
        if output_mode == "classification":
            for id, pred in zip(ids, preds):
                f.write("%d\t%s\n"%(id, labels[pred]))
        elif output_mode == "regression":
            for id, pred in zip(ids, preds):
                f.write("%d\t%.3f\n"%(id, pred))
    return 


def get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant):
##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
    # expected_flop = torch.mean( expected_flop )

    if flop_cur < flop_need - flop_tolerant:   # Too Small FLOP
        loss = - torch.log( expected_flop )
    #elif flop_cur > flop_need + flop_tolerant: # Too Large FLOP
    elif flop_cur > flop_need: # Too Large FLOP
        loss = torch.log( expected_flop )
    else: # Required FLOP
        loss = None
    if loss is None: return 0, 0
    else           : return loss, loss.item()

def get_model_flops(model, one_input):
    model.eval()
    input_ids, input_mask, segment_ids, label_ids, seq_lengths = one_input
    input_ids = torch.unsqueeze(input_ids, 0)
    input_mask = torch.unsqueeze(input_mask, 0)
    segment_ids = torch.unsqueeze(segment_ids, 0)
    label_ids = torch.unsqueeze(label_ids, 0)
    seq_lengths = torch.unsqueeze(seq_lengths, 0)
    # cache_inputs = torch.rand(*shape)
    if next(model.parameters()).is_cuda: 
        # cache_inputs = cache_inputs.cuda()
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        seq_lengths = seq_lengths.cuda()

    with torch.no_grad():
        _, _, _, flops = model('basic', input_ids, segment_ids, input_mask)
    return flops
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The student model dir.")
    parser.add_argument("--searched_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The searched model dir, for the mask.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_inference",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # added arguments
    parser.add_argument("--CE_loss",
                        action='store_true',
                        help="No distillation during fine-tuning")
    parser.add_argument("--arch_update_ratio",
                        default=1.0,
                        type=float,
                        help="Each backward can only update such ratio amount of alpha dimensions")
    parser.add_argument("--arch_learning_rate",
                        default=1,
                        type=float,
                        help="The learning rate for arch optimizer.")   
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)
    parser.add_argument('--int_distill',
                        action='store_true')
    parser.add_argument('--pred_distill',
                        action='store_true')
    parser.add_argument('--data_url',
                        type=str,
                        default="")
    parser.add_argument('--temperature',
                        type=float,
                        default=1.)
    
    # added arguments for search
    parser.add_argument('--gumbel_tau_max',
                        type=float,
                        default=5.,
                        help='The maximum tau for Gumbel.')
    parser.add_argument('--gumbel_tau_min',   
                        type=float,
                        default=0.1,
                        help='The minimum tau for Gumbel.')
    parser.add_argument("--search_embedding",
                        action='store_true',
                        help="Whether to search on the input embedding.")
    parser.add_argument("--search_qkv_hidden",
                        action='store_true',
                        help="Whether to search on the qkv hidden dimension.")
    parser.add_argument("--search_ff",
                        action='store_true',
                        help="Whether to search on the feadforward layer.")
    parser.add_argument("--search_heads",
                        action='store_true',
                        help="Whether to search on the multihead.")
    parser.add_argument("--search_sc",
                        action='store_true',
                        help="Whether to search on the skip connections.")
    parser.add_argument('--share_qkv_embedding_alpha',   
                        action='store_true',
                        help='Whether to share embedding alpha among qkv')
    parser.add_argument('--share_multihead_embedding_alpha',   
                        action='store_true',
                        help='Whether to share embedding alpha among all heads')
    parser.add_argument('--share_qkv_hidden_alpha',   
                        action='store_true',
                        help='Whether to share alpha among qkv')
    parser.add_argument('--share_multihead_hidden_alpha',   
                        action='store_true',
                        help='Whether to share alpha among all heads')
    parser.add_argument("--search_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for searching.")
    parser.add_argument("--keep_weights_from_search",
                        action='store_true',
                        help="Weights are transferred from searching model, 1111-1118 discussion")                
    parser.add_argument("--random_arch",
                        action='store_true',
                        help="Architecture is initialized randomly towards the given ratio")
    parser.add_argument("--random_arch_config",
                        type=str,
                        help="Path to config.json")                        
    parser.add_argument("--random_arch_ratio",
                        type=float,
                        help="FLOPS ratio used to initialize random architecture")
    parser.add_argument("--random_arch_dir",
                        type=str,
                        help="Path to store the randomized arch")
    parser.add_argument("--alpha_init_value",
                        type=int,
                        default=5,
                        help="Initial value of the alpha tensors")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mnli-mm": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    # intermediate distillation default parameters
    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst-2": {"num_train_epochs": 10, "max_seq_length": 64},
        "sts-b": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    acc_tasks = ["mnli", "mnli-mm", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]


    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 1:
        torch.cuda.empty_cache()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    torch.autograd.set_detect_anomaly(True)
    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if not args.pred_distill and not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if not args.do_eval and not args.do_test:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)
        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    if not args.do_test:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if not args.do_eval and not args.CE_loss:
        teacher_model = TinyBertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
        teacher_model.to(device)

    if args.random_arch:
        if 'random' in args.searched_model:
            searched_model = RandomAlphaMaskModel(device, args)
            if args.random_arch_dir != None:
                if not os.path.exists( args.random_arch_dir ):
                    os.mkdir( args.random_arch_dir )
                with torch.no_grad():
                    archs, flops = searched_model.get_archs()
                    output_arch_summary_file = os.path.join(args.random_arch_dir, "arch_summary.txt")
                    store_arch_summary(archs, output_arch_summary_file, searched_model.get_search_size(), None, None, total_flops=flops, num_attention_heads=searched_model.config.num_attention_heads)
        else:
            raise ValueError('args.searched_model = %s is not applicable when args.random_arch = %s'%(args.searched_model, args.random_arch))
    else:
        searched_model = DapBertForSequenceClassificationSearch.from_pretrained(args.searched_model, num_labels=num_labels, args=args)
    
    if args.keep_weights_from_search:
        student_model = searched_model
    else:
        if args.keep_weights_from_search:
            student_model = DepolyableBert(args.searched_model, num_labels, weights_dir=args.searched_model) # load arch. and weights from search model
        elif args.pred_distill or args.do_test or args.do_eval:
            student_model = DepolyableBert(args.student_model, num_labels) # load arch. and weights from student model
        else:
            student_model = DepolyableBert(args.searched_model, num_labels, weights_dir=args.student_model) # load arch. from search model, restore weights from student model
    
    if args.int_distill:
        multihead_indx = student_model.arch_indx["multihead"]

    student_model.to(device)
    if args.do_test:
        student_model.eval()

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer, output_mode, test=True)
        test_data, _ = get_tensor_data(output_mode, test_features, test=True)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        if args.do_inference:
            do_inference(student_model, args.task_name, test_dataloader, device, output_mode, num_labels, args.output_dir, processor.get_labels())
        else:
            do_test(student_model, args.task_name, test_dataloader, device, output_mode, num_labels, args.output_dir, processor.get_labels())
    elif args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        result_to_file(result, output_eval_file)
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        # if n_gpu > 1:
        #     student_model = torch.nn.DataParallel(student_model)
        #     teacher_model = torch.nn.DataParallel(teacher_model)
        # Prepare optimizer
        param_optimizer = list(student_model.named_parameters())
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not 'alpha' in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not 'alpha' in n], 'weight_decay': 0.0}
        ]
        schedule = 'warmup_linear'
        if not args.pred_distill:
            schedule = 'none'
        optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        
        # Prepare loss functions
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        best_dev_err = None
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            student_model.train()
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                
                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)
                    
                if not args.CE_loss:
                    with torch.no_grad():
                        teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)
                
                if args.CE_loss:
                    # cal training loss, cross entropy wrt labels
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        cls_loss = loss_fct(student_logits.view(-1, num_labels), label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    if global_step % 20 == 0:
                        print("target: %s"%('pred' if args.pred_distill else 'int'))
                        print("architecture cross entropy loss wrt labels: %f"%(cls_loss.item()))
                    

                    if args.gradient_accumulation_steps > 1:
                        cls_loss = cls_loss / args.gradient_accumulation_steps
                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                elif args.int_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att, layer in zip(student_atts, new_teacher_atts, range(student_layer_num)):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)
                        teacher_att = teacher_att[:, multihead_indx[layer],:,:] # select unpruned heads' attention maps from teacher
                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss = rep_loss + att_loss
                    tr_att_loss += att_loss.item()
                    tr_rep_loss += rep_loss.item()
                elif args.pred_distill:
                    if output_mode == "classification":
                        cls_loss = soft_cross_entropy(student_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                    loss = cls_loss
                    tr_cls_loss += cls_loss.item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1


                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step + 1) % args.eval_step == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    student_model.eval()

                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    result = {}
                    if args.pred_distill:
                        result = do_eval(student_model, task_name, eval_dataloader,
                                         device, output_mode, eval_labels, num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    result_to_file(result, output_eval_file)
                    student_model.train()
                    if args.pred_distill:
                        save_model = False
                        if task_name in acc_tasks and result['acc'] > best_dev_acc:
                            best_dev_acc = result['acc']
                            save_model = True

                        if task_name in corr_tasks and result['corr'] > best_dev_acc:
                            best_dev_acc = result['corr']
                            save_model = True

                        if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                            best_dev_acc = result['mcc']
                            save_model = True
                        if save_model:
                            logger.info("***** Save epoch %d model *****"%(epoch_))

                            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                            # if not args.pred_distill:
                            #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                            model_name = WEIGHTS_NAME
                            output_model_file = os.path.join(args.output_dir, model_name)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)



            # end of an epoch, see if need to save model
            if not args.pred_distill:
                save_model = False
                if best_dev_err == None or tr_loss < best_dev_err:
                    best_dev_err = tr_loss
                    save_model = True

            if save_model:
                logger.info("***** Save epoch %d model *****"%(epoch_))

                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                # if not args.pred_distill:
                #     model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                model_name = WEIGHTS_NAME
                output_model_file = os.path.join(args.output_dir, model_name)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

                # Test mnli-mm
                student_model.eval()
                if args.pred_distill and task_name == "mnli":
                    task_name = "mnli-mm"
                    processor = processors[task_name]()
                    if not os.path.exists(args.output_dir + '-MM'):
                        os.makedirs(args.output_dir + '-MM')

                    eval_examples = processor.get_dev_examples(args.data_dir)

                    eval_features = convert_examples_to_features(
                        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)

                    logger.info("***** Running mm evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                                    batch_size=args.eval_batch_size)

                    result = do_eval(student_model, task_name, eval_dataloader,
                                        device, output_mode, eval_labels, num_labels)

                    result['global_step'] = global_step

                    tmp_output_eval_file = os.path.join(args.output_dir + '-MM', "eval_results.txt")
                    result_to_file(result, tmp_output_eval_file)

                    task_name = 'mnli'
                student_model.train()

if __name__ == "__main__":
    main()
