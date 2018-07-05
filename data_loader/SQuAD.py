# -*- coding: utf-8 -*-
"""
Load SQuAD dataset.
"""
import json
import nltk
import random
import torch
import pickle
import numpy as np
from functools import partial
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .config import *
from util.list_utils import del_list_inplace


MAX_Q_LEN = 30  # maximum number of tokens in question
MAX_C_LEN = 400  # maximum number of tokens in context
MAX_C_S_NUM = 16  # maximum number of sentences in context
MAX_C_S_LEN = 100  # maximum number of tokens in context sentences
MAX_W_SIZE = 16  # maximum number of characters in a word


class Example(object):
    """
    Data sample.
    """

    def __init__(self, title, context,
                 question, question_id,
                 answer_start, answer_text):
        """
        Initialize a data sample
        :param title: title of the Wikipedia article
        :param context: a passage from the article
        :param question: a question text
        :param question_id: a unique question id
        :param answer_start: start character position in context
        :param answer_text: answer text
        """
        self.title = title
        self.context = context
        self.question = question
        self.question_id = question_id
        self.answer_start = answer_start
        self.answer_text = answer_text
        self.tokenized_context = None
        self.context_wids = None
        self.context_cids = None
        self.tokenized_question = None
        self.question_wids = None
        self.question_cids = None
        self.answer_tpos = None
        self.split_tokenized_context = None
        self.context_sent_wids = None
        self.context_sent_cids = None
        self.answer_spos = None
        self.answer_tpos_in_sent = None


def read_json(path, type,
              debug_mode=False, debug_len=10,
              delete_long_context=True,
              delete_long_question=True,
              longest_context=MAX_C_LEN,
              longest_question=MAX_Q_LEN):
    """
    Read SQuAD json file into a list of Example objects.
    :param path: data path
    :param type: "train" or other
    :param debug_mode: if debut_mode is True, only load a few samples
    :param debug_len: how many samples to load when debug
    :param delete_long_context: whether delete long context sample
    :param delete_long_question: whether delete long question sample
    :param longest_context: maximum context length to keep
    :param longest_question: maximum question length to keep
    :return: a list of Example instances
    """
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            if delete_long_context and \
                    len(nltk.word_tokenize(context)) > longest_context:
                continue
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                if delete_long_question and \
                        len(nltk.word_tokenize(question)) > longest_question:
                    continue
                question_id = qa["id"]

                if type == "train":
                    for ans in answers:
                        answer_start = int(ans["answer_start"])
                        answer_text = ans["text"]
                        e = Example(title, context, question, question_id,
                                    answer_start, answer_text)
                        examples.append(e)
                        if debug_mode and len(examples) >= debug_len:
                            return examples
                else:
                    answer_start_list = \
                        [ans["answer_start"] for ans in answers]
                    c = Counter(answer_start_list)
                    most_common_answer, freq = c.most_common()[0]
                    answer_text = None
                    answer_start = None
                    if freq > 1:
                        for i, ans_start in enumerate(answer_start_list):
                            if ans_start == most_common_answer:
                                answer_text = answers[i]["text"]
                                answer_start = answers[i]["answer_start"]
                                break
                    else:
                        answer_text = answers[random.choice(
                            range(len(answers)))]["text"]
                        answer_start = answers[random.choice(
                            range(len(answers)))]["answer_start"]
                    e = Example(title, context, question,
                                question_id, answer_start, answer_text)
                    examples.append(e)
                    if debug_mode and len(examples) >= debug_len:
                        return examples
    print("#examples :%s" % len(examples))
    return examples


def tokenize_context_with_answer(context, answer_text, answer_start):
    """
    Tokenize context and locate the answer token-level position
    after tokenized the context,
    as the original location is based on char-level
    :param context: context
    :param answer_text: answer text
    :param answer_start: answer start position (char level)
    :return: tokenized context,
        answer start token index,
        answer end token index (inclusive)
    """
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]
    tokenized_fore = nltk.word_tokenize(fore)
    tokenized_mid = nltk.word_tokenize(mid)
    tokenized_after = nltk.word_tokenize(after)
    tokenized_text = nltk.word_tokenize(answer_text)
    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None
    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token = len(tokenized_fore)
    answer_end_token = len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token


def tokenize_context(context, answer_text, answer_start):
    """
    Sentence tokenize and word tokenize context and locate
    the answer sentence level, global token level, and local
    token level position, after tokenized the context.
    :param context: context
    :param answer_text: answer text
    :param answer_start: answer start position (char level)
    :return: tokenized_context, 1D list, context tokens
        answer_tpos_start, int, answer start token position
        answer_tpos_end, int, answer end token position
        split_tokenized_context, 2D list, tokens of sentences in context
        answer_spos, int, answer in which sentence of context
        answer_tpos_start_in_sent, int, answer start token position in sentence
        answer_tpos_end_in_sent, int, answer end token position in sentence
    """
    # tokenize context without sentence split
    tokenized = tokenize_context_with_answer(
        context, answer_text, answer_start)
    if tokenized is None:
        return None
    else:
        tokenized_context, answer_tpos_start, answer_tpos_end =\
            tokenized

    answer_end = answer_start + len(answer_text) - 1

    # split sentence for context
    sents = nltk.sent_tokenize(context)
    num_sent = len(sents)
    sent_lengths = [len(s) for s in sents]
    sent_end_cpos = np.cumsum(sent_lengths)
    sent_end_cpos = (sent_end_cpos - np.ones(num_sent) +
                     np.array(list(range(num_sent))))

    # answer in which sentence
    answer_spos = []
    for i in range(num_sent):
        if answer_text in sents[i]:
            answer_spos.append(i)
    if len(answer_spos) == 0:
        print("answer span multi sentence")
        return None
    elif len(answer_spos) == 1:
        answer_spos = answer_spos[0]
    else:
        answer_spos = answer_spos[-1]
        for i in range(num_sent):
            if answer_end <= sent_end_cpos[i]:
                answer_spos = i
                break
    if answer_spos >= MAX_C_S_NUM:
        print("answer sentence position too long")
        return None

    # answer start char position in the sentence
    new_answer_start = answer_start
    if answer_spos > 0:
        new_answer_start = (answer_start -
                            sent_end_cpos[answer_spos - 1])
    answer_start_in_sent = [i for i in range(len(sents[answer_spos]))
                            if sents[answer_spos].startswith(answer_text, i)]
    if len(answer_start_in_sent) == 0:
        print("cannot find answer in split sentence")
        return None
    elif len(answer_start_in_sent) == 1:
        new_answer_start = answer_start_in_sent[0]
    else:
        new_answer_start = int(min(answer_start_in_sent,
                                   key=lambda x: abs(x - new_answer_start)))

    # tokenize all context sentences
    split_tokenized_context = []
    for i in range(len(sents)):
        if i == answer_spos:
            answer_sent_tokenize = tokenize_context_with_answer(
                sents[answer_spos], answer_text, new_answer_start)
            if answer_sent_tokenize is None:
                print("cannot find answer after tokenize answer sentence")
                return None
            else:
                (tokenized_answer_sent,
                 answer_tpos_start_in_sent,
                 answer_tpos_end_in_sent) = answer_sent_tokenize
                if len(tokenized_answer_sent) > MAX_C_S_LEN:
                    print("answer sentence too long")
                    return None
                split_tokenized_context.append(tokenized_answer_sent)
        else:
            sent_tokens = nltk.word_tokenize(sents[i])
            split_tokenized_context.append(sent_tokens)

    return (tokenized_context,
            int(answer_tpos_start), int(answer_tpos_end),
            split_tokenized_context,
            int(answer_spos),
            int(answer_tpos_start_in_sent), int(answer_tpos_end_in_sent))


def padding2d(seqs, pad_id, pad_size):
    padded_seqs = np.ones([len(seqs), pad_size]) * pad_id
    for i in range(len(seqs)):
        for j in range(min(len(seqs[i]), pad_size)):
            padded_seqs[i, j] = seqs[i][j]
    return padded_seqs


def padding3d(seqs, pad_id, pad_size1, pad_size2):
    padded_seqs = np.ones([len(seqs), pad_size1, pad_size2]) * pad_id
    for i in range(len(seqs)):
        for j in range(min(len(seqs[i]), pad_size1)):
            for k in range(min(len(seqs[i][j]), pad_size2)):
                padded_seqs[i, j, k] = seqs[i][j][k]
    return padded_seqs


def padding4d(seqs, pad_id, pad_size1, pad_size2, pad_size3):
    padded_seqs = np.ones([len(seqs), pad_size1, pad_size2, pad_size3]) *\
        pad_id
    for i in range(len(seqs)):
        for j in range(min(len(seqs[i]), pad_size1)):
            for k in range(min(len(seqs[i][j]), pad_size2)):
                for l in range(min(len(seqs[i][j][k]), pad_size3)):
                    padded_seqs[i, j, k, l] = seqs[i][j][k][l]
    return padded_seqs


class SQuAD(Dataset):
    def __init__(self, path, itow, wtoi, itoc, ctoi,
                 split="train", debug_mode=False, debug_len=10):
        self.insert_start = wtoi.get("<SOS>", None)
        self.insert_end = wtoi.get("<EOS>", None)
        self.PAD_wid = wtoi.get("<PAD>", None)
        self.PAD_cid = ctoi.get("<PAD>", None)
        self.wtoi = wtoi
        self.ctoi = ctoi
        self.itow = itow
        self.itoc = itoc
        self.split = split

        # load json
        self.examples = read_json(
            path, self.split, debug_mode=debug_mode, debug_len=debug_len)

        # tokenize context and question
        idxs_to_del = []
        idx = -1

        for e in self.examples:
            idx = idx + 1

            tokenized = tokenize_context(
                e.context, e.answer_text, e.answer_start)
            if tokenized is None:
                idxs_to_del.append(idx)
                continue
            (tokenized_context,
             answer_tpos_start, answer_tpos_end,
             split_tokenized_context,
             answer_spos,
             answer_tpos_start_in_sent, answer_tpos_end_in_sent) = tokenized
            e.tokenized_context = tokenized_context
            e.answer_tpos = (answer_tpos_start, answer_tpos_end)
            e.split_tokenized_context = split_tokenized_context
            e.answer_spos = answer_spos
            e.answer_tpos_in_sent = (answer_tpos_start_in_sent,
                                     answer_tpos_end_in_sent)

            e.tokenized_question = nltk.word_tokenize(e.question)

        del_list_inplace(self.examples, idxs_to_del)

        # numerical context and question
        for e in self.examples:
            e.question_wids = self._tokens2wids(
                e.tokenized_question, self.wtoi)
            e.question_cids = self._tokens2cids(
                e.tokenized_question, self.ctoi)
            e.context_wids = self._tokens2wids(
                e.tokenized_context, self.wtoi)
            e.context_cids = self._tokens2cids(
                e.tokenized_context, self.ctoi)
            e.context_sent_wids = []
            e.context_sent_cids = []
            for tokenized_sent in e.split_tokenized_context:
                e.context_sent_wids.append(
                    self._tokens2wids(tokenized_sent, self.wtoi))
                e.context_sent_cids.append(
                    self._tokens2cids(tokenized_sent, self.ctoi))

    def __getitem__(self, idx):
        e = self.examples[idx]
        return (e.question_wids, e.question_cids,
                e.context_wids, e.context_cids,
                e.context_sent_wids, e.context_sent_cids,
                e.answer_spos, e.answer_tpos, e.answer_tpos_in_sent)

    def __len__(self):
        return len(self.examples)

    def _tokens2cids(self, seq, ctoi):
        """
        Transform a list of tokens to a 2d list of char vector indexes.
        :param seq: tokenized word list
        :param ctoi: dictionary to map characters to char vector indexes
        :return: a 2d list, each item is the char indexes of each token
        """
        result = []
        for word in seq:
            # transform characters to char indexes
            result.append(self._tokens2wids(word, ctoi))
        return result

    def _tokens2wids(self, seq, wtoi, insert_sos=False, insert_eos=False):
        """
        Transform a list of tokens to a 1d list of word vector indexes.
        :param seq: tokenized word list
        :param wtoi: dictionary to map words to word vector indexes
        :param insert_sos: whether insert "<SOS>" at the beginning
        :param insert_eos: whether insert "<EOS>" ate the end
        :return: a 1d list, each item is the word vector index of each token
        """
        result = []
        if self.insert_start is not None and insert_sos:
            result.append(self.insert_start)
        for word in seq:
            result.append(wtoi.get(word, wtoi["<UNK>"]))
        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result

    def _create_collate_fn(self, batch_first=True):
        """
        Define how to pack a batch of samples.
        This functions defines what you will get for a batch of data.
        :param batch_first: whether the returned tensor is batch_first
        :return: each batch data will get the following
            batch_question_lengths, 1D LongTensor,
                number of tokens of each question
            batch_question_wids_padded, LongTensor,
                (batch_size, MAX_Q_LEN)
            batch_question_cids_padded, LongTensor,
                (batch_size, MAX_Q_LEN, MAX_W_SIZE)
            batch_context_lengths, 1D list,
                number of tokens of each context
            batch_context_wids_padded, LongTensor,
                (batch_size, MAX_C_LEN)
            batch_context_cids_padded, LongTensor,
                (batch_size, MAX_C_LEN, MAX_W_SIZE)
            batch_context_sent_lengths, 2D LongTensor,
                number of tokens of each sentence in each context
            batch_context_sent_wids_padded, LongTensor,
                (batch_size, MAX_C_S_NUM, MAX_C_S_LEN)
            batch_context_sent_cids_padded, LongTensor,
                (batch_size, MAX_C_S_NUM, MAX_C_S_LEN, MAX_W_SIZE)
            batch_answer_spos, int,
                sentence level position of answer in context
            batch_answer_tpos, tuple like (start, end)
                token level position of answer in context
            batch_answer_tpos_in_sent, tuple like (start, end)
                token level position of answer in answer sentence of context
        """
        def collate(examples, this):
            # same sequence with __getitem__
            (batch_question_wids, batch_question_cids,
             batch_context_wids, batch_context_cids,
             batch_context_sent_wids, batch_context_sent_cids,
             batch_answer_spos, batch_answer_tpos,
             batch_answer_tpos_in_sent) = zip(*examples)

            batch_question_lengths = torch.LongTensor(
                [len(s) for s in batch_question_wids])
            batch_question_wids_padded = torch.LongTensor(padding2d(
                batch_question_wids, this.PAD_wid,
                MAX_Q_LEN).astype(int))
            batch_question_cids_padded = torch.LongTensor(padding3d(
                batch_question_cids, this.PAD_cid,
                MAX_Q_LEN, MAX_W_SIZE).astype(int))
            batch_context_lengths = torch.LongTensor(
                [len(s) for s in batch_context_wids])
            batch_context_wids_padded = torch.LongTensor(padding2d(
                batch_context_wids, this.PAD_wid,
                MAX_C_LEN).astype(int))
            batch_context_cids_padded = torch.LongTensor(padding3d(
                batch_context_cids, this.PAD_cid,
                MAX_C_LEN, MAX_W_SIZE).astype(int))
            batch_context_sent_lengths = torch.LongTensor(padding2d(
                [[len(s) for s in context_sent]
                 for context_sent
                 in batch_context_sent_wids], 0, MAX_C_S_NUM))
            batch_context_sent_wids_padded = torch.LongTensor(padding3d(
                batch_context_sent_wids, this.PAD_wid,
                MAX_C_S_NUM, MAX_C_S_LEN).astype(int))
            batch_context_sent_cids_padded = torch.LongTensor(padding4d(
                batch_context_sent_cids, this.PAD_cid,
                MAX_C_S_NUM, MAX_C_S_LEN, MAX_W_SIZE).astype(int))

            batch_answer_spos = torch.LongTensor(batch_answer_spos)
            batch_answer_tpos = torch.LongTensor(batch_answer_tpos)
            batch_answer_tpos_in_sent = torch.LongTensor(
                batch_answer_tpos_in_sent)

            return (batch_question_lengths,
                    batch_question_wids_padded,
                    batch_question_cids_padded,
                    batch_context_lengths,
                    batch_context_wids_padded,
                    batch_context_cids_padded,
                    batch_context_sent_lengths,
                    batch_context_sent_wids_padded,
                    batch_context_sent_cids_padded,
                    batch_answer_spos,
                    batch_answer_tpos,
                    batch_answer_tpos_in_sent)

        return partial(collate, this=self)

    def get_dataloader(self, batch_size,
                       num_workers=4, shuffle=True, pin_memory=False):
        """
        Get PyTorch data loader for this dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(True),
                          num_workers=num_workers, pin_memory=pin_memory)


def read_dataset(json_file, itow, wtoi, itoc, ctoi,
                 cache_file, is_debug=False, split="train"):
    if os.path.isfile(cache_file):
        print("Read built %s dataset from %s" % (split, cache_file))
        dataset = pickle.load(open(cache_file, "rb"))
        print("Finished reading %s dataset from %s" % (split, cache_file))

    else:
        print("building %s dataset" % split)
        dataset = SQuAD(json_file, itow, wtoi, itoc, ctoi,
                        debug_mode=is_debug, split=split)
        pickle.dump(dataset, open(cache_file, "wb"))
    return dataset
