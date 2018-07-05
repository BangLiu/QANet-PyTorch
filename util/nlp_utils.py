# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
import gensim
import os
from .file_utils import pickle_dump_large_file, pickle_load_large_file
from nltk.tokenize.punkt import PunktSentenceTokenizer
import torch


def split_sentence(text, language):
    """
    Segment a input text into a list of sentences.
    :param text: a segmented input string.
    :param language: language type. "Chinese" or "English".
    :return: a list of segmented sentences.
    """
    if language == "Chinese":
        return split_chinese_sentence(text)
    elif language == "English":
        return split_english_sentence(text)
    else:
        print("Currently only support Chinese and English.")


def split_chinese_sentence(text):
    """
    Segment a input Chinese text into a list of sentences.
    :param text: a segmented input string.
    :return: a list of segmented sentences.
    """
    words = str(text).split()
    start = 0
    i = 0
    sents = []
    punt_list = '。!！?？;；~～'.decode('utf8')
    for word in words:
        word = word.decode("utf8")
        token = list(words[start:i + 2]).pop().decode("utf8")
        if word in punt_list and token not in punt_list:
            sents.append(words[start:i + 1])
            start = i + 1
            i += 1
        else:
            i += 1
            token = list(words[start:i + 2]).pop()
    if start < len(words):
        sents.append(words[start:])
    sents = [" ".join(x) for x in sents]
    return sents


def split_english_sentence(text):
    """
    Segment a input English text into a list of sentences.
    :param text: a segmented input string.
    :return: a list of segmented sentences.
    """
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)
    return sentences


def remove_OOV(text, vocab):
    """
    Remove OOV words in a text.
    """
    tokens = str(text).split()
    tokens = [word for word in tokens if word in vocab]
    new_text = " ".join(tokens)
    return new_text


def replace_OOV(text, replace, vocab):
    """
    Replace OOV words in a text with a specific word.
    """
    tokens = str(text).split()
    new_tokens = []
    for word in tokens:
        if word in vocab:
            new_tokens.append(word)
        else:
            new_tokens.append(replace)
    new_text = " ".join(new_tokens)
    return new_text


def remove_stopwords(text, stopwords):
    """
    Remove stop words in a text.
    """
    tokens = str(text).split()
    tokens = [word for word in tokens if word not in stopwords]
    new_text = " ".join(tokens)
    return new_text


def right_pad_zeros_2d(lst, max_len, dtype=np.int64):
    """
    Given a 2d list, padding or truncating each sublist to max_len.
    :param lst: input 2d list.
    :param max_len: maximum length.
    :return: padded list.
    """
    result = np.zeros([len(lst), max_len], dtype)
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            if j >= max_len:
                break
            result[i][j] = val
    return result


def right_pad_zeros_1d(lst, max_len):
    """
    Given a 1d list, padding or truncating each sublist to max_len.
    :param lst: input 1d list.
    :param max_len: maximum length.
    :return: padded list.
    """
    lst = lst[0:max_len]
    lst.extend([0] * (max_len - len(lst)))
    return lst


def load_w2v(fin, binary, vector_size,
             specials=["<UNK>", "<PAD>", "<SOS>", "<EOS>"]):
    """
    Load word vector file.
    :param fin: input word vector file name.
    :param binary: True of False.
    :param vector_size: vector length.
    :param sepcials: list of special words.
    :return: w2v tensor, dictionary and dimension
    """
    model = {}
    vocab = []

    model = gensim.models.KeyedVectors.load_word2vec_format(
        fin, binary=binary)
    vocab = set(model.wv.vocab.keys())

    wv = dict((k, model[k]) for k in vocab if len(model[k]) == vector_size)
    if specials is not None:
        for special in specials:
            wv[special] = np.zeros(vector_size, dtype=float)
            vocab.add(special)
    wv = OrderedDict(wv)
    wv_word2ix = {word: i for i, word in enumerate(wv)}
    wv_dim = vector_size
    wv_tensor = torch.Tensor(np.array(list(wv.values()))).view(-1, wv_dim)
    wv_vocab = vocab

    return wv_tensor, wv_word2ix, wv_vocab, wv_dim


def read_w2v(fin, binary, vector_size,
             specials=["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
             cache_file=None):
    """
    Read word vector file with potential cache file.
    :param fin: input word vector file name.
    :param binary: True of False.
    :param vector_size: vector length.
    :param sepcials: list of special words.
    :param cache_file: if offered, it is a pkl file that contains
                       the result of load_w2v.
    :return: w2v tensor, dictionary and dimension
    """
    if os.path.isfile(cache_file):
        print("Read built word vector from %s" % (cache_file))
        wv_tensor, wv_word2ix, wv_vocab, wv_dim = \
            pickle_load_large_file(cache_file)
        print("Finished reading word vector from %s" % (cache_file))

    else:
        print("Read word vector")
        wv_tensor, wv_word2ix, wv_vocab, wv_dim = load_w2v(
            fin, binary, vector_size, specials)
        print("Dump word vector to %s" % (cache_file))
        pickle_dump_large_file([wv_tensor, wv_word2ix, wv_vocab, wv_dim],
                               cache_file)
        print("Finished reading word vector")
    return wv_tensor, wv_word2ix, wv_vocab, wv_dim


if __name__ == "__main__":
    a = "这个 苹果 好哒 啊 ！ ！ ！ 坑死 人 了 。 你 是 谁 ？ 额 。 。 。 好吧 。"
    print(a)
    for b in split_sentence(a, "Chinese"):
        print(b)

    a = "Good morning! Let us start this lecture. What are you doing?"
    for b in split_sentence(a, "English"):
        print(b)

    text = "你 好 吗 老鼠"
    vocab = ["你", "好", "老鼠"]
    replace = "UNKNOWN"
    print(remove_OOV(text, vocab))
    print(replace_OOV(text, replace, vocab))

    a = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3], [5, 6, 7]]
    print(a)
    print(right_pad_zeros_2d(a, 5, dtype=np.int64))
    print(right_pad_zeros_1d([1, 2, 3, 4, 5], 10))
    print(right_pad_zeros_1d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 10))

    wv_tensor, wv_word2ix, wv_vocab, wv_dim = load_w2v(
        "../../../../../doc2graph/data/raw/Google-w2v/GoogleNews-vectors-negative300.bin",
        True, 300,
        specials=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
    print(wv_tensor.size())
    print(len(wv_vocab))
    print(wv_dim)
