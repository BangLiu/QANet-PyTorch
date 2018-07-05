# -*- coding: utf-8 -*-
"""
Test SQuAD.
"""
import matplotlib.pyplot as plt
from datetime import datetime
from util.nlp_utils import read_w2v
from SQuAD import *


def test_SQuAD_loader():
    # word vector configure
    data_folder = "../../../../datasets/"
    word_vocab_config = {
        "insert_start": "<SOS>",
        "insert_end": "<EOS>",
        "tokenization": "nltk",
        "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
        "embedding_file":
            data_folder + "original/Glove/glove.840B.300d.bin",
        "embedding_cache_file":
            data_folder + "original/Glove/glove.840B.300d.pkl",
        "embedding_binary": True,
        "embedding_dim": 300
    }
    char_vocab_config = {
        "insert_start": "<SOS>",
        "insert_end": "<EOS>",
        "tokenization": "nltk",
        "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
        "embedding_file":
            data_folder + "original/Glove/glove.840B.300d-char.txt",
        "embedding_cache_file":
            data_folder + "original/Glove/glove.840B.300d-char.pkl",
        "embedding_binary": False,
        "embedding_dim": 300
    }
    start = datetime.now()
    wv_tensor, wv_word2ix, wv_vocab, wv_dim = read_w2v(
        word_vocab_config["embedding_file"],
        word_vocab_config["embedding_binary"],
        word_vocab_config["embedding_dim"],
        word_vocab_config["specials"],
        word_vocab_config["embedding_cache_file"])
    cv_tensor, cv_word2ix, cv_vocab, cv_dim = read_w2v(
        char_vocab_config["embedding_file"],
        char_vocab_config["embedding_binary"],
        char_vocab_config["embedding_dim"],
        char_vocab_config["specials"],
        char_vocab_config["embedding_cache_file"])
    print("Time of loading word vector ", datetime.now() - start)

    debug = False
    batch_size_dev = 2
    batch_size_train = 2
    train_cache = data_folder + "processed/SQuAD/SQuAD%s.pkl" % (
        "_debug" if debug else "")
    dev_cache = data_folder + "processed/SQuAD/SQuAD_dev%s.pkl" % (
        "_debug" if debug else "")
    train_json = data_folder + "original/SQuAD/train-v1.1.json"
    dev_json = data_folder + "original/SQuAD/dev-v1.1.json"

    # data
    start = datetime.now()
    train_data = read_dataset(
        train_json, wv_vocab, wv_word2ix,
        cv_vocab, cv_word2ix, train_cache, debug)
    dev_data = read_dataset(
        dev_json, wv_vocab, wv_word2ix,
        cv_vocab, cv_word2ix, dev_cache, debug, split="dev")
    train_dataloader = train_data.get_dataloader(
        batch_size_train, shuffle=True, pin_memory=False)
    dev_dataloader = dev_data.get_dataloader(batch_size_dev)
    print("Time of loading dataset ", datetime.now() - start)
    return train_dataloader, dev_dataloader


def test_SQuAD_analyzer():
    # test read_json, analysis the characteristics of SQuAD data
    data_json = "../../../../datasets/original/SQuAD/train-v1.1.json"
    dev = read_json(path=data_json, type="dev",
                    debug_mode=False, debug_len=10,
                    delete_long_context=True,
                    delete_long_question=True,
                    longest_context=3000,
                    longest_question=300)
    num_examples = 0
    num_del_examples = 0
    question_length = []
    context_length = []
    answer_length = []
    context_sentence_num = []
    context_sentence_length = []
    for e in dev:
        tokenized = tokenize_context(e.context, e.answer_text, e.answer_start)
        if tokenized is None:
            num_del_examples = num_del_examples + 1
            continue
        (tokenized_context,
         answer_tpos_start, answer_tpos_end,
         split_tokenized_context,
         answer_spos,
         answer_tpos_start_in_sent, answer_tpos_end_in_sent) = tokenized

        num_examples = num_examples + 1
        question_length.append(len(nltk.word_tokenize(e.question)))
        context_length.append(len(tokenized_context))
        answer_length.append(answer_tpos_end - answer_tpos_start + 1)
        context_sentence_num.append(len(split_tokenized_context))
        context_sentence_length.extend([len(s) for s
                                        in split_tokenized_context])

    print("# examples: ", num_examples)
    print("# deleted examples: ", num_del_examples)
    plt.hist(question_length, 20, facecolor='blue', alpha=0.5)
    plt.show()
    plt.hist(context_length, 20, facecolor='blue', alpha=0.5)
    plt.show()
    plt.hist(answer_length, 20, facecolor='blue', alpha=0.5)
    plt.show()
    plt.hist(context_sentence_num, 20, facecolor='blue', alpha=0.5)
    plt.show()
    plt.hist(context_sentence_length, 20, facecolor='blue', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    test_SQuAD_loader()
    # test_SQuAD_analyzer()
