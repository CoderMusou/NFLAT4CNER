# --coding: utf-8-
import os
from functools import partial

from fastNLP import cache_results, Vocabulary, DataSet
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import ConllLoader

from utils.paths import yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path, data_filename
from utils.tools import Trie


def load_data(dataset_name, index_token=False, char_min_freq=1, bigram_min_freq=1, only_train_min_freq=True,
              char_dropout=0.01, bigram_dropout=0.01, dropout=0, label_type='ALL', refresh_data=False):
        type = label_type if label_type!='ALL' else ''
        cache_name = ('cache/NER_dataset_{}{}'.format(dataset_name, type))

        return load_ner(data_filename[dataset_name], yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path,
                 index_token, char_min_freq, bigram_min_freq, only_train_min_freq, char_dropout, bigram_dropout, dropout,
                 label_type, _cache_fp=cache_name, _refresh=refresh_data)

@cache_results(_cache_fp='cache/datasets', _refresh=False)
def load_ner(data_path, unigram_embedding_path, bigram_embedding_path, index_token, char_min_freq,
                   bigram_min_freq, only_train_min_freq, char_dropout, bigram_dropout, dropout, label_type='ALL'):
    loader = ConllLoader(['chars', 'target'])

    train_path = os.path.join(data_path['path'], data_path['train'])
    dev_path = os.path.join(data_path['path'], data_path['dev'])
    test_path = os.path.join(data_path['path'], data_path['test'])

    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}

    datasets = {}

    for k, v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k, v in datasets.items():
        print('{}:{}'.format(k, len(v)))

    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    for k, v in datasets.items():
        # ignore the word segmentation tag
        # v.apply_field(lambda x: ['<start>'] + [w[0] for w in x], 'chars', 'chars')
        # add s label
        # v.apply_field(lambda x: ['O'] + x, 'target', 'target')

        if label_type == 'NE':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NAM' else 'O' for w in x],
                          'target', 'target')
        if label_type == 'NM':
            v.apply_field(lambda x: [w if len(w) > 1 and w.split('.')[1] == 'NOM' else 'O' for w in x],
                          'target', 'target')

        # 将所有digit转为0
        v.apply_field(lambda chars:[''.join(['0' if c.isdigit() else c for c in char]) for char in chars],
            field_name='chars', new_field_name='chars')

        v.apply_field(get_bigrams, 'chars', 'bigrams')

    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))

    for k, v in datasets.items():
        v.add_seq_len('chars', new_field_name='seq_len')

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab

    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()), field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_dropout, only_norm_found_vector=True,
                                            min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, dropout=dropout)
        embeddings['char'] = unigram_embedding
    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=bigram_dropout, only_norm_found_vector=True,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq, dropout=dropout)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


def get_bigrams(chars):
    return [char1 + char2
            for char1, char2 in zip(chars, chars[1:] + ['<eos>'])]


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list


@cache_results(_cache_fp='need_to_defined_fp', _refresh=True)
def equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings, w_list, word_embedding_path=None,
                                   only_lexicon_in_train=False, word_char_mix_embedding_path=None,
                                   lattice_min_freq=1, only_train_min_freq=True, dropout=0):
    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result, bme_num = w_trie.get_lexicon(sentence)
        # print(result)
        if len(result) == 0:
            return [[-1, -1, '<non_word>']]
        return [[-1, -1, '<non_word>']] + result

    # def get_bme_num(chars, w_trie):
    #     sentence = ''.join(chars)
    #     result, bme_num = w_trie.get_lexicon(sentence)
    #     # print(result)
    #     return bme_num

    a = DataSet()
    # a.apply
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s, e, lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    import copy
    def get_max(x):
        max_num = [0, 0, 0]
        for item in x:
            max_num = [a if a > b else b for a, b in zip(max_num, item)]
        return max_num

    # max_num = [0, 0, 0]
    for k, v in datasets.items():
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons')
        # v.apply_field(partial(get_bme_num,w_trie=w_trie),'chars','bme_num')
        # for num in v.apply_field(get_max, field_name='bme_num'):
        #     max_num = [a if a > b else b for a, b in zip(max_num, num)]
        v.apply_field(copy.copy, 'chars', 'raw_chars')
        v.add_seq_len('lexicons', 'lex_num')
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    # print('max', max_num)
    #
    # def get_bme_feat(x, max_num=max_num):
    #     max_s = max(max_num)
    #     return [[1 if s == i else 0 for i in range(max_s+1)] for s in x]
    #
    # for k,v in datasets.items():
    #     v.apply_field(get_bme_feat,field_name='bme_num')

    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        # print('lexicons:{}'.format(lexicons))
        # print('lex_only:{}'.format(list(filter(lambda x:x[2],lexicons))))
        # print('result:{}'.format(result))
        return result

    def get_pos_s(ins):
        # lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len))# + lex_s

        return pos_s

    def get_pos_e(ins):
        # lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len))# + lex_e

        return pos_e

    # def norm_bme(x, max_num):
    #     for i in range(len(x)):
    #         x[i] = [(2 * b - a) / a for a, b in zip(max_num, x[i])]
    #     return x
    # def get_word_label(ins):
    #     string = "".join(ins['entities'])
    #     entities = ins['entities']
    #     label = []
    #     for word in ins['raw_words']:
    #         if word in entities:
    #             label.append([0, 1])
    #         elif word not in entities and sum([1 if c in string else 0 for c in word]):
    #             label.append([1, 0])
    #         else:
    #             label.append([0, 0])
    #     return [[0, 0] for i in range(ins['seq_len'])] + label

    for k, v in datasets.items():
        v.apply_field(lambda x: [m[2] for m in x], field_name='lexicons', new_field_name='raw_words')
        v.apply(concat, new_field_name='lattice')
        # v.set_input('lattice')
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        # v.apply_field(partial(norm_bme, max_num=max_num),  field_name='bme_num', new_field_name='bme_num')
        # v.set_input('pos_s', 'pos_e')

        # v.apply(get_word_label, new_field_name='word_label')

    word_vocab = Vocabulary()
    # word_vocab.add_word_lst(w_list)
    word_vocab.from_dataset(datasets['train'], field_name='raw_words',
                               no_create_entry_dataset=[datasets['dev'], datasets['test']])
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
                               no_create_entry_dataset=[datasets['dev'], datasets['test']])
    vocabs['lattice'] = lattice_vocab

    # if word_embedding_path is not None:
    #     word_embedding = StaticEmbedding(word_vocab, word_embedding_path,
    #                                         only_norm_found_vector=True, word_dropout=0, dropout=dropout)
    #     embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
                                            only_norm_found_vector=True, dropout=dropout,
                                            min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding
    # print(datasets['train'][77]['lattice'])
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(*(datasets.values()),
                                   field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(*(datasets.values()),
                                  field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(*(datasets.values()),
                                    field_name='raw_words', new_field_name='words')

    return datasets, vocabs, embeddings