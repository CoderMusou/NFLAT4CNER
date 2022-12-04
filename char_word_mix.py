# --coding: utf-8-
from utils.paths import yangjie_rich_pretrain_word_path, yangjie_rich_pretrain_unigram_path, \
    yangjie_rich_pretrain_char_and_word_path

output_f = open(yangjie_rich_pretrain_char_and_word_path, 'w', encoding='utf-8')

with open(yangjie_rich_pretrain_word_path, 'r', encoding='utf-8') as lexicon_f:
    lexicon_lines = lexicon_f.readlines()
    for l in lexicon_lines:
        l_split = l.strip().split()
        if len(l_split[0]) != 1:
            print(l.strip(), file=output_f)

with open(yangjie_rich_pretrain_unigram_path, 'r', encoding='utf-8') as char_f:
    char_lines = char_f.readlines()
    for l in char_lines:
        print(l.strip(), file=output_f)

output_f.close()
