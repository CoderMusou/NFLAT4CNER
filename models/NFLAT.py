from modules.infomer import InterFormerEncoder
from modules.transformer import TransformerEncoder

from torch import nn
import torch
import torch.nn.functional as F

from modules.utils import get_crf_zero_init


class NFLAT(nn.Module):
    def __init__(self, tag_vocab, char_embed, word_embed, num_layers, hidden_size, n_head, feedforward_dim, dropout,
                 max_seq_len, after_norm=True, attn_type='adatrans',  bi_embed=None, softmax_axis=-2,
                 char_dropout=0.5, word_dropout=0.5, fc_dropout=0.3, pos_embed=None, attn_dropout=0, four_pos_fusion='ff',
                 q_proj=True, k_proj=True, v_proj=True, r_proj=True, scale=False,
                 vocab=None, before=True, is_less_head=1):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()
        self.vocab = vocab

        self.char_embed = char_embed
        self.word_embed = word_embed
        char_embed_size = self.char_embed.embed_size
        word_embed_size = self.word_embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            char_embed_size += self.bi_embed.embed_size

        self.char_fc = nn.Linear(char_embed_size, hidden_size)
        self.word_fc = nn.Linear(word_embed_size, hidden_size)
        self.char_dropout = nn.Dropout(char_dropout)
        self.word_dropout = nn.Dropout(word_dropout)

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.before = before

        self.chars_transformer = TransformerEncoder(num_layers, hidden_size, n_head//is_less_head, feedforward_dim, dropout,
                                                    after_norm=after_norm, attn_type=attn_type,
                                                    scale=scale, attn_dropout=attn_dropout,
                                                    pos_embed=pos_embed)

        self.informer = InterFormerEncoder(num_layers, hidden_size, n_head, max_seq_len, feedforward_dim, softmax_axis=softmax_axis,
                                       four_pos_fusion=four_pos_fusion, fc_dropout=dropout, attn_dropout=attn_dropout,
                                       q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, r_proj=r_proj, scale=scale, vocab=vocab)


        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(hidden_size, len(tag_vocab))
        self.crf = get_crf_zero_init(len(tag_vocab))
        self.times = 0

    def _forward(self, chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        char_ids = chars
        word_ids = words

        chars_mask = chars.ne(0)
        chars = self.char_embed(chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            chars = torch.cat([chars, bigrams], dim=-1)
        chars = self.char_dropout(chars)
        chars = self.char_fc(chars)

        words = self.word_embed(words)
        words = self.word_dropout(words)
        words = self.word_fc(words)

        if self.before:
            chars.masked_fill_(~(chars_mask.unsqueeze(-1)), 0)
            chars = self.chars_transformer(chars, chars_mask)

        chars = self.informer(chars, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, char_ids, word_ids)

        if not self.before:
            chars.masked_fill_(~(chars_mask.unsqueeze(-1)), 0)
            chars = self.chars_transformer(chars, chars_mask)

        self.fc_dropout(chars)
        chars = self.out_fc(chars)

        logits = F.log_softmax(chars, dim=-1)

        if not self.training:
            paths, _ = self.crf.viterbi_decode(logits, chars_mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, chars_mask)
            return {'loss': loss}

    def forward(self, chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        return self._forward(chars, target, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams)

    def predict(self, chars, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=None):
        return self._forward(chars, None, words, pos_s, pos_e, lex_s, lex_e, seq_len, lex_num, bigrams=bigrams)
