import os

import torch
from torch.optim.lr_scheduler import LambdaLR

from models.NFLAT import NFLAT
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim, nn
from fastNLP import SpanFPreRecMetric, BucketSampler, LRScheduler, AccuracyMetric

import argparse
from modules.utils import set_rng_seed, MyEvaluateCallback
from utils.load_data import load_data, equip_chinese_ner_with_lexicon, load_yangjie_rich_pretrain_word_list
from utils.paths import *

device = 0
refresh_data = False
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'resume', 'ontonotes', 'msra'])
parser.add_argument('--lexicon_name', default='yj', choices=['lk','yj','tx'])
parser.add_argument('--only_lexicon_in_train', default=False)
parser.add_argument('--label_type', default='ALL', help='NE|NM|ALL')

args = parser.parse_args()
dataset = args.dataset

pos_embed = None

batch_size = 10
warmup_steps = 0.2
after_norm = 1
model_type = 'transformer'
normalize_embed = True

char_embed_dropout = 0.5
word_embed_dropout = 0.5
momentum = 0.9

dropout = 0.2
fc_dropout = 0.3
attn_dropout = 0

four_pos_fusion = 'ff_two'

crf_lr = 1

before = False
scale = False
k_proj = True
softmax_axis = -1
is_less_head = 1
use_bigram = 1

seed = 2022
set_rng_seed(seed)

num_layers = 1
attn_type = 'adatrans'
n_epochs = 100

if dataset == 'resume':
    attn_dropout = 0
    batch_size = 16
    char_embed_dropout = 0.6
    dropout = 0.2
    fc_dropout = 0
    is_less_head = 1
    lr = 0.002
    n_epochs = 50
    use_bigram = 1
    warmup_steps = 0.05
    word_embed_dropout = 0.5
    n_heads = 8
    head_dims = 16
elif dataset == 'weibo':
    attn_dropout = 0.2
    batch_size = 16
    char_embed_dropout = 0.6
    dropout = 0
    fc_dropout = 0.2
    is_less_head = 2
    lr = 0.003
    use_bigram = 1
    warmup_steps = 0.4
    word_embed_dropout = 0.4
    n_heads = 12
    head_dims = 16
elif dataset == 'ontonotes':
    batch_size = 16
    char_embed_dropout = 0.4
    dropout = 0.2
    fc_dropout = 0.2
    attn_dropout = 0.2
    is_less_head = 2
    lr = 0.0008
    use_bigram = 1
    warmup_steps = 0.1
    word_embed_dropout = 0.4
    n_heads = 8
    head_dims = 32
elif dataset == 'msra':
    attn_dropout = 0
    batch_size = 16
    char_embed_dropout = 0.4
    dropout = 0.2
    fc_dropout = 0
    is_less_head = 1
    lr = 0.002
    use_bigram = 1
    warmup_steps = 0.2
    word_embed_dropout = 0.4
    n_heads = 8
    head_dims = 32

args.init = 'uniform'

encoding_type = 'bmeso'
if args.dataset == 'weibo':
    encoding_type = 'bio'

d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

datasets, vocabs, embeddings = load_data(dataset, index_token=False, char_min_freq=1, bigram_min_freq=1,
                                         only_train_min_freq=1, char_dropout=0.01, label_type=args.label_type,
                                         refresh_data=refresh_data)

if args.lexicon_name == 'lk':
    word_path = lk_word_path
    word_char_mix_embedding_path = lk_word_path
    lex = 'lk'
elif args.lexicon_name == 'tx':
    word_path = tencet_word_path
    word_char_mix_embedding_path = tencet_word_path
    lex = 'tx'
else:
    word_path = yangjie_rich_pretrain_word_path
    word_char_mix_embedding_path = yangjie_rich_pretrain_char_and_word_path
    lex = 'yj'

w_list = load_yangjie_rich_pretrain_word_list(word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

type = args.label_type if args.label_type != 'ALL' else ''
cache_name = os.path.join('cache', ('dataset_{}_lex_{}{}'.format(
    args.dataset, lex, type)))
datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                              w_list, yangjie_rich_pretrain_word_path,
                                                              _refresh=refresh_data, _cache_fp=cache_name,
                                                              only_lexicon_in_train=args.only_lexicon_in_train,
                                                              word_char_mix_embedding_path=word_char_mix_embedding_path)

for i, dataset in datasets.items():
    dataset.set_input('chars', 'bigrams', 'target')
    dataset.set_input('words')
    dataset.set_input('seq_len', 'lex_num')
    dataset.set_input('pos_s', 'pos_e', 'lex_s', 'lex_e')
    dataset.set_target('seq_len', 'target')

max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))

if use_bigram:
    bi_embed = embeddings['bigram']
else:
    bi_embed = None

model = NFLAT(tag_vocab=vocabs['label'], char_embed=embeddings['lattice'], word_embed=embeddings['lattice'],
              num_layers=num_layers, hidden_size=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=dropout, max_seq_len=max_seq_len,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=bi_embed,
              char_dropout=char_embed_dropout,
              word_dropout=word_embed_dropout,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=scale,
              softmax_axis=softmax_axis,
              vocab=vocabs['lattice'],
              four_pos_fusion=four_pos_fusion,
              before=before,
              is_less_head=is_less_head,
              attn_dropout=attn_dropout)

params_nums = 0
for n,p in model.named_parameters():
    print('{}:{}'.format(n,p.size()))
    if 'char_embed' not in n and 'bi_embed' not in n:
        x = 1
        for size in p.size():
            x *= size
        params_nums += x
print('params_nums:', params_nums)

with torch.no_grad():
    print('{}init pram{}'.format('*'*15,'*'*15))
    for n,p in model.named_parameters():
        # print(n, p.size())
        if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except Exception as e:
                print(e)
                print(n)
                exit(1208)
    print('{}init pram{}'.format('*' * 15, '*' * 15))



crf_params = list(model.crf.parameters())
crf_params_ids = list(map(id,crf_params))
non_crf_params = list(filter(lambda x:id(x) not in crf_params_ids, model.parameters()))

param_ = [{'params': non_crf_params}, {'params': crf_params, 'lr': lr * crf_lr}]

optimizer = optim.SGD(param_, lr=lr, momentum=momentum)

callbacks = []

lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep)))
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = MyEvaluateCallback(datasets['test'])

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, lrschedule_callback, evaluate_callback])


print("-"*20)
print("Hyper-parameters")
print("lexicon_name:", args.lexicon_name)
print("n_heads:", n_heads)
print("head_dims:", head_dims)
print("num_layers:", num_layers)
print("lr:", lr)
print("attn_type:", attn_type)
print("n_epochs:", n_epochs)
print("batch_size:", batch_size)
print("warmup_steps:", warmup_steps)
print("model_type:", model_type)
print("n_epochs:", n_epochs)
print("momentum:", momentum)
print("seed:", seed)
print("-"*20)

print('parameter weight:')
print(model.state_dict()['informer.layer_0.ffn.0.weight'])


f1_metric = SpanFPreRecMetric(vocabs['label'], encoding_type=encoding_type)
acc_metric = AccuracyMetric()
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]

trainer = Trainer(datasets['train'], model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=n_epochs, dev_data=datasets['dev'],
                  metrics=metrics,
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=3000, save_path=None)
trainer.train(load_best_model=False)
