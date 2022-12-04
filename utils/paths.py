


yangjie_rich_pretrain_unigram_path = '../data/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_bigram_path = '../data/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_word_path = '../data/ctb.50d.vec'

yangjie_rich_pretrain_char_and_word_path = '../data/yangjie_word_char_mix.txt'
lk_word_path = '/data/ws/sgns.merge.word'
tencet_word_path = '/data/ws/Tencent_AILab_ChineseEmbedding.txt'



ontonote4ner_cn_path = '../data/OntoNote4NER'
msra_ner_cn_path = '../data/MSRANER'
resume_ner_path = '../data/ResumeNER'
weibo_ner_path = '../data/WeiboNER'

data_filename = {
            "weibo": {
                "path": '../data/WeiboNER',
                "train": "weiboNER_2nd_conll.train_deseg",
                "dev": "weiboNER_2nd_conll.dev_deseg",
                "test": "weiboNER_2nd_conll.test_deseg",
            },
            "resume": {
                "path": '../data/ResumeNER',
                "train": "train.char.bmes",
                "dev": "dev.char.bmes",
                "test": "test.char.bmes",
            },
            "ontonotes": {
                "path": '../data/OntoNote4NER',
                "train": "train.char.bmes_clip",
                "dev": "dev.char.bmes_clip",
                "test": "test.char.bmes_clip",
            },
            "msra": {
                "path": '../data/MSRANER',
                "train": "train_dev.char.bmes_clip",
                "dev": "test.char.bmes_clip",
                "test": "test.char.bmes_clip",
            }
        }
