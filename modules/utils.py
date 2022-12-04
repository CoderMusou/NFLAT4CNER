# 设置随机数种子
import math
import torch
from fastNLP import Callback, Tester, DataSet


def set_rng_seed(rng_seed:int = None, random:bool = True, numpy:bool = True,
                 pytorch:bool=True, deterministic:bool=True):
    """
    设置模块的随机数种子。由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        需要在fitlog.commit()或fitlog.set_log_dir()之后运行才会记录该rng_seed到log中
    :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个。
    :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
    :param bool, numpy: 是否将numpy的seed设置为rng_seed.
    :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
    :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True
    """
    if rng_seed is None:
        import time
        rng_seed = int(time.time()%1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    return rng_seed


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

def char_lex_len_to_mask(char_len, lex_len):
    """
    根据 seq_len 和 lex_len 生成三维 mask
    """
    batch_size = char_len.size(0)
    max_char_len = char_len.max().long()
    max_lex_len = lex_len.max().long()
    broadcast_char_len = torch.arange(max_char_len).expand(batch_size, -1).to(char_len)
    char_mask = broadcast_char_len.lt(char_len.unsqueeze(1))
    broadcast_lex_len = torch.arange(max_lex_len).expand(batch_size, max_char_len, -1).to(char_len)
    mask = broadcast_lex_len.lt(lex_len.unsqueeze(1).unsqueeze(1)).masked_fill(~char_mask.unsqueeze(-1), False)

    return mask


class MyEvaluateCallback(Callback):
    """
    通过使用该Callback可以使得Trainer在evaluate dev之外还可以evaluate其它数据集，比如测试集。每一次验证dev之前都会先验证EvaluateCallback
    中的数据。
    """

    def __init__(self, data=None, tester=None):
        """
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用Trainer中的metric对数据进行验证。如果需要传入多个
            DataSet请通过dict的方式传入。
        :param ~fastNLP.Tester,Dict[~fastNLP.DataSet] tester: Tester对象, 通过使用Tester对象，可以使得验证的metric与Trainer中
            的metric不一样。
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.intermediate_dev = []
        self.intermediate_test = []
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        self.intermediate_dev.append(list(eval_result['SpanFPreRecMetric'].values()))
        if len(self.testers) > 0:
            for idx, (key, tester) in enumerate(self.testers.items()):
                try:
                    eval_result = tester.test()
                    self.intermediate_test.append(list(eval_result['SpanFPreRecMetric'].values()))
                    if idx == 0:
                        indicator, indicator_val = _check_eval_results(eval_result)
                        if indicator_val>self.best_test_metric_sofar:
                            self.best_test_metric_sofar = indicator_val
                            self.best_test_epoch = self.epoch
                            self.best_test_sofar = eval_result
                    if better_result:
                        # torch.save(self.model.state_dict(), '/data/ws/NFLAT_best_model/best_model_parameter.pkl')
                        self.best_dev_test = eval_result
                        self.best_dev_epoch = self.epoch
                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception as e:
                    self.logger.error("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        print(self.intermediate_dev)
        print(self.intermediate_test)
        if self.best_test_sofar:
            self.logger.info("Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_test_sofar, self.best_test_epoch))
        if self.best_dev_test:
            self.logger.info("Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_dev_test, self.best_dev_epoch))

    def on_exception(self, exception):
        print(exception)

def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf
