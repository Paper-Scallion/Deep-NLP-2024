import os
import torch
import random
import numpy as np
import jieba
import opencc
import gensim
import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import utility
from model.transformer import TransformerDecoder

# 读取数据集
class DLDataset(Dataset):
    def __init__(self, args):
        self.data = list()
        cc = opencc.OpenCC("t2s")  # 初始化opencc，繁体转简体
        corpus_path = args.data_path  # 语料库路径
        title_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",")  # 读取语料库目录为list
        corpus_tokenized = list()
        # 对于目录中每一个书
        for title in title_list:
            book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read()  # 按对应编码格式打开文件并读取内容
            book = utility.erase_useless(book)  # 删除无用内容
            book = utility.preprocessing(book)  # 预处理
            book = cc.convert(book)  # 繁体转简体
            sentences = utility.split_sentences(book)  # 将文章拆分为单独句子，格式为list[str]
            self.data.extend(sentences)  # 加入到数据集中

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class Collate:
    def __init__(self, args):
        self.src_seq_len = args.src_seq_len  # 序列长度
        self.dictionary = args.dictionary  # 词典
        self.sos_id = args.sos_id  # 开始token_id
        self.eos_id = args.eos_id  # 结束token_id

    def collate_fn(self, batch):
        src_ids = []
        label_ids = []
        for i, src in enumerate(batch):  # src为未分词的句子
            src_tokenlized = [word for word in jieba.cut(src, cut_all=False) if utility.is_chinese_token(word)]  # 进行分词
            src_id = [self.dictionary.token2id[token] for token in src_tokenlized]  # 词list转换为id list
            if len(src_id) < self.src_seq_len:  # 若长度小于预设，则补全
                label_id = src_id[1:] + [self.eos_id] + [-100] * (self.src_seq_len - len(src_id))
                src_id = src_id + [0] * (self.src_seq_len - len(src_id))
            else:  # 否则取末端
                src_id = src_id[: self.src_seq_len]
                label_id = src_id[1:] + [self.eos_id]
            src_ids.append(src_id)  # 加入到总数据集中
            label_ids.append(label_id)
        src_ids = torch.tensor(np.array(src_ids)).long()
        label_ids = torch.tensor(np.array(label_ids)).long()
        data = {"src_ids": src_ids, "label_ids": label_ids}
        return data

class Args:
    data_path = "./chinese-corpus"  # 语料库路径
    save_dir = "./checkpoints/transformer_model.pt"  # 模型保存路径
    dictionary = gensim.corpora.Dictionary.load("./result/dictionary.bin")  # 词典路径
    dictionary.add_documents([["(sos)", "(eos)"]])  # 添加sos和eos
    n_trg_vocab = len(dictionary)  # 词典大小
    sos_id = 0  # 序列开始
    eos_id = 1  # 序列结束
    trg_pad_idx = 0  # 填充id
    d_word_vec = 512  # 词向量维度
    d_model = 512  # 特征维度
    d_inner = 1024  # 隐藏层维度
    n_layers = 6  # 解码器层数
    n_head = 8  # 头数
    d_k = d_model // n_head
    d_v = d_model // n_head
    dropout = 0.1  # 随机失效概率
    src_seq_len = 64  # 一次输入网络的token数
    epochs = 100  # 训练轮数
    train_batch_size = 20  # 训练batch
    test_batch_size = 5  # 测试batch
    learning_rate = 3e-5  # 学习率
    max_grad_norm = 5  # clip_grad_norm_参数
    weight_decay = 0.01  # 权重衰减
    adam_epsilon = 1e-8
    warmup_proportion = 0.1 # 学习率预热
    do_train = True # 进行训练

def predict(model, test_dataset, args):
    with torch.no_grad():
        for src in test_dataset:
            origin = src.strip()
            # src = origin[:20]  # 取前50个字，后续用于生成
            src_tokens = [word for word in jieba.cut(src, cut_all=False) if utility.is_chinese_token(word)]  # 分词
            input_tokens = src_tokens[: (len(src_tokens) // 2 if len(src_tokens) // 2 > 0 else 1)]  # 取前一半的词
            src = "".join(input_tokens)
            trg_id = None
            input_token_ids = [[args.dictionary.token2id[token] for token in input_tokens]]
            # print([args.id2token[i] for i in src_ids[0]])
            while trg_id != args.eos_id:
                # print(src_ids)
                if trg_id is None:
                    input_token_ids = torch.from_numpy(np.array(input_token_ids, dtype=np.int64)).cuda()
                if len(input_token_ids[0]) > args.src_seq_len:
                    break
                output = model(input_token_ids)
                output = output[:, -1, :].detach().cpu().numpy()
                output = np.argmax(output, -1).tolist()
                trg_id = output[0]
                input_token_ids = input_token_ids.detach().cpu().numpy().tolist()
                input_token_ids[0].append(trg_id)
                input_token_ids = torch.from_numpy(np.array(input_token_ids, dtype=np.int64)).cuda()

            if isinstance(input_token_ids, torch.Tensor):
                input_token_ids = input_token_ids.detach().cpu().numpy().tolist()

            print("提示词：" + src)
            print("真实标签：" + origin)
            print("提示词+生成："+ "".join([args.dictionary[i] for i in input_token_ids[0]]))
            print("\n")

def main():
    args = Args() # 参数配置
    data = DLDataset(args) # 加载数据集
    ratio = int(0.95 * len(data))
    train_dataset = data[:ratio] # 部分作为训练集
    test_dataset = data[ratio:] # 部分作为测试集
    collate = Collate(args)
    model = TransformerDecoder(
        n_trg_vocab=args.n_trg_vocab,
        trg_pad_idx=args.trg_pad_idx,
        d_word_vec=args.d_word_vec,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        trg_seq_len=args.src_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=1,
        collate_fn=collate.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=1,
        collate_fn=collate.collate_fn,
    )

    if os.path.exists(args.save_dir): # 加载模型参数
        model.load_state_dict(torch.load(args.save_dir, map_location=torch.device("cpu")), strict=True)
    model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # 交叉熵损失函数，忽略掉用于pad的id
    no_decay = ["bias", "LayerNorm.weight"]
    module = model.module if hasattr(model, "module") else model
    model_param = list(module.named_parameters()) # 模型参数

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {
            "params": [p for n, p in model_param if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon) # AdamW优化器
    total_step = args.epochs * len(train_loader) + 1 # 总训练步数
    scheduler = get_linear_schedule_with_warmup( # 学习率预热
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_step),
        num_training_steps=total_step,
    )
    global_step = 1 # 当前步数
    tq = tqdm.tqdm(total=total_step)
    for epoch in range(1, args.epochs + 1): # 训练轮数
        for step, batch in enumerate(train_loader):
            for k, v in batch.items():
                batch[k] = v.cuda()
            model.train() # 训练模式
            output = model(batch["src_ids"]) # 模型推理
            batch_size = output.size(0)
            seq_len = output.size(1)
            loss = criterion(output.view(batch_size * seq_len, -1), batch["label_ids"].view(-1)) # 计算损失值
            optimizer.zero_grad() # 梯度初始化为0
            loss.backward() # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # 梯度裁减
            optimizer.step() # 更新参数值
            scheduler.step() # 学习率调整
            cur_lr = optimizer.state_dict()["param_groups"][0]["lr"] # 当前学习率
            tq.set_postfix(loss=loss.item(), lr=cur_lr)
            tq.update(1)
            global_step += 1
            if global_step % 1000 == 0: # 每1000步评估模型
                predict(model, random.choices(test_dataset, k=10), args)
            if global_step % 10000 == 0: # 每10000步保存模型
                torch.save(model.state_dict(), args.save_dir)
    tq.close()

if __name__ == "__main__":
    main()
