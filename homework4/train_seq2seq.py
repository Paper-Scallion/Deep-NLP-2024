import torch
import torch.nn as nn
import torch.optim as optim
import gensim.corpora
import jieba
import opencc
import random
import tqdm
import os
import numpy as np

import utility
from model.seq2seq import Seq2SeqModel

class Args:
    dictionary = gensim.corpora.Dictionary.load("./result/dictionary.bin") # 读取词典
    save_dir = "./checkpoints/seq2seq_model.pt" # 模型保存路径
    vocab_size = len(dictionary) # 词典总词数
    bos_id = 0 # 开始标志token id
    eos_id = 1 # 结束标志token id
    pad_id = 2
    num_single = 128 # 一次输入的token数
    embed_dim = 512 # 模型输入embedding维度
    hidden_dim = 512 # 隐藏层维度
    num_layers = 6 # 层数
    learn_rate = 1e-5 # 学习率
    epochs = 100 # 训练轮数

def predict(model, test_dataset, args):
    model.eval()
    with torch.no_grad():
        for src in test_dataset:
            input_token_ids = [args.bos_id] + src[: len(src) // 2 if len(src) // 2 != 0 else 1] # 取前一半作为输入，并加入bos
            i = 0
            while i < len(input_token_ids): # 前一半使用真实标签，得到隐藏状态和最后的输出
                if i == 0:
                    hx = None
                input_tensor = torch.Tensor([input_token_ids[i]]).long().cuda()
                out, hx = model(input_tensor, hx)
                i += 1

            out = torch.argmax(out).unsqueeze(0)
            while True: # 后面完全使用当前输出和隐藏状态来推理
                out, hx = model(out, hx)
                out_id = out.clone().detach().cpu().numpy()
                out_id = np.argmax(out_id)
                input_token_ids.append(out_id)
                if out_id == args.eos_id or len(input_token_ids) >= 128:
                    break
                out = torch.argmax(out).unsqueeze(0)
            
            print("提示词：" + "".join([args.dictionary[token_id] for token_id in src[: len(src) // 2 if len(src) // 2 != 0 else 1]]))
            print("真实标签：" + "".join([args.dictionary[token_id] for token_id in src]))
            print("提示词+生成：" + "".join([args.dictionary[token_id] for token_id in input_token_ids]))
            print("\n")

def main():
    args = Args() # 参数配置
    # 参数配置
    pad_id = 2

    cc = opencc.OpenCC("t2s") # 初始化opencc，繁体转简体
    corpus_path = "./chinese-corpus" # 语料库路径
    title_list = open("{}/inf.txt".format(corpus_path), "r", encoding="gb18030").readline().split(",") # 读取语料库目录为list
    corpus_tokenized = list() # 所有语料
    # 对于目录中每一个书
    for title in title_list:
        book = open("{}/{}.txt".format(corpus_path, title), "r", encoding="gb18030").read() # 按对应编码格式打开文件并读取内容
        book = utility.erase_useless(book) # 删除无用内容
        book = utility.preprocessing(book) # 预处理
        book = cc.convert(book) # 繁体转简体
        sentences = utility.split_sentences(book)  # 将文章拆分为单独句子，格式为list[str]
        sentences_tokenized = [[args.dictionary.token2id[token] for token in jieba.cut(sentence, cut_all=False) if utility.is_chinese_token(token)] for sentence in sentences] # 句子转token id
        corpus_tokenized.extend(sentences_tokenized) # 加入到总数据集

    ratio = int(0.95 * len(corpus_tokenized))
    random.shuffle(corpus_tokenized)
    train_dataset = corpus_tokenized[:ratio] # 部分作为训练集
    test_dataset = corpus_tokenized[ratio:] # 部分作为测试集
    model = Seq2SeqModel(args) # 构建模型
    
    if os.path.exists(args.save_dir): # 加载模型参数
        model.load_state_dict(torch.load(args.save_dir, map_location=torch.device('cpu')), strict=True)
    else:
        model.init_weights() # 初始化权重
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate) # 优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    criterion = criterion.cuda() # 损失函数
    tq = tqdm.tqdm(total=args.epochs * len(train_dataset))
    global_step = 1 # 当前步数
    for e in range(args.epochs):
        select = list(range(len(train_dataset)))
        random.shuffle(select)
        for s in select:
            sentence_tokenized = train_dataset[s] # 选择第s个句子
            seq = [args.bos_id] + sentence_tokenized + [args.eos_id] # 加入开始和结束标志
            total_length = len(seq) # 总长度
            for i in range(total_length - 1):
                if i == 0: # 第一个输入不包含隐藏状态
                    hx = None
                model.train()
                input_seq = torch.Tensor([seq[i]]).long().cuda() # 当前位置为输入
                label_seq = torch.Tensor([seq[i + 1]]).long().cuda() # 下一位置为标签
                optimizer.zero_grad()
                rand_num = random.randint(1, 100)
                if 1 <= rand_num <= 70 or i == 0: # 一定几率使用真值训练
                    out, hx = model(input_seq, hx)
                else: # 一定几率使用上次输出训练
                    out = torch.argmax(out).unsqueeze(0)
                    out, hx = model(out, hx)
                loss = criterion(out, label_seq) # 计算损失值
                loss.backward()
                optimizer.step()
                hx0 = hx[0].detach()
                hx1 = hx[1].detach()
                hx = (hx0, hx1)
                tq.set_postfix(loss=loss.item(), lr=args.learn_rate)
                tq.update(1)
                global_step += 1
                if global_step % 1000 == 0: # 每1000步评估模型
                    predict(model, random.choices(test_dataset, k=10), args)
                if global_step % 10000 == 0: # 每10000步保存模型
                    torch.save(model.state_dict(), args.save_dir)
    tq.close()

if __name__ == "__main__":
    main()
