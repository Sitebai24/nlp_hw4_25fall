import os
import numpy as np
from transformers import T5TokenizerFast
from load_data import custom_clean_sql as clean_sql  # 确保该函数可用

# --------------------------------------------------------
# 1️⃣ Utility: 计算词汇表大小
# --------------------------------------------------------
def get_vocab_size(examples, tokenizer):
    vocab = set()
    for s in examples:
        vocab.update(tokenizer.encode(s, add_special_tokens=False))
    return len(vocab)

# --------------------------------------------------------
# 2️⃣ Utility: 计算平均长度和最大长度
# --------------------------------------------------------
def get_length_stats(examples, tokenizer):
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in examples]
    return np.mean(lengths), np.max(lengths)

# --------------------------------------------------------
# 3️⃣ 读入原始数据 (train.nl / dev.nl)
# --------------------------------------------------------
with open("data/train.nl", "r", encoding="utf-8") as f:
    train_examples = [line.strip() for line in f if line.strip()]

with open("data/dev.nl", "r", encoding="utf-8") as f:
    dev_examples = [line.strip() for line in f if line.strip()]

# --------------------------------------------------------
# 4️⃣ 读入对应 SQL 数据
# --------------------------------------------------------
with open("data/train.sql", "r", encoding="utf-8") as f:
    train_sql = [q.strip() for q in f if q.strip()]

with open("data/dev.sql", "r", encoding="utf-8") as f:
    dev_sql = [q.strip() for q in f if q.strip()]

# --------------------------------------------------------
# 5️⃣ 初始化 Tokenizer
# --------------------------------------------------------
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# --------------------------------------------------------
# 6️⃣ 统计：原始数据
# --------------------------------------------------------
mean_train_sen_length, max_train_sen_length = get_length_stats(train_examples, tokenizer)
mean_dev_sen_length, max_dev_sen_length = get_length_stats(dev_examples, tokenizer)
mean_train_sql_length, max_train_sql_length = get_length_stats(train_sql, tokenizer)
mean_dev_sql_length, max_dev_sql_length = get_length_stats(dev_sql, tokenizer)

vcb_train_nl_size = get_vocab_size(train_examples, tokenizer)
vcb_dev_nl_size = get_vocab_size(dev_examples, tokenizer)
vcb_train_sql_size = get_vocab_size(train_sql, tokenizer)
vcb_dev_sql_size = get_vocab_size(dev_sql, tokenizer)

# --------------------------------------------------------
# 7️⃣ 读入预处理后的数据 (with_schema + cleaned SQL)
# --------------------------------------------------------
with open("data/train_with_schema.nl", "r", encoding="utf-8") as f:
    train_preps = [i.strip() for i in f if i.strip()]

with open("data/dev_with_schema.nl", "r", encoding="utf-8") as f:
    dev_preps = [i.strip() for i in f if i.strip()]

cleaned_train_sql = [clean_sql(query) for query in train_sql]
cleaned_dev_sql = [clean_sql(query) for query in dev_sql]

mean_train_preps_sen, max_train_preps_sen = get_length_stats(train_preps, tokenizer)
mean_dev_preps_sen, max_dev_preps_sen = get_length_stats(dev_preps, tokenizer)
cleaned_train_sql_mean, cleaned_train_sql_max = get_length_stats(cleaned_train_sql, tokenizer)
cleaned_dev_sql_mean, cleaned_dev_sql_max = get_length_stats(cleaned_dev_sql, tokenizer)

vcb_train_nl_size_preps = get_vocab_size(train_preps, tokenizer)
vcb_dev_nl_size_preps = get_vocab_size(dev_preps, tokenizer)
vcb_train_sql_size_preps = get_vocab_size(cleaned_train_sql, tokenizer)
vcb_dev_sql_size_preps = get_vocab_size(cleaned_dev_sql, tokenizer)

# --------------------------------------------------------
# 8️⃣ 打印结果 (格式化对齐，便于复制到 LaTeX)
# --------------------------------------------------------
print("\n📊 Data Statistics (Before Pre-processing)")
print(f"{'Statistic':<45} {'Train':>10} {'Dev':>10}")
print("-" * 70)
print(f"{'Number of examples':<45} {len(train_examples):>10} {len(dev_examples):>10}")
print(f"{'Mean sentence length':<45} {mean_train_sen_length:>10.2f} {mean_dev_sen_length:>10.2f}")
print(f"{'Max sentence length':<45} {max_train_sen_length:>10} {max_dev_sen_length:>10}")
print(f"{'Mean SQL query length':<45} {mean_train_sql_length:>10.2f} {mean_dev_sql_length:>10.2f}")
print(f"{'Max SQL query length':<45} {max_train_sql_length:>10} {max_dev_sql_length:>10}")
print(f"{'Vocab size (natural language)':<45} {vcb_train_nl_size:>10} {vcb_dev_nl_size:>10}")
print(f"{'Vocab size (SQL)':<45} {vcb_train_sql_size:>10} {vcb_dev_sql_size:>10}")

print("\n📊 Data Statistics (After Pre-processing)")
print(f"{'Statistic':<45} {'Train':>10} {'Dev':>10}")
print("-" * 70)
print(f"{'Number of examples':<45} {len(train_preps):>10} {len(dev_preps):>10}")
print(f"{'Mean sentence length':<45} {mean_train_preps_sen:>10.2f} {mean_dev_preps_sen:>10.2f}")
print(f"{'Max sentence length':<45} {max_train_preps_sen:>10} {max_dev_preps_sen:>10}")
print(f"{'Mean SQL query length (cleaned)':<45} {cleaned_train_sql_mean:>10.2f} {cleaned_dev_sql_mean:>10.2f}")
print(f"{'Max SQL query length (cleaned)':<45} {cleaned_train_sql_max:>10} {cleaned_dev_sql_max:>10}")
print(f"{'Vocab size (natural language)':<45} {vcb_train_nl_size_preps:>10} {vcb_dev_nl_size_preps:>10}")
print(f"{'Vocab size (SQL)':<45} {vcb_train_sql_size_preps:>10} {vcb_dev_sql_size_preps:>10}")





# import os, random, re, string
# from collections import Counter
# from tqdm import tqdm
# import pickle
# import numpy as np 

# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence

# import nltk
# nltk.download('punkt')
# from transformers import T5TokenizerFast
# import torch


# # read in data from train.nl and dev.sql 

# with open("data/train.nl", "r", encoding="utf-8") as f:
#     train = f.readlines()

# with open("data/dev.nl", "r", encoding  = "utf-8") as d:
#     dev = d.readlines()

# # remove newlines from each line 

# train_examples = [line.strip() for line in train if line.strip()]
# dev_examples = [line.strip() for line in dev if line.strip()] 

# # print number of examples 
# print(len(train_examples))
# print(len(dev_examples))


# # calculate the mean sentence length 
# tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# train_sen_length = [len(tokenizer.encode(sen)) for sen in train_examples]
# mean_train_sen_length = np.mean(train_sen_length)
# print(mean_train_sen_length)

# dev_sen_length = [len(tokenizer.encode(sen)) for sen in dev_examples]
# mean_dev_sen_length = np.mean(dev_sen_length)
# print(mean_dev_sen_length)



# # calculte mean sql query length 
# with open("data/train.sql", "r", encoding="utf-8") as f:
#     train_sql = f.readlines()
# with open("data/dev.sql", "r", encoding = "utf-8") as d:
#     dev_sql = d.readlines()


# train_sql = [query.strip() for query in train_sql if query.strip()]
# dev_sql = [query.strip() for query in dev_sql if query.strip()]


# train_sql_length = [len(tokenizer.encode(query)) for query in train_sql]
# mean_train_sql_length = np.mean(train_sql_length)

# dev_sql_length = [len(tokenizer.encode(query)) for query in dev_sql]
# mean_dev_sql_length = np.mean(dev_sql_length) 

# print(mean_train_sql_length)
# print(mean_dev_sql_length)


# #  calcualte vocabulary for nl and sql 
# vcb_train_nl = set()
# for i in train_examples:
#     vcb_train_nl.update(tokenizer.encode(i))
# vcb_train_nl_size = len(vcb_train_nl)
# print(vcb_train_nl_size)

# vcb_dev_nl = set()

# for i in dev_examples:
#     vcb_dev_nl.update(tokenizer.encode(i))
# vcb_dev_nl_size = len(vcb_dev_nl)
# print(vcb_dev_nl_size)

# # calculate vocab size for sql for train and dev 

# vcn_train_sql = set()

# for i in train_sql:
#     vcn_train_sql.update(tokenizer.encode(i))
# vcb_train_sql_size = len(vcn_train_sql)
# print(vcb_train_sql_size)


# vcn_dev_sql = set()
# for i in dev_sql:
#     vcn_dev_sql.update(tokenizer.encode(i))
# vcn_dev_sql_size = len(vcn_dev_sql)
# print(vcn_dev_sql_size)



# # data stats after pre-processing 

# # import the cleaning function

# from load_data import custom_clean_sql as clean_sql 



# # 3. use the '_with_schema.nl' files 

# with open("data/train_with_schema.nl", "r", encoding = "utf-8") as f:
#     train_schema = f.readlines()

# with open("data/dev_with_schema.nl", "r", encoding ="utf-8") as d:
#     dev_schema = d.readlines()

# # remove newlines from each line 
# # strip removes trailing spaces new lines 
# train_preps = [i.strip() for i in train_schema if i.strip()]
# dev_preps = [i.strip() for i in dev_schema if i.strip()]

# # number of examples 
# print(len(train_preps))
# print(len(dev_preps))

# # mean sentence length 

# # get the length for each sentence and store it in a list, and then calculate the mean 

# train_preps_sen_length = [len(tokenizer.encode(i)) for i in train_preps]
# dev_preps_sen_length = [len(tokenizer.encode(i)) for i in dev_preps]


# mean_train_preps_sen = np.mean(train_preps_sen_length)
# mean_dev_preps_sen = np.mean(dev_preps_sen_length)
# print(f'mean train sentence after preprocessing: {mean_train_preps_sen: .2f}')
# print(f'mean dev sentence after preprocessing: {mean_dev_preps_sen: .2f}')


# # after cleaned, mean sql length 

# cleaned_train_sql = [clean_sql(query) for query in train_sql]
# cleaned_dev_sql = [clean_sql(query) for query in dev_sql]


# cleaned_train_sql_len = [len(tokenizer.encode(i)) for i in cleaned_train_sql]
# cleaned_dev_sql_len = [len(tokenizer.encode(i)) for i in cleaned_dev_sql]

# cleaned_train_sql_mean = np.mean(cleaned_train_sql_len)
# cleaned_dev_sql_len_mean = np.mean(cleaned_dev_sql_len)

# print(f"cleaned train sql mean: {cleaned_train_sql_mean:.2f}")

# print(f"cleaned dev sql mean: {cleaned_dev_sql_len_mean:.2f}")

# # vocab size 

# #  calcualte vocabulary for nl and sql
# vcb_train_nl_preps = set()
# for i in train_preps:
#     vcb_train_nl_preps.update(tokenizer.encode(i))
# vcb_train_nl_size_preps = len(vcb_train_nl_preps)
# print(f"Preps Train NL Vocab Size: {vcb_train_nl_size_preps}")

# vcb_dev_nl_preps = set()
# for i in dev_preps:
#     vcb_dev_nl_preps.update(tokenizer.encode(i))
# vcb_dev_nl_size_preps = len(vcb_dev_nl_preps)
# print(f"Preps Dev NL Vocab Size: {vcb_dev_nl_size_preps}")

# # calculate vocab size for sql for train and dev
# vcn_train_sql_preps = set()
# for i in cleaned_train_sql:
#     vcn_train_sql_preps.update(tokenizer.encode(i))
# vcb_train_sql_size_preps = len(vcn_train_sql_preps)
# print(f"Preps Train SQL Vocab Size: {vcb_train_sql_size_preps}")

# vcn_dev_sql_preps = set()
# for i in cleaned_dev_sql:
#     vcn_dev_sql_preps.update(tokenizer.encode(i))
# vcn_dev_sql_size_preps= len(vcn_dev_sql_preps)
# print(f"Preps Dev SQL Vocab Size: {vcn_dev_sql_size_preps}")





