import numpy as np
import pandas as pd
import jieba


def load_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            vocab[line] = i
    return vocab


def token_to_ids(words, word_ids):
    # 把一个词组 转成id
    # <s><e><unk>
    idxs = []
    start = word_ids['<s>']
    end = word_ids['<e>']
    idxs.append(start)
    for w in words:
        try:
            idx = word_ids[w]
        except Exception:
            idx = word_ids['<unk>']
        idxs.append(idx)
    idxs.append(end)
    return idxs


def spilt_en(en_str):
    en_str = en_str.strip() \
        .replace('"', ' " ') \
        .replace(',', ' , ') \
        .replace('.', ' . ') \
        .replace('?', ' ? ') \
        .replace('!', ' ! ')
    en_str = en_str.strip()
    en_words = en_str.split(' ')
    return en_words


def spilt_zh(zh_str):
    zh_str = zh_str.strip()
    zh_words = list(jieba.cut(zh_str))
    return zh_words


def pair(x, en_word_ids, zh_word_ids):
    '''
源语言单词ID序列，目标语言单词ID序列  下一个单词ID序列
([0, 18, 24, 14, 1161, 801, 16, 55, 82, 329, 1326, 4, 1],
 [0, 20, 84, 256, 30, 86, 21, 93, 6, 15, 111, 5504, 3164, 3],
 [20, 84, 256, 30, 86, 21, 93, 6, 15, 111, 5504, 3164, 3, 1])

 '<s> Two young , White males are outside near many bushes . <e>'
 '<s> Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche .'
 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche . <e>'
'''

    source_words = spilt_en(x[0])
    zh_words = spilt_zh(x[1])

    # 源语言单词ID序列
    source_words_token_ids = token_to_ids(source_words, en_word_ids)
    zh_oken_ids = token_to_ids(zh_words, zh_word_ids)
    # 目标语言单词ID序列
    goal_words_token_ids = zh_oken_ids[:-1]
    # 下一个单词ID序列
    next_words_token_ids = zh_oken_ids[1:]

    source_words_str = ' '.join(map(str, source_words_token_ids))
    goal_words_str = ' '.join(map(str, goal_words_token_ids))
    next_words_str = ' '.join(map(str, next_words_token_ids))
    x['data'] = source_words_str + ';' + goal_words_str + ';' + next_words_str
    return x


en_vocab_path = 'en_vocab.txt'
zh_vocab_path = 'zh_vocab.txt'
en_word_ids = load_vocab(en_vocab_path)
zh_word_ids = load_vocab(zh_vocab_path)

cmn_file = 'cmn.csv'
df_cmn = pd.read_csv(cmn_file, sep='\t', header=None)
df_cmn = df_cmn.apply(lambda x: pair(x, en_word_ids, zh_word_ids), axis=1)

data = pd.DataFrame()
data['data'] = df_cmn['data'].values.tolist()
data.to_csv('data.csv', header=False, sep='\t', index=False)
