import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")


# 分词操作 但是只能分英语 因为是根据空格来分词的 中文需要用jieba
# 标点符号也被分开
def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


# "title": "University_of_Notre_Dame",
# "paragraphs": [{
# "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold
#  dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper
# statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica
# of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a
# replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous
# in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome),
# is a simple, modern stone statue of Mary.",
# "qas": [{
# "answers": [{
# "answer_start": 515,
# "text": "Saint Bernadette Soubirous"
# }],
# 返回的是 问题处理完后的一些信息  和 问题 本身的一些信息
def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                # eg: Architecturally, the school has a Catholic character...
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # 返回单词列表
                # ['Architecturally', ',', 'the', 'school', 'has', 'a', 'Catholic', 'character', ...]
                context_tokens = word_tokenize(context)
                # 返回每个单词的字母列表
                # [['A', 'r', 'c', 'h', 'i', 't', 'e', 'c', 't', 'u', 'r', 'a', 'l', 'l', 'y'], [',']...
                context_chars = [list(token) for token in context_tokens]
                # 返回context 中每个单词的起始和结束坐标
                # [(0, 15), (15, 16), (17, 20), (21, 27)...
                spans = convert_idx(context, context_tokens)
                # 用于统计单词和字母的词频 ？ 到底是统计的是context的词的个数还是 问题中词的个数
                # 但是它直接加的是问题的长度 为什么这样做
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        # {'answer_start': 515, 'text': 'Saint Bernadette Soubirous'}
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        # 一个问题中多个答案的累加
                        answer_texts.append(answer_text)
                        answer_span = []
                        # 有点绕智商不够 但是作用是 找到answer 每个在context的出现的位置 即索引
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        # 返回的是在单词层面的索引 起始位置和结束位置
                        # [102, 103, 104]
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    # example 是一个问题的信息
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    # examples 是 整个文件中所有问题的信息
                    examples.append(example)
                    # total 文件中所有问题 的一个累加
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                # line = ", -0.082752 0.67204 -0.14987 -0.064983 0.056491 0.40228 ..." 加上前面的单词或者符号一共301维
                array = line.split()
                # 将每行的的单词或者字符取出 即 ，
                word = "".join(array[0:-vec_size])
                # vector = [-0.082752, 0.67204, -0.14987, -0.064983, 0.056491, 0.40228,...]
                vector = list(map(float, array[-vec_size:]))
                # 给出现次数大于limit的单词用向量表示
                # embedding_dict 存放的是每个词的向量
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        # 字母向量没有使用预训练向量而是使用随机正态分布
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    '''
    token2idx_dict ：单词->索引
    NULL:0
    OOV:1
    word:2
    ...
    embedding_dict : 单词->向量
    NULL:[0,0,0,0,...]
    OOV:[0,0,0,0,,...]
    word:[-0.082752, 0.67204, -0.14987, -0.064983, ...]
    ...
    idx2emb_dict : 索引->向量
    0:[0,0,0,0,...]
    1:[0,0,0,0,...]
    2:[-0.082752, 0.67204, -0.14987, -0.064983, ...]
    ...
    emb_mat : 所有的向量
    [[0,0,0,0,...],
     [0,0,0,0,,...],
     [-0.082752, 0.67204, -0.14987, -0.064983, ...],]
    '''
    # 存放的是单词和索引
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    # 生成一个 300维的0向量 保持和单词的向量维度一样
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


# 构建特征存储到records文件中返回 meta信息（问题的个数）
def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    # 为什么 test 和其他不一样 1000，400
    para_limit = config.test_para_limit if is_test else config.para_limit
    # 100，50
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    # 100，30
    ans_limit = 100 if is_test else config.ans_limit
    # 16
    char_limit = config.char_limit

    # 去掉段落，问题，答案，长度超过设置的
    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        # 每个原文长度限制为para_limit 个单词
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        # 每个原文中每个单词的字母长度限制为char_limit
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        # 每个问题长度限制为ques_limit 个单词
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        # 每个问题中的每个单词字母长度限制为char_limit
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        # 根据单词的不同形式 返回它的索引
        def _get_word(word):
            # capitalize()将字符串的第一个字母变成大写,其他字母变小写
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        # context_idxs 存放的是原文中每个位置对应的单词索引
        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        # ques_idxs 存放的是问题中每个位置对应的单词索引
        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
            "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    # train_examples是整个文件中问题的信息，train_eval是没有处理之前的一些问题的信息
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    '''
    word2idx_dict是 单词->索引 
    NULL:0
    OOV:1
    word:2
    char2idx_dict类似
    '''
    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # 存放的是单词向量 N * 300
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    # 存放的是字母向量 M * 64维度
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    # _eval存放的是原文,spans, 答案 和 uuid
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    # 存放的是 dev 问题个数
    save(config.dev_meta, dev_meta, message="dev meta")
    # 存放的是 test 问题个数
    save(config.test_meta, test_meta, message="test meta")
    # 存放的是 单词->索引 已经去重 {",":2,".":3,"the":4,"and":5,"to":6,"of":7,"a":8,"in":9,"\...}
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    # 存放的是 字母->索引 已经去重： {"A":2,"r":3,"c":4,"h":5,"i":6,"t":7,"e":8,...}
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
