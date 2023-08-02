import os, json
import random as rd
from copy import deepcopy
import re
from matplotlib.pylab import *
# from torchtext.data import Example
try:
    from torchtext.data import Example
except:
    pass
from tqdm import tqdm

import torch
import torch.utils.data
# import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F

import time 
from time import strftime, localtime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data -----------------------------------------------------------------------------------------------
def load_sign2sql_test(path_sign2sql):
    # Get data
    test_data = load_sign2text_data(path_sign2sql, mode='test')

    # Get table
    path_table = os.path.join(path_sign2sql, 'all.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1
    
    return test_data, table

def get_loader_sign2text_test(data_test, bS, shuffle_test=False):
    test_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_test,
        shuffle=shuffle_test,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return test_loader

def process_dataset(processor, data, tables):
    processed_dataset = []
    
    for idx, entry in tqdm(enumerate(data)):
        # if idx > 3:
        #     continue
        entry = processor.pipeline(entry, tables[entry['table_id']])


        processed_dataset.append(entry)
    return processed_dataset


# Load data -----------------------------------------------------------------------------------------------
def load_sign2text(path_sign2text):
    # Get data
    train_data = load_sign2text_data(path_sign2text, mode='train')
    dev_data = load_sign2text_data(path_sign2text, mode='dev')

    # Get table
    path_table = os.path.join(path_sign2text, 'all.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1
    
    return train_data, dev_data, table

def load_sign2text_data(path_sign2text, mode='train'):
    path_sql = os.path.join(path_sign2text, mode+'_tok.jsonl')
    data = []
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            if os.path.exists(t1['video_path']):
                data.append(t1)
    return data

def load_text2sql_v2(path_wikisql):
    # Get data
    train_data = load_text2sql_data_v2(path_wikisql, mode='train')
    dev_data = load_text2sql_data_v2(path_wikisql, mode='dev')

    # Get table
    path_table = os.path.join(path_wikisql, 'all.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1
    
    return train_data, dev_data, table

def load_text2sql_data_v2(path_wikisql, mode='train'):
    path_sql = os.path.join(path_wikisql, mode+'_tok.jsonl')
    data = []
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            data.append(t1)
    return data

def get_loader_sign2text(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )
    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader

def get_fields_sign2text_1(t1):
    # print(t1)
    nlu1 = t1['question']
    vid_path = t1['video_path']
    # vid_path = '/mnt/silver/zsj/data/sign2sql/dataset/length3_preprocessed/0.npy'
    video = np.load(vid_path)
    # if video.shape[0] == 0:
    #     print(nlu1)
    return nlu1, video

def get_fields_sign2text(t1s):
    nlu = []
    videos = []
    for t1 in t1s:
        nlu1, video = get_fields_sign2text_1(t1)
        
        nlu.append(nlu1)
        videos.append(video)
    return nlu, videos

def get_padded_batch_video(videos, max_vid_len=600):
    # sample rate 4 or 5 frame
    bS = len(videos)
    video_downsampled = []
    vid_lens = []
    vid_shape = None
    for vid in videos:
        if vid.shape[0] == 0:
            return None, None
        tmp = vid[::5]  # or 4
        video_downsampled.append(tmp)
        vid_lens.append(tmp.shape[0])
        if vid_shape is None:
            vid_shape = (tmp.shape[1], tmp.shape[2])
    video_array = np.zeros([bS, min(max(vid_lens), max_vid_len), vid_shape[0], vid_shape[1]])
    video_array_mask = np.zeros([bS, min(max(vid_lens), max_vid_len)])
    for b in range(bS):
        video_array[b, :min(vid_lens[b], max_vid_len)] = video_downsampled[b][:max_vid_len]
        video_array_mask[b, :min(vid_lens[b], max_vid_len)] = 1
    video_array = torch.from_numpy(video_array).type(torch.float32).to(device)
    video_array_mask = torch.from_numpy(video_array_mask==1).to(device)
    return video_array, video_array_mask

def get_db_table(t1s):
    t1 = t1s[0]
    db_id = t1['db_id']
    vega_zero_list = t1['trg'].split(' ')
    table_id = vega_zero_list[vega_zero_list.index('data') + 1]
    return db_id, table_id
def get_fields_text(t1s, vid_shape):
    src = []
    trg = []
    tok_types = []
    for t1 in t1s:
        src1 = t1['src']
        nl_len = len(re.findall('<N>.*</N>', src1)[0].split(' '))
        src1 = "<N> </N> " + " ".join(src1.split(' ')[nl_len:])
        src.append(src1)
        trg.append(t1['trg'])
        tok_types.append(("nl " * (vid_shape + 2)) + " ".join(t1['tok_types'].split(' ')[nl_len:]))
    return src, trg, tok_types

def get_fields_text1(t1s):
    src = []
    trg = []
    tok_types = []
    for t1 in t1s:
        src1 = t1['src']
        nl_len = len(re.findall('<N>.*</N>', src1)[0].split(' '))
        src1 = "<N> </N> " + " ".join(src1.split(' ')[nl_len:])
        src.append(src1)
        trg.append(t1['trg'])
        tok_types.append("nl nl " + " ".join(t1['tok_types'].split(' ')[nl_len:]))
    return src, trg, tok_types

def get_text_input(src, trg, tok_types, SRC, TRG, TOK_TYPES):
    fileds = [('src', SRC), ('trg', SRC), ('tok_types', TOK_TYPES)]
    examples = []
    for src1, trg1, tok_types1 in zip(src, trg, tok_types):
        example = Example.fromlist([src1, trg1, tok_types1], fields=fileds)
        examples.append(example)
    src = SRC.process([example.src for example in examples]).to(device)
    trg = TRG.process([example.trg for example in examples]).to(device)
    tok_types = TOK_TYPES.process([example.tok_types for example in examples]).to(device)

    return src, trg, tok_types

def get_input_output_token(tokenizer, nlu):
    # tokenizer: BERT tokenizer (subword)
    bS = len(nlu)
    input_text = []
    output_text = []
    text_lens = []
    for nlu1 in nlu:
        tokens = []
        tokens.append("[CLS]")
        tokens += tokenizer.tokenize(nlu1)
        tokens.append("[SEP]")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_text.append(ids[:-1])
        output_text.append(ids[1:])
        text_lens.append(len(ids)-1)
    input_text_array = []
    output_text_array = []
    text_mask_array = np.zeros([bS, max(text_lens)])
    for b in range(bS):
        input_text_array.append(
            input_text[b] + [0] * (max(text_lens) - text_lens[b])
        )
        output_text_array.append(
            output_text[b] + [0] * (max(text_lens) - text_lens[b])
        )
        text_mask_array[b, :text_lens[b]] = 1
    input_text_array = torch.tensor(input_text_array, dtype=torch.long, device=device)
    output_text_array = torch.tensor(output_text_array, dtype=torch.long, device=device)
    text_mask_array = torch.from_numpy(text_mask_array==1).to(device)
    return input_text_array, output_text_array, text_mask_array


def get_input_output_where_ids(tokenizer, sql):
    # sql: list of dict [{'conds':[[2, 0, 'ABC ABC'],[...]]}, {}]
    bS = len(sql)
    input_where_ids = []
    output_where_ids = []
    where_ids_lens = []
    for sql_i in sql:
        tokens = []
        conds = sql_i['conds']
        for cond in conds:
            tokens.append("[CLS]")
            tokens.append(str(cond[0]))
            tokens.append(str(cond[1]))
            tokens += tokenizer.tokenize(str(cond[2]))
        if len(tokens) == 0:
            tokens.append("[CLS]")  # no cond found!
        tokens.append("[SEP]")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_where_ids.append(ids[:-1])
        output_where_ids.append(ids[1:])
        where_ids_lens.append(len(ids)-1)
    
    input_where_array = []
    output_where_array = []
    where_mask_array = np.zeros([bS, max(where_ids_lens)])
    for b in range(bS):
        input_where_array.append(
            input_where_ids[b] + [0] * (max(where_ids_lens) - where_ids_lens[b])
        )
        output_where_array.append(
            output_where_ids[b] + [0] * (max(where_ids_lens) - where_ids_lens[b])
        )
        where_mask_array[b, :where_ids_lens[b]] = 1
    input_where_array = torch.tensor(input_where_array, dtype=torch.long, device=device)
    output_where_array = torch.tensor(output_where_array, dtype=torch.long, device=device)
    where_mask_array = torch.from_numpy(where_mask_array==1).to(device)
    return input_where_array, output_where_array, where_mask_array


def get_input_output_where_value_ids(tokenizer, sql):
    # sql: list of dict [{'conds':[[2, 0, 'ABC ABC'],[...]]}, {}]
    bS = len(sql)
    input_where_value_ids = []
    output_where_value_ids = []
    where_value_ids_lens = []
    for sql_i in sql:
        tokens = []
        conds = sql_i['conds']
        for cond in conds:
            tokens.append("[CLS]")
            tokens += tokenizer.tokenize(str(cond[2]))
        if len(tokens) == 0:
            tokens.append("[CLS]")  # no cond found!
        tokens.append("[SEP]")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_where_value_ids.append(ids[:-1])
        output_where_value_ids.append(ids[1:])
        where_value_ids_lens.append(len(ids)-1) # -1
    
    input_where_value_array = []
    output_where_value_array = []
    where_value_mask_array = np.zeros([bS, max(where_value_ids_lens)])
    for b in range(bS):
        input_where_value_array.append(
            input_where_value_ids[b] + [0] * (max(where_value_ids_lens) - where_value_ids_lens[b])
        )
        output_where_value_array.append(
            output_where_value_ids[b] + [0] * (max(where_value_ids_lens) - where_value_ids_lens[b])
        )
        where_value_mask_array[b, :where_value_ids_lens[b]] = 1
    input_where_value_array = torch.tensor(input_where_value_array, dtype=torch.long, device=device)
    output_where_value_array = torch.tensor(output_where_value_array, dtype=torch.long, device=device)
    where_value_mask_array = torch.from_numpy(where_value_mask_array==1).to(device)
    return input_where_value_array, output_where_value_array, where_value_mask_array


def printTime():

    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

    return