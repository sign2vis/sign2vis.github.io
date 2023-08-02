import json
import numpy as np
import pandas as pd
import os

import sqlite3

def split_dataset(json_file_path, out_root, ratio=0.15):
    all_data = []    
    with open(json_file_path, "r", encoding="utf-8") as f:
        for row in f.readlines():
            all_data.append(json.loads(row))
    
    N = len(all_data)
    print('%d in total.' % N)
    
    data_id = [x['id'] for x in all_data]
    data_db_id = [x['db_id'] for x in all_data]
    data_chart_type = [x['chart'] for x in all_data]
    data_hardness = [x['hardness'] for x in all_data]
    data_query = [x['query'] for x in all_data]
    data_question = [x['question'] for x in all_data]
    data_vega_zero = [x['vega_zero'] for x in all_data]


    train_id = []
    train_db_id = []
    train_chart_type = []
    train_hardness = []
    train_query = []
    train_question = []
    train_vega_zero = []

    dev_id = []
    dev_db_id = []
    dev_chart_type = []
    dev_hardness = []
    dev_query = []
    dev_question = []
    dev_vega_zero = []

    test_id = []
    test_db_id = []
    test_chart_type = []
    test_hardness = []
    test_query = []
    test_question = []
    test_vega_zero = []

    idx = np.random.permutation(N)

    for id in idx[:int(N*(1.0-ratio*2))]:
        train_id.append(data_id[id])
        train_db_id.append(data_db_id[id])
        train_chart_type.append(data_chart_type[id])
        train_hardness.append(data_hardness[id])
        train_query.append(data_query[id])
        train_question.append(data_question[id])
        train_vega_zero.append(data_vega_zero[id])
    
    for id in idx[int(N*(1.0-ratio*2)): int(N*(1.0-ratio))]:
        dev_id.append(data_id[id])
        dev_db_id.append(data_db_id[id])
        dev_chart_type.append(data_chart_type[id])
        dev_hardness.append(data_hardness[id])
        dev_query.append(data_query[id])
        dev_question.append(data_question[id])
        dev_vega_zero.append(data_vega_zero[id])
    
    for id in idx[int(N*(1.0-ratio)):]:
        test_id.append(data_id[id])
        test_db_id.append(data_db_id[id])
        test_chart_type.append(data_chart_type[id])
        test_hardness.append(data_hardness[id])
        test_query.append(data_query[id])
        test_question.append(data_question[id])
        test_vega_zero.append(data_vega_zero[id])
    
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    train_data = {
    'tvBench_id': train_id,
    'db_id': train_db_id,
    'chart': train_chart_type,
    'hardness': train_hardness,
    'query': train_query,
    'question': train_question,
    'vega_zero': train_vega_zero
    }
    
    df = pd.DataFrame(train_data)
    df.to_csv(os.path.join(out_root, 'train.csv'), index=None)

    dev_data = {
    'tvBench_id': dev_id,
    'db_id': dev_db_id,
    'chart': dev_chart_type,
    'hardness': dev_hardness,
    'query': dev_query,
    'question': dev_question,
    'vega_zero': dev_vega_zero
    }
    
    df = pd.DataFrame(dev_data)
    df.to_csv(os.path.join(out_root, 'dev.csv'), index=None)
    
    test_data = {
    'tvBench_id': test_id,
    'db_id': test_db_id,
    'chart': test_chart_type,
    'hardness': test_hardness,
    'query': test_query,
    'question': test_question,
    'vega_zero': test_vega_zero
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv(os.path.join(out_root, 'test.csv'), index=None)

    pass

if __name__ == "__main__":
    json_file_path = 'D:/axxwd/SIGN2VIS/nvBench/s2v_data.jsonl'
    out_root = './my_one_data'
    nl_temp = []
    with open("./text-davinci-003_data_final.jsonl", "r", encoding="utf-8") as f:
        for row in f.readlines():
            nl_temp.append(json.loads(row))
    with open("./NVBench_vegazero.json", "r", encoding="utf-8") as f:
        v_content = json.load(f)
    s2v_data = []
    with open("s2v_data.jsonl", 'w', encoding='utf=8') as f:
    # for i in range(1):
        count = 0
        for i,x in enumerate(nl_temp):
            y = x['question']
            # if y.find(' cod') != -1:
            #     count += 1
            # y = y.replace(' cod', ' code')
            # y = y.replace(' from high to low', '')
            # y = y.replace(' from low to high', '')
            # y = y[2:].capitalize()
            id = x['id']
            # print(y)
            now_data = {
            'id':id,
            'db_id': v_content[id]['db_id'],
            'chart': v_content[id]['chart'],
            'hardness': v_content[id]['hardness'],
            'query': v_content[id]['vis_query']['VQL'],
            'question': y,
            'vega_zero': v_content[id]['vega_zero']
            }
            # print(now_data)
            f.write(json.dumps(now_data) + '\n')
        print(count)
    # split_dataset(json_file_path, out_root)
