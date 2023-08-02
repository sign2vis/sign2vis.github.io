import json
import numpy as np
import pandas as pd
import os

import sqlite3

def split_dataset(json_file_path, out_root, ratio=0.15):
    with open(json_file_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    N = len(all_data.keys())
    print('%d in total.' % N)

    data_nl2vis = list(all_data.values())

    data_id = list(all_data.keys())
    data_db_id = [x['db_id'] for x in data_nl2vis]
    data_chart_type = [x['chart'] for x in data_nl2vis]
    data_hardness = [x['hardness'] for x in data_nl2vis]
    data_query = [x['vis_query']['VQL'] for x in data_nl2vis]
    data_question = [x['nl_queries'][0] for x in data_nl2vis]
    data_vega_zero = [x['vega_zero'] for x in data_nl2vis]


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
with open("./NVBench_short.json", "r", encoding="utf-8") as f:
    content = json.load(f)

key_words = ['select', 'from', 'where', 'group', 'having', 'order', 'limit', 'bin']

data_nl2vis = list(content.values())
key_nl2vis = list(content.keys())
# print(len(data_nl2vis))
# print(len(key_nl2vis))
data_db_id = [x['db_id'] for x in data_nl2vis]
data_chart_type = [x['chart'] for x in data_nl2vis]
data_hardness = [x['hardness'] for x in data_nl2vis]
data_query = [x['vis_query']['VQL'] for x in data_nl2vis]
data_xname = [x['vis_obj']['x_name'].lower() for x in data_nl2vis]
data_yname = [x['vis_obj']['y_name'].lower() for x in data_nl2vis]
data_nl = [x['nl_queries'][0] for x in data_nl2vis]
data_vega_zero = []
is_use = []
count = 0
count1 = 0
count2 = 0
count3 = 0
for i,x in enumerate(data_query):
    is_x_use = True
    # print(x)
    x_list = x.lower().split(' ')
    # print(x_list)
    x_vega_zero = 'mark [T] data [D] encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [S] topk [K] bin [B]'
    # ----------[T]
    if x_list[1] == 'pie':
        x_vega_zero = x_vega_zero.replace('[T]', 'arc')
    elif x_list[1] == 'scatter':
        x_vega_zero = x_vega_zero.replace('[T]', 'point')
    else:
        x_vega_zero = x_vega_zero.replace('[T]', x_list[1])

    st_id = x_list.index('from') + 1

    ed_id = None
    for j in range(len(x_list)-st_id):
        if x_list[st_id + j] in key_words:
            ed_id = st_id + j
            break

    # ----------[D]
    is_tables = False
    table_list = []
    for j in x_list[st_id:ed_id]:
        if j != '':
            table_list.append(j)
    if len(table_list) > 1:
        is_tables = True
    
    table_name = " ".join(table_list)
    # print('./database/'+data_db_id[i]+'/'+data_db_id[i]+'.sqlite')
    cnx = sqlite3.connect('./database/'+data_db_id[i]+'/'+data_db_id[i]+'.sqlite')
    try:
        data = pd.read_sql_query("SELECT * FROM " + table_name, cnx)
    except:
        is_x_use = False
        count3 += 1
    x_vega_zero = x_vega_zero.replace('[D]', table_name)
    # print(data_xname[i], data_yname[i])
    # ----------[X] [AggFunction] [Y]
    x_vega_zero = x_vega_zero.replace('[X]', data_xname[i].lower())

    x_name = data_xname[i].lower()
    y_name = data_yname[i].lower()
    
    if x_name.find('(') != -1:
        is_x_use = False
        count1 += 1
    ind = data_yname[i].find('(')

    if ind != -1:
        # print(data_yname[i][:ind].lower())
        if data_yname[i][:ind].lower() == 'avg':
            x_vega_zero = x_vega_zero.replace('[AggFunction]', 'mean')
        else:
            x_vega_zero = x_vega_zero.replace('[AggFunction]', data_yname[i][:ind].lower())
        end = data_yname[i].find(')')
        y_name = data_yname[i][ind+1:end].lower()
        if y_name == '*':
            x_vega_zero = x_vega_zero.replace('[Y]', data_xname[i].lower())
        else:
            x_vega_zero = x_vega_zero.replace('[Y]', y_name)
    else:
        x_vega_zero = x_vega_zero.replace('[AggFunction]', 'none')
        x_vega_zero = x_vega_zero.replace('[Y]', data_yname[i].lower())
    # ----------[Z] [G]
    try:
        gid = x_list.index('group')
        g_st_id = gid + 2
        g_ed_id = None
        for j in range(len(x_list)-g_st_id):
            if x_list[g_st_id + j] in key_words:
                g_ed_id = g_st_id + j
                break
        # print(g_st_id, g_ed_id)
        group_list = []
        for group_name in x_list[g_st_id:g_ed_id]:
            if group_name != '' and group_name != ',':
                group_list.append(group_name)
        # is_print = False
        for g_name in group_list:
            print(g_name, data_yname[i])
            if g_name.find('.') != -1:
                g_name = g_name[g_name.find('.') + 1:]
            if g_name == data_xname[i]:
                x_vega_zero = x_vega_zero.replace('[G]', 'x')
            elif g_name == data_yname[i]:
                x_vega_zero = x_vega_zero.replace('[G]', 'y')
            else:
                x_vega_zero = x_vega_zero.replace('[Z]', g_name)
                # is_print = True
        if x_vega_zero.find('[Z]') != -1:
            x_vega_zero = x_vega_zero.replace(' color [Z]', '')
        if x_vega_zero.find('[G]') != -1:
            x_vega_zero = x_vega_zero.replace(' group [G]', '')
        # if is_print:
        #     print(x_vega_zero)
    except:
        x_vega_zero = x_vega_zero.replace(' group [G]', '')
        x_vega_zero = x_vega_zero.replace(' color [Z]', '')
    # ----------[F]
    try:
        wid = x_list.index('where')
        w_st_id = wid + 1
        w_ed_id = None
        for j in range(len(x_list)-w_st_id):
            if x_list[w_st_id + j] in key_words:
                w_ed_id = w_st_id + j
                break
        where_list = []
        for j in x_list[w_st_id:w_ed_id]:
            if is_tables and j.find('.') != -1:
                where_list.append(j[j.find('.') + 1:])
            elif j != '':
                where_list.append(j)
        for j in where_list:
            if j.find('(') != -1:
                is_x_use = False
                count2 += 1
        x_vega_zero = x_vega_zero.replace('[F]', ' '.join(where_list))
        # print(' '.join(where_list))
        # if ' '.join(x_list[w_st_id:w_ed_id]).find('(') != -1:
        #     count += 1
        pass
    except:
        x_vega_zero = x_vega_zero.replace(' filter [F]', '')
        # print('no filter')
        pass
    # ----------[B]
    try:
        bid = x_list.index('bin')
        b_st_id = bid + 1
        b_ed_id = None
        for j in range(len(x_list)-b_st_id):
            if x_list[b_st_id + j] in key_words:
                b_ed_id = b_st_id + j
                break
        bin_list = []
        for j in x_list[b_st_id:b_ed_id]:
            if j.find('.') != -1:
                bin_list.append(j[j.find('.') + 1:])
            else:
                bin_list.append(j)
        if bin_list[0] == x_name:
            bin_list[0] = 'x'
        elif bin_list[0] == y_name:
            bin_list[0] = 'y'
        else:
            pass
        x_vega_zero = x_vega_zero.replace('[B]', ' '.join(bin_list))
        pass
    except:
        x_vega_zero = x_vega_zero.replace(' bin [B]', '')
        pass

    # ----------[S]
    try:
        oid = x_list.index('order')
        o_st_id = oid + 2
        o_ed_id = None
        for j in range(len(x_list)-o_st_id):
            if x_list[o_st_id + j] in key_words:
                o_ed_id = o_st_id + j
                break
        sort_list = []
        for j in x_list[o_st_id:o_ed_id]:
            if j.find('.') != -1:
                sort_list.append(j[j.find('.') + 1:])
            else:
                sort_list.append(j)
        if sort_list[0] == data_xname[i]:
            sort_list[0] = 'x'
        elif sort_list[0] == data_yname[i]:
            sort_list[0] = 'y'
        # print(' '.join(sort_list))
        x_vega_zero = x_vega_zero.replace('[S]', ' '.join(sort_list))
        pass
    except:
        x_vega_zero = x_vega_zero.replace(' sort [S]', '')
    
    # ----------[K]
    try:
        lid = x_list.index('limit')
        l_st_id = lid + 1
        x_vega_zero = x_vega_zero.replace('[K]', x_list[l_st_id])
    except:
        x_vega_zero = x_vega_zero.replace(' topk [K]', '')
    
    if x_vega_zero[-9:] == 'transform':
        x_vega_zero = x_vega_zero[:-10]
    data_vega_zero.append(x_vega_zero)
    is_use.append(is_x_use)
    # print(x_vega_zero)

for i in is_use:
    if i :
        count += 1
print(count, count1, count2, count3)

id_to_vega_zero = dict()
for each in ['train.csv', 'dev.csv', 'test.csv']:
    df = pd.read_csv('D:/fwq/Sevi/dataset/' + each)
    for index, row in df.iterrows():
        # print(row['tvBench_id'], row['vega_zero'])
        id_to_vega_zero[row['tvBench_id']] = row['vega_zero']
print(len(list(id_to_vega_zero.keys())))
print(len(is_use))

csv_id = []
csv_db_id = []
csv_chart = []
csv_hardness = []
csv_query = []
csv_question = []
csv_vega_zero = []

for i in range(len(is_use)):
    if is_use[i]:
        if key_nl2vis[i] not in id_to_vega_zero.keys():
            # print(key_nl2vis[i], data_vega_zero[i])
            pass
        else:
            if id_to_vega_zero[key_nl2vis[i]] != data_vega_zero[i]:
                print(id_to_vega_zero[key_nl2vis[i]], data_vega_zero[i], sep='\n')
        csv_id.append(key_nl2vis[i])
        csv_db_id.append(data_db_id[i])
        csv_chart.append(data_chart_type[i])
        csv_hardness.append(data_hardness[i])
        csv_query.append(data_query[i])
        csv_question.append(data_nl[i])
        csv_vega_zero.append(data_vega_zero[i])
print(len(csv_id))

new_content = dict()
for i,x in enumerate(csv_id):
    new_value = content[x]
    new_value['vega_zero'] = csv_vega_zero[i]
    # print(new_value['nl_queries'][0])
    # print(csv_question[i])
    new_que = []
    for y in csv_question[i].split(' '):
        if y != '' and y!= ' ':
            new_que.append(y)
    new_que = ' '.join(new_que)
    # print(new_que)
    new_value['nl_queries'][0] = new_que
    new_content[x] = new_value
print(len(new_content.keys()))
with open("./NVBench_vegazero.json", "w", encoding="utf-8") as f:
    json.dump(new_content, f, indent=True)

