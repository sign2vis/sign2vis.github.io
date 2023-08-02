import json
import numpy as np
import pandas as pd
import os
import datacompy
import sqlite3
def get_val(query, key, key_words):
    try:
        q_list = query.split(' ')
        # print(q_list, key)
        st_id = q_list.index(key) + 1
        ed_id = None
        for i in range(len(q_list) - st_id):
            if q_list[st_id + i] in key_words:
                ed_id = st_id + i
                break
        
        return ' '.join(q_list[st_id:ed_id])
    except:
        return None

def with_acc(pred, gold):
    key_words = ['mark', 'data', 'encoding','x', 'y', 'aggregate', 'color', 'transform', 'filter', 'group', 'sort', 'topk', 'bin']
    pred_sort = get_val(pred, 'sort', key_words[10:])

    # if pred_sort:
    #     pred = pred.replace(' sort ' + pred_sort, '')
    gold_sort = get_val(gold, 'sort', key_words[10:])
    if pred_sort and gold_sort:
        # print(pred_sort, gold_sort)
        pred = pred.replace(' sort ' + pred_sort, ' sort ' + gold_sort)
        # print(pred)
        # print(gold)
    return pred, gold

def vega_zero_to_VQL(query):
    key_words = ['mark', 'data', 'encoding','x', 'y', 'aggregate', 'color', 'transform', 'filter', 'group', 'sort', 'topk', 'bin']
    VQL = 'Visualize'
    chart_type = get_val(query, 'mark', key_words)
    if chart_type == 'arc':
        chart_type = 'pie'
    elif chart_type == 'point':
        chart_type = 'scatter'
    chart_type = chart_type.upper()
    VQL = VQL + ' ' + chart_type + ' SELECT'
    x = get_val(query, 'x', key_words[3:])
    y = get_val(query, 'aggregate', key_words[5:])
    # print(y, query)
    agg = y.split(' ')[0]
    y = ' '.join(y.split(' ')[1:])
    # if x == y:
    #     y = '*'
    if agg == 'mean':
        y = 'AVG' + '(' + y + ')'
    elif agg == 'none':
        y = y
    else:
        y = agg + '(' + y + ')'

    VQL = VQL + ' ' + x + ' , ' + y + ' FROM'
    table_id = get_val(query, 'data', key_words[1:])
    VQL = VQL + ' ' + table_id
    
    fil = get_val(query, 'filter', key_words[8:])
    if fil:
        VQL = VQL + ' WHERE' + ' ' + fil
    color = get_val(query, 'color', key_words[6:])
    group = get_val(query, 'group', key_words[9:])

    g_list = []
    if color:
        g_list.append(color)
    if group:
        g_list.append(x)
    if len(g_list) != 0:
        VQL = VQL + ' GROUP BY'
        VQL = VQL + ' ' + g_list[0]
        if len(g_list) > 1:
            VQL = VQL + ' , ' + g_list[1]
    sort = get_val(query, 'sort', key_words[10:])
    if sort:
        s_list = sort.split(' ')
        if s_list[0] == 'x':
            s_list[0] = x
        elif s_list[0] == 'y':
            s_list[0] = y
        else:
            pass
        VQL = VQL + ' ORDER BY ' + ' '.join(s_list) 
    
    topk = get_val(query, 'topk', key_words[10:])
    if topk:
        VQL = VQL + ' LIMIT ' + topk
    bin = get_val(query, 'bin', key_words[10:])
    if bin:
        b_list = bin.split(' ')
        b_list[0] = x
        VQL = VQL + ' BIN ' + ' '.join(b_list)
    # print(VQL)
    return VQL, chart_type, bin

def get_res(db_id, VQL):
    try:
        ed_id = VQL.split(' ').index('BIN')
    except:
        ed_id = None
    SQL = ' '.join(VQL.split(' ')[2:ed_id])
    db_url = './database/' + f'{db_id}/' + f'{db_id}.sqlite'
    cnx = sqlite3.connect(db_url)
    data = pd.read_sql_query(SQL, cnx)
    return data

def exx_acc(db_id, pred, gold):
    pred_VQL, pred_chart_type, pred_bin = vega_zero_to_VQL(pred)
    gold_VQL, gold_chart_type, gold_bin = vega_zero_to_VQL(gold)
    try:
        pred_res = get_res(db_id, pred_VQL)
        gold_res = get_res(db_id, gold_VQL)
        if pred_chart_type == gold_chart_type and pred_bin == gold_bin and pred_res.equals(gold_res):
            return True
        else:
            return False
    except:
        if pred_VQL == gold_VQL:
            print(pred_VQL, gold_VQL)
            return True
        else:
            return False



from utils.metrics import bleu, chrf, rouge

# def remove_sort(pre, trg):
#     keywords = ['topk', 'bin']
#     pre_list = pre.split(' ')
#     trg_list = trg.split(' ')
#     try:
#         st_id = pre_list.index('sort')
#         ed_id = None
#         for j in range(len(pre_list) - st_id):
            

#     return pre, trg

# def get_new_temp(line):
#     temp = 'mark [T] data [D] encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [S] topk [K] bin [B]'
#     if line.find('color') == -1:
#         temp = temp.replace(' color [Z]', '')
#     if line.find('filter') == -1:
#         temp = temp.replace(' filter [F]', '')
#     if line.find('group') == -1:
#         temp = temp.replace(' group [G]', '')
#     if line.find('sort') == -1:
#         temp = temp.replace(' sort [S]', '')
#     if line.find('topk') == -1:
#         temp = temp.replace(' topk [K]', '')
#     if line.find('bin') == -1:
#         temp = temp.replace(' bin [B]', '')
#     if temp[-9:] == 'transform':
#         temp = temp[:-10]
#     temp = temp.replace('[D]', line.split(' ')[line.split(' ').index('data') + 1])
#     return temp
trg = []
with open('./my_one_data/test.jsonl', 'r', encoding='utf-8') as f:
    for row in f.readlines():
        trg.append(json.loads(row))

# pre = []
# with open('./my_one_data/res_wi.txt', 'r', encoding='utf-8') as f:
#     for row in f.readlines():
#         pre.append(row.strip())


# with open('./my_one_data/new_temp.txt', 'w', encoding='utf-8') as f:
#     for row in pre:
#         f.write(get_new_temp(row) + '\n')

pre = []
with open('./text-davinci-003_new_pre_wi.txt', 'r', encoding='utf-8') as f:
    for row in f.readlines():
        pre.append(row.strip())

t_c = 0 
f_c = 0
t_ec = 0
f_ec = 0

exe_b = 0
exe_p = 0
exe_l = 0
exe_s = 0
exe_sb = 0
exe_gl = 0
exe_gs = 0
cnt_b = 0
cnt_p = 0
cnt_l = 0
cnt_s = 0
cnt_sb = 0
cnt_gl = 0
cnt_gs = 0

res = []
res_trg = []
# print(get_new_temp('mark arc data train encoding x time y aggregate none train_number transform filter destination = \'chennai\''))
for i in range(len(pre)):
    x_list = pre[i].replace('\"', '\'').split(' ')
    if x_list[1] == 'pie':
        x_list[1] = 'arc'
    elif x_list[1] == 'scatter':
        x_list[1] = 'point'
    elif x_list[1] == 'histogram':
        x_list[1] = 'bar'
        pass
    try:
        agg_i = x_list.index('aggregate') + 1
    except:
        ins = x_list.index('y') + 1
        x_list.insert(ins, 'none')
        x_list.insert(ins, 'aggregate')
        agg_i = x_list.index('aggregate') + 1
    if x_list[agg_i] == 'average' or x_list[agg_i] == 'avg':
        x_list[agg_i] = 'mean'
    
    try:
        sort_i = x_list.index('sort') + 1
        x_i = x_list.index('x') + 1
        if x_list[x_i] == x_list[sort_i]:
            x_list[sort_i] = 'x'
    except:
        pass
    
    db_id = trg[i]['db_id']
    chart = trg[i]['chart']
    if chart == 'Bar':
        cnt_b += 1
    elif chart == 'Pie':
        cnt_p += 1
    elif chart == 'Line':
        cnt_l += 1
    elif chart == 'Scatter':
        cnt_s += 1
    elif chart == 'Stacked Bar':
        cnt_sb += 1
    elif chart == 'Grouping Line':
        cnt_gl += 1
    else:
        cnt_gs += 1
    pred = " ".join(x_list)
    gold = " ".join(trg[i]['vega_zero'].replace('\"', '\'').split(' '))
    pred, gold = with_acc(pred, gold)
    res.append(pred)
    res_trg.append(gold)
    # " ".join(pre[i].replace('\"', '\'').split(' ')) == " ".join(trg[i]['vega_zero'].replace('\"', '\'').split(' '))
    # get_new_temp(" ".join(pre[i].replace('\"', '\'').split(' '))) == get_new_temp(" ".join(trg[i]['vega_zero'].replace('\"', '\'').split(' ')))
    if pred == gold:
        t_c += 1
    else:
        # print(" ".join(x_list))
        # print(" ".join(trg[i]['vega_zero'].replace('\"', '\'').split(' ')))
        # print(get_new_temp(" ".join(pre[i].replace('\"', '\'').split(' '))))
        # print(get_new_temp(" ".join(trg[i]['vega_zero'].replace('\"', '\'').split(' '))))
        f_c += 1
    if exx_acc(db_id, pred, gold):
        t_ec += 1
        if chart == 'Bar':
            exe_b += 1
        elif chart == 'Pie':
            exe_p += 1
        elif chart == 'Line':
            exe_l += 1
        elif chart == 'Scatter':
            exe_s += 1
        elif chart == 'Stacked Bar':
            exe_sb += 1
        elif chart == 'Grouping Line':
            exe_gl += 1
        else:
            exe_gs += 1
    else:
        f_ec += 1
print(len(res), len(res_trg))
print(t_c, f_c, t_c / len(pre), f_c / len(pre))
# Calculate BLEU scores
bleus = bleu(references=res_trg, hypotheses=res)
results_wo = {}
results_wo['bleu1'] = bleus['bleu1']
results_wo['bleu2'] = bleus['bleu2']
results_wo['bleu3'] = bleus['bleu3']
results_wo['bleu4'] = bleus['bleu4']
results_wo['chrf'] = chrf(references=res_trg, hypotheses=res)
results_wo['rouge'] = rouge(references=res_trg, hypotheses=res)
# x_vega_zero = 'mark [T] data [D] encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [S] topk [K] bin [B]'

print(results_wo)

print(t_ec, f_ec, t_ec / len(pre), f_ec / len(pre))

print('b', exe_b / (cnt_b))
print('p', exe_p / (cnt_p))
print('l', exe_l / (cnt_l))
print('s', exe_s / (cnt_s))
print('sb', exe_sb / (cnt_sb))
print('gl', exe_gl / (cnt_gl))
print('gs', exe_gs / (cnt_gs))

