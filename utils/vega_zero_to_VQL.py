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


if __name__ == '__main__':
    df = pd.read_csv('./my_last_data_final/test.csv')
    for index, row in df.iterrows():
        vega_zero = row['vega_zero']
        db_id = row['db_id']
        print(exx_acc(db_id, vega_zero, vega_zero))
        