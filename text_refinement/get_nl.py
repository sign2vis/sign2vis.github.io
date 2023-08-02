import json
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, errors='replace', line_buffering=True)
import re
import os
import openai
import time
import numpy as np
from nltk import word_tokenize
import enchant

d = enchant.Dict('en_US')

key_words = ['mark', 'encoding', 'filter', 'group', 'sort', 'topk', 'bin']

VALUE_NUM_SYMBOL = "{value}"

def choose_short(nl_list):
    nl_list.sort(key=lambda x:len(x.split(' ')), reverse=False)
    ans = None
    for y in nl_list:
        if (y != "") and (y[0] != ",") and (y[0] != "\u539f"):
                ans = y.strip('\n')
                break
    
    return ans

def strip_nl(nl):
    '''
    return keywords of nl query
    '''
    nl_keywords = []
    nl = nl.strip()
    nl = nl.replace(";"," ; ").replace(",", " , ").replace("?", " ? ").replace("\t"," ")
    nl = nl.replace("(", " ( ").replace(")", " ) ")
    
    str_1 = re.findall("\"[^\"]*\"", nl)
    str_2 = re.findall("\'[^\']*\'", nl)
    float_nums = re.findall("[-+]?\d*\.\d+", nl)
    
    values = str_1 + str_2 + float_nums
    for val in values:
        nl = nl.replace(val.strip(), VALUE_NUM_SYMBOL)
        # print(val)
    
    
    raw_keywords = nl.strip().split()
    for tok in raw_keywords:
        if "." in tok:
            to = tok.replace(".", " . ").split()
            to = [t.lower() for t in to if len(t)>0]
            nl_keywords.extend(to)
        elif "'" in tok and tok[0]!="'" and tok[-1]!="'":
            to = word_tokenize(tok)
            to = [t.lower() for t in to if len(t)>0]
            nl_keywords.extend(to)      
        elif len(tok) > 0:
            nl_keywords.append(tok.lower())
    j = 0
    for i in range(len(nl_keywords)):
        if nl_keywords[i].find('_') != -1:
            nl_x = []
            for v in nl_keywords[i].split('_'):
                if d.check(v):
                    nl_x.append(v)
            nl_keywords[i] = ' '.join(nl_x)
        if nl_keywords[i] == VALUE_NUM_SYMBOL:
            nl_keywords[i] = values[j].strip()
            j += 1

    return nl_keywords

def get_table_id(query_list):
    st_id = query_list.index('data') + 1
    ed_id = None
    for j in range(len(query_list) - st_id):
        if query_list[st_id + j] == 'encoding':
            ed_id = st_id + j
            break
    table_id = ' '.join(query_list[st_id:ed_id])
    return table_id

def get_fil(query_list):
    try:
        f_st_id = query_list.index('filter') + 1
        f_ed_id = None
        for j in range(len(query_list) - f_st_id):
            if query_list[f_st_id + j] in key_words:
                f_ed_id = f_st_id + j
                break
        return ' '.join(query_list[f_st_id:f_ed_id])
    except:
        return None


with open("./NVBench.json", "r", encoding="utf-8") as f:
    content = json.load(f)

with open("./NVBench_vegazero.json", "r", encoding="utf-8") as f:
    v_content = json.load(f)

key_nl2vis = list(v_content.keys())
count = 0
yy = []

id_list = []
vega_zero_list = []
for i,x in enumerate(key_nl2vis):
    vega_zero = v_content[x]['vega_zero']
    query_list = vega_zero.lower().split(' ')
    table_id = get_table_id(query_list)
    if len(table_id.split(' ')) > 1:
        continue
    vega_zero_list.append(vega_zero)
    id_list.append(x)
    count += 1
    # print(x,':')
    # for y in content[x]['nl_queries']:
    #     print(y)
    # print(vega_zero)
    if vega_zero.find('bin') != -1 and vega_zero.find('sort') != -1:
        yy += content[x]['nl_queries']
# print(count)

'A [T] chart showing the [AGG] of [Y] for each [X] with [F] in [S] order of [SC] with the top [K] [KC] and group by [C] and bin by [B]'
'A [T] chart showing [N] with [F] in [S] order of [SC] with the top [K] [KC] and group by [C] and bin by [B]'
count = 0
max_len = 0
nl_temp = []
for i,x in enumerate(vega_zero_list):
    x_list = x.split(' ')
    x_nl ='A [T] chart showing [N] with [F] in [S] order of [SC] with the top [K] [KC] and binning by [B], grouped by [C]'
    
    chart_type = x_list[x_list.index('mark') + 1]
    if chart_type == 'arc':
        chart_type = 'pie'
    if chart_type == 'point':
        chart_type = 'scatter'
    x_nl = x_nl.replace('[T]', chart_type)

    x_axis = x_list[x_list.index('x') + 1]
    y_axis = x_list[x_list.index('y') + 3]

    if x_axis == y_axis:
        pass
    
    x_t = True
    y_t = True

    x_s = x_axis.split('_')
    y_s = y_axis.split('_')

    for tok in x_s:
        if not d.check(tok):
            x_t = False
    
    for tok in y_s:
        if not d.check(tok):
            y_t = False

    x_axis = ' '.join(x_s)
    y_axis = ' '.join(y_s)

    # x_nl = x_nl.replace('[X]', x_axis)
    # x_nl = x_nl.replace('[Y]', y_axis)

    agg = x_list[x_list.index('aggregate') + 1]
    # if agg == 'none':
    #     x_nl = x_nl.replace(' [AGG] of', '')
    # elif agg == 'max':
    #     x_nl = x_nl.replace('[AGG]', 'maximum')
    # elif agg == 'min':
    #     x_nl = x_nl.replace('[AGG]', 'minimal')
    # elif agg == 'count':
    #     x_nl = x_nl.replace('[AGG]', 'amount')
    # else:
    #     x_nl = x_nl.replace('[AGG]', agg)
    
    fil = get_fil(x_list)
    if fil:
        fil = ' '.join(fil.split('_'))
        # x_nl = x_nl.replace('[F]', '[N]')
        x_nl = x_nl.replace(' with [F]', '')
    else:
        x_nl = x_nl.replace(' with [F]', '')
    
    try:
        top_k = x_list[x_list.index('topk') + 1]
    except:
        top_k = None
    
    sor = None
    col_s = None
    try:
        s_st_id = x_list.index('sort') + 1
        s_ed_id = None
        for j in range(len(x_list) - s_st_id):
            if x_list[s_st_id + j] in key_words:
                s_ed_id = s_st_id + j
                break
        col_s, sor = x_list[s_st_id:s_ed_id]
    except:
        col_s, sor = None, None
    # print(col_s, sor)
    if top_k:
        x_nl = x_nl.replace(' in [S] order of [SC]', '')
        x_nl = x_nl.replace('[K]', top_k)
        if col_s == 'x' and x_t:
            x_nl = x_nl.replace('[KC]', x_axis)
        elif col_s == 'y' and y_t:
            x_nl = x_nl.replace('[KC]', y_axis)
        else:
            pass
        x_nl = x_nl.replace('[KC]', '[N]')
    else:
        x_nl = x_nl.replace(' with the top [K] [KC]', '')
        if sor:
            if sor == 'asc':
                x_nl = x_nl.replace('[S]', 'ascending')
            else:
                x_nl = x_nl.replace('[S]', 'descending')
            if col_s == 'x' and x_t:
                x_nl = x_nl.replace('[SC]', x_axis)
            elif col_s == 'y' and y_t:
                if x_axis == y_axis:
                    # print(id_list[i])
                    x_nl = x_nl.replace('[SC]', 'the y-axis')
                elif agg != 'none':
                    x_nl = x_nl.replace('[SC]', 'the y-axis')
                else:
                    x_nl = x_nl.replace('[SC]', y_axis)
            else:
                pass
            x_nl = x_nl.replace('[SC]', '[N]')
        else:
            x_nl = x_nl.replace(' in [S] order of [SC]', '')
    
    try:
        z_t = True
        col_z = x_list[x_list.index('color') + 1]
        z_s = col_z.split('_')
        for tok in z_s:
            if not d.check(tok):
                z_t = False

        col_z = ' '.join(z_s)
    except:
        col_z = None
    
    if col_z:
        # x_nl = x_nl.replace('[C]', '[N]')
        if z_t:
            x_nl = x_nl.replace('[C]', col_z)
        else:
            x_nl = x_nl.replace('[C]', '[N]')
        pass
    else:
        x_nl = x_nl.replace(', grouped by [C]', '')
    
    try:
        b_st_id = x_list.index('bin') + 1
        b_ed_id = None
        for j in range(len(x_list) - b_st_id):
            if x_list[b_st_id + j] in key_words:
                b_ed_id = b_st_id + j
                break
        col_b, _, bin = x_list[b_st_id:b_ed_id]
    except:
        col_b, _, bin = None, None, None
    
    if bin:
        x_nl = x_nl.replace('[B]', bin)
    else:
        x_nl = x_nl.replace(' and binning by [B]', '')
    
    x_nl = x_nl[2:].capitalize()
    # print(id_list[i])
    # print(content[id_list[i]]['nl_queries'])
    all_nl = content[id_list[i]]['nl_queries']
    all_nl.sort(key=lambda x:len(x.split(' ')), reverse=False)
    if col_s == 'y' and y_t and x_axis != y_axis and agg != 'none':
        nl_temp.append({'id':id_list[i],
                    'Sentence':' '.join(strip_nl(choose_short(all_nl))),
                    'Template':x_nl})
    # print(x)
    count += len(x_nl.split(' '))
    max_len = max(max_len, len(x_nl.split(' ')))

# print(count / len(vega_zero_list), max_len)

# with open("nl_temp.jsonl", 'w', encoding='utf=8') as f:
#     for x in nl_temp:
#         print(x)
#         json.dump(x, f)
#         f.write('\n')

with open("nl.jsonl", 'w', encoding='utf=8') as f:
    for x in nl_temp:
        print(x)
        f.write(json.dumps(x) + '\n')