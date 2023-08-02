import os
import pandas as pd
import json
import re
sign2text_path = '/mnt/silver/guest/zgb/Sign2Vis/sign2text_test/sign2text_all_text_pred.txt'

def get_token_types(input_source):
    token_types = ''

    for ele in re.findall('<N>.*</N>', input_source)[0].split(' '):
        token_types += ' nl'

    for ele in re.findall('<C>.*</C>', input_source)[0].split(' '):
        token_types += ' template'
        
    table_num = len(re.findall('<D>.*</D>', input_source)[0].split(' ')) - len(re.findall('<COL>.*</COL>', input_source)[0].split(' ')) - len(re.findall('<VAL>.*</VAL>', input_source)[0].split(' ')) - 1

    token_types += ' table' * table_num

    for ele in re.findall('<COL>.*</COL>', input_source)[0].split(' '):
        token_types += ' col'

    for ele in re.findall('<VAL>.*</VAL>', input_source)[0].split(' '):
        token_types += ' value'

    token_types += ' table'

    token_types = token_types.strip()
    return token_types

que = []
with open(sign2text_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res = line.strip()
        res = res.replace(' .', '.')
        res = res.replace(' - ', '-')
        que.append(res)


test_path = '/mnt/silver/guest/zgb/Sign2Vis/ncNet/dataset/my_last_data'
test_df = pd.read_csv(os.path.join(test_path, 'test.csv'))

for index, row in test_df.iterrows():
    # print(row['question'], que[index].capitalize(), sep='\n')
    row['question'] = que[index].capitalize()
    # vega_zero_list = row['vega_zero'].split(' ')
    # table_name = vega_zero_list[vega_zero_list.index('data') + 1]
    # col_names = str(row['mentioned_columns']) if str(row['mentioned_columns']) != 'nan'  else ''
    # value_names = str(row['mentioned_values']) if str(row['mentioned_values']) != 'nan'  else ''

    # input_source = '<N> ' + row[
    #                     'question'] + ' </N>' + ' <C> ' + row['query_template'] + ' </C> ' + '<D> ' + table_name + ' <COL> ' + col_names + ' </COL>' + ' <VAL> ' + value_names + ' </VAL> </D>'
    # token_types = get_token_types(input_source)
    # row['source'] = input_source
    # row['token_types'] = token_types
    # print(row['source'])
    # row['question'] = que[int(index/2)]

test_df.to_csv('./test.csv', index=False)

# for index, row in test_df.iterrows():
#     # print(row['question'], que[int(index/2)], sep='\n')
#     row['question'] = que[int(index/2)]
#     vega_zero_list = row['vega_zero'].split(' ')
#     table_name = vega_zero_list[vega_zero_list.index('data') + 1]
#     col_names = str(row['mentioned_columns']) if str(row['mentioned_columns']) != 'nan'  else ''
#     value_names = str(row['mentioned_values']) if str(row['mentioned_values']) != 'nan'  else ''

#     input_source = '<N> ' + row[
#                         'question'] + ' </N>' + ' <C> ' + row['query_template'] + ' </C> ' + '<D> ' + table_name + ' <COL> ' + col_names + ' </COL>' + ' <VAL> ' + value_names + ' </VAL> </D>'
#     token_types = get_token_types(input_source)
#     row['source'] = input_source
#     row['token_types'] = token_types
#     print(row['source'])
#     # row['question'] = que[int(index/2)]

# test_df.to_csv('./test.csv', index=False)