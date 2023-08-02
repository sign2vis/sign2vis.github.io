import os
import openai
import time
import json
import pickle
import sqlite3
import re
import numpy as np
import pandas as pd
from nltk import word_tokenize
from gensim import corpora, models, similarities
VALUE_NUM_SYMBOL = "{value}"
TEMP = 'mark [T] data [D] encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [S] topk [K] bin [B]'
key = ""
openai.proxy = ""

def get_gpt_ans(model, prompt, key):
    # print(key)
    try:
        openai.api_key = key
        completion = openai.Completion.create(
                engine=model,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                prompt=prompt,
                max_tokens = 220
                )
        # print(completion)
        ans = completion.choices[0].text.strip()

        # print(completion)
    except:
        print(key)
        time.sleep(22)
        return get_gpt_ans(model, prompt, key)
    return ans

def get_new_temp(line, temp = TEMP):
    if line.find('color') == -1:
        temp = temp.replace(' color [Z]', '')
    if line.find('filter') == -1:
        temp = temp.replace(' filter [F]', '')
    if line.find('group') == -1:
        temp = temp.replace(' group [G]', '')
    if line.find('sort') == -1:
        temp = temp.replace(' sort [S]', '')
    if line.find('topk') == -1:
        temp = temp.replace(' topk [K]', '')
    if line.find('bin') == -1:
        temp = temp.replace(' bin [B]', '')
    if temp[-9:] == 'transform':
        temp = temp[:-10]
    temp = temp.replace('[D]', line.split(' ')[line.split(' ').index('data') + 1])
    return temp

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
        # if nl_keywords[i].find('_') != -1:
        #     nl_x = []
        #     for v in nl_keywords[i].split('_'):
        #         if d.check(v):
        #             nl_x.append(v)
        #     nl_keywords[i] = ' '.join(nl_x)
        if nl_keywords[i] == VALUE_NUM_SYMBOL:
            nl_keywords[i] = values[j].strip()
            j += 1

    return nl_keywords

def get_table(db_id, table_name, db_tables_columns_types):
    table_str = table_name + ' (\n'
    types = db_tables_columns_types[db_id][table_name]
    for col in types.keys():
        table_str += col + ', ' + types[col] + '\n'
    table_str += ')'
    return table_str

def get_first_example(db_tables_columns_types):
    example_data = ""

    db_id = "wine_1"
    table_name = "wine"
    table = get_table(db_id, table_name, db_tables_columns_types)
    query = "Bar chart showing the total number of wines whose price is greater than 100, grouped by year."
    g = "mark bar data wine encoding x year y aggregate count year color grape transform filter price > 100 group x sort x"
    temp = "mark bar data wine encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [X] topk [K] bin [B]"
    example_data += f"Table: {table}, \n"
    example_data += f"Sentence: {query}, \n"
    example_data += f"Template: {get_new_temp(g, temp)} \n"
    example_data += f"Result: {g} \n"

    db_id = "hr_1"
    table_name = "employees"
    table = get_table(db_id, table_name, db_tables_columns_types)
    query = "Bar chart showing the sum of employee id for employees whose salary is in the range of 8000 and 12000 and commission is not null or department number does not equal to 40 by hire date in descending order of employee id and binning by month."
    g = "mark bar data employees encoding x hire_date y aggregate sum employee_id transform filter salary between 8000 and 12000 and commission_pct != \"null\" or department_id != 40 sort y desc bin x by month"
    temp = "mark bar data employees encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] sort [Y] desc topk [K] bin [B]"
    example_data += f"Table: {table}, \n"
    example_data += f"Sentence: {query}, \n"
    example_data += f"Template: {get_new_temp(g, temp)} \n"
    example_data += f"Result: {g} \n"
    return example_data

        
def get_example(question, train_questions, train_results, db_tables_columns_types):

    dictionary = corpora.Dictionary([text.split() for text in train_questions])

    corpus = [dictionary.doc2bow(text.split()) for text in train_questions]

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    query_bow = dictionary.doc2bow(question.split())
    query_tfidf = tfidf[query_bow]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    sims = index[query_tfidf]

    top_indexes = sims.argsort()[-1:][::-1]
    example_data = ""
    for i in top_indexes:
        # print(i+1)
        query = train_questions[i].strip()
        g, db_id = train_results[i].strip().split('\t')
        table_name = g.split(' ')[g.split(' ').index('data') + 1]

        # table_names = []
        # from_tables = re.findall(r' FROM\s+([^\s,(]+)', g, re.IGNORECASE)
        # join_tables = re.findall(r' JOIN\s+([^\s,]+)', g, re.IGNORECASE)
        # table_names.extend(from_tables)
        # table_names.extend(join_tables)
        # table_names = [table_name.rstrip(')') for table_name in table_names]
        # table_names = [table_name.lstrip('(') for table_name in table_names]
        # table_names = list(set(table_names))
        
        table = get_table(db_id, table_name, db_tables_columns_types)
        example_data += f"Table: {table}, \n"
        example_data += f"Sentence: {query}, \n"
        example_data += f"Template: {get_new_temp(g)} \n"
        example_data += f"Result: {g} \n"
    return example_data

def get_prompt(now_data, train_questions, train_results, db_tables_columns_types):
    now_db_id = now_data['db_id']
    vega_zero_list = now_data['vega_zero'].split(' ')
    now_table_name = vega_zero_list[vega_zero_list.index('data') + 1]
    now_table = get_table(now_db_id, now_table_name, db_tables_columns_types)
    now_question = ' '.join(strip_nl(now_data['question']))
    temp = get_new_temp(now_data['vega_zero'], now_data['query_template'])
    prompt = ""
    prompt += get_first_example(db_tables_columns_types)
    prompt += get_example(now_question, train_questions, train_results, db_tables_columns_types)
    prompt += f"Table: {now_table}, \n"
    prompt += f"Sentence: {now_question}, \n"
    prompt += f"Template: {temp} \n"
    prompt += "Result: "
    return prompt


if __name__ == '__main__':
    pass
    with open("./db_tables_columns_types.json", "r", encoding="utf-8") as f:
        db_tables_columns_types = json.load(f)
    # print(get_table('architecture', 'mill', db_tables_columns_types))

    # train_data = []
    # with open("./my_one_data/train.jsonl", "r", encoding="utf-8") as f:
    #     for row in f.readlines():
    #         train_data.append(json.loads(row))
    # test_data = []
    # with open("./my_one_data/test.jsonl", "r", encoding="utf-8") as f:
    #     for row in f.readlines():
    #         test_data.append(json.loads(row))

    # new_temp = []
    # with open("./my_one_data/new_temp.txt", "r", encoding='utf-8') as f:
    #     for row in f.readlines():
    #         new_temp.append(row.strip())

    train_data = []
    with open("./my_last_data/train.jsonl", "r", encoding="utf-8") as f:
        for row in f.readlines():
            train_data.append(json.loads(row))

    with open("./my_last_data/dev.jsonl", "r", encoding="utf-8") as f:
        for row in f.readlines():
            train_data.append(json.loads(row))
    
    test_data = []
    df = pd.read_csv("./sign2text_data_final/test.csv")
    for index, row in df.iterrows():
        if index % 2 == 1:
            test_data.append(row)
    # test_data = []
    # with open("./sign2text_data/test.jsonl", "r", encoding="utf-8") as f:
    #     for row in f.readlines():
    #         test_data.append(json.loads(row))

    # new_temp = []
    # with open("./my_last_data/new_temp.txt", "r", encoding='utf-8') as f:
    #     for row in f.readlines():
    #         new_temp.append(row.strip())
    
    train_questions = []
    for x in train_data:
        que = x['question']
        train_questions.append(' '.join(strip_nl(que)))
    
    train_results = []
    for x in train_data:
        res = x['vega_zero']
        train_results.append(res + '\t' + x['db_id'])
    
    # print(get_first_example(db_tables_columns_types))
    
    for i,x in enumerate(test_data):
        # if i != 565:
        #     continue

        # print(x['question'], new_temp[i], sep='\n')
        prompt = get_prompt(x, train_questions, train_results, db_tables_columns_types)
        # print(prompt)
        # break
        model = 'text-davinci-003'
        
        ans = get_gpt_ans(model, prompt, key)
        
        print(ans)

        with open(f"./{model}_new_pre_wi.txt", "a", encoding="utf-8") as f:
            f.write(ans + '\n')
