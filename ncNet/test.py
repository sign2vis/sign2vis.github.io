__author__ = "Yuyu Luo"

'''
This script handles the testing process.
We evaluate the ncNet on the benchmark dataset.
'''
import json
import torch
import torch.nn as nn

from model.VisAwareTranslation import translate_sentence, translate_sentence_with_guidance, postprocessing, get_all_table_columns
from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab
import sqlite3
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

import argparse

from utils.metrics import bleu, chrf, rouge

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
    db_url = './dataset/database/' + f'{db_id}/' + f'{db_id}.sqlite'
    cnx = sqlite3.connect(db_url)
    data = pd.read_sql_query(SQL, cnx)
    return data

def exe_acc(db_id, pred, gold):
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
            # print(pred_VQL, gold_VQL)
            return True
        else:
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test.py')

    parser.add_argument('-model', required=False, default='./save_models/last_model_best_wo.pt',
                        help='Path to model weight file')
    parser.add_argument('-data_dir', required=False, default='./dataset/my_last_data_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('-db_info', required=False, default='./dataset/database_information.csv',
                        help='Path to database tables/columns information, for building vocab')
    parser.add_argument('-test_data', required=False, default='./dataset/sign2text_data_final/test.csv',
                        help='Path to testing dataset, formatting as csv')
    parser.add_argument('-db_schema', required=False, default='./dataset/db_tables_columns.json',
                        help='Path to database schema file, formatting as json')
    parser.add_argument('-db_tables_columns_types', required=False, default='./dataset/db_tables_columns_types.json',
                        help='Path to database schema file, formatting as json')

    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_input_length', type=int, default=150)
    parser.add_argument('-show_progress', required=False, default=False, help='True to show details during decoding')
    opt = parser.parse_args()
    print("the input parameters: ", opt)

    SEED = 1

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, my_max_length = build_vocab(
        data_dir=opt.data_dir,
        db_info=opt.db_info,
        batch_size=opt.batch_size,
        max_input_length=opt.max_input_length
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256  # it equals to embedding dimension
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    print("------------------------------\n| Build encoder of the ncNet ... | \n------------------------------")
    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  TOK_TYPES,
                  my_max_length
                  )

    print("------------------------------\n| Build decoder of the ncNet ... | \n------------------------------")
    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device,
                  my_max_length
                  )

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    print("------------------------------\n| Build the ncNet structure... | \n------------------------------")
    ncNet = Seq2Seq(enc, dec, SRC, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)  # define the transformer-based ncNet

    print("------------------------------\n| Load the trained ncNet ... | \n------------------------------")
    ncNet.load_state_dict(torch.load(opt.model, map_location=device))


    print("------------------------------\n|          Testing  ...      | \n------------------------------")


    db_tables_columns = get_all_table_columns(opt.db_schema)
    db_tables_columns_types = get_all_table_columns(opt.db_tables_columns_types)

    all_true = []
    all_false = []
    wi_true = []

    only_nl_cnt = 0
    only_nl_match = 0
    only_exe_match = 0
    nl_template_cnt = 0
    nl_template_match = 0
    template_exe_match = 0

    cnt_e = 0
    cnt_m = 0
    cnt_h = 0
    cnt_eh = 0
    match_e = 0
    match_m = 0
    match_h = 0
    match_eh = 0
    exe_e = 0
    exe_m = 0
    exe_h = 0
    exe_eh = 0

    cnt_b = 0
    cnt_p = 0
    cnt_l = 0
    cnt_s = 0
    cnt_sb = 0
    cnt_gl = 0
    cnt_gs = 0
    exe_b = 0
    exe_p = 0
    exe_l = 0
    exe_s = 0
    exe_sb = 0
    exe_gl = 0
    exe_gs = 0
    match_b = 0
    match_p = 0
    match_l = 0
    match_s = 0
    match_sb = 0
    match_gl = 0
    match_gs = 0

    test_df = pd.read_csv(opt.test_data)

    res_e = []
    res_m = []
    res_h = []
    res_eh = []
    res_e_trg = []
    res_m_trg = []
    res_h_trg = []
    res_eh_trg = []

    res_wo = []
    res_wi = []
    res_all = []
    res_wo_trg = []
    res_wi_trg = []
    res_all_trg = []

    res_b = []
    res_p = []
    res_l = []
    res_s = []
    res_sb = []
    res_gl = []
    res_gs = []
    res_b_trg = []
    res_p_trg = []
    res_l_trg = []
    res_s_trg = []
    res_sb_trg = []
    res_gl_trg = []
    res_gs_trg = []

    wi_jsonl = []
    wo_jsonl = []
    for index, row in tqdm(test_df.iterrows()):
        wi = False
        wo = False

        chart = row['chart']
        hardness = row['hardness']
        gold_query = row['labels'].lower()

        # if ' '.join(gold_query.replace('"', "'").split()) != 'mark bar data employees encoding x first_name y aggregate mean salary transform filter first_name like \'%m\' group x sort x asc':
        #     continue

        src = row['source']

        # i_list = src.split(' ')
        # i_list[i_list.index('<C>') + 4] = '[D]'
        # src = ' '.join(i_list).lower()
        src = src.lower()
        # print(src)

        tok_types = row['token_types']

        vega_zero_list = gold_query.split(' ')
        st_id = vega_zero_list.index('data') + 1
        ed_id = None
        for j in range(len(vega_zero_list) - st_id):
            if vega_zero_list[st_id + j] == 'encoding':
                ed_id = st_id + j
        table_name = ' '.join(vega_zero_list[st_id:ed_id])


        try:
            translation, attention, enc_attention = translate_sentence_with_guidance(
                row['db_id'], table_name, src, SRC, TRG, TOK_TYPES, tok_types, SRC,
                ncNet, db_tables_columns, db_tables_columns_types, device, my_max_length, show_progress=opt.show_progress
            )
        except: 
            translation, attention, enc_attention = translate_sentence(
                src, SRC, TRG, TOK_TYPES, tok_types,
                ncNet, device, my_max_length
            )

        pred_query = ' '.join(translation).replace(' <eos>', '').lower()
        old_pred_query = pred_query

        # p_list = pred_query.split(' ')
        # p_list[p_list.index('data') + 1] = table_name
        # pred_query = ' '.join(p_list)

        # print(pred_query)
        # print(gold_query)



        res_all_trg.append(' '.join(gold_query.replace('"', "'").split()))

        if '[t]' not in src:
            # with template
            pred_query = postprocessing(gold_query, pred_query, True, src)

            nl_template_cnt += 1
            res_wi.append(' '.join(pred_query.replace('"', "'").split()))
            res_wi_trg.append(' '.join(gold_query.replace('"', "'").split()))
            res_all.append(' '.join(pred_query.replace('"', "'").split()))

            if hardness == 'Easy':
                cnt_e += 1
                res_e.append(' '.join(pred_query.replace('"', "'").split()))
                res_e_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif hardness == 'Medium':
                cnt_m += 1
                res_m.append(' '.join(pred_query.replace('"', "'").split()))
                res_m_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif hardness == 'Hard':
                cnt_h += 1
                res_h.append(' '.join(pred_query.replace('"', "'").split()))
                res_h_trg.append(' '.join(gold_query.replace('"', "'").split()))
            else:
                cnt_eh += 1
                res_eh.append(' '.join(pred_query.replace('"', "'").split()))
                res_eh_trg.append(' '.join(gold_query.replace('"', "'").split()))  
            
            if chart == 'Bar':
                cnt_b += 1
                res_b.append(' '.join(pred_query.replace('"', "'").split()))
                res_b_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif chart == 'Pie':
                cnt_p += 1
                res_p.append(' '.join(pred_query.replace('"', "'").split()))
                res_p_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif chart == 'Line':
                cnt_l += 1
                res_l.append(' '.join(pred_query.replace('"', "'").split()))
                res_l_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif chart == 'Scatter':
                cnt_s += 1
                res_s.append(' '.join(pred_query.replace('"', "'").split()))
                res_s_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif chart == 'Stacked Bar':
                cnt_sb += 1
                res_sb.append(' '.join(pred_query.replace('"', "'").split()))
                res_sb_trg.append(' '.join(gold_query.replace('"', "'").split()))
            elif chart == 'Grouping Line':
                cnt_gl += 1
                res_gl.append(' '.join(pred_query.replace('"', "'").split()))
                res_gl_trg.append(' '.join(gold_query.replace('"', "'").split()))
            else:
                cnt_gs += 1
                res_gs.append(' '.join(pred_query.replace('"', "'").split()))
                res_gs_trg.append(' '.join(gold_query.replace('"', "'").split()))
            
            if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                nl_template_match += 1
                if hardness == 'Easy':
                    match_e += 1
                elif hardness == 'Medium':
                    match_m += 1
                elif hardness == 'Hard':
                    match_h += 1
                else:
                    match_eh += 1

                if chart == 'Bar':
                    match_b += 1
                elif chart == 'Pie':
                    match_p += 1
                elif chart == 'Line':
                    match_l += 1
                elif chart == 'Scatter':
                    match_s += 1
                elif chart == 'Stacked Bar':
                    match_sb += 1
                elif chart == 'Grouping Line':
                    match_gl += 1
                else:
                    match_gs += 1
            else:
                pass

            try:
                exe_acc_res = exe_acc(row['db_id'], ' '.join(pred_query.replace('"', "'").split()), ' '.join(gold_query.replace('"', "'").split()))
            except:
                exe_acc_res = False
            if exe_acc_res:
                wi = True
                template_exe_match += 1
                if hardness == 'Easy':
                    exe_e += 1
                elif hardness == 'Medium':
                    exe_m += 1
                elif hardness == 'Hard':
                    exe_h += 1
                else:
                    exe_eh += 1
                
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
                pass

            wi_jsonl.append({
                'id': row['tvBench_id'],
                'pred': ' '.join(pred_query.replace('"', "'").split()),
                'gold': ' '.join(gold_query.replace('"', "'").split()),
                'match': (' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split())),
                'exe': exe_acc_res,
                'question': row['question'],
                'hardness': row['hardness'],
                'chart': chart
            })
            if row['tvBench_id'] == '509':
                print('wi:', ' '.join(pred_query.replace('"', "'").split()))

        if '[t]' in src:
            # without template
            pred_query = postprocessing(gold_query, pred_query, False, src)

            only_nl_cnt += 1
            res_wo.append(' '.join(pred_query.replace('"', "'").split()))
            res_wo_trg.append(' '.join(gold_query.replace('"', "'").split()))
            res_all.append(' '.join(pred_query.replace('"', "'").split()))
            if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                only_nl_match += 1

            else:
                # print(row['question'])
                # print(row['source'])
                # print(' '.join(pred_query.replace('"', "'").split()))
                # print(' '.join(gold_query.replace('"', "'").split()))
                pass
            try:
                exe_acc_res = exe_acc(row['db_id'], ' '.join(pred_query.replace('"', "'").split()), ' '.join(gold_query.replace('"', "'").split()))
            except:
                exe_acc_res = False
            if exe_acc_res:
                only_exe_match += 1
                wo = True
            else:
                pass

            wo_jsonl.append({
                'id': row['tvBench_id'],
                'pred': ' '.join(pred_query.replace('"', "'").split()),
                'gold': ' '.join(gold_query.replace('"', "'").split()),
                'match': (' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split())),
                'exe': exe_acc_res,
                'question': row['question'],
                'hardness': row['hardness'],
                'chart': chart
            })
            if row['tvBench_id'] == '509':
                print('wo:', ' '.join(pred_query.replace('"', "'").split()))

        # break
        # if index > 100:
        #     break
    
    # Calculate BLEU scores
    bleus = bleu(references=res_wo_trg, hypotheses=res_wo)
    results_wo = {}
    results_wo['bleu1'] = bleus['bleu1']
    results_wo['bleu2'] = bleus['bleu2']
    results_wo['bleu3'] = bleus['bleu3']
    results_wo['bleu4'] = bleus['bleu4']
    results_wo['chrf'] = chrf(references=res_wo_trg, hypotheses=res_wo)
    results_wo['rouge'] = rouge(references=res_wo_trg, hypotheses=res_wo)

    bleus = bleu(references=res_wi_trg, hypotheses=res_wi)
    results_wi = {}
    results_wi['bleu1'] = bleus['bleu1']
    results_wi['bleu2'] = bleus['bleu2']
    results_wi['bleu3'] = bleus['bleu3']
    results_wi['bleu4'] = bleus['bleu4']
    results_wi['chrf'] = chrf(references=res_wi_trg, hypotheses=res_wi)
    results_wi['rouge'] = rouge(references=res_wi_trg, hypotheses=res_wi)

    bleus = bleu(references=res_all_trg, hypotheses=res_all)
    results_all = {}
    results_all['bleu1'] = bleus['bleu1']
    results_all['bleu2'] = bleus['bleu2']
    results_all['bleu3'] = bleus['bleu3']
    results_all['bleu4'] = bleus['bleu4']
    results_all['chrf'] = chrf(references=res_all_trg, hypotheses=res_all)
    results_all['rouge'] = rouge(references=res_all_trg, hypotheses=res_all)
    print("========================================================")
    print('ncNet w/o chart template:', only_nl_match / only_nl_cnt)
    print('ncNet w/o chart template:', only_exe_match / only_nl_cnt)
    print(results_wo)
    print('ncNet with chart template:', nl_template_match / nl_template_cnt)
    print('ncNet with chart template:', template_exe_match / nl_template_cnt)
    print(results_wi)
    print('ncNet overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
    print(results_all)

    results = dict()
    bleus = bleu(references=res_e_trg, hypotheses=res_e)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_e_trg, hypotheses=res_e)
    print(results)
    print('easy:', match_e / cnt_e)
    print('easy:', exe_e / cnt_e)

    bleus = bleu(references=res_m_trg, hypotheses=res_m)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_m_trg, hypotheses=res_m)
    print(results)
    print('medium:', match_m / cnt_m)
    print('medium:', exe_m / cnt_m)

    bleus = bleu(references=res_h_trg, hypotheses=res_h)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_h_trg, hypotheses=res_h)
    print(results)
    print('hard:', match_h / cnt_h)
    print('hard:', exe_h / cnt_h)

    bleus = bleu(references=res_eh_trg, hypotheses=res_eh)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_eh_trg, hypotheses=res_eh)
    print(results)
    print('e hard:', match_eh / cnt_eh)
    print('e hard:', exe_eh / cnt_eh)
    
    bleus = bleu(references=res_b_trg, hypotheses=res_b)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_b_trg, hypotheses=res_b)
    print(results)
    print('Bar:', (match_sb + match_b) / (cnt_sb + cnt_b))
    print('Bar:', (exe_sb + exe_b) / (cnt_sb + cnt_b))

    bleus = bleu(references=res_p_trg, hypotheses=res_p)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_p_trg, hypotheses=res_p)
    print(results)
    print('Pie:', match_p / cnt_p)
    print('Pie:', exe_p / cnt_p)

    bleus = bleu(references=res_l_trg, hypotheses=res_l)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_l_trg, hypotheses=res_l)
    print(results)
    print('Line:', (match_gl + match_l) / (cnt_gl + cnt_l))
    print('Line:', (exe_gl + exe_l) / (cnt_gl + cnt_l))

    bleus = bleu(references=res_s_trg, hypotheses=res_s)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_s_trg, hypotheses=res_s)
    print(results)
    print('Scatter:', (match_gs + match_s) / (cnt_gs + cnt_s))
    print('Scatter:', (exe_gs + exe_s) / (cnt_gs + cnt_s))

    bleus = bleu(references=res_sb_trg, hypotheses=res_sb)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_sb_trg, hypotheses=res_sb)
    print(results)
    print('Stacked Bar:', match_sb / cnt_sb)
    print('Stacked Bar:', exe_sb / cnt_sb)

    bleus = bleu(references=res_gl_trg, hypotheses=res_gl)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_gl_trg, hypotheses=res_gl)
    print(results)
    print('Grouping Line:', match_gl / cnt_gl)
    print('Grouping Line:', exe_gl / cnt_gl)

    bleus = bleu(references=res_gs_trg, hypotheses=res_gs)
    results['bleu4'] = bleus['bleu4']
    results['rouge'] = rouge(references=res_gs_trg, hypotheses=res_gs)
    print(results)
    print('Grouping Scatter:', match_gs / cnt_gs)
    print('Grouping Scatter:', exe_gs / cnt_gs)

    # with open('./test_res/res_wo.txt', 'w', encoding='utf-8') as f:
    #     for x in res_wo:
    #         f.write(x + '\n')
    
    # with open('./test_res/res_wi.txt', 'w', encoding='utf-8') as f:
    #     for x in res_wi:
    #         f.write(x + '\n')
    
    # with open('./test_res/res_trg.txt', 'w', encoding='utf-8') as f:
    #     for x in res_wi_trg:
    #         f.write(x + '\n')
    
    with open('./test_res/wi_no.jsonl', 'w', encoding='utf-8') as f:
        for x in wi_jsonl:
            f.write(json.dumps(x) + '\n')

    with open('./test_res/wo_no.jsonl', 'w', encoding='utf-8') as f:
        for x in wo_jsonl:
            f.write(json.dumps(x) + '\n')
