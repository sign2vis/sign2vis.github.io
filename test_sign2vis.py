import torch
import torch.nn as nn

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
from sign2vis.model.VisAwareTranslation import translate_s2v, postprocessing, get_all_table_columns
from sign2vis.utils.utils_sign2vis import *
from sign2vis.model.slt import SLTModel
from sign2vis.model.sign2vis import *
from sign2vis.model.Encoder import AllEncoder
from sign2vis.model.Decoder import Decoder
from ncNet.preprocessing.build_vocab import build_vocab_only
from sign2vis.utils.metrics import bleu, chrf, rouge
import sqlite3
import numpy as np
import random
import time
import math
import os
# import matplotlib.pyplot as plt
import pandas as pd
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    db_url = '/mnt/silver/guest/zgb/MySign2Vis/ncNet/dataset/database/' + f'{db_id}/' + f'{db_id}.sqlite'
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
            print(pred_VQL, gold_VQL)
            return True
        else:
            return False

def get_data(sign_path, args):
    csv_path = args.data_dir
    bS = args.batch_size
    with_temp = args.with_temp

    df = pd.read_csv(os.path.join(csv_path, 'test.csv'))
    now_data = []
    for index, row in df.iterrows():
        id = row['tvBench_id']
        que = row['question']
        src = row['source']
        trg = row['labels']
        tok_types = row['token_types']
        db_id = row['db_id']
        hardness = row['hardness']
        chart = row['chart']
        video_path = os.path.join(sign_path, id + '.npy')
                
        if with_temp == 0 and src.find('[T]') == -1:
            continue
        elif with_temp == 1 and src.find('[T]') != -1:
            continue
        else:
            pass
        if not os.path.exists(video_path):
            # print(id, que)
            continue
        now_data.append({'id':id,
                         'question':que,
                         'src':src,
                         'trg':trg,
                         'tok_types':tok_types,
                         'video_path':video_path,
                         'db_id': db_id, 
                         'hardness': hardness,
                         'chart': chart
                         })
    test_data = now_data


    # test_data = test_data[::100]

    print(len(test_data))
    test_loader = get_loader_sign2text_test(test_data, bS)
    return test_data, test_loader

def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def evaluate(model, data_loader, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            _, videos = get_fields_sign2text(batch)
            video_array, video_array_mask = get_padded_batch_video(videos)
            
            src, trg, tok_types = get_fields_text(batch, video_array.shape[1])
            src, trg, tok_types = get_text_input(src, trg, tok_types, SRC, TRG, TOK_TYPES)

            output, _ = model(video_array, video_array_mask, src, trg[:, :-1], tok_types, SRC)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=False, default='./ncNet/dataset/my_last_data_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('--with_temp', type=int, required=False, default=2,
                        help='Which template to use, 0:empty, 1:fill, 2:all')
    parser.add_argument('--db_info', required=False, default='./ncNet/dataset/database_information.csv',
                        help='Path to database tables/columns information, for building vocab')
    parser.add_argument('--output_dir', type=str, default='./sign2vis_v2_save/')

    parser.add_argument('--epoch', type=int, default=100,
                        help='the number of epoch for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--max_input_length', type=int, default=150)
    parser.add_argument('--dr_enc', default=0.1, type=float, help="Dropout rate.")

    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    
    parser.add_argument("--trained", default=True, action='store_true')

    args = parser.parse_args()
    
    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    BERT_PT_PATH = '/mnt/silver/zsj/data/sign2sql/model/annotated_wikisql_and_PyTorch_bert_param'
    SLT_PT_PATH = '/mnt/silver/guest/zgb/MySign2Vis/sign2text_save/new_slt_model_best.pt'
    S2V_PT_PATH = '/mnt/silver/guest/zgb/MySign2Vis/sign2vis_v2_save/s2v_wo_model_best.pt'
    path_sign2text = '/mnt/silver/guest/zgb/MySign2Vis/new_npy_data'
    
    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, my_max_length =  build_vocab_only(
        data_dir=args.data_dir,
        db_info=args.db_info,
        max_input_length=args.max_input_length
    )

    test_data, test_loader = get_data(path_sign2text, args)
    db_tables_columns = get_all_table_columns("/mnt/silver/guest/zgb/MySign2Vis/ncNet/dataset/db_tables_columns.json")
    db_tables_columns_types = get_all_table_columns("/mnt/silver/guest/zgb/MySign2Vis/ncNet/dataset/db_tables_columns_types.json")
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256 # it equals to embedding dimension
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    print("------------------------------\n| Build encoder of the text ... | \n------------------------------")
    enc = AllEncoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  TOK_TYPES,
                  800
                 )
    print("------------------------------\n| Build decoder of the model ... | \n------------------------------")
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

    print("------------------------------\n| Build the pretrained slt... | \n------------------------------")
    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    
    # Get SLT model
    model_slt = SLTModel(model_bert.embeddings, HID_DIM, 1e-4, 3)  # TODO: train sign2text with dropout 0.1
    model_slt = model_slt.to(device)
    # load pretrained slt pt !!!
    if torch.cuda.is_available():
        res = torch.load(SLT_PT_PATH)
    else:
        res = torch.load(SLT_PT_PATH, map_location='cpu')
    model_slt.load_state_dict(res['model'])

    print("------------------------------\n| Build the sign2vis model... | \n------------------------------")
    model_s2v = Sign2vis_ts_Model(model_slt, enc, dec, INPUT_DIM, SRC, SRC_PAD_IDX, TRG_PAD_IDX, HID_DIM, device, args.dr_enc)
    model_s2v = model_s2v.to(device)
    if args.trained:
        if torch.cuda.is_available():
            res = torch.load(S2V_PT_PATH)
        else:
            res = torch.load(S2V_PT_PATH, map_location='cpu')
        model_s2v.load_state_dict(res)
        print('load pretrained s2v')

    print("------------------------------\n| Testing start ... | \n------------------------------")
    model_s2v.eval()
    res_wo = []
    res_wi = []
    res_all = []
    res_wo_trg = []
    res_wi_trg = []
    res_all_trg = []
    res_e = []
    res_m = []
    res_h = []
    res_eh = []
    res_e_trg = []
    res_m_trg = []
    res_h_trg = []
    res_eh_trg = []
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

    only_nl_cnt = 0
    only_nl_match = 0
    only_nl_exe = 0

    nl_template_cnt = 0
    nl_template_match = 0
    nl_template_exe = 0

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
    match_b = 0
    match_p = 0
    match_l = 0
    match_s = 0
    match_sb = 0
    match_gl = 0
    match_gs = 0
    exe_b = 0
    exe_p = 0
    exe_l = 0
    exe_s = 0
    exe_sb = 0
    exe_gl = 0
    exe_gs = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            _, videos = get_fields_sign2text(batch)
            video_array, video_array_mask = get_padded_batch_video(videos)
            
            src, trg, tok_types = get_fields_text(batch, video_array.shape[1])
            sentence = src[0].lower()
            gold_query = trg[0]
            src, trg, tok_types = get_text_input(src, trg, tok_types, SRC, TRG, TOK_TYPES)
            db_id, table_id = get_db_table(batch)
            hardness = batch[0]['hardness']
            chart = batch[0]['chart']
            # for src1 in src:
            #     print(src1)
            # p = []
            # for x in src1:
            #     p.append(SRC.vocab.itos[x])
            # print(p)
            # for trg1 in trg:
            #     print(trg1)
            # p = []
            # for x in trg1:
            #     p.append(TRG.vocab.itos[x])
            # print(p)            
            # print(db_id, table_id)
            enc_src, src_mask = model_s2v.forward_encoder(video_array, video_array_mask, src, trg[:, :-1], tok_types, SRC)
            src = sentence
            translation = translate_s2v(enc_src, src_mask, db_id, table_id, sentence, TRG, model_s2v,
                                      db_tables_columns, db_tables_columns_types, device, max_len=150, show_progress = False)
        

            pred_query = ' '.join(translation).replace(' <eos>', '').lower()
            old_pred_query = pred_query

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
                    exe_acc_res = exe_acc(db_id, ' '.join(pred_query.replace('"', "'").split()), ' '.join(gold_query.replace('"', "'").split()))
                except:
                    exe_acc_res = False
                if exe_acc_res:
                    nl_template_exe += 1
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
                    'id': batch[0]['id'],
                    'pred': ' '.join(pred_query.replace('"', "'").split()),
                    'gold': ' '.join(gold_query.replace('"', "'").split()),
                    'match': (' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split())),
                    'exe': exe_acc_res,
                    'question': batch[0]['question'],
                    'hardness': hardness,
                    'chart': chart
                })

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
                    pass

                try:
                    exe_acc_res = exe_acc(db_id, ' '.join(pred_query.replace('"', "'").split()), ' '.join(gold_query.replace('"', "'").split()))
                except:
                    exe_acc_res = False
                if exe_acc_res:
                    only_nl_exe += 1
                
                wo_jsonl.append({
                    'id': batch[0]['id'],
                    'pred': ' '.join(pred_query.replace('"', "'").split()),
                    'gold': ' '.join(gold_query.replace('"', "'").split()),
                    'match': (' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split())),
                    'exe': exe_acc_res,
                    'question': batch[0]['question'],
                    'hardness': hardness,
                    'chart': chart
                })


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

    # bleus = bleu(references=res_all_trg, hypotheses=res_all)
    # results_all = {}
    # results_all['bleu1'] = bleus['bleu1']
    # results_all['bleu2'] = bleus['bleu2']
    # results_all['bleu3'] = bleus['bleu3']
    # results_all['bleu4'] = bleus['bleu4']
    # results_all['chrf'] = chrf(references=res_all_trg, hypotheses=res_all)
    # results_all['rouge'] = rouge(references=res_all_trg, hypotheses=res_all)
    print("========================================================")
    print('ncNet w/o chart template:', only_nl_match / only_nl_cnt)
    print('ncNet w/o chart template:', only_nl_exe / only_nl_cnt)
    print(results_wo)
    print('ncNet with chart template:', nl_template_match / nl_template_cnt)
    print('ncNet with chart template:', nl_template_exe / nl_template_cnt)
    print(results_wi)

    with open('./test_res/wi_s2v.jsonl', 'w', encoding='utf-8') as f:
        for x in wi_jsonl:
            f.write(json.dumps(x) + '\n')

    with open('./test_res/wo_s2v.jsonl', 'w', encoding='utf-8') as f:
        for x in wo_jsonl:
            f.write(json.dumps(x) + '\n')

    # print('ncNet overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
    # print(results_all)
    
    # results = dict()
    # bleus = bleu(references=res_e_trg, hypotheses=res_e)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_e_trg, hypotheses=res_e)
    # print(results)
    # print('easy:', match_e / cnt_e)
    # print('easy:', exe_e / cnt_e)

    # bleus = bleu(references=res_m_trg, hypotheses=res_m)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_m_trg, hypotheses=res_m)
    # print(results)
    # print('medium:', match_m / cnt_m)
    # print('medium:', exe_m / cnt_m)

    # bleus = bleu(references=res_h_trg, hypotheses=res_h)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_h_trg, hypotheses=res_h)
    # print(results)
    # print('hard:', match_h / cnt_h)
    # print('hard:', exe_h / cnt_h)

    # bleus = bleu(references=res_eh_trg, hypotheses=res_eh)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_eh_trg, hypotheses=res_eh)
    # print(results)
    # print('e hard:', match_eh / cnt_eh)
    # print('e hard:', exe_eh / cnt_eh)

    # bleus = bleu(references=res_b_trg, hypotheses=res_b)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_b_trg, hypotheses=res_b)
    # print(results)
    # print('Bar:', match_b / cnt_b)
    # print('Bar:', exe_b / cnt_b)

    # bleus = bleu(references=res_p_trg, hypotheses=res_p)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_p_trg, hypotheses=res_p)
    # print(results)
    # print('Pie:', match_p / cnt_p)
    # print('Pie:', exe_p / cnt_p)

    # bleus = bleu(references=res_l_trg, hypotheses=res_l)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_l_trg, hypotheses=res_l)
    # print(results)
    # print('Line:', match_l / cnt_l)
    # print('Line:', exe_l / cnt_l)

    # bleus = bleu(references=res_s_trg, hypotheses=res_s)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_s_trg, hypotheses=res_s)
    # print(results)
    # print('Scatter:', match_s / cnt_s)
    # print('Scatter:', exe_s / cnt_s)

    # bleus = bleu(references=res_sb_trg, hypotheses=res_sb)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_sb_trg, hypotheses=res_sb)
    # print(results)
    # print('Stacked Bar:', match_sb / cnt_sb)
    # print('Stacked Bar:', exe_sb / cnt_sb)

    # bleus = bleu(references=res_gl_trg, hypotheses=res_gl)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_gl_trg, hypotheses=res_gl)
    # print(results)
    # print('Grouping Line:', match_gl / cnt_gl)
    # print('Grouping Line:', exe_gl / cnt_gl)

    # bleus = bleu(references=res_gs_trg, hypotheses=res_gs)
    # results['bleu4'] = bleus['bleu4']
    # results['rouge'] = rouge(references=res_gs_trg, hypotheses=res_gs)
    # print(results)
    # print('Grouping Scatter:', match_gs / cnt_gs)
    # print('Grouping Scatter:', exe_gs / cnt_gs)