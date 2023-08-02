import os, sys, argparse, re, json
import pandas as pd

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sign2vis.utils.utils_sign2vis import *
from sign2vis.model.slt import *
from sign2vis.utils.metrics import bleu, chrf, rouge

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument('-data_dir', required=False, default='./ncNet/dataset/my_last_data_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('-with_temp', required=False, default=1,
                        help='Which template to use, 0:empty, 1:fill, 2:all')
    
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_infer', default=False, action='store_true')
    parser.add_argument('--infer_loop', default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')

    parser.add_argument('--tepoch', default=50, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    # parser.add_argument('--fine_tune',
    #                     default=False,
    #                     action='store_true',
    #                     help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='SLT1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int,  # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=3, type=int, help="The number of Transformer layers in Encoder or Decoder.")
    parser.add_argument('--dr', default=0.1, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=256, type=int, help="The dimension of hidden vector in the SLTModel.")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


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

def get_models(args, BERT_PT_PATH, trained=False, path_model=None):
    print(f"Batch_size = {args.batch_size * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: False")
    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)

    # Get SLT model
    # model = SLTModel(model_bert.embeddings, args.hS, args.lr, args.lS)
    model = SLTModel(model_bert.embeddings, args.hS, args.dr, args.lS)
    model = model.to(device)

    if trained:
        assert path_model != None

        if torch.cuda.is_available():
            print(f"load model's checkpoint")
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])
    
    return model, tokenizer

def get_data(sign_path, args):
    data = []
    csv_path = args.data_dir
    bS = args.batch_size
    with_temp = args.with_temp
    
    each = 'test.csv'
    df = pd.read_csv(os.path.join(csv_path, each))
    now_data = []
    for index, row in df.iterrows():
        id = row['tvBench_id']
        que = row['question']
        src = row['source']
        trg = row['labels']
        tok_types = row['token_types']
        video_path = os.path.join(sign_path, id + '.npy')
        if with_temp == 0 and src.find('[T]') == -1:
            continue
        elif with_temp == 1 and src.find('[T]') != -1:
            continue
        if not os.path.exists(video_path):
            print(id, que)
            continue


        now_data.append({'id':id,
                        'question':que,
                        'src':src,
                        'trg':trg,
                        'tok_types':tok_types,
                        'video_path':video_path
        })



    test_data = now_data
    print(len(test_data))
    # test_data = test_data[::16]
    test_loader = get_loader_sign2text_test(test_data, bS, shuffle_test=False)
    return test_data, test_loader

def test(data_loader, model, tokenizer, 
         st_pos=0, beam_size=5):
    model.eval()
    
    cnt = 0
    all_text_pred = []
    all_text_trg = []
    for iB, t in enumerate(data_loader):
    
        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, videos = get_fields_sign2text(t)
        # nlu : natural language
        # videos : video npys
        
        video_array, video_array_mask = get_padded_batch_video(videos)
        if video_array == None:
            continue
        input_text_array, output_text_array, text_mask_array = get_input_output_token(tokenizer, nlu)
        # video_array: [B, T_video, 144, 144]
        # input_text_array: [B, T_text]

        # Inference SLT Beam
        stacked_txt_output = inference_slt_beam(model, tokenizer, video_array, video_array_mask, beam_size)
        texts = []
        pad_id = 0
        for ts in stacked_txt_output:
            if pad_id in ts:
                ids = ts[:ts.tolist().index(pad_id)][:-1]  # remove <PAD> & <EOS>
            else:
                ids = ts[:-1]  # remove <EOS>
            tokens = tokenizer.convert_ids_to_tokens(ids)
            cur = tokenizer.concat_tokens(tokens)
            texts.append(cur)
        all_text_pred += texts
        # print(texts[0])
        text_trg = []
        output_text_array, text_mask_array = output_text_array.cpu(), text_mask_array.cpu()
        for i in range(output_text_array.size(0)):
            ids = output_text_array[i, text_mask_array[i]][:-1]  # remove <PAD> & <EOS>
            tokens = tokenizer.convert_ids_to_tokens(ids)
            cur = tokenizer.concat_tokens(tokens)
            text_trg.append(cur)
        all_text_trg += text_trg
    
    # Calculate BLEU scores
    bleus = bleu(references=all_text_trg, hypotheses=all_text_pred)
    results = {}
    results['bleu1'] = bleus['bleu1']
    results['bleu2'] = bleus['bleu2']
    results['bleu3'] = bleus['bleu3']
    results['bleu4'] = bleus['bleu4']
    results['chrf'] = chrf(references=all_text_trg, hypotheses=all_text_pred)
    results['rouge'] = rouge(references=all_text_trg, hypotheses=all_text_pred)
    return results, all_text_pred, all_text_trg


def print_result(print_data, dname):
    print(f'{dname} results ------------')
    print(print_data)


def save_text_pred_dev(save_root, all_text_pred, dname):
    with open(os.path.join(save_root, dname), 'w') as f:
        for text in all_text_pred:
            f.write(text+'\n')


if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    BERT_PT_PATH = '/mnt/silver/zsj/data/sign2sql/model/annotated_wikisql_and_PyTorch_bert_param'
    path_sign2text = '/mnt/silver/guest/zgb/MySign2Vis/new_npy_data'

    path_save_for_evaluation = './sign2text_test'
    if not os.path.exists(path_save_for_evaluation):
        os.mkdir(path_save_for_evaluation)
    
    path_pretrained_slt_model = './sign2text_save'
    
    ## 3. Load data

    test_data, test_loader = get_data(path_sign2text, args)

    ## 4. Build & Load models
    
    if args.trained:
        # To start from the pre-trained models, un-comment following lines.
        path_model_slt = os.path.join(path_pretrained_slt_model, 'new_slt_model_best.pt')
        model, tokenizer = get_models(args, BERT_PT_PATH, 
                                      trained=True, path_model=path_model_slt)
    else:
        # error
        pass
    printTime()
    ## 5. Test
    with torch.no_grad():
        results, all_text_pred, all_text_trg = test(test_loader,
                                          model,
                                          tokenizer,
                                          st_pos=0,
                                          beam_size=5)
    printTime()
    print_result(results, 'Sign to Text')

    save_text_pred_dev(path_save_for_evaluation, all_text_pred, 'sign2text_all_text_pred.txt')
    save_text_pred_dev(path_save_for_evaluation, all_text_trg, 'sign2text_all_text_trg.txt')

