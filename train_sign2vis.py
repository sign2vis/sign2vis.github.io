import torch
import torch.nn as nn

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel

from sign2vis.utils.utils_sign2vis import *
from sign2vis.model.slt import SLTModel
from sign2vis.model.sign2vis import *
from sign2vis.model.Encoder import AllEncoder
from sign2vis.model.Decoder import Decoder
from ncNet.preprocessing.build_vocab import build_vocab_only


import numpy as np
import random
import time
import math
import os
# import matplotlib.pyplot as plt
import pandas as pd
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(sign_path, args):
    data = []
    csv_path = args.data_dir
    bS = args.batch_size
    with_temp = args.with_temp

    for each in ['train.csv', 'dev.csv']:
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
                                 'video_path':video_path
                })
            data.append(now_data)


    train_data, dev_data = data
    # train_data = train_data[::16]
    # dev_data = dev_data[::16]
    print(len(train_data), len(dev_data))
    train_loader, dev_loader = get_loader_sign2text(train_data, dev_data, bS, shuffle_train=True)
    return train_data, dev_data, train_loader, dev_loader

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

def train(model, data_loader, optimizer, criterion, clip, SRC, TRG, TOK_TYPES, accumulate_gradients=1):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        # print(batch)
        _, videos = get_fields_sign2text(batch)
        video_array, video_array_mask = get_padded_batch_video(videos)
        # print(video_array.shape, video_array_mask.shape)
        src, trg, tok_types = get_fields_text(batch, video_array.shape[1])
        # print(src)
        src, trg, tok_types = get_text_input(src, trg, tok_types, SRC, TRG, TOK_TYPES)
        # print(src.shape, trg.shape, tok_types.shape)

        optimizer.zero_grad()

        output, _ = model(video_array, video_array_mask, src, trg[:, :-1], tok_types, SRC)

        # print('output', output, output.shape)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        # loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # optimizer.step()
        
                # Calculate gradient
        if i % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if accumulate_gradients == 1:
                optimizer.step()
        elif i % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        epoch_loss += loss.item()
 
    return epoch_loss / len(data_loader)

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
    parser.add_argument('--data_dir', required=False, default='./ncNet/dataset/my_new_last_data_final/',
                        help='Path to dataset for building vocab')
    parser.add_argument('--with_temp', type=int, required=False, default=2,
                        help='Which template to use, 0:empty, 1:fill, 2:all')
    parser.add_argument('--db_info', required=False, default='./ncNet/dataset/database_information.csv',
                        help='Path to database tables/columns information, for building vocab')
    parser.add_argument('--output_dir', type=str, default='./sign2vis_v2_save/')

    parser.add_argument('--epoch', type=int, default=100,
                        help='the number of epoch for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--max_input_length', type=int, default=150)
    parser.add_argument('--dr_enc', default=0.1, type=float, help="Dropout rate.")

    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    
    parser.add_argument("--trained", default=False, action='store_true')

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
    NC_PT_PATH = '/mnt/silver/guest/zgb/MySign2Vis/ncNet/save_models/model_best.pt'
    S2V_PT_PATH = '/mnt/silver/guest/zgb/MySign2Vis/sign2vis_v2_save/s2v_all_model_best.pt'
    path_sign2text = '/mnt/silver/guest/zgb/MySign2Vis/new_npy_data'
    
    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, my_max_length =  build_vocab_only(
        data_dir=args.data_dir,
        db_info=args.db_info,
        max_input_length=args.max_input_length
    )

    train_data, dev_data, train_loader, dev_loader = get_data(path_sign2text, args)
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
    model_s2v = Sign2vis_Model(model_slt, enc, dec, INPUT_DIM, SRC, SRC_PAD_IDX, TRG_PAD_IDX, HID_DIM, device, args.dr_enc)
    model_s2v = model_s2v.to(device)
    if args.trained:
        if torch.cuda.is_available():
            res = torch.load(S2V_PT_PATH)
        else:
            res = torch.load(S2V_PT_PATH, map_location='cpu')
        model_s2v.load_state_dict(res)
        
        # encoder_dict = model_s2v.encoder.state_dict()
        # pretrained_encoder_dict = {k: v for k, v in res.items() if k in encoder_dict}
        # encoder_dict.update(pretrained_encoder_dict)
        # model_s2v.encoder.load_state_dict(encoder_dict)

        # decoder_dict = model_s2v.decoder.state_dict()
        # pretrained_decoder_dict = {k: v for k, v in res.items() if k in decoder_dict}
        # decoder_dict.update(pretrained_decoder_dict)
        # model_s2v.decoder.load_state_dict(decoder_dict)
        print('load pretrained s2v')


    print("------------------------------\n| Init for training ... | \n------------------------------")
    # model_s2v.apply(initialize_weights)

    LEARNING_RATE = args.learning_rate

    optimizer = torch.optim.Adam(model_s2v.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = args.epoch
    CLIP = 1

    train_loss_list, valid_loss_list = list(), list()

    best_valid_loss = float('inf')

    print("------------------------------\n| Training start ... | \n------------------------------")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model_s2v, train_loader, optimizer, criterion, CLIP, SRC, TRG, TOK_TYPES, args.accumulate_gradients)
        # print(train_loss)

        valid_loss = evaluate(model_s2v, dev_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save the best trained model
        if valid_loss < best_valid_loss:
            print('△○△○△○△○△○△○△○△○\nSave the BEST model!\n△○△○△○△○△○△○△○△○△○')
            best_valid_loss = valid_loss
            torch.save(model_s2v.state_dict(), args.output_dir + 's2v_all_model_best.pt')

        # # save model on each epoch
        # print('△○△○△○△○△○△○△○△○\nSave ncNet!\n△○△○△○△○△○△○△○△○△○')
        # torch.save(ncNet.state_dict(), opt.output_dir + 'model_' + str(epoch + 1) + '.pt')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        # plt.plot(train_loss_list)
        # plt.plot(valid_loss_list)
        # plt.show()


