import numpy as np
import torch
import torch.nn as nn
from .AttentionForcing import create_visibility_matrix
from sign2vis.modules.transformer import make_transformer_encoder, make_transformer_decoder, \
    PositionalEncoding, subsequent_mask, Embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sign2vis_Model(nn.Module):
    '''
    A transformer-based Seq2Seq model.
    '''
    def __init__(self,
                 pretrained_slt_model,
                 encoder,
                 decoder,
                 input_dim,
                 SRC,
                 src_pad_idx,
                 trg_pad_idx,
                 d_model,
                 device,
                 dropout=0.1, 
                 num_layers=3, 
                 num_heads=8):
        super().__init__()
        self.pretrained_slt_model = pretrained_slt_model
        for param in self.pretrained_slt_model.parameters():
            param.requires_grad = False
        self.video_embedding = nn.Linear(d_model, d_model, bias=False)
        self.tok_embedding = nn.Embedding(input_dim, d_model)

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    '''
    The source mask is created by checking where the source sequence is not equal to a <pad> token. 
    It is 1 where the token is not a <pad> token and 0 when it is. 
    It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, 
    which of shape [batch size, n heads, seq len, seq len].
    '''

    def make_visibility_matrix(self, src, vid_len, SRC):
        '''
        building the visibility matrix here
        '''
        # src = [batch size, src len]
        batch_matrix = []
        for each_src in src:
            nl_src = torch.from_numpy(np.array([SRC.vocab.stoi[SRC.pad_token]] * vid_len)).to(device)
            each_src = torch.cat((each_src[0:1], nl_src, each_src[1:2], each_src[ 2:]), dim=0)
            v_matrix = np.ones(each_src.shape * 2, dtype=int)
            n_heads_matrix = [v_matrix] * 8 # TODO: 8 is the number of heads ...
            batch_matrix.append(np.array(n_heads_matrix))
        batch_matrix = np.array(batch_matrix)

        # batch_matrix = [batch size, n_heads, src_len, src_len]
        return torch.tensor(batch_matrix).to(device)

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask
    
    def concat_video_src(self, x1_embed, x1_mask, x2_embed, x2_mask):
        x_embed = torch.cat((x2_embed[:, 0:1, :], x1_embed, x2_embed[:, 1:2, :], x2_embed[:, 2:, :]), dim=1)
        x_mask = torch.cat((x2_mask[:, :, :, 0:1], x1_mask.unsqueeze(1).unsqueeze(2), x2_mask[:, :, :, 1:2], x2_mask[:, :, :, 2:]), dim=3)
        return x_embed, x_mask
    
    def forward(self, video_array, video_array_mask, src, trg, tok_types, SRC):

        x1_embed = self.pretrained_slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
        x1_embed = self.pretrained_slt_model.encoder(self.pretrained_slt_model.position_encoding(x1_embed), mask=video_array_mask.unsqueeze(-2))
        x1_embed = self.video_embedding(x1_embed)
        x2_embed = self.tok_embedding(src)

        # print(x1_embed.shape, x2_embed.shape)

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # print(video_array_mask.shape, src_mask.shape)
        x_embed, x_mask = self.concat_video_src(x1_embed, video_array_mask, x2_embed, src_mask)

        # print(x_embed.shape, x_mask.shape, tok_types.shape)

        batch_visibility_matrix = self.make_visibility_matrix(src, x1_embed.shape[1], SRC)

        # print(x_embed.shape, x_mask.shape, tok_types.shape, batch_visibility_matrix.shape)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        x_hidden, enc_attention = self.encoder(x_embed, x_mask, tok_types, batch_visibility_matrix)

        # x_hidden = [batch size, src len, hid dim]

        #print(trg.shape, x_hidden.shape, trg_mask.shape, x_mask.shape)

        # print(x_hidden.shape)

        output, attention = self.decoder(trg, x_hidden, trg_mask, x_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


    def forward_encoder(self, video_array, video_array_mask, src, trg, tok_types, SRC):

        x1_embed = self.pretrained_slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
        x1_embed = self.pretrained_slt_model.encoder(self.pretrained_slt_model.position_encoding(x1_embed), mask=video_array_mask.unsqueeze(-2))
        x1_embed = self.video_embedding(x1_embed)
        x2_embed = self.tok_embedding(src)

        # print(x1_embed.shape, x2_embed.shape)

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # print(video_array_mask.shape, src_mask.shape)
        x_embed, x_mask = self.concat_video_src(x1_embed, video_array_mask, x2_embed, src_mask)

        # print(x_embed.shape, x_mask.shape, tok_types.shape)

        batch_visibility_matrix = self.make_visibility_matrix(src, x1_embed.shape[1], SRC)

        # print(x_embed.shape, x_mask.shape, tok_types.shape, batch_visibility_matrix.shape)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        x_hidden, enc_attention = self.encoder(x_embed, x_mask, tok_types, batch_visibility_matrix)

        return x_hidden, x_mask
