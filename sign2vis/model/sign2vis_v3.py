import numpy as np
import torch
import torch.nn as nn
from .AttentionForcing import create_visibility_matrix
from sign2vis.modules.transformer import make_transformer_encoder, make_transformer_decoder, \
    PositionalEncoding, subsequent_mask, Embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sign2visModel(nn.Module):
    '''
    A transformer-based Seq2Seq model.
    '''
    def __init__(self,
                 pretrained_slt_model,
                 encoder,
                 decoder,
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
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.all_encoder = make_transformer_encoder(N_layer=num_layers,
                                                d_model=d_model,
                                                d_ff=d_model*4,
                                                heads=num_heads,
                                                dropout=dropout,
                                                ffn_layer='ffn',
                                                first_kernel_size=1)
        self.encoder = encoder
        self.segment_embedding = nn.Embedding(2, d_model)
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

    def make_visibility_matrix(self, src, SRC):
        '''
        building the visibility matrix here
        '''
        # src = [batch size, src len]
        batch_matrix = []
        for each_src in src:
            v_matrix = create_visibility_matrix(SRC, each_src)
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
    
    def gen_segments(self, l_seg1, l_seg2, device):
        seg_ids = [1] * l_seg1 + [0] * l_seg2
        seg_ids = torch.tensor(seg_ids, dtype=torch.long).to(device)
        return seg_ids

    def concat_video_src(self, x1_embed, x1_mask, x2_embed, x2_mask):
        x_embed = torch.cat((x1_embed, x2_embed), dim=1)
        x_mask = torch.cat((x1_mask, x2_mask), dim=1)
        return x_embed, x_mask
    
    def forward(self, video_array, video_array_mask, src, trg, tok_types, SRC):

        x1_embed = self.pretrained_slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
        x1_embed = self.pretrained_slt_model.encoder(self.pretrained_slt_model.position_encoding(x1_embed), mask=video_array_mask.unsqueeze(-2))
        x1_embed = self.video_embedding(x1_embed)

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        batch_visibility_matrix = self.make_visibility_matrix(src, SRC)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src, enc_attention = self.encoder(src, src_mask, tok_types, batch_visibility_matrix)


        # enc_src = [batch size, src len, hid dim]

        x2_embed = enc_src
        
        x_embed, x_mask = self.concat_video_src(x1_embed, video_array_mask, x2_embed, src_mask.squeeze(1).squeeze(1))
        seg_ids = self.gen_segments(x1_embed.size(1), x2_embed.size(1), x_embed.device)

        x_embed = x_embed + self.segment_embedding(seg_ids.unsqueeze(0)) 
        x_hidden = self.all_encoder(self.position_encoding(x_embed), mask=x_mask.unsqueeze(-2))

        output, attention = self.decoder(trg, x_hidden, trg_mask, x_mask.unsqueeze(-2).unsqueeze(-2))

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention