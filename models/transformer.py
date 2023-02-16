import torch
import torch.nn as nn

class TransformerForecaster(nn.Module):
    def __init__(self, config):
        super(TransformerForecaster, self).__init__()

        self.num_nodes = config["num_nodes"]
        self.pred_len = config["max_pred"]

        self.pos_encode_dim = config["hidden_dim"]
        self.input_encoder = nn.Linear(2, config["hidden_dim"])
        self.trans_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.trans_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.pred_model = nn.Linear(config["hidden_dim"], 1)

        self.prior_info = nn.Parameter(torch.randn(config["hidden_dim"]))

    def encode_time(self, t):
        # t: (B, N_t)
        i = torch.arange(self.pos_encode_dim // 2, device=t.device)
        denom = 0.1**(2*i / self.pos_encode_dim) # (self.pos_encode_dim/2,)
        f = t.unsqueeze(-1) / denom

        encoding = torch.cat((torch.sin(f), torch.cos(f)),
                dim=-1) # (B, N_T, pos_encoding_dim)
        return encoding

    def forward(self, batch, cond_length):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        pos_enc = self.encode_time(batch.t) #(B, N_T, d_h)
        pos_enc_repeated = pos_enc.repeat_interleave(self.num_nodes,
                dim=0) # (BN, N_T, d_h)
        enc_input = torch.cat((batch.y, batch.mask.unsqueeze(-1)),
                dim=-1) # (B, N_T, d_h)

        # Only encode conditioning length
        trans_input = self.input_encoder(enc_input[:,:cond_length]) +\
            pos_enc_repeated[:,:cond_length] # (B, N_T', d_h)

        # Treat unobserved times as padding
        enc_mask = batch.mask.to(bool).logical_not()[
                :,:cond_length] # True when batch.mask is 0 (unobs.) (B, N_T')

        # Add on static prior info representation
        # (Fixes cases where nothing obs. in encoding seq)
        prior_info_rs = self.prior_info.view(1,1,self.prior_info.shape[0]
                ).repeat_interleave(trans_input.shape[0], dim=0)
        trans_input = torch.cat((trans_input, prior_info_rs), dim=1) # (B, N_T'+1, d_h)
        extra_mask = torch.zeros((enc_mask.shape[0],1),
                device=enc_mask.device).to(bool) # (B,1)
        enc_mask = torch.cat((enc_mask, extra_mask), dim=1) #(B, N_T'+1)

        encoded_rep = self.trans_encoder(trans_input,
                src_key_padding_mask=enc_mask) # (B, N_T', d_h)

        # Input to decoder is only time encoding
        dec_input = pos_enc_repeated[:,cond_length:(cond_length+self.pred_len)]
        decoded_rep = self.trans_decoder(dec_input, encoded_rep,
                memory_key_padding_mask=enc_mask) # (B, max_pred, d_h)

        pred = self.pred_model(decoded_rep) # (B, max_pred, 1)

        # Pad in case of short output len
        actual_pred_len = pred.shape[1]
        if actual_pred_len < self.pred_len:
            pred_padding = torch.zeros(pred.shape[0],
                    (self.pred_len - actual_pred_len), 1, device=pred.device)
            pred = torch.cat((pred, pred_padding), dim=1)

        return pred

