import torch
import torch.nn as nn

class TransformerJointForecaster(nn.Module):
    def __init__(self, config):
        super(TransformerJointForecaster, self).__init__()

        self.num_nodes = config["num_nodes"]
        self.pred_len = config["max_pred"]

        self.pos_encode_dim = config["hidden_dim"]
        self.input_encoder = nn.Linear(2*self.num_nodes, config["hidden_dim"])
        self.trans_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.trans_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(config["hidden_dim"], 4, config["hidden_dim"],
                    batch_first=True), config["gru_layers"])

        self.pred_model = nn.Linear(config["hidden_dim"], self.num_nodes)

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

        N_T = batch.t.shape[1]
        y = batch.y.view(-1, self.num_nodes, N_T).transpose(1,2) # (B, N_T, N)
        mask_input = batch.mask.view(-1, self.num_nodes,
                N_T).transpose(1,2) # (B, N_T, N)
        enc_input = torch.cat((y, mask_input),
                dim=-1) # (B, N_T, 2N)

        # Only encode conditioning length
        trans_input = self.input_encoder(enc_input[:,:cond_length]) +\
            pos_enc[:,:cond_length] # (B, N_T', d_h)

        # Add on static prior info representation
        # (Fixes cases where nothing obs. in encoding seq)
        prior_info_rs = self.prior_info.view(1,1,self.prior_info.shape[0]
                ).repeat_interleave(trans_input.shape[0], dim=0)
        trans_input = torch.cat((trans_input, prior_info_rs), dim=1) # (B, N_T'+1, d_h)

        encoded_rep = self.trans_encoder(trans_input) # (B, N_T', d_h)

        # Input to decoder is only time encoding
        dec_input = pos_enc[:,cond_length:(cond_length+self.pred_len)] # (B, max_pred)
        decoded_rep = self.trans_decoder(dec_input, encoded_rep) # (B, max_pred, d_h)

        pred = self.pred_model(decoded_rep) # (B, max_pred, N)

        # Pad in case of short output len
        actual_pred_len = pred.shape[1]
        if actual_pred_len < self.pred_len:
            pred_padding = torch.zeros(pred.shape[0],
                    (self.pred_len - actual_pred_len), self.num_nodes,
                    device=pred.device)
            pred = torch.cat((pred, pred_padding), dim=1) # (B, max_pred, N)

        # Reshape pred
        pred = pred.transpose(1,2).reshape(-1, self.pred_len, 1) # (BN, max_pred, 1)
        return pred

