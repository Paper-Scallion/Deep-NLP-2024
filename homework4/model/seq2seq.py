import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, args):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, args.num_layers)
        self.linear = nn.Linear(args.hidden_dim, args.vocab_size)

    def forward(self, x, hx):
        x = self.embedding(x)
        out, hx = self.lstm(x, hx)
        out = self.linear(out)
        return out, hx
    
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.05, 0.05)
