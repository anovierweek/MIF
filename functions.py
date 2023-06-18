import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import time

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"




class BiLSTM_Attention(nn.Module):

    def __init__(self,embedding_dim,hidden_dim,n_layers=1,dropout =0.2,bidirectional=False):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional=bidirectional
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,num_layers=n_layers, bidirectional=self.bidirectional, dropout=0,batch_first=True)       
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))



        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.rnn(x) # final_hidden_state:[n_layers*2, batch, hidden_dim]
        
        
        final_hidden = F.relu(self.linear(self.dropout(final_hidden_state.squeeze())) )#[batch, seq_len, hidden_dim]

        return output,self.fc(final_hidden)

class CrossModal_Attention(nn.Module):

    def __init__(self,middle_hidden_size,embedding,input_embeddig,attention_nhead = 8, attention_dropout = 0.2,dropout=0.2):

        super(CrossModal_Attention, self).__init__()
        self.middle_hidden_size = middle_hidden_size
        self.attention_nhead = attention_nhead
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)        #单词数，嵌入向量维度
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.middle_hidden_size,
            num_heads=self.attention_nhead, 
            dropout=self.attention_dropout,
        )

        self.bilstm_t = BiLSTM_Attention(embedding_dim =input_embeddig[0],hidden_dim=self.middle_hidden_size,dropout=self.dropout)
        self.bilstm_a = BiLSTM_Attention(embedding_dim =input_embeddig[0],hidden_dim=self.middle_hidden_size,dropout=self.dropout)
        self.bilstm_v = BiLSTM_Attention(embedding_dim =input_embeddig[1],hidden_dim=self.middle_hidden_size,dropout=self.dropout)
        
        self.trans_t = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.middle_hidden_size*embedding[0], self.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) 
        self.trans_v = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.middle_hidden_size*embedding[2], self.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        self.trans_a = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.middle_hidden_size*embedding[1], self.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        


    def forward(self, text, visual, audio):
        t,ht = self.bilstm_t(text)
        v,hv = self.bilstm_v(visual)
        a,ha = self.bilstm_a(audio)
        t = t.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        a = a.permute(1, 0, 2)
        tv_attention_out, _ = self.attention(v,t,t)
        tv_attention_out = torch.mean(tv_attention_out, dim=0).squeeze(0)
        vt_attention_out, _ = self.attention(t,v,v) # torch.Size([seq_len, batch_size, hidden_dim])
        vt_attention_out = torch.mean(vt_attention_out, dim=0).squeeze(0)
        ta_attention_out, _ = self.attention(a,t,t)
        ta_attention_out = torch.mean(ta_attention_out, dim=0).squeeze(0)
        at_attention_out, _ = self.attention(t,a,a)
        at_attention_out = torch.mean(at_attention_out, dim=0).squeeze(0)
        av_attention_out, _ = self.attention(v,a,a)
        av_attention_out = torch.mean(av_attention_out, dim=0).squeeze(0)
        va_attention_out, _ = self.attention(a,v,v)
        va_attention_out = torch.mean(va_attention_out, dim=0).squeeze(0)

        tv_attention =  torch.add(tv_attention_out,vt_attention_out)/2
        ta_attention =  torch.add(ta_attention_out,at_attention_out)/2
        va_attention =  torch.add(av_attention_out,va_attention_out)/2

        text_prob_vec = torch.cat([ht,tv_attention], dim=1) 
        visual_prob_vec = torch.cat([hv,va_attention], dim=1)  
        audio_prob_vec = torch.cat([ha, ta_attention], dim=1)
        
        
        return text_prob_vec,visual_prob_vec,audio_prob_vec


class UniModal_Attention(nn.Module):

    def __init__(self,middle_hidden_size,embedding,input_embeddig,attention_nhead = 8,attention_dropout = 0.2,dropout=0.2):

        super(UniModal_Attention, self).__init__()
        self.middle_hidden_size = middle_hidden_size
        self.attention_nhead = attention_nhead
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.embedding = embedding        
        self.bilstm = BiLSTM_Attention(embedding_dim =input_embeddig,hidden_dim=self.middle_hidden_size,dropout=self.dropout)
        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.middle_hidden_size,
            num_heads=self.attention_nhead, 
            dropout=self.attention_dropout,
        )
        self.trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.middle_hidden_size*self.embedding, self.middle_hidden_size),
            nn.ReLU(inplace=True)
        )



    def forward(self, input):
        # 输入x  torch.Size([batch_size, seq_len, hidden_dim])
        x,h = self.bilstm(input)
        x = x.permute(1, 0, 2) # torch.Size([seq_len, batch_size, hidden_dim])
        attention_out, _ = self.attention(x,x,x) # torch.Size([seq_len, batch_size, hidden_dim])
        attention_out = torch.mean(attention_out, dim=0).squeeze(0) # torch.Size([batch_size, hidden_dim])
        output = torch.cat([h,attention_out], dim=1)
        return output,h
