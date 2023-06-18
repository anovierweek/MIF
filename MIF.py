"""
From: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""
import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchvision 
from transformers import BertModel, BertConfig

from BertTextEncoder import BertTextEncoder
from functions import *
from transformers import Wav2Vec2Model
wav2vec = Wav2Vec2Model.from_pretrained("/usr/zhoujie/myjupyter/MMSA/MISSA/wav2vec2-base/chinese-wav2vec2-base").to(device="cuda")    # 用于提取通用特征，768维

__all__ = ['MIF']

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

# let's define a simple model that can deal with multimodal variable length sequence
class MIF(nn.Module):
    def __init__(self, config):
        super(MIF, self).__init__()

        assert config.use_bert == True

        self.config = config
        self.text_size = config.feature_dims[0]
        self.visual_size = config.feature_dims[2]
        self.acoustic_size = config.feature_dims[1]
        self.weight=config.weight
        self.hidden_size = config.hidden_size


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        # self.input_sizes = input_sizes = [39,4,41]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes if config.train_mode == "classification" else 1
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between
    
        if config.use_bert:
            # text subnets
            self.bertmodel = BertTextEncoder(language=config.language, use_finetune=config.use_finetune)

        self.vrnn1 = rnn(hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1], out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[0], out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # Linear Layer to same sized space
        ##########################################
        self.project_tt = nn.Sequential()
        self.project_tt.add_module('project_t_dropout', nn.Dropout(dropout_rate))
        self.project_tt.add_module('project_t1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size*2))
        self.project_tt.add_module('project_t_activation', self.activation)
        self.project_tt.add_module('project_t2', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size*2))
        # self.project_tt.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_vv = nn.Sequential()
        self.project_vv.add_module('project_v_dropout', nn.Dropout(dropout_rate))
        self.project_vv.add_module('project_v1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size*2))
        self.project_vv.add_module('project_v_activation', self.activation)
        self.project_vv.add_module('project_v2', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size*2))
        # self.project_vv.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_aa = nn.Sequential()
        self.project_aa.add_module('project_a_dropout', nn.Dropout(dropout_rate))
        self.project_aa.add_module('project_a1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size*2))
        self.project_aa.add_module('project_a_activation', self.activation)
        self.project_aa.add_module('project_a2', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size*2))
        # self.project_aa.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        self.norm = nn.LayerNorm(config.hidden_size*2)

        ##########################################
        # private encoders
        ##########################################
        self.private_t = UniModal_Attention(middle_hidden_size=config.hidden_size,attention_nhead=config.num_head,embedding=config.seq_lens[0],input_embeddig =input_sizes[0],attention_dropout=config.attention_dropout,dropout=config.dropout)
        self.private_a = UniModal_Attention(middle_hidden_size=config.hidden_size,attention_nhead=config.num_head,embedding=config.seq_lens[1],input_embeddig =input_sizes[0],attention_dropout=config.attention_dropout,dropout=config.dropout)
        self.private_v = UniModal_Attention(middle_hidden_size=config.hidden_size,attention_nhead=config.num_head,embedding=config.seq_lens[2],input_embeddig =input_sizes[1],attention_dropout=config.attention_dropout,dropout=config.dropout)

        ##########################################
        # shared encoder
        ##########################################
        self.shared = CrossModal_Attention(middle_hidden_size=config.hidden_size,attention_nhead=config.num_head,embedding=config.seq_lens,input_embeddig =input_sizes,attention_dropout=config.attention_dropout,dropout=config.dropout)

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size*2, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size*2, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6*2, out_features=self.config.hidden_size*6))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*6, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm(hidden_sizes[0])
        self.vvlayer_norm = nn.LayerNorm(hidden_sizes[1])
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm(hidden_sizes[0])

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.hidden_size*2, nhead=2)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, text, acoustic, visual):
        # text[batch_size,3,39]  acoustic[batch_size,400,33]  visual[batch_size,55,709]
        # bert_sent_mask : consists of seq_len of 1, followed by padding of 0.
        bert_sent, bert_sent_mask, bert_sent_type = text[:,0,:], text[:,1,:], text[:,2,:]

        batch_size = text.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(text) # [batch_size, seq_len, 768]

            # Use the mean value of bert of the front real sentence length as the final representation of text.
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output) #[batch_size,39,768]
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len # [batch_size,768]

            # utterance_text = masked_output #[batch_size,39,768]
            # utterance_text = bert_output
        utterance_audio = wav2vec(acoustic.reshape([batch_size,-1]))['last_hidden_state']     # torch.Size([batch_size, 41, 768])，模型出来是一个BaseModelOutput的结构体。
        for param in wav2vec.parameters():
                param.requires_grad = True

        lengths = mask_len.squeeze().int().detach().cpu().view(-1)
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm) #final_h1v 和 final_h2v  torch.Size([2, batch_size, 709])

        utterance_video = torch.cat((final_h1v, final_h2v), dim=0).permute(1, 0, 2) # torch.Size([batch_size, 4, 709])

        # # extract features from acoustic modality
        # final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        # utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        utterance_text = self.tlayer_norm(masked_output)
        utterance_video = self.vvlayer_norm(utterance_video)
        utterance_audio = self.alayer_norm(utterance_audio)



        # Shared-private encoders
        self.shared_private(utterance_text,utterance_video, utterance_audio)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)# torch.Size([batch_size, 4])
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )# torch.Size([batch_size, 4])
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0) # torch.Size([6,batch_size, hidden_dim*2])
        memory = h
        h = self.transformer_decoder(h,memory) # torch.Size([6,batch_size, hidden_dim*2])
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1) # torch.Size([batch_size, 6*hidden_dim*2])
        o = self.fusion(h)# torch.Size([batch_size, 5])
        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)# torch.Size([batch_size, hidden_dim*2])
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)# torch.Size([batch_size, hidden_dim])
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t,utterance_v,utterance_a):
        # Private-shared components

        self.utt_private_t,self.utt_t_orig = self.private_t(utterance_t) # torch.Size([batch_size, hidden_dim*2*2])
        self.utt_private_v,self.utt_v_orig = self.private_v(utterance_v) # torch.Size([batch_size, hidden_dim*2*2])
        self.utt_private_a,self.utt_a_orig = self.private_a(utterance_a) # torch.Size([batch_size, hidden_dim*2*2])
        self.utt_shared_t,self.utt_shared_v,self.utt_shared_a = self.shared(utterance_t,utterance_v,utterance_a)
        

    def forward(self, text, audio, video):
        output = self.alignment(text, audio, video)
        tmp = {
            "M": output
        }
        return tmp