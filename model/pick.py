# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:54 PM

from typing import *

import torch
import torch.nn as nn
import numpy as np
from . import encoder,graph,decoder

from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class PICKModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_module = kwargs['embedding']
        encoder_module = kwargs['encoder']
        graph_module = kwargs['graph']
        decoder_module = kwargs['decoder']
        self.make_model(embedding_module, encoder_module, graph_module, decoder_module)

    def make_model(self, embedding_module, encoder_module, graph_module, decoder_module):
        # Given the params of each component, creates components.

        embedding_module['args']['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = getattr(nn,embedding_module['type'])(**embedding_module['args'])

        encoder_module['args']['transformer']['args']['transformer_layer']['args']['d_model'] = embedding_module['args']['embedding_dim']
        self.encoder = getattr(encoder,encoder_module['type'])(**encoder_module['args'])

        graph_module['args']['graph_learning']['args']['in_dim'] = encoder_module['args']['image_encoder']['args']['output_channels']
        graph_module['args']['graph_convolution']['args']['out_dim'] = encoder_module['args']['image_encoder']['args']['output_channels']
        self.graph = getattr(graph,graph_module['type'])(**graph_module['args'])

        decoder_module['args']['bilstm']['args']['input_size'] = encoder_module['args']['transformer']['args']['transformer_layer']['args']['d_model']
        if decoder_module['args']['bilstm']['args']['bidirectional']:
            decoder_module['args']['mlp']['args']['in_dim'] = decoder_module['args']['bilstm']['args']['hidden_size'] * 2
        else:
            decoder_module['args']['mlp']['args']['in_dim'] = decoder_module['args']['bilstm_module']['args']['hidden_size']
        decoder_module['args']['mlp']['args']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_module['args']['crf']['args']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = getattr(decoder,decoder_module['type'])(**decoder_module['args'])

    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        '''
        # filter out padding value, (B*N, T, D)
        input = input * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        mean_out = sum_out.div(text_len)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        # input
        image_segments = kwargs['image_segments']  # (B, 3, H, W)
        relation_features = kwargs['relation_features']  # initial relation embedding (B, N, N, 6)
        text_segments = kwargs['text_segments']  # text segments (B, N, T)
        text_length = kwargs['text_length']  # (B, N)
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = kwargs['mask']  # (B, N, T)
        boxes_coordinate = kwargs['boxes_coordinate']  # (B, num_boxes, 8)

        ##### Forward Begin #####
        ### Encoder module ###
        # word embedding
        text_emb = self.word_emb(text_segments)

        # src_key_padding_mask is text padding mask, True is padding value (B*N, T)
        # graph_node_mask is mask for graph, True is valid node, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        # set of nodes, (B*N, T, D)
        x = self.encoder(images_segments=image_segments, transcripts=text_emb, src_key_padding_mask=src_key_padding_mask)

        ### Graph module ###
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        # (B*N, T, D) -> (B*N, D)
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
        # (B*N, 1)，True is valid node
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        # (B*N, D), filter out not valid node
        x_gcn = x_gcn * graph_node_mask.byte()

        # initial adjacent matrix (B, N, N)
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # (B, 1)
        # (B, N, D)
        x_gcn = x_gcn.reshape(B, N, -1)
        # (B, N, D), (B, N, N), (B,)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

        ### Decoder module ###
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length,
                                                        iob_tags_label)
        ##### Forward End #####

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}

        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
