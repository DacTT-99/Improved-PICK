# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/7/2020 5:54 PM

from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops import roi_pool

from . import resnet

import numpy as np

class Encoder(nn.Module):
    def __init__(self,
                 char_embedding_dim: int,
                 out_dim: int,
                 image_feature_dim: int = 512,
                 nheaders: int = 8,
                 nlayers: int = 6,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 image_encoder: str = 'resnet50',
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        """
        Convert image segments and text segments to node embedding.

        Parameters
        ----------
        char_embedding_dim : int
            parameter d_model of Transformer
        out_dim : int
            Same with Dimmension of model text embedding
        image_feature_dim : int, optional
            out_chanels of CNN, by default 512
        nheaders : int, optional
            Number of headers in Transformer Encoder Layer, by default 8
        nlayers : int, optional
            Number of Transformer Encoder Layers stacked, by default 6
        feedforward_dim : int, optional
            Dimension of Feedforward Layer in Transformer, by default 2048
        dropout : float, optional
            Dropout rate , by default 0.1
        max_len : int, optional
            The maximum length of text sequence, by default 100
        image_encoder : str, optional
            CNN model name, by default 'resnet50'
        roi_pooling_mode : str, optional
            Mode of Region of Interest algorithm, by default 'roi_align'
        roi_pooling_size : Tuple[int, int], optional
            Size of result of RoI algorithm, by default (7, 7)
        """
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], 'roi pooling model: {} not support.'.format(
            roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size)  # (h, w)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                               nhead=nheaders,
                                                               dim_feedforward=feedforward_dim,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)

        if image_encoder == 'resnet18':
            self.cnn = resnet.resnet18(output_channels=image_feature_dim)
        elif image_encoder == 'resnet34':
            self.cnn = resnet.resnet34(output_channels=image_feature_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet.resnet50(output_channels=image_feature_dim)
        elif image_encoder == 'resnet101':
            self.cnn = resnet.resnet101(output_channels=image_feature_dim)
        elif image_encoder == 'resnet152':
            self.cnn = resnet.resnet152(output_channels=image_feature_dim)
        else:
            raise NotImplementedError()

        self.pooling = nn.Conv2d(out_dim,out_dim,[3,1])
        self.bn = nn.BatchNorm2d(out_dim)

        self.norm = nn.LayerNorm(out_dim)

        # Compute the positional encodings once in log space.
        position_embedding = torch.zeros(max_len, char_embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2).float() *
                             -(math.log(10000.0) / char_embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)  # 1, 1, max_len, char_embedding_dim
        self.register_buffer('position_embedding', position_embedding)

        self.pe_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)

    def forward(self, images_segments: torch.Tensor, transcripts: torch.Tensor, src_key_padding_mask: torch.Tensor):
        '''
        :param images: whole_images, shape is (B, N, H, W, C), where B is batch size, N is the number of segments of
                the documents, H is height of image, W is width of image, C is channel of images (default is 3).
        :param boxes_coordinate: boxes coordinate, shape is (B, N, 8),
                where 8 is coordinates (x1, y1, x2, y2, x3, y3, x4, y4).
        :param transcripts: text segments, shape is (B, N, T, D), where T is the max length of transcripts,
                                D is dimension of model.
        :param src_key_padding_mask: text padding mask, shape is (B*N, T), True for padding value.
            if provided, specified padding elements in the key will be ignored by the attention.
            This is an binary mask. When the value is True, the corresponding value on the attention layer of Transformer
            will be filled with -inf.
        need_weights: output attn_output_weights.
        :return: set of nodes X, shape is (B*N, T, D)
        '''

        # B : batch size
        # N : Number of boxes in single image
        # T : Max length of transcripts

        B, N, T, D = transcripts.shape

        # get image embedding using cnn : (B*N, C, H, W)
        images_segments_embedding = torch.flatten(images_segments,0,1)
        # image segments embedding: (B*N, C, H/16, W/16)
        images_segments_embedding = self.cnn(images_segments_embedding)

        # get transcript embedding using transformer

        # add positional embedding : (B, N, T, D)
        transcripts_segments = self.pe_dropout(transcripts + self.position_embedding[:, :, :transcripts.size(2), :])
        # (B*N, T ,D)
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)
        # (T, B*N, D)
        transcripts_segments = transcripts_segments.transpose(0,1).contiguous()
        # Transcripts segments embedding : (T, B*N, D)
        transcripts_segments_embedding = self.transformer_encoder(transcripts_segments,src_key_padding_mask=src_key_padding_mask)
        # (B*N,T, D)
        transcripts_segments_embedding = transcripts_segments_embedding.transpose(0,1)
        # (B*N, D, T)
        transcripts_segments_embedding = transcripts_segments_embedding.transpose(1,2).contiguous()
        # (B*N, D, 1, T)
        transcripts_segments_embedding = transcripts_segments_embedding.unsqueeze(dim=2)

        #Concatenation : (B*N, D, 5, T)
        out = torch.cat([images_segments_embedding,transcripts_segments_embedding],dim=2)

        # pooling : (B*N, D, 1, T)
        out = self.pooling(out)
        # (B*N, D, T)
        out = out.squeeze()
        # ()
        #out = out.transpose(0, 1).contiguous()

        # (T, B*N, D)
        #out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)

        # (B*N, T, D)
        out = out.transpose(1, 2).contiguous()
        out = self.norm(out)
        out = self.output_dropout(out)

        return out
