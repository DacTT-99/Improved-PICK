from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align,roi_pool

from . import resnet


class Encoder(nn.Module):

    def __init__(self,
                image_encoder: dict,
                 transformer: dict,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        """
        Convert image segments and text segments to node embedding.

        Parameters
        ----------
        image_encoder : dict
            image_encoder kwargs
        transformer : dict
            transformer kwargs
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            Max sequence lenght of each segments, by default 100
        roi_pooling_mode : str, optional
            perform region_of_interest algorithm in RCNN , by default 'roi_align'
        roi_pooling_size : Tuple[int, int], optional
            size of region pooling, by default (7, 7)

        """
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], 'roi pooling model: {} not support.'.format(
            roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size)  # (h, w)

        transformer_layer = transformer['args']['transformer_layer']
        char_embedding_dim = transformer_layer['args']['d_model']
        out_dim = transformer_layer['args']['d_model']
        image_feature_dim = image_encoder['args']['output_channels']

        transformer_encoder_layer = getattr(nn,transformer_layer['type'])(transformer_layer['args'])
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,num_layers=transformer['args']['num_layers'])

        try:
            self.cnn = getattr(resnet,image_encoder['type'])(image_encoder['args'])
        except:
            raise NotImplementedError()

        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)

        self.projection = nn.Linear(2 * out_dim, out_dim)
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

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor,
                src_key_padding_mask: torch.Tensor):
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

        B, N, T, D = transcripts.shape

        # get image embedding using cnn
        # (B, 3, H, W)
        _, _, origin_H, origin_W = images.shape

        # image embedding: (B, C, H/16, W/16)
        images = self.cnn(images)
        _, C, H, W = images.shape

        # generate rois for roi pooling, rois shape is (B, N, 5), 5 means (batch_index, x0, y0, x1, y1)
        rois_batch = torch.zeros(B, N, 5, device=images.device)
        # Loop on the every image.
        for i in range(B):  # (B, N, 8)
            # (N, 8)
            doc_boxes = boxes_coordinate[i]
            # (N, 4)
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i

        spatial_scale = float(H / origin_H)
        # use roi pooling get image segments
        # (B*N, C, roi_pooling_size, roi_pooling_size)
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            image_segments = roi_pool(images, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)

        # (B*N, D, 1, 1)
        image_segments = F.relu(self.bn(self.conv(image_segments)))
        # # (B*N, D,)
        image_segments = image_segments.squeeze()

        # (B*N, 1, D)
        image_segments = image_segments.unsqueeze(dim=1)

        # add positional embedding
        transcripts_segments = self.pe_dropout(transcripts + self.position_embedding[:, :, :transcripts.size(2), :])
        # (B*N, T ,D)
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)

        # (B*N, T, D)
        image_segments = image_segments.expand_as(transcripts_segments)

        # here we first add image embedding and text embedding together,
        # then as the input of transformer to get a non-local fusion features, different from paper process.
        out = image_segments + transcripts_segments

        # (T, B*N, D)
        out = out.transpose(0, 1).contiguous()

        # (T, B*N, D)
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)

        # (B*N, T, D)
        out = out.transpose(0, 1).contiguous()
        out = self.norm(out)
        out = self.output_dropout(out)

        return out

class Encoder_v2(nn.Module):
    def __init__(self,
                 image_encoder: dict,
                 transformer: dict,
                 dropout: float = 0.1,
                 max_len: int = 100):
        """
        Convert image segments and text segments to node embedding.

        Parameters
        ----------
        image_encoder : dict
            image encode kwargs
        transformer : dict
            transformer kwargs
        dropout : float, optional
            dropout rate, by default 0.1
        max_len : int, optional
            Max sequence lenght of each segments, by default 100
        """
        super().__init__()

        self.dropout = dropout
        transformer_layer = transformer['args']['transformer_layer']
        char_embedding_dim = transformer_layer['args']['d_model']
        out_dim = transformer_layer['args']['d_model']

        transformer_encoder_layer = getattr(nn,transformer_layer['type'])(transformer_layer['args'])
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer,num_layers=transformer['args']['num_layers'])

        try:
            self.cnn = getattr(resnet,image_encoder['type'])(image_encoder['args'])
        except:
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
        # (B*N, T, D)
        out = out.transpose(1, 2).contiguous()
        out = self.norm(out)
        out = self.output_dropout(out)

        return out
