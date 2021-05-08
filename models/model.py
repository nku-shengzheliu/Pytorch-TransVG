from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
#from torch.utils.data.distributed import DistributedSampler

from models.detr import build_detr, build_VLFusion

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

def load_weights(model, load_path):
    # 加载DETR模型的Transformer Encoder层与backbone层ResNet50的预训练参数
    dict_trained = torch.load(load_path)['model']
    # new_list = list(model.state_dict().keys())
    # trained_list = list(dict_trained.keys())
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model


class TransVG(nn.Module):
    def __init__(self, jemb_drop_out=0.1, bert_model='bert-base-uncased',tunebert=True, args=None):
        super(TransVG, self).__init__()
        # self.coordmap = coordmap
        # self.emb_size = emb_size
        # self.NFilm = NFilm
        # self.intmd = intmd
        # self.mstage = mstage
        # self.convlstm = convlstm
        self.tunebert = tunebert
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = build_detr(args)
        # if args.resume==None and args.pretrain==None:
        self.visumodel = load_weights(self.visumodel, './saved_models/detr-r50-e632da11.pth')
        
        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)

        ## Visual-linguistic Fusion model
        self.vlmodel = build_VLFusion(args)
        self.vlmodel = load_weights(self.vlmodel, './saved_models/detr-r50-e632da11.pth')
        
        ## Prediction Head
        self.Prediction_Head = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          #nn.Dropout(jemb_drop_out),
          nn.Linear(256, 4),)
        for p in self.Prediction_Head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, mask, word_id, word_mask):
        ## Visual Module
        batch_size = image.size(0)
        fv = self.visumodel(image, mask)

        ## Language Module
        all_encoder_layers, _ = self.textmodel(word_id, \
            token_type_ids=None, attention_mask=word_mask)
        ## Sentence feature TODO:这里取了最后四层，TransVG没说
        fl = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4
        if not self.tunebert:
            ## fix bert during training
            # raw_flang = raw_flang.detach()
            fl = fl.detach()

        ## Visual-linguistic Fusion Module
        x = self.vlmodel(fv, fl)

        ## Prediction Head
        outbox = self.Prediction_Head(x)  # (x; y;w; h)
        outbox = outbox.sigmoid()*2.-0.5

        return outbox