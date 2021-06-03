# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_vis_transformer, build_transformer
from .position_encoding import build_position_encoding

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, img, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        # pos: position encoding
        features, pos = self.backbone(samples)  # pos:list, pos[-1]: [64, 256, 20, 20]

        src, mask = features[-1].decompose()  # src:[64, 2048, 20, 20]  mask:[64,20,20]
        assert mask is not None
        out = self.transformer(self.input_proj(src), mask, pos[-1])
        
        return out
    
class VLFusion(nn.Module):
    def __init__(self, transformer, pos):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: no use
            """
        super().__init__()
        #self.num_queries = num_queries
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model
        self.pr = nn.Embedding(1, hidden_dim)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # self.v_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        # self.l_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
        self.v_proj = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),)
        self.l_proj = torch.nn.Sequential(
          nn.Linear(768, 256),
          nn.ReLU(),)

    def forward(self, fv, fl):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        bs, c, h, w = fv.shape
        _, _, l = fl.shape

        pv = self.v_proj(fv.view(bs, c, -1).permute(0,2,1))  # [bs,400,256]
        pl = self.l_proj(fl)  # [bs, 40, 256]
        pv = pv.permute(0,2,1)  # [bs,256,400]
        pl = pl.permute(0,2,1)  # [bs,256,40]

        # pv = self.v_proj(fv)  # [bs, 256, 20, 20]
        # pv = pv.view(bs, 256, -1)  # [bs, 256, 400]

        # fl = fl.unsqueeze(0).permute(0,3,1,2)  # [1, 768, bs, 40]
        # pl = self.l_proj(fl)  # [1, 256, bs, 40]
        # pl = pl.squeeze().permute(1, 0, 2).view(bs, 256, -1)  # [bs, 256, 40]

        pr = self.pr.weight # [1, 256]
        pr = pr.expand(bs,-1).unsqueeze(2)  # [bs, 256, 1]

        x0 = torch.cat((pv, pl), dim=2)
        x0 = torch.cat((x0, pr), dim=2)  # [bs, 256, 441]
        
        pos = self.pos(x0).to(x0.dtype)  # [bs, 441, 256]
        mask = torch.zeros([bs, x0.shape[2]]).cuda()
        mask = mask.bool()  # [bs, 441]
        
        out = self.transformer(x0, mask, pos)  # [441, bs, 256]
        
        return out[-1]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detr(args):

    device = torch.device(args.device)
    backbone = build_backbone(args) # ResNet 50
    transformer = build_vis_transformer(args)

    model = DETR(
        backbone,
        transformer,
    )
    return model

def build_VLFusion(args):

    device = torch.device(args.device)
    
    transformer = build_transformer(args)

    pos = build_position_encoding(args, position_embedding = 'learned')

    model = VLFusion(
        transformer,
        pos,
    )
    return model


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    # * DETR
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400+40, type=int,
                        help="Number of query slots in VLFusion")
    parser.add_argument('--pre_norm', action='store_true')
    
    args = parser.parse_args()
    model = build_VLFusion(args)
    model.cuda()
    img = torch.ones((8,256,20,20)).cuda()
    lang = torch.ones((8, 40, 768)).cuda()
    out = model(img, lang)

    print(out.shape)  # torch.Size([64, 256, 20, 20])
