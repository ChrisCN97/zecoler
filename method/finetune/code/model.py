# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)  # 分开
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 合并
        self.dense.weight.data.normal_(0, 0.1)  # new add
        # self.ln = nn.LayerNorm(config.hidden_size)  # new
        l2n = int(config.hidden_size/2)
        self.dense2 = nn.Linear(config.hidden_size, l2n)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(l2n, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)  # 分开
        x = self.dropout(x)
        x = self.dense(x)
        # x = self.ln(x)  # new
        x = torch.tanh(x)
        x = self.dropout(x)  #
        x = self.dense2(x)  #
        x = torch.tanh(x)  #
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None):
        embedding = self.encoder.get_input_embeddings()
        outputs = embedding(input_ids).view(-1,self.args.block_size)
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
      
        
 
        


