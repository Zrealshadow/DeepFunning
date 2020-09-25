'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-25 14:33:49
 * @desc 
'''

import torch
import torch.nn as nn



''' ------------------------- DGCNN Model --------------------------- '''

'''
DGCNN Dilate Gated Convolutional Neural NetWork 
Source: https://kexue.fm/archives/5409
forward calculate:
    f = sigmoid()
    Y = X * [ 1 - f(conv1d_2(X))] + Conv1d_2(X) * f(conv1d_2(X))

init Params:
    input : 
        h_dim : size of word_embedding / x.shape[-1]
        dilation : dilation rate
        k_size : kernel size of convolution layer
        drop_gate :

forward Params:
    inputs: [ x, mask]
        x : batch_size * word_emb_size * seq_max_len 
        mask : batch_size * seq_max_len [0,1]
    
    output: [ xx, mask]
        xx : batch_size * word_emb_size * seq_max_len
'''

class DilatedGatedConv1D(nn.Module):
    '''
    DGCNN
    '''
    def __init__(self,h_dim,dilation,k_size=3,drop_gate=0.1):
        super(DilatedGatedConv1D,self).__init__()
        self.h_dim = h_dim
        self.dilation = dilation
        self.kernel_size = k_size
        self.dropout=nn.Dropout(p=drop_gate)
        self.padding=self.dilation *(self.kernel_size-1)//2
        #input  batch_size , Channel_in , seq_len
        self.conv1=nn.Conv1d(in_channels=self.h_dim,out_channels=self.h_dim,\
            kernel_size=self.kernel_size,dilation=dilation,padding=self.padding)
        self.conv2=nn.Conv1d(in_channels=self.h_dim,out_channels=self.h_dim,\
            kernel_size=self.kernel_size,dilation=dilation,padding=self.padding)

    def forward(self,inputs):
        x,mask=inputs
        #x batch_size , word_emb_size , seq_max_len
        #mask 部分置0 batch_size , seq_max_len 
        mask_=mask.unsqueeze(1).expand(-1,x.shape[1],-1)
        
        x=x*mask_
        # x,mask: batch_size , word_emb_size , seq_max_len

        x1=self.conv1(x)
        x2=self.conv2(x)
        # batch_size , word_emb_size, seq_max_len

        x2=self.dropout(x2)
        x2=torch.sigmoid(x2)

        #add resnet and multiply a gate for this resnet layer
        xx=(1-x2)* x + x2 * x1
        #batch_size , word_emb_size, seq_max_len
        # xx=xx.permute(0,2,1).contiguous()
        return [xx,mask]


''' ------------------------- Transformer Model --------------------------- '''

from deepfunning.model.transformer import make_model
'''
** This Module I have not trained **

Args:
    src_vocab source data vocabulary list's size
    tag_vocab : target data vocabulary list's size
    N: encoderlayer in Encoder
    d_model: word embedding size
    d_ff: full connected layer size
    h: the num of multiheadedAttention Layer's head
    dropout: dropout probability
Returns:

public method :
    forward(src,tgt,src_mask,tgt_mask):
        Args:
            src : source text  batch_size * max_seq_len_src
            tgt : target text  batch_size * max_seq_len_tgt
            src_mask : mask [1,0] batch_size * max_seq_len_src
            tgt_mask : mask [1,0] batch_size * max_seq_len_tgt
        Returns:
            x : predict Target text batch_size * max_seq_len_tgt * vocab_size

'''
def Transformer(src_vocab,tag_vocab,N=6,d_model=512,d_ff=2058,h=8,dropout=0.1):
    return make_model(src_vocab,tag_vocab,N,d_model,d_ff,h,dropout)


''' ------------------------- CRF Model --------------------------- '''

import deepfunning.model.crf as crf
'''
Args:
    tag_nums : num of labels
    is_padding : this is_padding means whether start_tag and end_tag are included in tag_nums
            if included, there is no need to pad, but start_tag,end_tag's index in tag_nums should be assigned.
            if not incldued, it will automatically label these tag

public method:
    forward(features,length,tags):
        Args:
            features : Extracted Features by Bert or BiLSTM | Batch_size * Max_seq_len * num_tag
            length : it is like mask, but it's a list to record true length of every text 
            tags : Labeled Tags according to text | Batch_size * Max_seq_len
        Returns:
            loss : Tensor loss
    
    get_batch_best_path(self,inputs,lengths):
        Description: Use to decoding
        Args:
            inputs : Batch_size * max_seq_len * num_tag
            lengths : record length of every text
        Returns:
            batch_best_path : list of every text's predicted tags | batch_size * len ,len is not unchanged
'''
def CRF(tag_nums,is_padding=False,start_tag=None,end_tag=None):
    return crf.CRF(tag_nums,is_padding=False,start_tag=None,end_tag=None)


'''------------------------ Hidden Markov Model -------------------------'''

import deepfunning.model.hmm as hmm
'''
Args:
    N: The num of hidden states
    M: The num of observed states

Public Method:
    train(word_lists,tag_lists,word2id,tag2id):
        Args:
            word_lists : list  batch_size * word_list
            tag_lists : list batch_size * tag_list
            word2id : dict 
            tag2id  : dict
        Returns:
            null
    test(test_word_lists,word2id,tag2id):
        Args:
            test_word_lists : list batch_size * word_list
            word2id : dict
            tag2id : dict
        Returns:
            pred_tag_list: list batch_size * word_list
'''
def HMM(N,M):
    return hmm.HMM(N,M)
