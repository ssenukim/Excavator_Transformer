import torch 
import torch.nn as nn 
import copy 
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_len=256, device=torch.device("cuda:0")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)
        
    def forward(self, x):
        x = x.contiguous().view(x.size(0), x.size(1), 1)
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out                
    
class TransformerEmbedding(nn.Module):
    def __init__(self, pos_embed):
        super(TransformerEmbedding, self).__init__()
        self.embedding = pos_embed 
        
    def forward(self, x):
        out = self.embedding(x)
        #print("after embedding: ", out, out.shape)
        return out 

def calculate_attention(query, key, value, mask=None):
    # query, key, value: (n_batch, h, seq_len, d_k)
    d_k = key.shape[-1]
        
    attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    #print("attention score(mask X): ", attention_score, attention_score.shape)  
    if mask is not None:
        #print(mask.shape, attention_score.shape)
        attention_score = attention_score.masked_fill(mask==0, 1e-9)
        #print("attention score: ", attention_score, attention_score.shape)    
    attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
    out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
    return out  
          
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc).to('cuda:0')  #(d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc).to('cuda:0')
        self.v_fc = copy.deepcopy(qkv_fc).to('cuda:0')
        self.out_fc = out_fc.to('cuda:0') 
    '''    
    def calculate_attention(query, key, value, mask=None):
        # query, key, value: (n_batch, h, seq_len, d_k)
        d_k = key.shape[-1]
        
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)

            
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out
    '''
    def forward(self, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc):       
            out = fc(x)             #calculate_attention 의 입력 모양에 맞춰주기위해
            out = out.contiguous().view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out 

        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)
       
        out = calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)  # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        #print(out, out.shape)
        return out 
        
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, fc1, fc2):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1.to('cuda:0')  #(d_embed, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = fc2.to('cuda:0')  #(d_ff, d_embed)

    def forward(self, x):
        out = x 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out 

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention 
        self.position_ff = position_ff

    def forward(self, src):
        out = src
        out = self.self_attention(query=out, key=out, value=out)
        out = self.position_ff(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff


    def forward(self, tgt, encoder_out, tgt_mask):
        out = tgt
        out = self.self_attention(query=out, key=out, value=out, mask=tgt_mask)
        out = self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=None)

        return out
    
class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):  # n_layer: Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    def forward(self, src):
        out = src
        for layer in self.layers:
            out = layer(out)
        return out

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer 
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
    
    def forward(self, tgt, encoder_out, tgt_mask):
        out = tgt        
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask)
            #print("decoder out: ", out, out.shape)
        
        return out 
        
class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed 
        self.tgt_embed = tgt_embed 
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator 

    def encode(self, src):
        out = self.encoder(self.src_embed(src))
        return out

    def decode(self, tgt, encoder_out, tgt_mask):
        out = self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask)
        return out

    def forward(self, src, tgt):
        tgt_mask = self.make_subsequent_mask(tgt, tgt)
        #print("mask: ", tgt_mask, tgt_mask.shape)
        encoder_out = self.encode(src)
        #print("encoder out: ", encoder_out, encoder_out.shape)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask)
        #print(decoder_out.shape)
        out = self.generator(decoder_out)
        
        return out

    def make_subsequent_mask(self, query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0)
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask 
    
    def generate(self, src, tgt, stop_angle=0.25):
        self.eval()
        result = np.zeros((1, 3))
        batch_size = src.shape[0]
        
        while result[-1, 2] < stop_angle:
            outputs = self.forward(src, tgt).to('cpu')
            outputs = outputs.squeeze(dim = 0)
            outputs = outputs.transpose(0, 1)
            outputs = outputs.detach().numpy()
            print(outputs.shape, result.shape)
            result = np.concatenate((result, outputs), axis=0)
            src[:, 6] = src[:, 6] + 0.01 
        
        result = np.delete(result, 0, axis=0)
        return result
    
    def save(self, PATH, state_dict=True):
        if state_dict==True:
            torch.save(self.state_dict(), PATH)
        else:
            torch.save(self, PATH)
        return
        
def build_model(tgt_vocab_size=3, device=torch.device("cuda:0"), max_len=256, d_embed=1, n_layer=3, d_model=24, h=8, d_ff_1=128, d_ff_2=32):
    
    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len, device=device)
    src_embed = TransformerEmbedding(pos_embed= copy.deepcopy(pos_embed))
    tgt_embed = TransformerEmbedding(pos_embed= copy.deepcopy(pos_embed))
    attention = MultiHeadAttentionLayer(d_model= d_model, h= h, qkv_fc= nn.Linear(d_embed, d_model), out_fc= nn.Linear(d_model, d_embed))
    position_ff = PositionWiseFeedForwardLayer(fc1= nn.Linear(d_embed, d_ff_1), fc2= nn.Linear(d_ff_1, d_embed))     
    encoder_block = EncoderBlock(self_attention= copy.deepcopy(attention), position_ff= copy.deepcopy(position_ff))
    decoder_block = DecoderBlock(self_attention=copy.deepcopy(attention), cross_attention= copy.deepcopy(attention), position_ff= copy.deepcopy(position_ff))
    encoder = Encoder(encoder_block= encoder_block, n_layer= n_layer)
    decoder = Decoder(decoder_block= decoder_block, n_layer= n_layer)
    generator = nn.Sequential(
        nn.Linear(d_embed, d_ff_2),
        nn.ReLU(),
        nn.Linear(d_ff_2, d_embed)
    )
    model = Transformer(
        src_embed= src_embed, 
        tgt_embed= tgt_embed,
        encoder= encoder, 
        decoder= decoder,
        generator= generator
    ).to(device)
    
    model.device = device 
    return model 