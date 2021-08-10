import torch
from torch import nn
import torchvision
import pretrainedmodels
import json
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_choice = 1

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=1):
        super(Encoder, self).__init__()        
        self.enc_image_size = encoded_image_size
            
        if encoder_choice==1:
            vgg16 = torchvision.models.vgg16(pretrained = True)
            self.features_nopool = nn.Sequential(*list(vgg16.features.children())[:-1])
            self.features_pool = list(vgg16.features.children())[-1]
            self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1]) 

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
    def forward(self, images):
        global encoder_choice
        
        if encoder_choice==1:
            x = self.features_nopool(images)
            x_pool = self.features_pool(x)
            x_feat = x_pool.view(x_pool.size(0), -1)

            y = self.classifier(x_feat)
            return y
            
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1)  
        return out

 
        
    

class DecoderWithAttention_choice(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention_choice, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        embed_dim = decoder_dim
        self.embed_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        
        self.decode_step1 = nn.LSTMCell(decoder_dim, decoder_dim, bias=True)
        self.decode_step2 = nn.LSTMCell(decoder_dim + decoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.get_img = nn.Linear(encoder_dim, decoder_dim)
        self.img_forward = nn.Linear(encoder_dim, decoder_dim)  
        self.img_backward = nn.Linear(encoder_dim, decoder_dim)  

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()  

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        h = self.init_h(encoder_out)  
        c = self.init_c(encoder_out)
        return h, c

    def get_img_rep(self, encoder_out):
        img = self.get_img(encoder_out)  
        return img
        
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, encoder_dim)  

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  
        img = self.get_img_rep(encoder_out)
        h1, c1 = torch.zeros_like(img), torch.zeros_like(img)
        h2, c2 = self.init_hidden_state(encoder_out)
            
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.decode_step1(embeddings[:batch_size_t, t, :] * img[:batch_size_t],(h1[:batch_size_t], c1[:batch_size_t]))                  
            h2, c2 = self.decode_step2(torch.cat([embeddings[:batch_size_t, t, :], h1[:batch_size_t]], dim = 1), (h2[:batch_size_t], c2[:batch_size_t]))    
            preds = self.fc(self.dropout(h2))

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind
