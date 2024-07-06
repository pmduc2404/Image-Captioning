import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_dim, dropout = 0.5, grad = False):
        super(Encoder, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)
        
        if not grad:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        feature = self.resnet(x)
        return self.dropout(self.relu(feature))
    

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers, device, encoder, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.device = device
        self.encoder = encoder.to(device)
    
    def forward(self, image, caption):
        features = self.encoder(image)
        
        embeddings = self.dropout(self.embed(caption))
       
        embeddings = torch.cat((features.unsqueeze(1),embeddings), dim=1)
        
        outputs, state = self.lstm(embeddings)
        outputs = self.linear(outputs)
        
        return outputs
    
#     def forward(self, image, captions):
#         features = self.encoder(image)
#         embeddings = self.dropout(self.embed(captions))
#         embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

#         batch_size = features.size(0)
#         captions_length = captions.size(1)
#         vocab_size = self.linear.out_features

#         outputs = torch.zeros(batch_size, captions_length, vocab_size).to(self.device)
#         input = features.unsqueeze(1)

#         state = None
        
#         for i in range(captions_length):
#             output, state = self.lstm(input, state)
#             output = self.linear(output)
#             outputs[:, i, :] = output.squeeze(1)

#             top1 = output.argmax(2)
#             input = self.dropout(self.embed(top1))

#         return outputs
