import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


def calculate_flat_conv_dim(cnn_encoder, tensor_shape):
    dummy_input = torch.zeros(tensor_shape)
    with torch.no_grad():
        encoded_output = cnn_encoder(dummy_input)
    flat_conv_dim = int(np.prod(encoded_output.shape[1:]))
    return flat_conv_dim, encoded_output.shape[2], encoded_output.shape[3]

enc_layers = 16
dec_layers = 16
class CNN_Encoder(nn.Module): 
    def __init__(self, in_channels, drop_out=0):
        super(CNN_Encoder, self).__init__()
        # Initial convolution layers with more filters and smaller kernel size for finer feature extraction
        self.conv1 = nn.Conv2d(in_channels, enc_layers, kernel_size=2, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(enc_layers)
        self.drop1 = nn.Dropout(drop_out)


        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.drop1(self.relu(self.bn1(self.conv1(x))))
        return x

class Embed(nn.Module):
    def __init__(self, in_channels, embedding_dim, flat_conv_dim, drop_out=0):
        super(Embed, self).__init__()
        self.cnn_encoder = CNN_Encoder(in_channels, drop_out)
        self.linear = nn.Linear(flat_conv_dim, embedding_dim)
        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.linear(x))
        x = self.relu(x)
        return x


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_dim, output_size=(5, 5), dropout_rate=0.5):
        super(CNN_Decoder, self).__init__()
        
        # Fully connected layer to expand the embedding
        self.fc = nn.Linear(embedding_dim, dec_layers * 5 * 5) 
        
        # ReLU activation and dropout
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate)

        # Deconvolution layers to upsample to the desired output size
        self.deconv1 = nn.ConvTranspose2d(dec_layers,1 , kernel_size=3, stride=1, padding=1)  # Output: (64, 10, 10)

    def forward(self, x):
        # Pass through the fully connected layer
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        
        x = x.view(x.size(0), dec_layers, 5, 5)
        
        x = self.deconv1(x)  # Output: (batch_size, 64, 10, 10)

        return x
    

class AutoEncoder(nn.Module):
    def __init__(self, tensor_shape, embedding_dim, output_size=(5, 5), dec_drop_out=0.5, enc_drop_out = 0):
        super(AutoEncoder, self).__init__()
        self.in_channels = tensor_shape[1]
        self.flat_conv_dim, self.encoded_height, self.encoded_width = calculate_flat_conv_dim(CNN_Encoder(self.in_channels), tensor_shape)
        self.encoder = Embed(self.in_channels, embedding_dim, self.flat_conv_dim, enc_drop_out)
        self.embedding_dim = embedding_dim
        self.transition_function = Parameter(torch.zeros(4, 1, embedding_dim))
        self.transition_function = nn.init.xavier_normal_(self.transition_function)

        self.decoder = CNN_Decoder(embedding_dim, output_size, dec_drop_out)

    def transition(self,tensor,action): #take in (X,Y) tensor
        temp = tensor.repeat(4,1) #hard coding 4 actions
        temp = temp.view(4,-1,self.embedding_dim) * self.transition_function
        temp = temp[action]
        return temp
    
    def forward(self, input_state,action, input_next_state,option):
        if option == 2:
            input_latent_state = self.encoder(input_state)
            input_next_latent_state = self.encoder(input_next_state) #save for latent to latent loss
            decoded_input_state = self.decoder(input_latent_state) #ground state
            transitioned_latent_state = self.transition(input_latent_state,action)
            return decoded_input_state,  transitioned_latent_state, input_next_latent_state
        if option == 1:
            input_latent_state = self.encoder(input_state)
            decoded_input_state = self.decoder(input_latent_state) #ground state 
            transitioned_latent_state = self.transition(input_latent_state,action)
            decoded_next_state = self.decoder(transitioned_latent_state) #option 1 difference
            return decoded_input_state, decoded_next_state