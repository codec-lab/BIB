import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


def calculate_flat_conv_dim(encoder, tensor_shape):
    dummy_input = torch.zeros(tensor_shape)
    with torch.no_grad():
        encoded_output = encoder(dummy_input)
    flat_conv_dim = int(np.prod(encoded_output.shape[1:]))
    return flat_conv_dim, encoded_output.shape[2], encoded_output.shape[3]

#CNN encoder reduces size of image but doesnt change shape
# class CNN_Encoder(nn.Module): 
#     def __init__(self, in_channels):
#         super(CNN_Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
#         self.relu = nn.ReLU(True)
    
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         return x

# #Embed uses CNN to reduce size, and then flattens output to size of embedding_dim
# class Embed(nn.Module):
#     def __init__(self, in_channels, embedding_dim, flat_conv_dim):
#         super(Embed, self).__init__()
#         self.cnn_encoder = CNN_Encoder(in_channels)
#         self.linear = nn.Linear(flat_conv_dim, embedding_dim) ####First nn init
#         self.relu = nn.ReLU(True)
#     def forward(self, x):
#         x = self.cnn_encoder(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         x = self.relu(x)
#         return x

class CNN_Encoder(nn.Module): 
    def __init__(self, in_channels):
        super(CNN_Encoder, self).__init__()
        # Initial convolution layers with more filters and smaller kernel size for finer feature extraction
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Residual connection, adjusting to match the number of output channels
        self.residual_conv = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        # Convolutional layers with batch normalization and ReLU activation
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Residual connection
        residual = self.residual_conv(x)
        
        # Continue with the third convolution layer
        x = self.relu(self.bn3(self.conv3(x)))

        # Adding the residual to the output
        x = x + residual
        
        return x

# Embed uses CNN to reduce size and then flattens output to size of embedding_dim
class Embed(nn.Module):
    def __init__(self, in_channels, embedding_dim, flat_conv_dim):
        super(Embed, self).__init__()
        self.cnn_encoder = CNN_Encoder(in_channels)
        self.linear = nn.Linear(flat_conv_dim, embedding_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x

# class CNN_Decoder(nn.Module):
#     def __init__(self, embedding_dim, flat_conv_dim, h2, w2, channel_size, dropout_rate=0.5):
#         super(CNN_Decoder, self).__init__()
#         self.fc = nn.Linear(embedding_dim, flat_conv_dim)

#         self.relu = nn.ReLU(True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
#         self.deconv2 = nn.ConvTranspose2d(16, channel_size, kernel_size=8, stride=4)
#         self.h2 = h2
#         self.w2 = w2
    
#     def forward(self, x):
#         x = self.relu(self.fc(x))
#         x = self.dropout(x)  # Apply dropout after the fully connected layer
#         x = x.view(x.size(0), 32, self.h2, self.w2)  # Ensure dimensions match the encoder's output
#         x = self.relu(self.deconv1(x))
#         x = self.dropout(x)  # Apply dropout after the first deconvolution layer
#         x = self.deconv2(x)
#         return x

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_dim, output_size=(20, 20), dropout_rate=0.5):
        super(CNN_Decoder, self).__init__()
        
        # Fully connected layer to expand the embedding
        self.fc = nn.Linear(embedding_dim, 128 * 5 * 5)
        
        # ReLU activation and dropout
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate)

        # Deconvolution layers to upsample to the desired output size
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: (64, 10, 10)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # Output: (32, 20, 20)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)                    # Output: (1, 20, 20)

    def forward(self, x):
        # Pass through the fully connected layer
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        
        # Reshape to (batch_size, 128, 5, 5)
        x = x.view(x.size(0), 128, 5, 5)
        
        # Upsample through deconvolution layers
        x = self.relu(self.deconv1(x))  # Output: (batch_size, 64, 10, 10)
        x = self.relu(self.deconv2(x))  # Output: (batch_size, 32, 20, 20)
        
        # Final convolution to get the desired output size
        x = self.final_conv(x)  # Output: (batch_size, 1, 20, 20)
        
        return x#.squeeze()  # Squeeze the channel dimension if not needed


    
class MLPRewardFn(nn.Module): #s,a reward function
    def __init__(self, embed_dim, num_actions):
        super(MLPRewardFn, self).__init__()
        self.embedding_dim = embed_dim
        self.num_actions = num_actions
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, num_actions) #removed relu
        )

    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        return self.mlp(x).view(-1, self.num_actions)

    
class TreeQN(nn.Module):
    def __init__(self, input_shape, embedding_dim, num_actions, tree_depth, gamma,decode_dropout,t1=True):
        super(TreeQN, self).__init__()
        self.in_channels = input_shape[1]
        self.batch_size = input_shape[0] 
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.tree_depth = tree_depth
        self.gamma = gamma
        self.decode_dropout = decode_dropout
        self.t1 = t1
        #parameters needed to instantiate models with correct shapes and/or plug into other math calculatinos 

        self.cnn_encoder = CNN_Encoder(self.in_channels)
        self.flat_conv_dim, self.h1, self.h2 = calculate_flat_conv_dim(self.cnn_encoder, input_shape)
        #flat_conv_dim is shape after passing through the CNN encoder
        #Used to pass into the encoder (flatten the output of the CNN encoder)

        #flat_conv_dim, h1, h2 also used to instantiate the decoder

        #self.decoder = CNN_Decoder(embedding_dim, self.flat_conv_dim, self.h1, self.h2, self.in_channels,self.decode_dropout)
        self.decoder = CNN_Decoder(embedding_dim, output_size=(self.h2, self.h2), dropout_rate=self.decode_dropout)

        self.encoder = Embed(self.in_channels, self.embedding_dim, self.flat_conv_dim)
        if self.t1:
            print("Einsum Transiton")
            self.transition_fun = Parameter(torch.zeros(embedding_dim, embedding_dim, num_actions))
            self.transition_fun = nn.init.xavier_normal_(self.transition_fun) 
        else:
            print("Addition Transition")
            self.transition_fun = Parameter(torch.rand(num_actions, 1, embedding_dim))

        

        #self.reward_fun = MLPRewardFn(embedding_dim, num_actions)
        self.goal = Parameter(torch.rand(embedding_dim))
        self.goal_beta = Parameter(torch.tensor(0.1))

    def reward_fun (self, tensor):
        return -torch.norm(tensor - self.goal, dim=1) * self.goal_beta
        #self.reward_fun = lambda z: -torch.norm(z - self.goal, dim=1) * self.goal_beta



        #self.value_fun = nn_init(nn.Linear(embedding_dim, 1),w_scale=0.1) #literally its just this except with the w_scale


    def tree_transition(self,tensor): #take in (X,Y) tensor
        if self.t1:
            #print('temp 1 shape',tensor.shape)
            temp = (torch.einsum("ij,jab->iba", tensor, self.transition_fun))
            #print('temp 2 shape',temp.shape)
        #temp = temp.contiguous() #not sure if this is necessary
        else:
            #print('temp 1 shape',tensor.shape)
            temp = tensor.repeat(self.num_actions,1)
            #print('temp 2 shape',temp.shape)
            temp = temp.view(self.num_actions,-1,self.embedding_dim) * self.transition_fun
            #print('temp 3 shape',temp.shape)
        next_state = temp#.detach()#!
        return next_state
        

    def transition(self,tensor): #take in (X,Y) tensor
        temp = tensor.repeat(self.num_actions,1)
        temp = temp.view(self.num_actions,-1,self.embedding_dim) * self.transition_fun
        return temp

    def tree_plan(self, tensor):
        tree_result = {
            "embeddings": [tensor],
            "values" : [],
            "rewards" : []
        }
        for i in range(self.tree_depth):

            tensor = self.tree_transition(tensor) # -> 4 next states
            reward = self.reward_fun(tensor.view(-1,self.embedding_dim)) # Just S reward
            tree_result['rewards'].append(reward)

            tensor = tensor.view(-1, self.embedding_dim)
            tree_result['embeddings'].append(tensor)

            #tree_result['values'].append(self.value_fun(tensor))
        return tree_result
    
    #gets q scores by weighing the expected value of the next state
    # def tree_backup(self, tree_result):
    #     all_backup_values = []
    #     backup_values = tree_result["rewards"][-1] #(num_actions*batch_size)^depth 
    #     for i in range(1, self.tree_depth + 1):
    #         one_step_backup = tree_result['rewards'][-i] + self.gamma*backup_values 
    #         if i < self.tree_depth:
    #             one_step_backup = one_step_backup.view(self.batch_size, -1, self.num_actions)

    #             all_backup_values.insert(0, F.softmax(one_step_backup,dim=2).view(-1,1))
    #             max_backup = (one_step_backup * F.softmax(one_step_backup, dim = 2)).sum(dim = 2)
    #             #max backup is the expected value of the next state



    #             backup_values = max_backup.view(-1, 1)
    #         else:
    #             backup_values = one_step_backup
    #     #backup_values = backup_values.view(self.batch_size, self.num_actions) #q final
    #     all_backup_values.insert(0,F.softmax(backup_values, dim = 1).view(-1,1))
    #     return backup_values, all_backup_values

    def tree_backup(self,tree_result):

        fourth_rewards = tree_result["rewards"][-1] #256 Rewards (softmax things later)
        fourth_vibes = fourth_rewards.view(-1,self.num_actions).sum(dim=1) #64 Vibes
        third_rewards = tree_result["rewards"][-2] + self.gamma*fourth_vibes #64 Rewards
        third_vibes = third_rewards.view(-1,self.num_actions).sum(dim=1) #16 Vibes
        second_rewards = tree_result["rewards"][-3] + self.gamma*third_vibes #16 Rewards
        second_vibes = second_rewards.view(-1,self.num_actions).sum(dim=1) #4 Vibes
        first_rewards = tree_result["rewards"][-4] + self.gamma*second_vibes #4 Rewards
        
        transition_1_probs = F.softmax(first_rewards,dim=0).unsqueeze(-1) #4 transition probs (4,1)

        transition_2_probs = F.softmax(second_rewards.view(-1,self.num_actions),dim=1) #Softmax every group of 4 actions (1,4,4)
        transition_2_probs = transition_2_probs * transition_1_probs
        transition_2_probs = transition_2_probs.unsqueeze(-1) #4 transition probs (4,4,1)

        transition_3_probs = F.softmax(third_rewards.view(-1,self.num_actions,self.num_actions),dim=2) #Softmax every group of 4 actions (4,4,4)
        transition_3_probs = transition_3_probs * transition_2_probs
        transition_3_probs = transition_3_probs.unsqueeze(-1) #4 transition probs (4,4,4,1)

        transition_4_probs = F.softmax(fourth_rewards.view(-1,self.num_actions,self.num_actions,self.num_actions),dim=3) #Softmax every group of 4 actions (4,4,4,4)
        transition_4_probs = transition_4_probs * transition_3_probs 

        return [transition_1_probs, transition_2_probs, transition_3_probs, transition_4_probs]

    

    def forward(self, tensor):
        tensor = self.encoder(tensor)
        #decoded_tensor = self.decoder(tensor)

        # if self.normalise_state:
        #     tensor = tensor / tensor.pow(2).sum(-1, keepdim=True).sqrt()

        tree_result = self.tree_plan(tensor)
        transition_probs = self.tree_backup(tree_result)
        #return q values like original paper and now also the decoded next_states from each action
        decoded_values = []
        for embedding in tree_result['embeddings']:
            decoded_values.append(self.decoder(embedding))
        return decoded_values, transition_probs 
 