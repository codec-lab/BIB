import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

#torch.set_default_dtype(torch.float64)
def ortho_init(tensor, scale=1.0):


    shape = tensor.size()
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:
        flat_shape = (shape[0] * shape[2] * shape[3], shape[1])  # NCHW
    else:
        raise NotImplementedError

    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    w = (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    
    # Instead of in-place copy, use torch.no_grad()
    with torch.no_grad():
        tensor.copy_(torch.FloatTensor(w))
    return tensor


def nn_init(module, w_init=ortho_init, w_scale=1.0, b_init=nn.init.constant, b_scale=0.0):
    w_init(module.weight, w_scale)
    b_init(module.bias, b_scale)
    return module


def calculate_flat_conv_dim(encoder, tensor_shape):
    dummy_input = torch.zeros(tensor_shape)
    with torch.no_grad():
        encoded_output = encoder(dummy_input)
    flat_conv_dim = int(np.prod(encoded_output.shape[1:]))
    return flat_conv_dim, encoded_output.shape[2], encoded_output.shape[3]

#CNN encoder reduces size of image but doesnt change shape
class CNN_Encoder(nn.Module): #did everything except some w2 division thing, but shapes will be same
    def __init__(self, in_channels):
        super(CNN_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu = nn.ReLU(True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

#Embed uses CNN to reduce size, and then flattens output to size of embedding_dim
class Embed(nn.Module):
    def __init__(self, in_channels, embedding_dim, flat_conv_dim):
        super(Embed, self).__init__()
        self.cnn_encoder = CNN_Encoder(in_channels)
        self.linear = nn_init(nn.Linear(flat_conv_dim, embedding_dim),w_scale = np.sqrt(2)) ####First nn init
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.cnn_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x

#Reverse process of Embed to get a predicted output state of same shape as initial input
class CNN_Decoder(nn.Module):
    def __init__(self, embedding_dim, flat_conv_dim, h2, w2, channel_size):
        super(CNN_Decoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, flat_conv_dim)
        self.relu = nn.ReLU(True)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, channel_size, kernel_size=8, stride=4)
        self.h2 = h2
        self.w2 = w2
    
    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 32, self.h2, self.w2)  # Ensure dimensions match the encoder's output
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

    
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
    def __init__(self, input_shape, embedding_dim, num_actions, tree_depth, td_lambda,gamma,normalise_state = True):
        super(TreeQN, self).__init__()
        self.in_channels = input_shape[1]
        self.batch_size = input_shape[0] 
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.tree_depth = tree_depth
        self.td_lambda = td_lambda
        self.gamma = gamma
        self.normalise_state = normalise_state
        #parameters needed to instantiate models with correct shapes and/or plug into other math calculatinos 

        self.cnn_encoder = CNN_Encoder(self.in_channels)
        self.flat_conv_dim, self.h1, self.h2 = calculate_flat_conv_dim(self.cnn_encoder, input_shape)
        #flat_conv_dim is shape after passing through the CNN encoder
        #Used to pass into the encoder (flatten the output of the CNN encoder)

        #flat_conv_dim, h1, h2 also used to instantiate the decoder

        self.decoder = CNN_Decoder(embedding_dim, self.flat_conv_dim, self.h1, self.h2, self.in_channels)

        self.encoder = Embed(self.in_channels, self.embedding_dim, self.flat_conv_dim)

        self.transition_fun = Parameter(torch.zeros(embedding_dim, embedding_dim, num_actions))
        self.transition_fun = nn.init.xavier_normal_(self.transition_fun) 

        self.reward_fun = MLPRewardFn(embedding_dim, num_actions)
        self.value_fun = nn_init(nn.Linear(embedding_dim, 1),w_scale=0.1) #literally its just this except with the w_scale


    def tree_transition(self,tensor): #trying no tanh
        temp = nn.Tanh()(torch.einsum("ij,jab->iba", tensor, self.transition_fun))
        temp = temp.contiguous() #not sure if this is necessary
        next_state = temp.detach()#!
        #experimenting with residual connection and normalization
        next_state = tensor.unsqueeze(1).expand_as(next_state) + next_state
        next_state = next_state / next_state.pow(2).sum(-1, keepdim=True).sqrt()
        return next_state
        

    def tree_plan(self, tensor):
        tree_result = {
            "embeddings": [tensor],
            "values" : [],
            "rewards" : []
        }
        for i in range(self.tree_depth):
            reward = self.reward_fun(tensor) # -> 4 (s,a) rewards
            print('reward',reward.shape)
            tree_result['rewards'].append(reward.view(-1,1))
            #print(f'Before transition: {tensor.shape}',i)
            tensor = self.tree_transition(tensor) # -> 4 next states
            #print(f'After transition: {tensor.shape}',i)
            tensor = tensor.view(-1, self.embedding_dim)
            tree_result['embeddings'].append(tensor)

            tree_result['values'].append(self.value_fun(tensor))
        
        return tree_result
    
    #gets q scores by weighing the expected value of the next state
    def tree_backup(self, tree_result):
        all_backup_values = []
        backup_values = tree_result["values"][-1] #(num_actions*batch_size)^depth 
        for i in range(1, self.tree_depth + 1):
            one_step_backup = tree_result['rewards'][-i] + self.gamma*backup_values 
            if i < self.tree_depth:
                one_step_backup = one_step_backup.view(self.batch_size, -1, self.num_actions)

                all_backup_values.insert(0, F.softmax(one_step_backup,dim=2).view(-1,1))
                max_backup = (one_step_backup * F.softmax(one_step_backup, dim = 2)).sum(dim = 2)
                #max backup is the expected value of the next state



                backup_values = ((1-self.td_lambda) * tree_result['values'][-i-1] + 
                                 (self.td_lambda) * max_backup.view(-1, 1))
                #td_lambda is the weight of the expected value of the next state
            else:
                backup_values = one_step_backup
        backup_values = backup_values.view(self.batch_size, self.num_actions) #q final
        all_backup_values.insert(0,F.softmax(backup_values, dim = 1).view(-1,1))
        return backup_values, all_backup_values
#save backup values at each transition (maybe consider later when considering trajectory prediction) 
#can get policy at each t 
    def forward(self, tensor):
        tensor = self.encoder(tensor)
        #decoded_tensor = self.decoder(tensor)

        if self.normalise_state:
            tensor = tensor / tensor.pow(2).sum(-1, keepdim=True).sqrt()

        tree_result = self.tree_plan(tensor)
        one_step_policy, all_policies = self.tree_backup(tree_result)
        #return q values like original paper and now also the decoded next_states from each action
        decoded_values = []
        for embedding in tree_result['embeddings']:
            decoded_values.append(self.decoder(embedding))
        return decoded_values, all_policies 
    
#z0 -> [a1,a2,a3,a4] -> t -> [z1,z1a,z1b,z1c] -> ... [q1,q2,q3,q4]
#                            [decode(z1),decode(z1a),decode(z1b),decode(z1c)]
#                            [next_state_prediction, messy, messy, messy]
#deocde(z0) + rest of the loss
#z0 -> [a1,a2,a3,a4] -> t -> [z0,z1a,z1b,z1c] -> [z2a,z2b,z2c,z2d]... [q1,q2,q3,q4]
#                            
