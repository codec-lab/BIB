import torch 
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F
def view_tensor(tensor, cmap='viridis', colorbar=True, title=None, xlabel=None, ylabel=None):
    """
    Visualizes a 2D tensor using Matplotlib.

    Args:
        tensor (torch.Tensor): The 2D tensor to visualize.
        cmap (str): The color map to use for visualization (default is 'viridis').
        colorbar (bool): Whether to display a color bar (default is True).
        title (str): The title of the plot (default is None).
        xlabel (str): The label for the x-axis (default is None).
        ylabel (str): The label for the y-axis (default is None).
    """
    if tensor.ndim != 2:
        raise ValueError("The input tensor must be 2D (Height x Width).")

    # Convert the tensor to numpy for Matplotlib compatibility
    tensor_np = tensor.numpy()

    # Create the plot
    plt.imshow(tensor_np, cmap=cmap, interpolation='nearest')
    
    # Add a colorbar if requested
    if colorbar:
        plt.colorbar()

    # Add title and labels if provided
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Show the plot
    plt.show()
###############FOR IRL
def get_start_points(goal_point,distance = 4):
    x, y = goal_point
    start_points = []
    for i in range(5):
        for j in range(5):
            if abs(i-x) + abs(j-y) == distance:
                start_points.append((i,j))
    return start_points

def get_all_start_points(distance = 4):
    train_start_points = []
    test_start_points = []
    for i in range(5):
        for j in range(5):
            if (i+j) % 2 == 0:
                goal_point = (i,j)
                train_start_points.append([goal_point, get_start_points(goal_point, distance)]) 
            else:
                goal_point = (i,j)
                test_start_points.append([goal_point, get_start_points(goal_point, distance)])
    return train_start_points, test_start_points

def trajectory_to_tensor(trajectory):
    tensor_traj = []
    goal_point = trajectory[-1]
    for i in range(len(trajectory)):
        x, y = trajectory[i] #x,y point
        tensor = torch.zeros(1, 5, 5)
        tensor[0][x][y] = 1
        tensor[0][goal_point[0]][goal_point[1]] = -1
        tensor_traj.append(tensor)
    return tensor_traj

def get_all_possible(squeeze=False):
    train_data = []
    test_data = []
    for i in range(25): #For each point on grid (assume goal point)
        goal_point = (i//5, i%5)
        for j in range(25):
            if i == j: ###OR include this??
                continue
            start_point = (j//5, j%5)
            grid = torch.zeros(5,5)
            grid[goal_point[0]][goal_point[1]] = -1
            grid[start_point[0]][start_point[1]] = 1
            if (goal_point[0] + goal_point[1]) % 2 == 0:
                train_data.append(grid.unsqueeze(0)) #For channel dimension
            else:
                test_data.append(grid.unsqueeze(0))
    if squeeze:
        for i in range(len(train_data)):
            train_data[i] = train_data[i].squeeze(0)
        for i in range(len(test_data)):
            test_data[i] = test_data[i].squeeze(0)
    return train_data, test_data

def batch_data(data,batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        #Use torch.cat to concatenate the tensors
        batch = torch.stack(data[i:i+batch_size])
        batches.append(batch)
    return batches

def get_all_possible_batched(batch_size):
    train_data, test_data = get_all_possible()
    train_batches = batch_data(train_data, batch_size)
    test_batches = batch_data(test_data, batch_size)
    return train_batches, test_batches

#####FOR INSPECT######
##Train auteoncoder
def train_autoencoder(model, optimizer, all_train, all_test, epochs=120):
    assert len(all_train[0].shape) == 4 #batch x channel by height by width
    t1 = torch.zeros_like(all_train[0][0])
    t2 = torch.zeros_like(all_train[0][0])
    t2[0][0] = 1
    baseline_loss = F.mse_loss(t1,t2).item()
    for i in range(epochs): 
        model.train()
        avg_loss =0
        train_count = 0
        valid_count = 0
        for traj in random.sample(all_train, len(all_train)):
            noisy_traj = traj.clone() + torch.randn_like(traj) * 0.1
            train_count +=1
            decoded_tensor = model(noisy_traj,None, None,option = 4)#model.decoder(encodeded_tensor)
            loss = F.mse_loss(decoded_tensor, noisy_traj) #Should this be with the true tensor?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        if i % 30 ==0 or i == epochs-1:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for traj in all_test:
                    valid_count +=1
                    decoded_tensor = model(traj,None,None, option =4)#model.decoder(encodeded_tensor)
                    loss = F.mse_loss(decoded_tensor, traj)
                    valid_loss += loss.item()
            print('VALID LOSS',i,(valid_loss/valid_count)/baseline_loss)
            print('Train Loss',i,(avg_loss/train_count)/baseline_loss,'\n') #Want less than 1


##Get inspect data
def get_fixed_goal_data(model,index = -1): 
    '''
    returns data for all 25 positions so just take the results
    '''
    model.eval()
    with torch.no_grad():
        all_latent_tensors = []
        all_decode_tensors = []
        all_grid_tensors = []
        for i in range(25): #For each goal point
            latent_tensors = []
            decode_tensors = []
            grid_tensors = []
            for k in range(5): #For each row, column
                for j in range(5):
                    goal_point = (i//5, i%5) #Fixed
                    start_point = (k, j)
                    # if goal_point == start_point: #For now, include goal on its own
                    #     continue
                    grid = torch.zeros(5,5)
                    grid[start_point[0]][start_point[1]] = 1
                    grid[goal_point[0]][goal_point[1]] = -1 #if goal is same, it'll overlap here

                    grid_tensors.append(grid.clone().detach()) #For sanity check
                    latent_tensor = model.encoder(grid.unsqueeze(0).unsqueeze(0))
                    latent_tensors.append(latent_tensor.detach())
                    decoded_tensor = model.decoder(latent_tensor)
                    decode_tensors.append(decoded_tensor.detach())
            all_latent_tensors.append(latent_tensors)
            all_decode_tensors.append(decode_tensors)
            all_grid_tensors.append(grid_tensors)
        if index >= 0:
            return all_latent_tensors[index], all_decode_tensors[index], all_grid_tensors[index]
        return all_latent_tensors, all_decode_tensors, all_grid_tensors
    
def get_sas_data():
    train_data, test_data = get_all_possible(squeeze=True)
    def fill_data(data):
        return_sas = []
        for i in range(len(data)): 
            argmax = data[i].argmax()
            y,x = (argmax//5, argmax%5)
            argmin = data[i].argmin()
            y_min,x_min = (argmin//5, argmin%5)
            possible_actions = [] #0 is up, 1 is down, 2 is left, 3 is right
            if y > 0:
                possible_actions.append(0)
            if y < 4:
                possible_actions.append(1)
            if x > 0:
                possible_actions.append(2)
            if x < 4:
                possible_actions.append(3)
            for action in possible_actions:
                state = data[i]
                next_state = torch.zeros_like(state)
                if action == 0:
                    next_state[y-1][x] = 1
                elif action == 1:
                    next_state[y+1][x] = 1
                elif action == 2:
                    next_state[y][x-1] = 1
                elif action == 3:
                    next_state[y][x+1] = 1
                next_state[y_min][x_min] = -1 #goal state. Apply after agent moves
                return_sas.append((state, action, next_state))
        return return_sas
    return fill_data(train_data), fill_data(test_data)

