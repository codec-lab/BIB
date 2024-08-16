import random
import torch
import torch.nn.functional as F
import pandas as pd

def get_start(size):
    goal_point = (size//2, size//2)
    start_point = -1
    while True:
        start_point = (random.randint(0,size), random.randint(0,size))
        if goal_point != start_point:
            break
    return start_point, goal_point

def hard_policy(state,goal_point):
    goal_x, goal_y = goal_point
    x,y = state
    x_right = goal_x > x # if goal is right
    x_left = goal_x < x # if goal is left
    y_up = goal_y > y # if goal is above
    y_down = goal_y < y # if goal is below
    possible_next_states = []
    if x_right:
        possible_next_states.append((x+1,y))
    if x_left:
        possible_next_states.append((x-1,y))
    if y_up:
        possible_next_states.append((x,y+1))
    if y_down:
        possible_next_states.append((x,y-1))
    if len(possible_next_states) == 0:
        return -1
    return random.choice(possible_next_states)

def point_to_tensor(point,goal,size,noise=0):
    x,y = point
    x_goal, y_goal = goal
    tensor = torch.zeros(size,size)
    scale = 1
    tensor[x][y] = 1 * scale

    tensor[x_goal][y_goal] = -1 * scale

    tensor = tensor + torch.randn(tensor.size()) * noise #0.1 0.01
    
    return tensor

def get_trajectory(size,start_point = None, goal_point = None):
    trajectory = []
    if start_point is None:
        start, goal = get_start(size)
    else:
        start, goal = start_point, goal_point
    trajectory.append(start)
    while start != goal:
        start = hard_policy(start,goal)
        trajectory.append(start)
    if len(trajectory) != 6:
        return get_trajectory(size)
    trajectory = trajectory[:5] #dont turn into goal
    return [point_to_tensor(p,goal,size).unsqueeze(0).unsqueeze(0) for p in trajectory]

def max_starting_points(size):
    start_points = []
    for i in range(10000):
        start_points.append(get_start(size)[0])
    start_points = set(start_points)
    goal_point = (size//2, size//2)
    return start_points, goal_point

def get_data(size = 20):
    s, goal_point = max_starting_points(size)
    start_points = list(s)
    train_start_points = start_points[:len(start_points)//2]
    test_start_points = start_points[len(start_points)//2:]

    train_data = [get_trajectory(size,start_point,goal_point) for start_point in train_start_points]
    valid_data = [get_trajectory(size,start_point,goal_point) for start_point in test_start_points]
    return train_data, valid_data


### Testing
#tr = {'rewards':[torch.rand(4),torch.rand(16),torch.rand(64),torch.rand(256)]}
# def tree_backup(tree_result):

#     fourth_rewards = tree_result["rewards"][-1] #256 Rewards (softmax things later)
#     fourth_vibes = fourth_rewards.view(-1,4).sum(dim=1) #64 Vibes
#     third_rewards = tree_result["rewards"][-2] + gamma*fourth_vibes #64 Rewards
#     third_vibes = third_rewards.view(-1,4).sum(dim=1) #16 Vibes
#     second_rewards = tree_result["rewards"][-3] + gamma*third_vibes #16 Rewards
#     second_vibes = second_rewards.view(-1,4).sum(dim=1) #4 Vibes
#     first_rewards = tree_result["rewards"][-4] + gamma*second_vibes #4 Rewards
    
#     transition_1_probs = F.softmax(first_rewards,dim=0).unsqueeze(-1) #4 transition probs (4,1)

#     transition_2_probs = F.softmax(second_rewards.view(-1,4),dim=1) #Softmax every group of 4 actions (1,4,4)
#     transition_2_probs *= transition_1_probs
#     transition_2_probs = transition_2_probs.unsqueeze(-1) #4 transition probs (4,4,1)

#     transition_3_probs = F.softmax(third_rewards.view(-1,4,4),dim=2) #Softmax every group of 4 actions (4,4,4)
#     transition_3_probs *= transition_2_probs
#     transition_3_probs = transition_3_probs.unsqueeze(-1) #4 transition probs (4,4,4,1)

#     transition_4_probs = F.softmax(fourth_rewards.view(-1,4,4,4),dim=3) #Softmax every group of 4 actions (4,4,4,4)
#     transition_4_probs *= transition_3_probs 

#     return [transition_1_probs, transition_2_probs, transition_3_probs, transition_4_probs]
# probs = tree_backup(tr)
# for p in probs:
#     print(p.sum(),p.shape)

def validate(model, valid_data,weighted=True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for t in valid_data:
    # Get reconstruction loss to help ground abstract state
            decoded_values, transition_probabilities = model(t[0])
            decode_loss = F.mse_loss(decoded_values[0], t[0], reduction='sum')

            # Flatten transition probabilities to then weigh with loss of each predicted state at each layer
            first = transition_probabilities[0].view(-1,1,1,1)
            second = transition_probabilities[1].view(-1,1,1,1)
            third = transition_probabilities[2].view(-1,1,1,1)
            fourth = transition_probabilities[3].view(-1,1,1,1)


            if weighted:
                first_loss = (F.mse_loss(decoded_values[1], t[1], reduction='none') * first).sum().item()
                second_loss = (F.mse_loss(decoded_values[2], t[2], reduction='none') * second).sum().item()
                third_loss = (F.mse_loss(decoded_values[3], t[3], reduction='none') * third).sum().item()
                fourth_loss = (F.mse_loss(decoded_values[4], t[4], reduction='none') * fourth).sum().item()
            else:
                first_loss = (F.mse_loss(decoded_values[1][first.argmax()].squeeze(0),t[1].squeeze(0).squeeze(0)))
                second_loss = (F.mse_loss(decoded_values[2][second.argmax()].squeeze(0),t[2].squeeze(0).squeeze(0)))
                third_loss = (F.mse_loss(decoded_values[3][third.argmax()].squeeze(0),t[3].squeeze(0).squeeze(0)))
                fourth_loss = (F.mse_loss(decoded_values[4][fourth.argmax()].squeeze(0),t[4].squeeze(0).squeeze(0)))

            total_loss += decode_loss + first_loss + second_loss + third_loss + fourth_loss

    return total_loss / len(valid_data)


def validate_autoencoder(model,valid_data):
    model.eval()
    total_loss = 0
    for t in valid_data:
        for sample in t:
            encoding = model.encoder(sample)
            decoding = model.decoder(encoding)
            reconstruction_loss = F.mse_loss(decoding, sample)
            total_loss += reconstruction_loss.item()
    return total_loss

def train_autoencoder(model,optimizer,train_data,valid_data,epochs,lambda_reg=0):
    raw_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for t in random.sample(train_data, len(train_data)):
            for sample in t:
                optimizer.zero_grad() 
                encoding = model.encoder(sample)
                decoding = model.decoder(encoding)
                
                # Compute reconstruction loss (MSE)
                reconstruction_loss = F.mse_loss(decoding, sample)
                
                # Compute L2 regularization (sum of squared weights)
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.sum(param ** 2)
                
                # Combine losses
                loss = reconstruction_loss + lambda_reg * l2_reg
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            raw_losses.append(total_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss},validation loss: {validate_autoencoder(model,valid_data)}')

def store_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append([name, param.grad.norm().item()])
    return gradients

def t_view(tensor,shrink =4):
    return torch.round(tensor[0][0][shrink:-shrink, shrink:-shrink])
def action_viewer(model,start_state,actions,shrink=4):
    with torch.no_grad():
        enc_state = model.encoder(start_state)
        transition = model.tree_transition(enc_state)
        print('True Original Start State:\n', t_view(start_state,shrink))
        print("Max True Original Start State:", ((start_state[0][0].argmax()//20).item(), (start_state[0][0].argmax()%20).item()))
        print("Decoded Next States")
        for action in actions:
            print('Action:', action)
            print('Next State:\n', t_view(model.decoder(transition.squeeze(0))[action].unsqueeze(0),shrink))
            #get coordinate of max value
            print('Max Value Decoded:', ((model.decoder(transition.squeeze(0))[action][0].argmax()//20).item(), (model.decoder(transition.squeeze(0))[action][0].argmax()%20).item()))

def get_greedy_path(probs):
    best_first_action = probs[0].argmax().item()
    best_second_action = probs[1][best_first_action].argmax().item()
    best_third_action = probs[2][best_first_action][best_second_action].argmax().item()
    best_fourth_action = probs[3][best_first_action][best_second_action][best_third_action].argmax().item()
    return best_first_action, best_second_action, best_third_action, best_fourth_action

def view_greedy_path(model,start_state,shrink=4):
    with torch.no_grad():
        decoded_states, transition_probabilities = model(start_state[0])
        print("True Path")
        for s in start_state: 
            print((s[0][0].argmax()//20).item(), (s[0][0].argmax()%20).item())
        actions = get_greedy_path(transition_probabilities)
        action_viewer(model,start_state[0],actions,shrink)

def get_action_df(model,data):
    action_sets = []
    for i in range(len(data)):
        _,probs = model(data[i][0])
        actions = get_greedy_path(probs)
        action_sets.append(actions)
    return pd.DataFrame(action_sets, columns=['Action 1', 'Action 2', 'Action 3', 'Action 4'])