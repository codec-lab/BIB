import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

from torch.optim import Adam
from torch.optim import RMSprop

from treeQN.treeqn_traj_simplest import TreeQN
import random
import argparse

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

def point_to_tensor(point,goal,size):
    x,y = point
    x_goal, y_goal = goal
    tensor = torch.zeros(size+2,size+2)
    scale = 1
    tensor[x][y] = 1 * scale
    tensor[x+1][y] = 1 * scale
    tensor[x][y+1] = 1 * scale
    tensor[x+1][y+1] = 1
    tensor[x_goal][y_goal] = -1 * scale
    tensor[x_goal+1][y_goal] = -1 * scale
    tensor[x_goal][y_goal+1] = -1 * scale
    tensor[x_goal+1][y_goal+1] = -1 * scale
    return tensor

def get_trajectory(size = 18,start_point = None, goal_point = None):
    trajectory = []
    if start_point is None:
        start, goal = get_start(size)
    else:
        start, goal = start_point, goal_point
    trajectory.append(start)
    while start != goal:
        start = hard_policy(start,goal)
        trajectory.append(start)
    if len(trajectory) != 5:
        return get_trajectory(size)
    return [point_to_tensor(p,goal,size).unsqueeze(0).unsqueeze(0) for p in trajectory]

def max_starting_points(size = 18):
    start_points = []
    for i in range(10000):
        start_points.append(get_start(18)[0])
    start_points = set(start_points)
    goal_point = (size//2, size//2)
    return start_points, goal_point

def validate(model, valid_data,num_actions):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for t in valid_data:
            decoded_values, all_policies = model(t[0])
            decode_loss = F.mse_loss(decoded_values[0], t[0], reduction='sum')

            first_policy = all_policies[0]
            second_policy = all_policies[1].view(num_actions, -1)
            third_policy = all_policies[2].view(num_actions, num_actions, -1)
            fourth_policy = all_policies[3].view(num_actions, num_actions, num_actions, -1)

            second_layer_probs = first_policy * second_policy
            third_layer_probs = second_layer_probs * third_policy
            fourth_layer_probs = third_layer_probs * fourth_policy

            first = torch.flatten(first_policy).view(num_actions, 1, 1, 1)
            second = torch.flatten(second_layer_probs).view(num_actions**2, 1, 1, 1)
            third = torch.flatten(third_layer_probs).view(num_actions**3, 1, 1, 1)
            fourth = torch.flatten(fourth_layer_probs).view(num_actions**4, 1, 1, 1)

            first_loss = (F.mse_loss(decoded_values[1], t[1], reduction='none') * first).sum()
            second_loss = (F.mse_loss(decoded_values[2], t[2], reduction='none') * second).sum()
            third_loss = (F.mse_loss(decoded_values[3], t[3], reduction='none') * third).sum()
            fourth_loss = (F.mse_loss(decoded_values[4], t[4], reduction='none') * fourth).sum()

            l2w, l3w, l4w = 1, 1, 1
            total_loss += first_loss + second_loss * l2w + third_loss * l3w + fourth_loss * l4w + decode_loss

    return (total_loss / len(valid_data)).item()

def store_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append([name, param.grad.norm().item()])
    return gradients




def train(model, optimizer, train_data, valid_data, num_actions, grad_clip, epochs):
    all_gradients = []
    raw_losses = []
    avg_train_loss = 0
    avg_valid_loss = 0
    for epoch in range(epochs):  # epochs
        model.train()  # Set the model to training mode
        avg_loss = 0
        temp_loss = 0
        temp_raw_loss = 0
        sample_count = 0

        avg_raw_loss = 0
        for t in random.sample(train_data, len(train_data)):  # sample through all data in random order each epoch
            # Get reconstruction loss to help ground abstract state
            decoded_values, all_policies = model(t[0])
            decode_loss = F.mse_loss(decoded_values[0], t[0], reduction='sum')

            # Get transition probabilities for each state
            first_policy = all_policies[0]
            second_policy = all_policies[1].view(num_actions, -1)
            third_policy = all_policies[2].view(num_actions, num_actions, -1)
            fourth_policy = all_policies[3].view(num_actions, num_actions, num_actions, -1)

            # These should all add to 1 (in testing there seems to be some small rounding error)
            second_layer_probs = first_policy * second_policy
            third_layer_probs = second_layer_probs * third_policy
            fourth_layer_probs = third_layer_probs * fourth_policy

            # Flatten transition probabilities to then weigh with loss of each predicted state at each layer
            first = torch.flatten(first_policy).view(num_actions, 1, 1, 1)
            second = torch.flatten(second_layer_probs).view(num_actions**2, 1, 1, 1)
            third = torch.flatten(third_layer_probs).view(num_actions**3, 1, 1, 1)
            fourth = torch.flatten(fourth_layer_probs).view(num_actions**4, 1, 1, 1)

            first_loss = (F.mse_loss(decoded_values[1], t[1], reduction='none') * first).sum()
            second_loss = (F.mse_loss(decoded_values[2], t[2], reduction='none') * second).sum()
            third_loss = (F.mse_loss(decoded_values[3], t[3], reduction='none') * third).sum()
            fourth_loss = (F.mse_loss(decoded_values[4], t[4], reduction='none') * fourth).sum()

            # For experimenting with different weights on different layers
            raw_loss = (first_loss + second_loss + third_loss + fourth_loss).detach().item()
            raw_losses.append(raw_loss)
            l2w, l3w, l4w = 1, 1, 1
            total_loss = first_loss + second_loss * l2w + third_loss * l3w + fourth_loss * l4w + decode_loss

            # break if total loss is nan
            if torch.isnan(total_loss):
                raise ValueError("NAN LOSS")

            temp_loss += total_loss
            temp_raw_loss += raw_loss
            sample_count += 1

            if sample_count % 1 == 0:
                optimizer.zero_grad()
                temp_loss.backward()

                # Monitor gradients before clipping and stepping
                all_gradients.append(store_gradients(model))

                # Uncomment if you want to use gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                avg_loss += temp_loss.item()
                avg_raw_loss += temp_raw_loss
                temp_loss = 0
                temp_raw_loss = 0

        # To handle the case where the number of samples is not a multiple of 1
        if sample_count % 1 != 0:
            optimizer.zero_grad()
            temp_loss.backward()
            
            # Monitor gradients before clipping and stepping
            all_gradients.append(store_gradients(model))

            # Uncomment if you want to use gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            avg_loss += temp_loss.item()
            avg_raw_loss += temp_raw_loss

        avg_train_loss = avg_loss / len(train_data)
        #avg_train_raw_loss = avg_raw_loss / len(train_data)

        # Perform validation
        avg_valid_loss = validate(model, valid_data, num_actions)
    return all_gradients[-1], avg_train_loss, avg_valid_loss

def parse_int_list(arg):
    return [int(x) for x in arg.strip('[]').split(',')]
def parse_float_list(arg):
    return [float(x) for x in arg.strip('[]').split(',')]



def main():
    parser = argparse.ArgumentParser(description='Train TreeQN model')
    parser.add_argument('--num_actions', type=parse_int_list, default=[4], help='Number of actions')
    parser.add_argument('--embedding_dim', type=parse_int_list, default=[2], help='Embedding dimension')
    parser.add_argument('--gamma', type=parse_float_list, default=[1.0], help='Discount factor')
    parser.add_argument('--learning_rate', type=parse_float_list, default=[1e-4], help='Learning rate')
    parser.add_argument('--grad_clip', type=parse_float_list, default=[1.0], help='Gradient clipping value')
    parser.add_argument('--epochs', type=parse_int_list, default=[30], help='Number of training epochs')

    args = parser.parse_args()
    #print num actions

    num_actions = args.num_actions
    embedding_dim = args.embedding_dim
    gamma = args.gamma
    learning_rate = args.learning_rate
    grad_clip = args.grad_clip
    epochs = args.epochs


    s, goal_point = max_starting_points()
    start_points = list(s)
    train_start_points = start_points[:len(start_points)//2]
    test_start_points = start_points[len(start_points)//2:]

    train_data = [get_trajectory(18,start_point,goal_point) for start_point in train_start_points]
    valid_data = [get_trajectory(18,start_point,goal_point) for start_point in test_start_points]
    for na in num_actions:
        for ed in embedding_dim: 
            for gam in gamma:
                for lr in learning_rate:
                    for gc in grad_clip:
                        for ep in epochs:
                            print("Number of Actions: ", na)
                            print("Embedding Dimension: ", ed)
                            print("Discount Factor: ", gam)
                            print("Learning Rate: ", lr)
                            print("Gradient Clipping: ", gc)
                            print("Epochs: ", ep)
                            model = TreeQN(input_shape = torch.zeros(1,1,20,20).shape,num_actions=na, tree_depth =4, embedding_dim=ed, td_lambda=1, gamma=gam)
                            optimizer = Adam(model.parameters(), lr=lr)
                            try:
                                gradient, train_loss, valid_loss = None, None, None
                                for i in range(10):
                                    try:
                                        gradient, train_loss, valid_loss = train(model= model, optimizer=optimizer, train_data=train_data, valid_data=valid_data, num_actions=na, grad_clip=gc, epochs=ep)
                                        print("Train Loss: ", train_loss)
                                        print("Validation Loss: ", valid_loss)
                                        
                                        #write final loss to file and parser args
                                        with open('final_losses2.txt', 'a') as f:
                                            f.write(f"Number of Actions: {na}\n")
                                            f.write(f"Embedding Dimension: {ed}\n")
                                            f.write(f"Discount Factor: {gam}\n")
                                            f.write(f"Learning Rate: {lr}\n")
                                            f.write(f"Gradient Clipping: {gc}\n")
                                            f.write(f"Epochs: {ep}\n")
                                            #f.write(f"Gradient Norms: {gradient}\n")
                                            f.write(f"Train Loss: {train_loss}\n")
                                            f.write(f"Validation Loss: {valid_loss}\n")
                                            f.write("\n\n")
                                        break
                                    except:
                                        raise ValueError("NAN LOSS")
                            except:
                                with open('final_losses.txt', 'a') as f:
                                    f.write(f"Number of Actions: {na}\n")
                                    f.write(f"Embedding Dimension: {ed}\n")
                                    f.write(f"Discount Factor: {gam}\n")
                                    f.write(f"Learning Rate: {lr}\n")
                                    f.write(f"Gradient Clipping: {gc}\n")
                                    f.write(f"Epochs: {ep}\n")
                                    f.write(f"Could not train model. Nan Values\n")
                                    f.write("\n\n")
                                print("Could not train model")
                                continue
                                

if __name__ == '__main__':
    main()