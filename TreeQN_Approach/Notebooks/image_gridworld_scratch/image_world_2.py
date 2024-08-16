import torch

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


def hard_policy(start_state, goal_point):
    def find_trajectories(x, y, goal_x, goal_y, path):
        # Base case: if the current state is the goal, return the path
        if (x, y) == (goal_x, goal_y):
            return [path]
        
        trajectories = []

        # Move right if possible
        if x < goal_x:
            trajectories += find_trajectories(x + 1, y, goal_x, goal_y, path + [(x + 1, y)])
        
        # Move left if possible
        if x > goal_x:
            trajectories += find_trajectories(x - 1, y, goal_x, goal_y, path + [(x - 1, y)])
        
        # Move up if possible
        if y < goal_y:
            trajectories += find_trajectories(x, y + 1, goal_x, goal_y, path + [(x, y + 1)])
        
        # Move down if possible
        if y > goal_y:
            trajectories += find_trajectories(x, y - 1, goal_x, goal_y, path + [(x, y - 1)])

        return trajectories
    
    x, y = start_state
    goal_x, goal_y = goal_point

    # Start finding trajectories with the initial path being the start state
    return find_trajectories(x, y, goal_x, goal_y, [start_state])

def trajectory_to_tensor(trajectory):
    tensor_traj = []
    goal_point = trajectory[-1]
    for i in range(len(trajectory)):
        x, y = trajectory[i] #x,y point
        tensor = torch.zeros(1, 5, 5)
        tensor[0][x][y] = 1
        tensor[0][goal_point[0]][goal_point[1]] = -1
        #add noise
        # noise = torch.randn(1, 5, 5) * 0.1
        # tensor += noise
        tensor_traj.append(tensor)
    return tensor_traj

def get_train_test_data(distance=4,batch_size=5): #not using right now
    train_start_points, test_start_points = get_all_start_points(distance)
    train_trajectories = []
    for goal_point, start_points in train_start_points:
        for start_point in start_points:
            train_trajectories += hard_policy(start_point, goal_point)

    test_trajectories = []
    for goal_point, start_points in test_start_points:
        for start_point in start_points:
            test_trajectories += hard_policy(start_point, goal_point)



    train_tensor_trajectories = [trajectory_to_tensor(traj) for traj in train_trajectories]
    test_tensor_trajectories = [trajectory_to_tensor(traj) for traj in test_trajectories]

    stacked_train_trajectories = []
    for traj in train_tensor_trajectories:
        stacked_train_trajectories.append(torch.stack(traj))

    stacked_test_trajectories = []
    for traj in test_tensor_trajectories:
        stacked_test_trajectories.append(torch.stack(traj))

    # combined_train_tensors = []
    # # Iterate through the list in steps of 5
    # for i in range(0, len(stacked_train_trajectories), batch_size):
    #     # Combine 5 tensors at a time along the batch dimension (dim=0)
    #     combined_tensor = torch.cat(stacked_train_trajectories[i:i+batch_size], dim=0)
    #     combined_train_tensors.append(combined_tensor)

    # combined_test_tensors = []
    # for i in range(0, len(stacked_test_trajectories), batch_size):
    #     combined_tensor = torch.cat(stacked_test_trajectories[i:i+batch_size], dim=0)
    #     combined_test_tensors.append(combined_tensor)
        
    return stacked_train_trajectories, stacked_test_trajectories