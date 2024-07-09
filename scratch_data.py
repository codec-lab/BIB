import json
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

def index_data(json_list, path_list):
    print(f'processing files {len(json_list)}')
    data_tuples = []
    for j, v in tqdm(zip(json_list, path_list)):
        with open(j, 'r') as f:
            state = json.load(f)
            #state is misnomer. State is all 9 trials in episode
        ep_lens = [len(x) for x in state]
        past_len = 0
        for e, l in enumerate(ep_lens):
            #e is episode number, l is length of episode
            data_tuples.append([])
            # skip first 30 frames and last 83 frames
            #are some trials faster or slower??? its l-83, so the last 83 in all are skipped
            for f in range(30, l - 83):
                # find action taken; 
                f0x, f0y = state[e][f]['agent'][0]
                f1x, f1y = state[e][f + 1]['agent'][0]
                dx = (f1x - f0x) / 2.
                dy = (f1y - f0y) / 2.
                action = [dx, dy]
                # action = ACTION_LIST.index([dx, dy])
                data_tuples[-1].append((v, past_len + f, action))
            assert len(data_tuples[-1]) > 0
            past_len += l
    return data_tuples
#gives dx, dy actions



#old for just training only
#gets all data regardless of e or u in folder (make sure to point to only train folder)
#only training on 1 type at a time though... does paper do all at once? I think it loops through types but only 1 is in the loop but only like 50% confident
def get_train_tuples(data_path,type,process_data):
    path_list = []
    json_list = []

    #get the list of mp4 path and json path for whatever "type" (preference, etc) is in the directory path
    print(f'reading files of type {type}')
    paths = [os.path.join(data_path, type, x) for x in os.listdir(os.path.join(data_path, type)) if
                x.endswith(f'.mp4')]
    jsons = [os.path.join(data_path, type, x) for x in os.listdir(os.path.join(data_path, type)) if
                x.endswith(f'.json') and 'index' not in x]

    paths = sorted(paths)
    jsons = sorted(jsons)

    path_list += paths
    json_list += jsons


    if process_data:
        data_tuples = index_data(json_list, path_list) 
        index_dict = {'data_tuples': data_tuples}
        with open(os.path.join(data_path, f'index_bib_{type}_train.json'), 'w') as fp:
            json.dump(index_dict, fp)
    else:   
        with open(os.path.join(data_path, f'index_bib_{type}_train.json'), 'r') as fp:
            index_dict = json.load(fp)
        data_tuples = index_dict['data_tuples']


    return data_tuples

def get_data_tuples(path,type,process_data):

    path_list_exp = []
    json_list_exp = []
    path_list_un = []
    json_list_un = []


    print(f'reading files of type {type}')
    paths_expected = sorted([os.path.join(path, type, x) for x in os.listdir(os.path.join(path, type)) if
                                x.endswith(f'e.mp4')])
    jsons_expected = sorted([os.path.join(path, type, x) for x in os.listdir(os.path.join(path, type)) if
                                x.endswith(f'e.json') and 'index' not in x])
    paths_unexpected = sorted([os.path.join(path, type, x) for x in os.listdir(os.path.join(path, type)) if
                                x.endswith(f'u.mp4')])
    jsons_unexpected = sorted([os.path.join(path, type, x) for x in os.listdir(os.path.join(path, type)) if
                                x.endswith(f'u.json') and 'index' not in x])

    path_list_exp += paths_expected
    json_list_exp += jsons_expected
    path_list_un += paths_unexpected
    json_list_un += jsons_unexpected

    if process_data:
        data_expected = index_data(json_list_exp, path_list_exp)
        index_dict = {'data_tuples': data_expected}
        with open(os.path.join(path, f'index_bib_test_{type}e.json'), 'w') as fp:
            json.dump(index_dict, fp)

        data_unexpected = index_data(json_list_un, path_list_un)
        index_dict = {'data_tuples': data_unexpected}
        with open(os.path.join(path, f'index_bib_test_{type}u.json'), 'w') as fp:
            json.dump(index_dict, fp)
    else:
        with open(os.path.join(path, f'index_bib_test_{type}e.json'), 'r') as fp:
            index_dict = json.load(fp)
        data_expected = index_dict['data_tuples']

        with open(os.path.join(path, f'index_bib_test_{type}u.json'), 'r') as fp:
            index_dict = json.load(fp)
        data_unexpected = index_dict['data_tuples']


    return data_expected, data_unexpected #tuples of data for expected and unexpected

#used on the path in datatuples to go from file path to frame
def _get_frame(video, frame_idx,size = (84,84)): #size is the size of the frame, 84x84 used as scale from paper
    cap = cv2.VideoCapture(video)
    # read frame at id and resize
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    _, frame = cap.read()

    if size is not None:
        assert frame is not None, f'frame is empty {frame_idx}, {video}'
        frame = cv2.resize(frame, size)
    frame = torch.tensor(frame).permute(2, 0, 1)
    # return frames as a torch tensor f x c x w x h
    frame = frame.to(torch.float32) / 255.
    cap.release()
    return frame

#trial = [idx * num_trials + t for t in range(num_trials)] #old school way 
#for clarity
def get_episode_trial_indices(episode_index, episode_length = 9):
    return [episode_index * episode_length + t for t in range(episode_length)]

def get_trial(data_tuples, trial_indices, num_transitions=30,action_range = 10, step=1):
    states = []
    actions = []
    trial_len = []
    #for trial in episode really
    for t in trial_indices:
        trial_len += [(t, n) for n in range(0, len(data_tuples[t]), step)]
        #gets trial_num, frame_num for each trial in episode
    random.shuffle(trial_len)
    #interesting that its shuffled. !

    #trial len is actually all the frames in the whole episode (between 30 and l-83)

    if len(trial_len) < num_transitions:
        return None, None, False
    #num_transitions is like the cap. Don't get trials that are too short
    #its also the max amount of frames to get?

    for t, n in trial_len[:num_transitions]:
        video = data_tuples[t][n][0]
        #video is the video path at trial t, frame n
        states.append(_get_frame(video, data_tuples[t][n][1]))
        #append cv2 frame to states

        # actions are pooled over frames
        # if there are not enough frames left in the trial, use the remaining frames
        #action at each state is like the average of the next 10 actions, but still each state is standalone
        if len(data_tuples[t]) > n + action_range:
            actions_xy = [d[2] for d in data_tuples[t][n:n + action_range]]
        else:
            actions_xy = [d[2] for d in data_tuples[t][n:]]

        actions_xy = np.array(actions_xy)
        actions_xy = np.mean(actions_xy, axis=0)
        actions.append(actions_xy)
    states = torch.stack(states, dim=0)
    actions = torch.tensor(np.array(actions))
    return states, actions, True
#simply returns each video state and the pooled action things for a series of trial indices
#the bool is if the trial is too short