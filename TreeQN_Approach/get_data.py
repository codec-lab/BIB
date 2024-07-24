import json
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

NR_EPS = 9
NR_GPUs = torch.cuda.device_count()
BATCH_SIZE = 6
NR_WORKERS = NR_GPUs * 2
NR_EPOCHS = 6
VAL_INTERVAL = int(4000 / BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIM = 32
DEPTH = 5
DIM_HEAD = 32
HEADS = 8
MLP_DIM = 256

MAX_LENGTH = 90
SUBSAMPLE_FACTOR = 5
TOP_K = 3

IMG_SZ = 84
IMG_SZ_COMPRESSED = 21
IMG_SZ_RAW = 200
IMG_CHANNELS = 3

class FrameDataset(torch.utils.data.Dataset):


#On init, it calls index data basically... except...
    def __init__(self, path, device, types=None, mode="train", process_data=1, train_split=0.8, val_split=0.05,
                 test_split=0.15):
        self.device = device
        self.path, self.types, self.mode = path, types, mode
        self.json_list, self.path_list = [], []
        self.train_split, self.val_split, self.test_split = train_split, val_split, test_split
        assert train_split + val_split + test_split == 1
        # read video files
        self._read_video_files()
        self.data_tuples = []
        # process json files to extract frame indices for training bib_data
        if process_data:
            for t in types:
                self._process_data(t)
        else:
            for t in types:
                with open(os.path.join(self.path, f'index_dict_{mode}_{t}.json'), 'r') as fp:
                    index_dict = json.load(fp)
                self.data_tuples.extend(index_dict['data_tuples'])


#create the index_data json. Used right away -- makes sense
    def _process_data(self, t):
        # index videos to make frame retrieval easier
        print('processing files')
        for j, v in zip(self.json_list, self.path_list):
            try:
                with open(j, 'r') as f:
                    state = json.load(f)
            except UnicodeDecodeError as e:
                print(f'file skipped {j} with {e}')
                continue
            ep_lens = [len(state[str(x)]) for x in range(NR_EPS)]
            all_eps = []
            first_actions = []
            #for each json,mp4,
                #for each episode in file (9 episodes)
            for e in range(NR_EPS):
                this_ep = []
                first_action = -1
                past_len = sum(ep_lens[:e])
                for f in range(ep_lens[e] - 83):
                    f0y, f0x = state[str(e)][str(f)]
                    f1y, f1x = state[str(e)][str(f + 1)]
                    dx = (f1x - f0x)
                    dy = (f1y - f0y)
                    action = (dx, dy)
                    if action != (0, 0) and first_action == -1: #Start recording actions once its moving
                        first_action = f + past_len
                        this_ep.append((f0x + 6, f0y + 6))
                    if first_action != -1: #append all other actions until end of episode (length - 83)
                        this_ep.append(action) #THIS EP is getting the actual actions
                if len(this_ep) == 0:
                    this_ep.append((f0x + 6, f0y + 6))
                all_eps.append(this_ep) 
                first_actions.append(first_action) #ONLY TIME first_action is appened 
            #For each json,mp4 file...
            #video file name
            #data tuple is vid file, first action of each trial in ep, actual actions in each trial
            #data_tuple[n] is a whole episode. data_tuple[n][2][m] is trial m actions in episode n
            self.data_tuples.append((v, first_actions, all_eps))
        index_dict = {'data_tuples': self.data_tuples}
        with open(os.path.join(self.path, f'index_dict_{self.mode}_{t.split("/")[-1]}.json'), 'w') as fp:
            json.dump(index_dict, fp)

#just for train/test/val
    def _fill_lists(self, t, start, stop):
        self.path_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                           if
                           x.endswith(f'e.mp4')][start:stop]
        self.json_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                           if
                           x.endswith(f'e.json')][start:stop]

    def _read_video_files(self):
        for t in self.types:
            print(f'reading files of type {t} in {self.mode}')
            type_length = len(os.listdir(os.path.join(self.path, t))) // 2
            print(type_length)
            if self.mode == 'eval':
                self.path_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                                   if
                                   x.endswith(f'e.mp4') or x.endswith(f'u.mp4')]
                self.json_list += [os.path.join(self.path, t, x) for x in sorted(os.listdir(os.path.join(self.path, t)))
                                   if
                                   x.endswith(f'e.json') or x.endswith(f'u.json')]
                continue
            elif self.mode == 'train':
                start = 0
                stop = int(type_length * self.train_split)
            elif self.mode == 'test':
                start = int(type_length * self.train_split)
                stop = int(type_length * (self.train_split + self.test_split))
            elif self.mode == 'val':
                start = int(type_length * (self.train_split + self.test_split))
                stop = type_length
            self._fill_lists(t, start, stop)

    @staticmethod
    def _get_frames(data_tuples): #input is data_tuple[ep], really like it should be "data_tuple"
        #data_tuple[ep] is video_file, first action index of each trial in ep, actions following each first
        sub_sampled_traces = []
        printer = False
        #yeah wtf are the sub sampled traces?
        for ep_i, ep in enumerate(data_tuples[2]): #the actions in each episode for each trial
            agent_pos = torch.Tensor(ep)
            #max_length = 90, sub_sample_factor = 5
            trace_length = min(MAX_LENGTH, max(2, agent_pos.shape[0] // SUBSAMPLE_FACTOR + 1))
            #linspace, get to agent_pos.shape in trace length steps
            ls = torch.linspace(0, agent_pos.shape[0] - 1, trace_length).int().unique()
            #img_sz_raw = 200, img_sz_compressed = 21
            x_coords = (agent_pos[:, 0].cumsum(dim=0) / IMG_SZ_RAW * IMG_SZ_COMPRESSED).long()
            y_coords = (agent_pos[:, 1].cumsum(dim=0) / IMG_SZ_RAW * IMG_SZ_COMPRESSED).long()
            path = y_coords * IMG_SZ_COMPRESSED + x_coords
            sub_sampled_trace = torch.index_select(path, 0, ls)

            if printer:
                print('printing for trial 0 of ep ' + str(ep_i))
                print('agent_pos_shape', agent_pos.shape)
                print('agent_pos', agent_pos)
                print('ls', ls)
                print('path', path)
                print('sub_sampled_trace', sub_sampled_trace)
                printer = False

            sub_sampled_traces.append(sub_sampled_trace)

        video_filename = data_tuples[0]
        #print(video_filename)
        cap = cv2.VideoCapture(video_filename)
        frames = []
        for ep_i, ep in enumerate(data_tuples[2]):
            steps = torch.Tensor(sub_sampled_traces[ep_i].shape[0], IMG_CHANNELS, IMG_SZ, IMG_SZ)
            counter = 0
            ls = torch.linspace(data_tuples[1][ep_i], data_tuples[1][ep_i] + len(ep), sub_sampled_traces[ep_i].shape[0])
            #print(ep_i)
            #print('ls', ls, sub_sampled_traces[ep_i].shape[0])
            for frame in ls:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame.item())
                _, frame = cap.read()
                frame = cv2.resize(frame, (IMG_SZ, IMG_SZ))
                frame = torch.tensor(frame).permute(2, 0, 1) / 255.
                steps[counter] = frame
                counter += 1
            frames.append(steps)
        cap.release()
        return sub_sampled_traces, frames

    def __getitem__(self, idx):
        target, step_through = self._get_frames(self.data_tuples[idx])
        return target, step_through

    def __len__(self):
        return len(self.data_tuples)

# def index_data(json_list, path_list):
#     print(f'processing files {len(json_list)}')
#     data_tuples = []
#     for j, v in tqdm(zip(json_list, path_list)):
#         with open(j, 'r') as f:
#             state = json.load(f)
#             #state is misnomer. State is all 9 trials in episode
#         ep_lens = [len(x) for x in state]
#         past_len = 0
#         for e, l in enumerate(ep_lens):
#             #e is episode number, l is length of episode
#             data_tuples.append([])
#             # skip first 30 frames and last 83 frames
#             #are some trials faster or slower??? its l-83, so the last 83 in all are skipped
#             for f in range(30, l - 83):
#                 # find action taken; 
#                 f0x, f0y = state[e][f]['agent'][0]
#                 f1x, f1y = state[e][f + 1]['agent'][0]
#                 dx = (f1x - f0x) / 2.
#                 dy = (f1y - f0y) / 2.
#                 action = [dx, dy]
#                 # action = ACTION_LIST.index([dx, dy])
#                 data_tuples[-1].append((v, past_len + f, action))
#             assert len(data_tuples[-1]) > 0
#             past_len += l
#     return data_tuples
# #gives dx, dy actions

# class FrameDataset():

#     def __init__(self, path, types,action_range = 10, num_transitions = 30, step = 1):
#         self.path = path
#         self.types = types 
#         self.action_range = action_range #actions pooled over 10 frames, default value from paper
#         self.num_transitions = num_transitions # default from paper, confused still on what it is. the 30 before the test?
#         #looks like 30 frames from across all trials in episode. 
#         #idx = 0
#         self.num_trials = 9
#         self.step = step
#         self.data_tuples = self.get_data_tuples(self.path, self.types)

#     def get_data_tuples(self,data_path,types):
#         path_list = []
#         json_list = []

#         #get the list of mp4 path and json path for whatever "type" (preference, etc) is in the directory path
#         for t in types:
#             print(f'reading files of type {t}')
#             paths = [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
#                         x.endswith(f'.mp4')]
#             jsons = [os.path.join(self.path, t, x) for x in os.listdir(os.path.join(self.path, t)) if
#                         x.endswith(f'.json') and 'index' not in x]

#             paths = sorted(paths)
#             jsons = sorted(jsons)

#             path_list += paths
#             json_list += jsons

#         data_tuples = []
#         data_tuples = index_data(json_list, path_list) 
#         return data_tuples

#         #used on the path in datatuples to go from file path to frame
#     def _get_frame(self,video, frame_idx,size = (84,84)): #size is the size of the frame, 84x84 used as scale from paper
#         cap = cv2.VideoCapture(video)
#         # read frame at id and resize
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         _, frame = cap.read()

#         if size is not None:
#             assert frame is not None, f'frame is empty {frame_idx}, {video}'
#             frame = cv2.resize(frame, size)
#         frame = torch.tensor(frame).permute(2, 0, 1)
#         # return frames as a torch tensor f x c x w x h
#         frame = frame.to(torch.float32) / 255.
#         cap.release()
#         return frame
    
#     #trial = [idx * num_trials + t for t in range(num_trials)] #old school way 
#     #for clarity
#     def get_episode_trial_indices(self,episode_index, episode_length = 9):
#         return [episode_index * episode_length + t for t in range(episode_length)]

#     # def get_trial(self,data_tuples, trial_indices, num_transitions,action_range = 10, step=1):
#     #     states = []
#     #     actions = []
#     #     trial_len = []
#     #     #for trial in episode really
#     #     for t in trial_indices:
#     #         trial_len += [(t, n) for n in range(0, len(data_tuples[t]), step)]
#     #         #gets trial_num, frame_num for each trial in episode
#     #     random.shuffle(trial_len)
#     #     #interesting that its shuffled. !

#     #     #trial len is actually all the frames in the whole episode (between 30 and l-83)

#     #     if len(trial_len) < num_transitions:
#     #         return None, None, False
#     #     #num_transitions is like the cap. Don't get trials that are too short
#     #     #its also the max amount of frames to get?

#     #     for t, n in trial_len[:num_transitions]:
#     #         video = data_tuples[t][n][0]
#     #         #video is the video path at trial t, frame n
#     #         states.append(self._get_frame(video, data_tuples[t][n][1]))
#     #         #append cv2 frame to states

#     #         # actions are pooled over frames
#     #         # if there are not enough frames left in the trial, use the remaining frames
#     #         #action at each state is like the average of the next 10 actions, but still each state is standalone
#     #         if len(data_tuples[t]) > n + action_range:
#     #             actions_xy = [d[2] for d in data_tuples[t][n:n + action_range]]
#     #         else:
#     #             actions_xy = [d[2] for d in data_tuples[t][n:]]

#     #         actions_xy = np.array(actions_xy)
#     #         actions_xy = np.mean(actions_xy, axis=0)
#     #         actions.append(actions_xy)
#     #     states = torch.stack(states, dim=0)
#     #     actions = torch.tensor(np.array(actions))
#     #     return states, actions, True
#     #simply returns each video state and the pooled action things for a series of trial indices
#     #the bool is if the trial is too short
#     def get_trial(self,idx):
#         states = []
#         actions = []
#         trial_len = []
#         trial_indices = self.get_episode_trial_indices(idx)
#         for t in trial_indices:
#             trial_len += [(t, n) for n in range(0, len(self.data_tuples[t]), self.step)]
#         random.shuffle(trial_len)
#         if len(trial_len) < self.num_transitions:
#             return None, None, False
#         for t, n in trial_len[:self.num_transitions]:
#             video = self.data_tuples[t][n][0]
#             states.append(self._get_frame(video, self.data_tuples[t][n][1]))
#             if len(self.data_tuples[t]) > n + self.action_range:
#                 actions_xy = [d[2] for d in self.data_tuples[t][n:n + self.action_range]]
#             else:
#                 actions_xy = [d[2] for d in self.data_tuples[t][n:]]
#             actions_xy = np.array(actions_xy)
#             actions_xy = np.mean(actions_xy, axis=0)
#             actions.append(actions_xy)
#         states = torch.stack(states, dim=0)
#         actions = torch.tensor(np.array(actions))
#         return states, actions