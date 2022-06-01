import os
import random


class VideoSelector:
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.algo_ls = [algo for algo in os.listdir(video_folder) if "DS_Store" not in algo]
        self.video_ls = [video for video in os.listdir(os.path.join(video_folder, self.algo_ls[0])) if "DS_Store" not in video]
        self.algo_len, self.video_len = list(range(len(self.algo_ls))), len(self.video_ls)
        self.video_index = 0

    def select(self):
        video_idx = self.video_index
        #video_idx = random.randint(0, self.video_len)
        #import pdb; pdb.set_trace()
        if video_idx >= self.video_len:
             import pdb; pdb.set_trace()
            
        algo1, algo2 = random.sample(self.algo_len, 2)  
        if random.randint(0, 10) > 5:
            video_path2 = os.path.join(self.video_folder, self.algo_ls[algo1], self.video_ls[video_idx])
            video_path1 = os.path.join(self.video_folder, self.algo_ls[algo2], self.video_ls[video_idx])
        else:
            video_path1 = os.path.join(self.video_folder, self.algo_ls[algo1], self.video_ls[video_idx])
            video_path2 = os.path.join(self.video_folder, self.algo_ls[algo2], self.video_ls[video_idx])
        print("Selected video:\n {}\n {}\n\n".format(video_path1, video_path2))
        self.video_index = self.video_index + 1
        return video_path1, video_path2




