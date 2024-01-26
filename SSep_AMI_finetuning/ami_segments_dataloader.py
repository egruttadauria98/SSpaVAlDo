import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile


class AMI_segments(Dataset):

    dataset_name = "AMI_segments"

    def __init__(
        self,
        split,
        parent_path="path/to/data",
        csv_name="metadata",
        n_src=2, 
        use_sum_mixture=False,
        return_id=False
    ):
        self.return_id = return_id

        if split not in ['train', 'test', 'valid', 'train_test']:
          raise Exception("Split must be either train, test or valid")

        if n_src == 2:
            print("Working with segments of size 5 seconds")
        elif n_src == 3:
            print("Working with segments of size 35 seconds")
        else:
            raise Exception("n_src must be either 2 or 3")


        parent_path = os.path.join(parent_path, f"random_{n_src}spks")
        #self.csv_path = os.path.join(parent_path, csv_name+"_"+split+".csv")
        #self.csv_path = os.path.join(parent_path, csv_name+"_"+split+"_new.csv")
        self.csv_path = os.path.join(parent_path, csv_name)

        self.n_src = n_src
        self.use_sum_mixture = use_sum_mixture

        # Open csv file
        self.df = pd.read_csv(self.csv_path, on_bad_lines='skip')
        initial_len = len(self.df)

        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()
            print(f"\n\n\nATTENTION: THERE ARE DUPLICATES IN THE {split} DATAFRAME")
            print(f"Initial len: {initial_len}")
            print(f"Final len: {len(self.df)}\n\n\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        try:
            # Get the row in dataframe
            row = self.df.iloc[idx]

            # Get the sources
            s1_path = row['path_source_1']
            s1, _ = sf.read(s1_path, dtype="float32")
            
            if len(s1.shape) > 1:
                s1 = np.sum(s1, axis=1)/s1.shape[1]

            s2_path = row['path_source_2']
            s2, _ = sf.read(s2_path, dtype="float32")
            if len(s2.shape) > 1:
                s2 = np.sum(s2, axis=1)/s2.shape[1]

            if self.n_src == 3:
                s3_path = row['path_source_3']
                s3, _ = sf.read(s3_path, dtype="float32")
                if len(s3.shape) > 1:
                    s3 = np.sum(s3, axis=1)/s3.shape[1]
                sources_list = [s1, s2, s3]
            else:
                sources_list = [s1, s2]

            if self.use_sum_mixture:
                if self.n_src == 3:
                    mixture = s1+s2+s3
                else:
                    mixture = s1+s2
            else:
                mixture_path = row['path_mixture']
                mixture, _ = sf.read(mixture_path, dtype="float32")
                if len(mixture.shape) > 1:
                    mixture = np.sum(mixture, axis=1)/mixture.shape[1]

            # Convert to torch tensor
            mixture = torch.from_numpy(mixture)
            # Stack sources
            sources = np.vstack(sources_list)
            # Convert sources to tensor
            sources = torch.from_numpy(sources)
            if not self.return_id:
                return mixture, sources
            
            raise NotImplementedError("The IDs of each recording are not retrieved at the moment")
            # 5400-34479-0005_4973-24515-0007.wav
            #id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
            #return mixture, sources, [id1, id2]
        except:
            print(f"\n\nError with index {idx}\n\n")
            new_idx = random.randint(0, len(self.df)-1)
            return self.__getitem__(new_idx)
