import torch
import os
import pandas as pd
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    #root_dir='/dtu/blackhole/1d/214141/ufc10',
    root_dir='./ufc10',
    split='train', 
    transform=None
):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = v2.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    #root_dir='/dtu/blackhole/1d/214141/ufc10',
    root_dir='./ufc10',
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [v2.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)


        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

def get_image_dataloaders(root_dir, transform=None, batch_size=8):
    # Define transforms with normalization
    if transform is None:
      transform = v2.Compose([
        v2.Resize((64, 64)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
      ])

    transform_val = v2.Compose([
        v2.Resize((64, 64)),
        v2.ToTensor()
      ])

    # Create train and validation datasets
    train_dataset = FrameImageDataset(
        root_dir=root_dir,
        split='train',
        transform=transform
    )

    val_dataset = FrameImageDataset(
        root_dir=root_dir,
        split='val',
        transform=transform_val
    )

    test_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='test',
        transform=transform_val
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def get_video_dataloaders(root_dir, transform=None, batch_size=8):
    # Define transforms with normalization
    if transform is None:
      transform = v2.Compose([
        v2.Resize((64, 64)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
      ])

    transform_val = v2.Compose([
        v2.Resize((64, 64)),
        v2.ToTensor()
      ])
    
    # Create train and validation datasets
    train_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        stack_frames=True
    )

    val_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='val',
        transform=transform_val,
        stack_frames=True
    )

    test_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='test',
        transform=transform_val,
        stack_frames=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader