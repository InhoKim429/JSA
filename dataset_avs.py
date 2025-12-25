import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import csv

import pdb

import random
import torchaudio
import json

def load_all_bboxes(annotation_dir, format='s4'):
    gt_bboxes = {}
    if format == 's4':
        with open('metadata/s4_box_processed.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            gt_bboxes[annotation['image']] = annotation['gt_box']

    elif format == 'ms3':
        with open('metadata/ms3_box_processed.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            gt_bboxes[annotation['image']] = annotation['gt_box']
    return gt_bboxes

def bbox2gtmap(bboxes):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
        gt_map += temp

    gt_map[gt_map > 0] = 1
    return gt_map

def load_spectrogram_torchaudio(path, dur=5.0, transformations=None):
    waveform, sample_rate = torchaudio.load(path)
    new_sample_rate = 16000

    if sample_rate != new_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        waveform = resampler(waveform)

    num_required_samples = int(new_sample_rate * dur)
    if waveform.shape[1] < num_required_samples:
        repeats = (num_required_samples // waveform.shape[1]) + 1
        waveform = waveform.repeat(1, repeats)
        waveform = waveform[:, :num_required_samples]
        start = 0.0

    else:
        start = (waveform.shape[1] - num_required_samples) // 2
        waveform = waveform[:, start:start + num_required_samples]

    if transformations:
        waveform = transformations(waveform)

    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=512,
        hop_length=160,
        center=True,
        power=2.0
    )

    spectrogram = spectrogram_transform(waveform)
    # Convert stereo to mono by taking mean across channels
    spectrogram = torch.mean(spectrogram, dim=0, keepdim=True)
    spectrogram = torch.log(spectrogram + 1e-7)
    return spectrogram, start / sample_rate

class MS3Dataset(Dataset):
    """Dataset for multiple sound source segmentation"""
    def __init__(self, audio_path, image_path, mask_path, all_bboxes, testset, audio_length):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.all_bboxes = all_bboxes
        self.testset = testset
        self.audio_length = audio_length
        self.mask_num = 5

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        self.aud_transform = transforms.Compose([
            transforms.Normalize(mean=[0.0], std=[12.0])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        return

    def __getitem__(self, index):
        file = self.testset[index]
        # bboxes = self.all_bboxes[file]
        audio_fn = os.path.join(self.audio_path, file + '.wav')
        spectrogram, audio_ss = load_spectrogram_torchaudio(audio_fn, dur=self.audio_length)
        spectrogram = self.aud_transform(spectrogram)

        imgs, audio, masks = [], [], []
        for id in range(1, self.mask_num + 1):
            try:
                img_fn = os.path.join(self.image_path, file + '_%d.png' %id)
                img = Image.open(img_fn).convert('RGB')
                img = self.img_transform(img)

                mask_fn = os.path.join(self.mask_path, file + '_%d.png' %id)
                mask = Image.open(mask_fn).convert('P')
                mask = self.mask_transform(mask)
            
            except:
                break

            imgs.append(img)
            audio.append(spectrogram)
            masks.append(mask)

        imgs_tensor = torch.stack(imgs, dim=0)
        audio_tensor = torch.stack(audio, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_tensor, masks_tensor, file, '_'

    def __len__(self):
        return len(self.testset)
    
class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, audio_path, image_path, mask_path, all_bboxes, testset, audio_length):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.all_bboxes = all_bboxes
        self.testset = testset
        self.audio_length = audio_length
        self.mask_num = 5

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        self.aud_transform = transforms.Compose([
            transforms.Normalize(mean=[0.0], std=[12.0])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        return

    def __getitem__(self, index):
        file = self.testset[index]

        audio_fn = os.path.join(self.audio_path, file + '.wav')
        spectrogram, audio_ss = load_spectrogram_torchaudio(audio_fn, dur=self.audio_length)
        spectrogram = self.aud_transform(spectrogram)

        imgs, audio, masks = [], [], []
        for id in range(1, self.mask_num + 1):
            try:
                img_fn = os.path.join(self.image_path, file + '_%d.png' %id)
                img = Image.open(img_fn).convert('RGB')
                img = self.img_transform(img)

                mask_fn = os.path.join(self.mask_path, file + '_%d.png' %id)
                mask = Image.open(mask_fn).convert('P')
                mask = self.mask_transform(mask)
            
            except:
                break

            imgs.append(img)
            audio.append(spectrogram)
            masks.append(mask)

        # masks_tensor = torch.stack([torch.from_numpy(mask).unsqueeze(0) for mask in masks], dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        imgs_tensor = torch.stack(imgs, dim=0)
        audio_tensor = torch.stack(audio, dim=0)

        return imgs_tensor, audio_tensor, masks_tensor, file, '_'

    def __len__(self):
        return len(self.testset)

def get_ms3_dataset(args):
    audio_path = os.path.join(args.test_data_path, 'ms3', 'audio_wav')
    image_path = os.path.join(args.test_data_path, 'ms3', 'visual_frames')
    mask_path = os.path.join(args.test_data_path, 'ms3', 'gt_masks')
    metadata_path = os.path.join('metadata', 'ms3_meta_data.csv')

    audio_length = args.aud_length
    testset = set([item[0] for item in csv.reader(open(metadata_path))])

    testset = sorted(list(testset))
    for file in testset:
        if not os.path.exists(os.path.join(audio_path, file + '.wav')):
            testset.remove(file)

    all_bboxes = None # load_all_bboxes(args.test_data_path, format='ms3')
    return MS3Dataset(audio_path, image_path, mask_path, all_bboxes, testset, audio_length)

def get_s4_dataset(args):
    audio_path = os.path.join(args.test_data_path, 's4', 'audio_wav')
    image_path = os.path.join(args.test_data_path, 's4', 'visual_frames')
    mask_path = os.path.join(args.test_data_path, 's4', 'gt_masks')
    metadata_path = os.path.join('metadata', 's4_meta_data.csv')

    audio_length = args.aud_length
    testset = set([item[0] for item in csv.reader(open(metadata_path))])

    testset = sorted(list(testset))

    for file in testset:
        if not os.path.exists(os.path.join(audio_path, file + '.wav')):
            testset.remove(file)

    all_bboxes = load_all_bboxes(args.test_data_path, format='s4')
    return S4Dataset(audio_path, image_path, mask_path, all_bboxes, testset, audio_length)

if __name__ == "__main__":
    import argparse

    args = argparse.Namespace()
    args.test_data_path = '/data04/inho/preprocessed_dataset/AVSBench/'
    args.aud_length = 5.0
    train_dataset = get_s4_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)
    
    print(len(train_dataset))

    for img, spec, mask, file, _ in train_dataloader:
        # img, spec, mask = batch_data # [b, 5, 3, 224, 224], [b, 1, 257, 501], [b, 1, 1, 224, 224]
        pdb.set_trace()
    pdb.set_trace()