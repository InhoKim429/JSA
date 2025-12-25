import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torch

import cv2
import torchaudio

# augment on audio maps
class FrequencyMask(object):
    """
    Implements frequency masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     FrequencyMask(max_width=10, use_mean=False),
        >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the frequency mask is to be applied.
        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(object):
    """
    Implements time masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)
    
      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     TimeMask(max_width=10, use_mean=False),
        >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the time mask is to be applied.
        Returns:
            Tensor: Transformed image with Time Mask.
        """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string

def load_image(path):
    return Image.open(path).convert('RGB')

def load_spectrogram(path, dur=3., rand=False):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base

    if rand:
        audio_ss = max(random.uniform(0.4, 0.6) * (audio_dur - dur), 0)
    else:
        audio_ss = max(float(audio_dur)/2 - dur/2, 0)
        # mid_ss = max(float(audio_dur)/2 - dur/2, 0)

    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram, audio_ss

def load_spectrogram_torchaudio(path, dur=3.0, rand=False, transformations=None):
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
        if rand:
            max_start = waveform.shape[1] - num_required_samples
            start = int(random.uniform(0.4, 0.6) * max_start)
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
    spectrogram = torch.log(spectrogram + 1e-7)
    return spectrogram, start / sample_rate

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs): 
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    return gt_bboxes

def get_all_classes():
    class_list = []
    all_classes={}
    with open('metadata/vggss.json') as json_file:
        annotations = json.load(json_file)
    for annotation in annotations:
        class_name = annotation["class"]
        # all_classes[annotation['file']] = class_name
        if class_name not in class_list:
            class_list.append(class_name)
    
    print('get class amount:', len(class_list))
    for annotation in annotations:
        class_name = annotation["class"]
        all_classes[annotation['file']] = int(class_list.index(class_name))

    return all_classes

def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map

class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, label_dict, image_path, audio_path,
                 audio_dur=3., all_bboxes=None, bbox_format='flickr',
                 hard_img=False, hard_aud=False, rand_aud=False, mode='eval'):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path
        self.label_dict = label_dict
        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format

        self.hard_img = hard_img
        self.hard_aud = hard_aud
        self.rand_aud = rand_aud
        self.mode = mode

        self.eval_img_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        self.eval_aud_transform = transforms.Compose([
            transforms.Normalize(mean=[0.0], std=[12.0])])
        
        if self.hard_img:
            self.img_transform = transforms.Compose([
                transforms.Resize(int(224 * 1.1), Image.BICUBIC),
                transforms.RandomCrop((224, 224)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(int(224 * 1.1), Image.BICUBIC),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        if self.hard_aud:
            self.aud_transform = transforms.Compose([
                FrequencyMask(max_width=10, use_mean=False),
                TimeMask(max_width=10, use_mean=False),
                transforms.Normalize(mean=[0.0], std=[12.0])])
        
        else:
            self.aud_transform = transforms.Compose([
                transforms.Normalize(mean=[0.0], std=[12.0])])
        return

    def __len__(self):
        return len(self.image_files)
    
    def getitem_train(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = load_image(img_fn)
        hard_frame = self.img_transform(frame)

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = np.load(audio_fn)
        spectrogram = np.log(spectrogram + 1e-8)
        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = self.aud_transform(spectrogram)

        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        dict_label = file_id[:-7]
        label = self.label_dict[dict_label] if dict_label in self.label_dict else '_'
        return hard_frame, spectrogram, bboxes, file_id, label
    
    def getitem_eval(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = load_image(img_fn)
        frame = self.eval_img_transform(frame)

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram, audio_ss = load_spectrogram_torchaudio(audio_fn, dur=self.audio_dur, rand=False)
        spectrogram = self.eval_aud_transform(spectrogram)

        bboxes = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        label = self.label_dict[file_id] if file_id in self.label_dict else '_'
        return frame, spectrogram, bboxes, file_id, label

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.getitem_train(idx)
        
        else:
            return self.getitem_eval(idx)

def get_train_dataset(args, hard_img, hard_aud, rand_aud):
    audio_path = os.path.join(args.train_data_path, 'audio') # f"{args.train_data_path}/audio/"
    image_path = os.path.join(args.train_data_path, 'frames') # f"{args.train_data_path}/frames/"
    img_format = '.jpg'

    audio_length = args.aud_length

    # List directory
    audio_files = {fn.split('.npy')[0] for fn in os.listdir(audio_path) if fn.endswith('.npy')} # numpy load
    # audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')} # torchaudio load
    image_files = {fn.split(img_format)[0] for fn in os.listdir(image_path) if fn.endswith(img_format)}
    avail_files = audio_files.intersection(image_files)
    csv_dir = os.path.join('metadata_temp/vggsound.csv')

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        # print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = [dt+'.npy' for dt in avail_files]
    # audio_files = [dt+'.wav' for dt in avail_files]
    image_files = [dt+img_format for dt in avail_files]

    label_dict = {item[0]: item[2] for item in csv.reader(open(csv_dir))}

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        label_dict=label_dict,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=audio_length,
        hard_img=hard_img,
        hard_aud=hard_aud,
        rand_aud=rand_aud,
        mode='train'
    )

def get_test_dataset(args, test_set):
    root_dir = '/data04/inho/preprocessed_dataset/'
    if test_set == 'flickr':
        testcsv = 'metadata/flickr_test_SLAVC.csv'
        audio_path = os.path.join(root_dir, "Flickr/test/audio/")
        image_path = os.path.join(root_dir, "Flickr/test/frames/")
        test_gt_path = os.path.join(root_dir, "Flickr/test/Annotations/")
    elif test_set == 'vggss':
        testcsv = 'metadata/vggss_test.csv'
        audio_path = os.path.join(root_dir, "VGGSound/audio/")
        image_path = os.path.join(root_dir, "VGGSound/frames/")
        test_gt_path = "metadata/vggss.json"
    elif test_set == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
        audio_path = os.path.join(root_dir, "VGGSound/audio/")
        image_path = os.path.join(root_dir, "VGGSound/frames/")
        test_gt_path = "metadata/vggss_heard_test.json"
    elif test_set == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
        audio_path = os.path.join(root_dir, "VGGSound/audio/")
        image_path = os.path.join(root_dir, "VGGSound/frames/")
        test_gt_path = "metadata/vggss_unheard_test.json"
    else:
        raise NotImplementedError
    
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss',
                   'avs_ms3': 'avs',
                   'avs_s4': 'avs'}[test_set]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]

    audio_dur = args.aud_length

    # Bounding boxes
    all_bboxes = load_all_bboxes(test_gt_path, format=bbox_format)

    if test_set == 'vggss':
        with open(test_gt_path) as f:
            json_data = json.load(f)
            label_dict = {item['file']: item['class'] for item in json_data}
    else:
        label_dict = {}

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        label_dict = label_dict,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=audio_dur,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
        hard_img=False,
        hard_aud=False,
        rand_aud=False,
        mode='eval'
    )

def inverse_normalize(tensor):

    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor

# if __name__ == '__main__':
#     from scipy.io import wavfile
#     dir = '/data04/inho/preprocessed_dataset/flickr_trainset/audio/'
#     audio = '9998522424.wav'

#     # dir = '/data04/inho/preprocessed_dataset/Flickr/test/audio/'
#     # audio = '2349424942.wav'

#     # dir = '/data04/inho/preprocessed_dataset/VGGSound/audio/'
#     # audio = 'zpWuikVorYg_000032.wav'

#     path = os.path.join(dir, audio)
#     duration = 3.0
#     mode = 'train'

#     samplerate, samples = wavfile.read(path)
    
#     if mode == 'test':
#         samp_lower = float(len(samples)//22050)/2 - duration/2
#         samp_higher = float(len(samples)//22050)/2 + duration/2

#         audio = samples[int(samp_lower*22050):int(samp_higher*22050)]

#         audio = audio / 32767

#     else:
#         audio = samples / 32767

#     # Repeat if audio is too short
#     if audio.shape[0] < samplerate * duration:
#         n = int(samplerate * duration / audio.shape[0]) + 1
#         audio = np.tile(audio, n)
#     audio = audio[:int(samplerate * duration)]

#     frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
#     spectrogram = np.log(spectrogram + 1e-7)
#     print(spectrogram.shape)

if __name__ == '__main__':
    dir = '/data04/inho/preprocessed_dataset/flickr_trainset/audio/'
    audio = '9998522424.wav'

    # dir = '/data04/inho/preprocessed_dataset/Flickr/test/audio/'
    # audio = '2349424942.wav'

    # dir = '/data04/inho/preprocessed_dataset/VGGSound/audio/'
    # audio = 'zpWuikVorYg_000032.wav'

    path = os.path.join(dir, audio)

    # path = '/data04/inho/preprocessed_dataset/VGGSound/audio/3-M_MA09Tgc_000018.wav'
    dur = 3.0

    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)
    print(samplerate)

    import pdb ; pdb.set_trace()

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    import pdb; pdb.set_trace()

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    # noverlap = 326 -> 257 x 256
    # spectrogram = load_spectrogram(file_path, dur=3.)
    # print(times)
    # print(spectrogram.shape)

    import pdb; pdb.set_trace()
