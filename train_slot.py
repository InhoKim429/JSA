import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import get_train_dataset, get_test_dataset, get_all_classes
from dataset_avs import get_ms3_dataset, get_s4_dataset
import random
import wandb

import test_model
from datetime import datetime

import model_slot
import utils

# from torch.utils.tensorboard import SummaryWriter   

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='noname', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)
    # parser.add_argument('--wandb', action='store_true')
    
    # hyper-params
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--aud_length', default=5.0, type=float)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regu")
    parser.add_argument("--lam1", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=1.0)
    parser.add_argument("--lam3", type=float, default=1.0)
    parser.add_argument("--infer_sharpening", type=float, default=0.1)
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--reciprocal_k", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup epochs")
    parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'])
    # parser.add_argument("--scheduler", action='store_true', help='use scheduler or not')
    # parser.add_argument("--resume", action='store_true')

    # Distributed params
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=None)

    # Evaluation
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')   
    parser.add_argument('--wandb',               type=str, default='false')
    parser.add_argument('--hard_img',            type=str, default='false')
    parser.add_argument('--hard_aud',            type=str, default='false')
    parser.add_argument('--rand_aud',            type=str, default='false')
    parser.add_argument("--scheduler",           type=str, default='false')
    parser.add_argument("--resume",              type=str, default='false')
    parser.add_argument('--save_visualizations', type=str, default='false')

    args = parser.parse_args()

    args.wandb = args.wandb in {'True', 'true'}
    args.hard_img = args.hard_img in {'True', 'true'}
    args.hard_aud = args.hard_aud in {'True', 'true'}
    args.rand_aud = args.rand_aud in {'True', 'true'}
    args.scheduler = args.scheduler in {'True', 'true'}
    args.resume = args.resume in {'True', 'true'}
    args.save_visualizations = args.save_visualizations in {'True', 'true'}

    if args.experiment_name == 'noname':
        now = datetime.now()
        now = now.strftime('%m_%d_%H_%M_%S')
        args.experiment_name = "slotmse_%s_%s_%s" %(args.trainset, args.testset, now)
    return args


def setup_seed(seed):
    print("Random seed: %d" %(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def main(args):
    setup_seed(args.seed)

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    if os.path.exists(model_dir):
        print('WARNING: Directory already exists.')
        # exit()

    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    model = model_slot.mymodel(args)
    print('Model loaded.')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # Optimizer
    print('optimizer: {}, use scheduler: {}'.format(args.optimizer, args.scheduler))
    if args.optimizer == 'adam':
        optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)
    elif args.optimizer == 'sgd':
        optimizer, scheduler = utils.build_optimizer_and_scheduler_sgd(model, args)
    else:
        exit()

    scaler = torch.cuda.amp.GradScaler()

    print('train dataset: ', args.trainset)
    train_dataset = get_train_dataset(args, hard_img=args.hard_img, hard_aud=args.hard_aud, rand_aud=args.rand_aud)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    print('train sample amount:', len(train_loader))

    vggss_set = get_test_dataset(args, 'vggss')
    vggss_loader = torch.utils.data.DataLoader(vggss_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('vggss sample amount:', len(vggss_loader))

    vggss_heard_set = get_test_dataset(args, 'vggss_heard')
    vggss_heard_loader = torch.utils.data.DataLoader(vggss_heard_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('vggss_heard sample amount:', len(vggss_heard_loader))

    vggss_unheard_set = get_test_dataset(args, 'vggss_unheard')
    vggss_unheard_loader = torch.utils.data.DataLoader(vggss_unheard_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('vggss_unheard sample amount:', len(vggss_unheard_loader))

    flickr_set = get_test_dataset(args, 'flickr')
    flickr_loader = torch.utils.data.DataLoader(flickr_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('flickr sample amount:', len(flickr_loader))

    ms3_set = get_ms3_dataset(args)
    ms3_loader = torch.utils.data.DataLoader(ms3_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('ms3 sample amount:', len(ms3_loader))

    s4_set = get_s4_dataset(args)
    s4_loader = torch.utils.data.DataLoader(s4_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=False, persistent_workers=args.workers > 0)
    print('s4 sample amount:', len(s4_loader))

    object_saliency_model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )
    object_saliency_model = object_saliency_model.cuda(args.gpu)

    wandbRun = wandb.init(project = 'SSL_slot_OGL_%s' %(args.trainset),
                          config = vars(args),
                          name = args.experiment_name,
                          anonymous='allow',
                          mode= 'online' if args.wandb else 'disabled')
    
    start_epoch = 0
    vggss_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vggss_heard_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vggss_unheard_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flickr_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ms3_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    s4_best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(start_epoch, args.epochs):
        train(train_loader, model, optimizer, scaler, epoch, args)

        if args.scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

        metrics = {
            'train/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        }
        wandb.log(metrics)

        # Evaluate
        if args.testset == 'flickr':
            flickr_best = validate(flickr_loader, 'flickr', model, object_saliency_model, epoch, flickr_best, model_dir, args)
        elif args.testset == 'vggss':
            vggss_best = validate(vggss_loader, 'vggss', model, object_saliency_model, epoch, vggss_best, model_dir, args)
        elif args.testset == 'vggss_heard':
            vggss_heard_best = validate(vggss_heard_loader, 'vggss_heard', model, object_saliency_model, epoch, vggss_heard_best, model_dir, args)
        elif args.testset == 'vggss_unheard':
            vggss_unheard_best = validate(vggss_unheard_loader, 'vggss_unheard', model, object_saliency_model, epoch, vggss_unheard_best, model_dir, args)
        elif args.testset == 'ms3':
            ms3_best = validate(ms3_loader, 'ms3', model, object_saliency_model, epoch, ms3_best, model_dir, args)
        elif args.testset == 's4':
            s4_best = validate(s4_loader, 's4', model, object_saliency_model, epoch, s4_best, model_dir, args)
        elif args.testset == 'all':
            flickr_best = validate(flickr_loader, 'flickr', model, object_saliency_model, epoch, flickr_best, model_dir, args)
            vggss_best = validate(vggss_loader, 'vggss', model, object_saliency_model, epoch, vggss_best, model_dir, args)
        elif args.testset == 'all_heard':
            vggss_heard_best = validate(vggss_heard_loader, 'vggss_heard', model, object_saliency_model, epoch, vggss_heard_best, model_dir, args)
            vggss_unheard_best = validate(vggss_unheard_loader, 'vggss_unheard', model, object_saliency_model, epoch, vggss_unheard_best, model_dir, args)
        # Checkpoint
        ckp = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1}
        torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
        print(f"Latest model saved to {model_dir}")
        
    wandbRun.finish()
    print(args)
    return

def train(train_loader, model, optimizer, scaler, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    info_losses = AverageMeter('Info', ':.3f')
    con_losses = AverageMeter('Con', ':.3f')
    div_losses = AverageMeter('Div', ':.3f')
    att_losses = AverageMeter('Att', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, info_losses, con_losses, div_losses, att_losses],
        prefix="Warmup: [{}]".format(epoch) if epoch < args.warmup else "Train: [{}]".format(epoch),
    )

    end = time.time()
    for i, (frame, spec, bboxes, file_id, label) in enumerate(train_loader):
        batch_size = frame.size(0)
        data_time.update(time.time() - end)

        if args.gpu is not None:
            frame = frame.cuda(args.gpu, non_blocking=True)
            spec  = spec.cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            info_loss, recon_loss, div_loss, att_loss = model(frame.float(), spec.float())
            if epoch < args.warmup:
                loss = info_loss
            else:
                loss = info_loss + args.lam1 * recon_loss + args.lam2 * div_loss + args.lam3 * att_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        info_losses.update(info_loss.item(), batch_size)
        con_losses.update(recon_loss.item(), batch_size)
        div_losses.update(div_loss.item(), batch_size)
        att_losses.update(att_loss.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
            break
    
    metrics = {
        'train/info_loss': info_losses.avg,
        'train/recon_loss': con_losses.avg,
        'train/div_loss': div_losses.avg,
        'train/att_loss': att_losses.avg,
        'epoch': epoch
    }
    wandb.log(metrics)
    return

def validate(test_loader, testset, model, object_saliency_model, epoch, best, model_dir,args):
    model.eval()
    object_saliency_model.eval()
    
    mAP, auc, \
    mAP_img, auc_img, \
    mAP_ogl, auc_ogl, \
    mAP_orig_obj, auc_orig_obj, \
    mAP_aud_orig_obj, auc_aud_orig_obj, \
    mAP_all_combined, auc_all_combined = \
        test_model.validate_img_aud(test_loader, model, object_saliency_model, './%s/%s/%s' %('final', args.trainset, testset), testset, epoch, args)
    
    if mAP > best[0]: # If sound source localization is the best.
        ckp = {'model': model.state_dict(),
            'epoch': epoch+1}
        torch.save(ckp, os.path.join(model_dir, '%s_best.pth' %(testset)))
        print(f"Best model saved to {model_dir}")

    best[0] = max(mAP, best[0])
    best[1] = max(auc, best[1])
    best[2] = max(mAP_img, best[2])
    best[3] = max(auc_img, best[3])
    best[4] = max(mAP_ogl, best[4])
    best[5] = max(auc_ogl, best[5])
    best[6] = max(mAP_aud_orig_obj, best[6])
    best[7] = max(auc_aud_orig_obj, best[7])
    best[8] = max(mAP_all_combined, best[8])
    best[9] = max(auc_all_combined, best[9])

    # Just for logging
    print('AUD_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP:.4f}', f'{auc:.4f}', f'{best[0]:.4f}', f'{best[1]:.4f}')
    print('OBJ_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_img:.4f}', f'{auc_img:.4f}', f'{best[2]:.4f}', f'{best[3]:.4f}')
    print('OGL_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_ogl:.4f}', f'{auc_ogl:.4f}', f'{best[4]:.4f}', f'{best[5]:.4f}')
    print('ORIG_OBJ_%s/cIoU, auc' %(testset), f'{mAP_orig_obj:.4f}', f'{auc_orig_obj:.4f}')
    print('AUD_ORIG_OBJ_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_aud_orig_obj:.4f}', f'{auc_aud_orig_obj:.4f}', f'{best[6]:.4f}', f'{best[7]:.4f}')
    print('ALL_COMBINED_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_all_combined:.4f}', f'{auc_all_combined:.4f}', f'{best[8]:.4f}', f'{best[9]:.4f}')

    metrics = {
        'origin_%s/cIoU' %(testset): mAP,
        'origin_%s/auc' %(testset): auc,
        'origin_%s/best_cIoU' %(testset): best[0],
        'origin_%s/best_auc' %(testset): best[1],
        
        'OBJ_%s/cIoU' %(testset): mAP_img,
        'OBJ_%s/auc' %(testset): auc_img,
        'OBJ_%s/best_cIoU' %(testset): best[2],
        'OBJ_%s/best_auc' %(testset): best[3],

        'OGL_%s/cIoU' %(testset): mAP_ogl,
        'OGL_%s/auc' %(testset): auc_ogl,
        'OGL_%s/best_cIoU' %(testset): best[4],
        'OGL_%s/best_auc' %(testset): best[5],

        'ORIG_OBJ_%s/cIoU' %(testset): mAP_orig_obj,
        'ORIG_OBJ_%s/auc' %(testset): auc_orig_obj,

        'AUD_ORIG_OBJ_%s/cIoU' %(testset): mAP_aud_orig_obj,
        'AUD_ORIG_OBJ_%s/auc' %(testset): auc_aud_orig_obj,
        'AUD_ORIG_OBJ_%s/best_cIoU' %(testset): best[6],
        'AUD_ORIG_OBJ_%s/best_auc' %(testset): best[7],

        'ALL_COMBINED_%s/cIoU' %(testset): mAP_all_combined,
        'ALL_COMBINED_%s/auc' %(testset): auc_all_combined,
        'ALL_COMBINED_%s/best_cIoU' %(testset): best[8],
        'ALL_COMBINED_%s/best_auc' %(testset): best[9],
        'epoch': epoch
    }
    wandb.log(metrics)
    return best

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
