import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from dataset import get_test_dataset, inverse_normalize
from dataset_avs import get_ms3_dataset, get_s4_dataset
import cv2
from tqdm import tqdm
import shutil
from datetime import datetime
import model_slot
import json
import random

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='noname', help='experiment name (used for checkpointing and logging)')

    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)
    
    # hyper-params
    parser.add_argument('--aud_length', default=5.0, type=float)

    # training/evaluation parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--infer_sharpening", type=float, default=0.1)
    parser.add_argument("--num_slots", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument('--wandb', type=str, default='false')

    # Distributed params
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=None)

    # Evaluation
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    args = parser.parse_args()

    with open(os.path.join(args.model_dir, args.experiment_name, 'configs.json'), 'r') as f:
        config_dict = json.load(f)
    config_namespace = argparse.Namespace(**config_dict)

    for key, value in vars(args).items():
        setattr(config_namespace, key, value)
    
    return config_namespace

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
    setup_seed(12345)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    from torchvision.models import resnet18
    object_saliency_model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1') # resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    audio_visual_model = model_slot.mymodel(args)
    audio_visual_model.cuda(args.gpu)
    object_saliency_model.cuda(args.gpu)

    # Load weights
    ckp_fn = os.path.join(model_dir, '%s_best.pth' %(args.testset))
    # ckp_fn = os.path.join(model_dir, 's4_best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {ckp_fn}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    if args.testset == 'ms3':
        testdataset = get_ms3_dataset(args)
    elif args.testset == 's4':
        testdataset = get_s4_dataset(args)
    else:
        testdataset = get_test_dataset(args, args.testset)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Loaded dataloader.")

    mAP, auc, \
    mAP_img, auc_img, \
    mAP_ogl, auc_ogl, \
    mAP_orig_obj, auc_orig_obj, \
    mAP_aud_orig_obj, auc_aud_orig_obj, \
    mAP_all_combined, auc_all_combined = \
    validate_img_aud(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args.testset, -1, args)
    
    print('AUD_%s/cIoU, auc' %(args.testset), f'{mAP:.4f}', f'{auc:.4f}')
    print('OBJ_%s/cIoU, auc' %(args.testset), f'{mAP_img:.4f}', f'{auc_img:.4f}')
    print('OGL_%s/cIoU, auc' %(args.testset), f'{mAP_ogl:.4f}', f'{auc_ogl:.4f}')
    print('ORIG_OBJ_%s/cIoU, auc' %(args.testset), f'{mAP_orig_obj:.4f}', f'{auc_orig_obj:.4f}')
    print('AUD_ORIG_OBJ_%s/cIoU, auc' %(args.testset), f'{mAP_aud_orig_obj:.4f}', f'{auc_aud_orig_obj:.4f}')
    print('ALL_COMBINED_%s/cIoU, auc' %(args.testset), f'{mAP_all_combined:.4f}', f'{auc_all_combined:.4f}')
    return

@torch.no_grad()
def validate_img_aud(testdataloader, audio_visual_model, object_saliency_model, viz_dir, testset, epoch, args):
    audio_visual_model.eval()
    object_saliency_model.eval()

    evaluator_aud = utils.Evaluator()
    evaluator_img = utils.Evaluator()
    evaluator_aud_img = utils.Evaluator()
    evaluator_orig_obj = utils.Evaluator()
    evaluator_aud_orig_obj = utils.Evaluator()
    evaluator_all_combined = utils.Evaluator()

    for step, (image, spec, bboxes, name, label) in enumerate(tqdm(testdataloader)):
        # Handle 5D image tensor by combining first two dimensions
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
            spec = spec.unsqueeze(0)
            bboxes = bboxes.unsqueeze(0)

        if len(image.size()) == 5:
            b, n, c, h, w = image.size()
            image = image.reshape(b*n, c, h, w)
            b, n, c, f, t = spec.size()
            spec = spec.reshape(b*n, c, f, t)
            b, n, c, h, w = bboxes.size()
            bboxes = bboxes.reshape(b*n, c, h, w)
            bboxes = bboxes.squeeze(1)

            expanded_names = []
            for n in name:
                expanded_names.extend([n] * 5)
            name = expanded_names

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        
        with torch.no_grad():
            heatmap_img, heatmap_aud = audio_visual_model(image.float(), spec.float())
            img_feat = object_saliency_model(image)

        heatmap_aud = F.interpolate(heatmap_aud, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_aud = heatmap_aud.data.cpu().numpy()

        # Compute S_OBJ
        heatmap_img = F.interpolate(heatmap_img, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_img = heatmap_img.data.cpu().numpy()

        original_obj = F.interpolate(img_feat, size=(224, 224), mode='bicubic', align_corners=False)
        original_obj = original_obj.data.cpu().numpy()

        bboxes = bboxes.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_aud = utils.normalize_img(heatmap_aud[i, 0])
            pred_img = utils.normalize_img(heatmap_img[i, 0])
            pred_aud_img = utils.normalize_img(pred_aud * args.alpha + pred_img * (1 - args.alpha))
            pred_orig_obj = utils.normalize_img(original_obj[i, 0])
            pred_aud_orig_obj = utils.normalize_img(pred_aud * args.alpha + pred_orig_obj * (1 - args.alpha))
            pred_all_combined = utils.normalize_img(pred_aud * args.alpha + pred_img * (1 - args.alpha) * 0.5 + pred_orig_obj * (1 - args.alpha) * 0.5)

            gt_map = bboxes[i]
            threshold = 0.6

            _, _, _, aud_infer_map = evaluator_aud.cal_CIOU(pred_aud, gt_map, name[i], threshold)
            _, _, _, img_infer_map = evaluator_img.cal_CIOU(pred_img, gt_map, name[i], threshold)
            _, _, _, aud_img_infer_map = evaluator_aud_img.cal_CIOU(pred_aud_img, gt_map, name[i], threshold)
            _, _, _, orig_obj_infer_map = evaluator_orig_obj.cal_CIOU(pred_orig_obj, gt_map, name[i], threshold)
            _, _, _, aud_orig_obj_infer_map = evaluator_aud_orig_obj.cal_CIOU(pred_aud_orig_obj, gt_map, name[i], threshold)
            _, _, _, all_combined_infer_map = evaluator_all_combined.cal_CIOU(pred_all_combined, gt_map, name[i], threshold)

            if epoch == -2: # drawing tool
                os.makedirs(os.path.join(viz_dir, name[i]), exist_ok=True)

                denorm_image = inverse_normalize(image[i]).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'image.jpg'), denorm_image)

                # visualize bboxes on raw images
                # gt_boxes_img = utils.visualize(denorm_image, gt_map) #, test_set=testset)
                overlay = np.zeros_like(denorm_image)
                overlay[gt_map == 1] = [0, 0, 255]
                gt_boxes_img = cv2.addWeighted(denorm_image, 1, overlay, 0.9, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'gt_boxes.jpg'), gt_boxes_img)

                # visualize predicted segmentation masks
                overlay = np.zeros_like(denorm_image)
                overlay[aud_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_av.jpg'), highlighted_image)

                overlay = np.zeros_like(denorm_image)
                overlay[img_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_obj.jpg'), highlighted_image)

                overlay = np.zeros_like(denorm_image)
                overlay[aud_img_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_aud_img.jpg'), highlighted_image)

                # visualize heatmaps
                heatmap_pred_aud = np.uint8(pred_aud*255)
                heatmap_pred_aud = cv2.applyColorMap(heatmap_pred_aud[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_aud, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_aud.jpg'), fin)

                heatmap_pred_img = np.uint8(pred_img*255)
                heatmap_pred_img = cv2.applyColorMap(heatmap_pred_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_img, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_img.jpg'), fin)

                heatmap_pred_aud_img = np.uint8(pred_aud_img*255)
                heatmap_pred_aud_img = cv2.applyColorMap(heatmap_pred_aud_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_aud_img, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_aud_img.jpg'), fin)

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc
    
    mAP, _, auc = compute_stats(evaluator_aud)
    mAP_img, _, auc_img = compute_stats(evaluator_img)
    mAP_ogl, _, auc_ogl = compute_stats(evaluator_aud_img)
    mAP_obj, _, auc_obj = compute_stats(evaluator_orig_obj)
    mAP_aud_orig_obj, _, auc_aud_orig_obj = compute_stats(evaluator_aud_orig_obj)
    mAP_all_combined, _, auc_all_combined = compute_stats(evaluator_all_combined)
    if epoch == -1:
        model_dir = os.path.join(args.model_dir, args.experiment_name)
        save_all_metrics(evaluator_aud, evaluator_img, evaluator_aud_img, model_dir)

    return mAP, auc, \
           mAP_img, auc_img, \
           mAP_ogl, auc_ogl, \
           mAP_obj, auc_obj, \
           mAP_aud_orig_obj, auc_aud_orig_obj, \
           mAP_all_combined, auc_all_combined  

def save_sorted_metrics(evaluator, output_path):
    """Save sorted metrics cIoU for each file to a text file"""
    metrics = []
    for filename, ciou in evaluator.file_ciou.items():
        infer_ratio = evaluator.infer_ratio[filename]
        metrics.append((filename, ciou, infer_ratio))
    
    # Sort by cIoU in descending order
    metrics.sort(key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w') as f:
        f.write('Filename\tcIoU\tInfer Ratio\n')
        for filename, ciou, infer_ratio in metrics:
            f.write(f'{filename}\t{ciou:.4f}\t{infer_ratio:.4f}\n')

def save_all_metrics(evaluator_av, evaluator_obj, evaluator_av_obj, viz_dir):
    """Save sorted metrics cIoU for each model's predictions"""
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save metrics for each model
    save_sorted_metrics(evaluator_av, os.path.join(viz_dir, 'av_metrics.txt'))
    save_sorted_metrics(evaluator_obj, os.path.join(viz_dir, 'obj_metrics.txt'))
    save_sorted_metrics(evaluator_av_obj, os.path.join(viz_dir, 'av_obj_metrics.txt'))

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

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)

