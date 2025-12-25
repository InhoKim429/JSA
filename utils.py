import os
import json
from torch.optim import *
import numpy as np
from sklearn import metrics

import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []
        self.file_ciou = {}
        self.infer_ratio = {}

    def cal_CIOU(self, infer, gtmap, file_name, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))
        
        self.ciou.append(ciou)
        self.file_ciou[file_name] = ciou
        self.infer_ratio[file_name] = np.sum(infer_map) / (224 * 224)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0))), infer_map

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []
        self.file_ciou = {}
        self.infer_ratio = {}

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value

def visualize(raw_image, boxes, test_set='flickr'):
    import cv2
    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:
        if test_set == 'vggss':
            box = box[0]
        xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return boxes_img[:, :, ::-1]

def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler

def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_iou(iou_list, suffix, output_dir):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(output_dir,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def visualize_query_key(image, audq, imgk, label, query_index, key_index=None):
    if key_index is None:
        key_index = query_index
    try:
        print(label[key_index])
    except:
        pass

    os.makedirs('./utils_qual', exist_ok=True)

    img = image[key_index].permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())

    plt.imsave('./utils_qual/qual_img.jpg', img.cpu().detach().numpy())
    plt.cla()
    plt.clf()
    plt.close()

    # need to change
    query = audq[query_index]
    key = imgk[key_index]
    
    cross_dots = torch.einsum('id,jd->ij', query, key) * (512 ** -0.5)
    cross_dots = torch.unsqueeze(cross_dots, dim=0)
    cross_attn = cross_dots.softmax(dim=1) + 1e-8
    cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True)

    for slot_idx in range(audq.size(1)):
        slot_attn = cross_attn[0, slot_idx, ...].unsqueeze(0).unsqueeze(0)
        slot_attn = slot_attn.reshape(1, 1, 7, 7)
        slot_attn = F.interpolate(slot_attn, size=(224, 224), mode='bicubic', align_corners=False)
        slot_attn = normalize_img(slot_attn)
        slot_attn = slot_attn[0, 0, ...]

        heatmap_img = np.uint8(slot_attn.cpu().detach().numpy()*255)
        heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(img.cpu().detach().numpy()*255), 0.2, 0)
        cv2.imwrite('./utils_qual/qual_attn_%d.jpg' %(slot_idx), fin)
    return


def visualize_attention(image, cross_attn, label, img_index):
    print(label[img_index])
    batch_size = image.size(0)
    
    img = image[img_index].permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())

    # need to change
    fg_slot_attn = cross_attn[img_index, 0, ...].unsqueeze(0).unsqueeze(0)
    gray_slot_attn = cross_attn[img_index, 1, ...].unsqueeze(0).unsqueeze(0)
    bg_slot_attn = cross_attn[img_index, 2, ...].unsqueeze(0).unsqueeze(0)

    fg_slot_attn = fg_slot_attn.reshape(1, 1, 7, 7)
    fg_slot_attn = F.interpolate(fg_slot_attn, size=(224, 224), mode='bicubic', align_corners=False)

    bg_slot_attn = bg_slot_attn.reshape(1, 1, 7, 7)
    bg_slot_attn = F.interpolate(bg_slot_attn, size=(224, 224), mode='bicubic', align_corners=False)

    gray_slot_attn = gray_slot_attn.reshape(1, 1, 7, 7)
    gray_slot_attn = F.interpolate(gray_slot_attn, size=(224, 224), mode='bicubic', align_corners=False)
    
    fg_slot_attn = normalize_img(fg_slot_attn) # , cross_attn.max(), cross_attn.min())
    bg_slot_attn = normalize_img(bg_slot_attn) # , cross_attn.max(), cross_attn.min())
    gray_slot_attn = normalize_img(gray_slot_attn) # , cross_attn.max(), cross_attn.min())

    # Stack the attention maps and normalize along slot dimension
    # stacked_attn = torch.stack([fg_slot_attn[0,0], bg_slot_attn[0,0], gray_slot_attn[0,0]], dim=0)
    # normalized_attn = F.softmax(stacked_attn, dim=0)
    
    # fg_slot_attn = normalized_attn[0:1,None]  
    # bg_slot_attn = normalized_attn[1:2,None]
    # gray_slot_attn = normalized_attn[2:3,None]

    fg_slot_attn = fg_slot_attn[0, 0, ...]
    bg_slot_attn = bg_slot_attn[0, 0, ...]
    gray_slot_attn = gray_slot_attn[0, 0, ...]

    plt.imsave('qual_img.jpg', img.cpu().detach().numpy())
    plt.cla()
    plt.clf()
    plt.close()

    heatmap_img = np.uint8(fg_slot_attn.cpu().detach().numpy()*255)
    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(img.cpu().detach().numpy()*255), 0.2, 0)
    cv2.imwrite('qual_fg_attn.jpg', fin)

    heatmap_img = np.uint8(bg_slot_attn.cpu().detach().numpy()*255)
    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(img.cpu().detach().numpy()*255), 0.2, 0)
    cv2.imwrite('qual_bg_attn.jpg', fin)

    heatmap_img = np.uint8(gray_slot_attn.cpu().detach().numpy()*255)
    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(img.cpu().detach().numpy()*255), 0.2, 0)
    cv2.imwrite('qual_gray_attn.jpg', fin)
    return

def calculate_mean(sim, num_slots):
    self_mask = torch.zeros_like(sim, dtype=torch.bool)
    nonself_mask = torch.ones_like(sim, dtype=torch.bool)
    b = sim.size(0)
    
    for i in range(b):
        self_mask[i, i] = True
        nonself_mask[i, i] = False
    
    self_masked_tensor = sim[self_mask].reshape(-1, num_slots, num_slots)
    self_mean_tensor = self_masked_tensor.mean(dim=0)
    nonself_masked_tensor = sim[nonself_mask].reshape(-1, num_slots, num_slots)
    nonself_mean_tensor = nonself_masked_tensor.mean(dim=0)
    
    return self_mean_tensor, nonself_mean_tensor

def get_potential_false_negative(img_slot, aud_slot, max_idx=None, k=20):
    # img : B x c x h x w
    # aud : B x c x t
    B = img_slot.size(0)
    if max_idx is None:
        first_img_slot = img_slot[:, 0, :]
        first_aud_slot = aud_slot[:, 0, :]
    else:
        first_img_slot = img_slot[torch.arange(B), max_idx, :]
        first_aud_slot = aud_slot[torch.arange(B), max_idx, :]

    img_sim = torch.einsum('nc,mc->nm', first_img_slot, first_img_slot)
    aud_sim = torch.einsum('nc,mc->nm', first_aud_slot, first_aud_slot)

    _, img_topk = torch.topk(img_sim, k=k, dim=1)
    _, aud_topk = torch.topk(aud_sim, k=k, dim=1)

    aud_indices = torch.stack([torch.bincount(row, minlength=B) for row in aud_topk])
    img_indices = torch.stack([torch.bincount(row, minlength=B) for row in img_topk])
    potential_false_negative = torch.logical_and(aud_indices, img_indices)
    reciprocal_false_negative = potential_false_negative * potential_false_negative.T
    reciprocal_false_negative.fill_diagonal_(0.0)
    reciprocal_false_negative = torch.logical_not(reciprocal_false_negative)
    return img_sim, aud_sim, reciprocal_false_negative

def print_all_reciprocal(reciprocal, label, file_name):
    f = open(file_name, 'w')
    for idx in range(len(reciprocal)):
        for i in range(len(reciprocal[idx])):
            if reciprocal[idx, i] == 0 or idx == i:
                print(i, label[i], file=f)
        print('-----------------------------------', file=f)
    f.close()
    return

def tsne(slots, label, slot_idx=0):
    wanted_slot = slots[:, slot_idx, :].cpu().detach()

    tsne = TSNE(n_components=2, random_state=42)
    tensor_tsne = tsne.fit_transform(wanted_slot)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(label)

    unique_labels = len(np.unique(label))
    cmap = plt.get_cmap('tab10', unique_labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], s=10, c=encoded_labels, cmap=cmap)

    handles, _ = scatter.legend_elements()
    legend_labels = label_encoder.classes_
    plt.legend(handles, legend_labels, title="Classes")

    plt.title('t-SNE plot')
    plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')

def tsne_condition(slots, labels, key=None, slot_idx=0):
    def check(key, label):
        return key in label
    wanted_slot = slots[:, slot_idx, :].cpu().detach()

    tsne = TSNE(n_components=2, random_state=42)
    tensor_tsne = tsne.fit_transform(wanted_slot)

    if key is not None:
        condition = np.array([check(key, label) for label in labels])
    else:
        condition = np.zeros(len(labels))
    colors = np.where(condition, 'red', 'blue')

    plt.figure(figsize=(40, 30))
    # plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], s=10)
    plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], s=10, c=colors)

    for i, label in enumerate(labels):
        plt.text(tensor_tsne[i, 0], tensor_tsne[i, 1], label, fontsize=8, ha='right')

    plt.title('t-SNE plot')
    plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.cla()

def tsne_modal(img_slots, aud_slots, labels, key=None, slot_idx=0):
    def check(key, label):
        return key in label
    img_slots_wanted = img_slots[:, slot_idx, :].cpu().detach()
    aud_slots_wanted = aud_slots[:, slot_idx, :].cpu().detach()
    wanted_slots = torch.cat([img_slots_wanted, aud_slots_wanted], dim=0)
    modal = torch.cat([torch.zeros(256), torch.ones(256)], dim=0)

    tsne = TSNE(n_components=2, random_state=42)
    tensor_tsne = tsne.fit_transform(wanted_slots)

    colors = np.where(modal, 'red', 'blue')

    plt.figure(figsize=(40, 30))
    # plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], s=10)
    plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], s=10, c=colors)

    for i, label in enumerate(labels + labels):
        plt.text(tensor_tsne[i, 0], tensor_tsne[i, 1], label, fontsize=8, ha='right')

    plt.title('t-SNE plot')
    plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.cla()
