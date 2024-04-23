

# VLDadaptor

## Installation

####  Our work is based on Python 3.7 and Pytorch 1.7.1+cu111 due to the  [CLIP requirement](https://github.com/openai/CLIP). The hardware is Nvidia Tesla V100 single GPU. Give a big thanks to Dr. Li WuYang with his work [SIGMA](https://github.com/CityU-AIM-Group/SIGMA). We use it as baseline.

#### Basic Installation

```bash
conda create -n vldadaptor  python==3.7 -y
conda activate vldadaptor
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/GingerCohle/VLDadaptor.git
cd VLDadaptor
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
conda install ipython
pip install ninja yacs cython matplotlib tqdm 
pip install --no-deps torchvision==0.2.1 
python setup.py build_ext install
cd ../..
pip install opencv-python==3.4.17.63
pip install scikit-learn
pip install scikit-image
python setup.py build develop
pip install Pillow==7.1.0
pip install tensorflow tensorboardX
pip install ipdb
```

#### CLIP Installation (China Region)

```bash
pip install ftfy regex tqdm
pip install git+https://gitee.com/lazybeauty/CLIP.git
```

#### CLIP Installation (Other Regions)

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Change Repo

#### We have two repos for our work. VC is for Cityscapes and VP is for Clipart. If you want to do Cityscapes2Foggy Cityscapes task, you can follow the above codes

```bash
cd VC
pip install Pillow==7.0.0
python setup.py build develop
pip install Pillow==7.1.0
```

#### Otherwise for Pascal VOC2Clipart

```bash
cd VP
pip install Pillow==7.0.0
python setup.py build develop
pip install Pillow==7.1.0
```

####  Or you can use two separate Conda environments for VC and VP respectively.

## Training

#### Foggy Cityscapes Training (Cuda Device 0 In VC folder)

#### and Clipart Training (Cuda Device 0 In VP folder)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py --config-file configs/SIGMA/sigma_res50_cityscapace_to_foggy.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train_net_da.py   --config-file configs/sigma_plus_plus/pascal_to_clipart_res101.yaml
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/SIGMA/sigma_res50_cityscapace_to_foggy.yaml MODEL.WEIGHT $model path$
```

## Code Inside
![image](https://github.com/GingerCohle/VLDadaptor/assets/37873318/fb131219-f661-4139-a28c-099009dab432)


#### In fcos_core/modeling/rpn/fcos/graph_matching_head.py, the prompt templates  and other components are in.

#### DMCKD

```python
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = features.view(features.shape[0],1, -1)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1).long()
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
```

#### DMCD

```python
    if nodes_2 is not None:
        # print((torch.cat(gt_cls)-1).tolist())
        # src CLIP op
        # visualize imgae w GT
        images_s, images_t = images  # unnormal
        to_pil = transforms.ToPILImage()
        mean = [-i for i in self.cfg.INPUT.PIXEL_MEAN]
        std = [1.0, 1.0, 1.0]
        image_list = []
        # denormalize
        for img in images_s.tensors:
            img = Fvis.normalize(img.squeeze(0), mean=mean,
                                 std=std)
            img = img[[2, 1, 0]] * 255
            img_np = to_pil(img.cpu().detach())
            image_list.append(img_np)
        reg_list = []
        cls_list = []
        for reg, cls in zip(gt_reg, gt_cls):
            reg_np = reg.cpu().numpy().astype(int)
            reg_list.append(reg_np)
            gt_np = cls.cpu().numpy().astype(int)
            cls_list.append(gt_np)
        clip_crop_feat_src_batch = []
        clip_crop_label_src_batch = []
        for img_v, reg_v, cls_v in zip(image_list, reg_list, cls_list):
            # img_tmp = img_v
            reg_tmp = reg_v
            cls_tmp = cls_v
            # print(cls_v)
            preprocessed = []
            for i in range(len(reg_tmp)):
                x1, y1, x2, y2 = reg_tmp[i][0], reg_tmp[i][1], reg_tmp[i][2], reg_tmp[i][3]
                # lbl = str(cls_v[i])
                # cv2.rectangle(img_v, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在目标框上方打上标签
                # cv2.putText(img_v, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # print(x1, y1, x2, y2)
                cropped_image = img_v.crop((x1, y1, x2, y2))
                w, h = cropped_image.size
                # print(cropped_image.size)
                # print('*'*10)
                if w > 10 and h >10:
                    clip_crop_label_src_batch.append(cls_tmp[i])
                    croped = self.preprocess(cropped_image)
                    preprocessed.append(croped)
            if len(preprocessed) >0:
                preprocessed = torch.stack(preprocessed).to(self.device)
                clip_crop_feat_src_batch.append(torch.nn.functional.normalize(self.clip_model.encode_image(preprocessed), p=2, dim=1))
        clip_crop_feat_src_batch = torch.cat(clip_crop_feat_src_batch)
        clip_crop_label_src_batch = torch.tensor(clip_crop_label_src_batch).cuda()
        # Align
        # src region feat
        clip_regionfeat = self.projection(torch.cat([nodes_1, nodes_2]))
        clip_regionfeat = torch.nn.functional.normalize(clip_regionfeat, p=2, dim=1)
        clip_label = torch.cat([labels_1, labels_2])

        text_features = self.text_features_for_classes_list
        clip_crop_label_src_batch = clip_crop_label_src_batch - 1
```

#### PMB

```python
def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):
    with torch.no_grad():
        for cls in sr_labels.unique().long():
            bs = sr_nodes[sr_labels == cls].detach()
            # lvl_sum
            text_features = self.text_features_for_classes_list
            clip_bs_regionfeat = self.projection(bs)
            clip_bs_regionfeat = torch.nn.functional.normalize(clip_bs_regionfeat, p=2, dim=1)
            cls_score_text = clip_bs_regionfeat @ text_features.T
            bs_cls_score = cls_score_text.softmax(dim=-1)[:, cls].unsqueeze(0).T
            bs_cls = torch.sum(bs * bs_cls_score, dim=0)
            momentum_lvl = torch.nn.functional.cosine_similarity(bs_cls.unsqueeze(0),
                                                                 self.src_seed[cls].unsqueeze(0))
            self.src_seed[cls] = self.src_seed[cls] * momentum_lvl + bs_cls * (1.0 - momentum_lvl)

    with torch.no_grad():
        if tg_nodes is not None:
            for cls in tg_labels.unique().long():
                clip_bs_regionfeat = self.projection(bs)
                clip_bs_regionfeat = torch.nn.functional.normalize(clip_bs_regionfeat, p=2, dim=1)
                cls_score_text = clip_bs_regionfeat @ text_features.T
                bs_cls_score = cls_score_text.softmax(dim=-1)[:, cls].unsqueeze(0).T
                bs_cls = torch.sum(bs * bs_cls_score, dim=0)
                momentum_lvl = torch.nn.functional.cosine_similarity(bs_cls.unsqueeze(0),
                                                                     self.tgt_seed[cls].unsqueeze(0))
                self.tgt_seed[cls] = self.tgt_seed[cls] * momentum_lvl + bs_cls * (1.0 - momentum_lvl)
```

## Visualization

### TSNE

### Detection (For [5151](https://github.com/voxel51/fiftyone))

#### StillTuned !!!!!, If you want the visualization Code, I can record a video for Visualization.
