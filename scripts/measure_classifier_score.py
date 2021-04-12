import os
import argparse
import random
import pickle
from shutil import copyfile
from typing import Optional, Callable, List

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TVF
from torchvision import transforms
import torch
from torchvision.models import wide_resnet50_2
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


class ImagePathsDataset(VisionDataset):
    def __init__(self, img_paths: List[os.PathLike], transform: Callable):
        self.transform = transform
        self.imgs_paths = img_paths

    def __len__(self):
        return len(self.imgs_paths) * 2

    def __getitem__(self, idx: int):
        image = pil_loader(self.imgs_paths[idx // 2])
        image = self.transform(image)
        w = image.shape[2]

        if idx % 2 == 0:
            half = image[:, :, :w//2]
            y = 0
        else:
            half = image[:, :, w//2:]
            y = 1

        return {"img": half, "label": y}


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    accs = []
    losses = []

    for batch in dataloader:
        img, label = batch['img'].to(device), batch['label'].to(device)
        preds = model(img).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(preds, label.float())
        acc = ((preds.sigmoid() > 0.5).long() == label).float().mean()

        losses.append(loss.item())
        accs.append(acc.item())

    return np.mean(losses), np.mean(accs)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.15, help='Amount of training images')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='Amount of training images')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/inference')
    args = parser.parse_args()

    data_dir = args.data_dir
    all_img_names = [f for f in os.listdir(data_dir) if os.path.splitext(f)[1].lower() in Image.EXTENSION]
    random.shuffle(all_img_names)
    NUM_TRAIN_IMGS = int(args.train_ratio * len(all_img_names))
    NUM_VAL_IMGS = int(args.val_ratio * len(all_img_names))
    img_paths_train = [os.path.join(data_dir, f) for f in all_img_names[:NUM_TRAIN_IMGS]]
    img_paths_val = [os.path.join(data_dir, f) for f in all_img_names[NUM_TRAIN_IMGS:NUM_TRAIN_IMGS+NUM_VAL_IMGS]]
    model = wide_resnet50_2(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)

    optim = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4},
    ])

    transform_train = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = ImagePathsDataset(img_paths_train, transform=transform_train)
    dataset_val = ImagePathsDataset(img_paths_val, transform=transform_val)
    batch_size = args.batch_size
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=5)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=5)

    device = 'cuda'
    model = model.to(device)

    total_num_epochs = args.num_epochs

    for epoch in range(total_num_epochs):
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))

        for i, batch in pbar:
            model.train()
            img, label = batch['img'].to(device), batch['label'].to(device)
            preds = model(img).squeeze(1)

            loss = F.binary_cross_entropy_with_logits(preds, label.float())
            acc = ((preds.sigmoid() > 0.5).long() == label).float().mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_description(f'Epoch {epoch}. Loss: {loss.detach().item():.03f}. Acc: {acc.cpu().item():.03f}')

        val_loss, val_acc = validate(model, dataloader_val)
        print(f'Val loss: {val_loss:.03f}. Val acc: {val_acc: .03f}')

    ### Testing ###
    img_paths_test = [os.path.join(data_dir, f) for f in all_img_names[NUM_TRAIN_IMGS+NUM_VAL_IMGS:]]

    transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_test = ImagePathsDataset(img_paths_test, transform=transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=10)
    scores = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            img = batch['img'].to(device)
            preds = model(img).sigmoid()

            # We compute the scores as the maximum between left and right
            # Because even if one side is easily predictable, then it will be very
            # difficult to connect/continue it
            curr_scores = preds.view(-1, 2).max(dim=1)[0]
            scores.extend(curr_scores.cpu().tolist())

    assert len(scores) == len(img_paths_test)

    print(f'[{data_dir}] Average score on the test set:', np.mean(scores))

    save_dir = f'{data_dir}_spatial_inv'
    dataset_name = os.path.basename(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    ### Preprocessing data and saving ###
    with open(f'{save_dir}/{dataset_name}_scores.pkl', 'wb') as f:
        pickle.dump(scores, f)

    for threshold in [0.5, 0.7, 0.95, 0.99]:
        final_img_paths = np.array(img_paths_test)[np.array(scores) < threshold].tolist()
        target_dir = f'{save_dir}/{dataset_name}_t_{threshold}'
        os.makedirs(target_dir, exist_ok=True)

        for src_img_path in tqdm(final_img_paths):
            trg_img_path = os.path.join(target_dir, os.path.basename(src_img_path))
            copyfile(src_img_path, trg_img_path)
