import os
import json
import glob

import torch
import numpy as np
# import torchvision.transforms as transforms
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.ops.boxes import masks_to_boxes
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from skimage import draw
from PIL import Image

def annotation_to_target(annotation_list, img_height, img_width, img_index):

    num_doors = len(annotation_list)

    # assign label of "1"
    labels = torch.ones((num_doors,), dtype=torch.int64)

    # image_id = index
    image_id = img_index

    # find all masks of all door instance
    all_masks = torch.empty((num_doors,img_height,img_width),dtype=torch.uint8)
    for i in range(0,num_doors):
        annotation = annotation_list[i]
        # separate coordinates into row and col coordinates
        x,y = zip(*annotation["points"])
        row_coords = np.array(y).astype(int)
        col_coords = np.array(x).astype(int)

        # create mask of zeros
        mask = np.zeros((img_height,img_width), dtype=np.uint8)
        
        # Draw filled polygons
        rr, cc = draw.polygon(row_coords, col_coords, mask.shape)
        mask[rr,cc] = 1

        # convert from numpy array to pytorch tensor
        all_masks[i,:,:] = torch.from_numpy(mask)

    # convert masks to bounding box
    boxes = masks_to_boxes(all_masks)

    # get areas of bounding boxes
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # all instances are not crowded
    iscrowd = torch.zeros((num_doors,), dtype=torch.int64)
        
    target = {}
    target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img_height,img_width))
    target["masks"] = tv_tensors.Mask(all_masks)
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    return target



def get_dataloader(dataset_dir, batch_size=1, split='train'):
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(degrees=180),
            transforms.RandomPerspective(distortion_scale=0.3),

            # transforms.ToTensor(),

            transforms.ToDtype(torch.float, scale=True),
            transforms.ToPureTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.ToDtype(torch.float, scale=True),
            transforms.ToPureTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = DoorDataset(dataset_dir, split=split, transform=transform)
    if dataset[0] is None:
        raise NotImplementedError('No data found, check dataset.py and implement __getitem__() in DoorDataset class!')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'), collate_fn=lambda x: tuple(zip(*x)))

    return dataloader

class DoorDataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(DoorDataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        all_json_list = glob.glob(os.path.join(dataset_dir, "*.json"))

        all_image_names = []
        all_annotations = []

        # get all annotations
        for json_path in all_json_list:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            

            image_name = json_data["imagePath"]
            annotation_data = []
            # get coordinates of mask shape of each image
            for shape in json_data["shapes"]:
                mask_shape = {}
                mask_shape["label"] = shape["label"]
                mask_shape["points"] = shape["points"]
                annotation_data.append(mask_shape)
                
            all_image_names.append(image_name)
            all_annotations.append(annotation_data)

        self.image_names = all_image_names

        if self.split != 'test':
            self.annotations = all_annotations

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        fullpath = os.path.join(self.dataset_dir, self.image_names[index])
        image = Image.open(fullpath)
        image_tensor = tv_tensors.Image(image)

        target = None
        if self.split != "test":
            anno = self.annotations[index]
            w,h = image.size
            target = annotation_to_target(anno, h, w, index)

        if self.transform is not None:
            transform_image, transform_target = self.transform(image_tensor, target)

        return transform_image, transform_target
