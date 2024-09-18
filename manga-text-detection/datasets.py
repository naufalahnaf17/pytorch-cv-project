import os
import zipfile
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch

def download_dataset():
    if os.path.isdir("./data") == False:
        command_kaggle = "kaggle datasets download -d naufalahnaf17/manga-text-detection -p data/"
        os.system(command_kaggle)

        zip_file_path = "./data/manga-text-detection.zip"
        extract_folder = "./data"

        with zipfile.ZipFile(zip_file_path,'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def get_dataloader(batch_size=2):
    dataset_path = "./data/"

    train_path = os.path.join(dataset_path,'train')
    valid_path = os.path.join(dataset_path,'valid')
    test_path = os.path.join(dataset_path,'test')

    train_anno = os.path.join(train_path,'_annotations.coco.json')
    valid_anno = os.path.join(valid_path,'_annotations.coco.json')
    test_anno = os.path.join(test_path,'_annotations.coco.json')

    class MangaDataset(CocoDetection):
        def __init__(self,root,anno,transform=None):
            super(MangaDataset,self).__init__(root,anno)
            self.transform = transform
        
        def __getitem__(self,index):
            image,target = super(MangaDataset,self).__getitem__(index)
            if self.transform is not None:
                image = self.transform(image)
            
            boxes = []
            labels = []
            
            for obj in target:
                xmin,ymin,width,height = obj['bbox']
                if width > 0 and height > 0:
                    boxes.append([xmin,ymin,xmin+width,ymin+height])
                    labels.append(obj['category_id'])
            
            target = {
                "boxes" : torch.tensor(boxes,dtype=torch.float32),
                "labels" : torch.tensor(labels,dtype=torch.int64),
                "image_id" : torch.tensor(target[0]["image_id"],dtype=torch.int64)
            }
            
            return image,target
            
    def transform(image):
        return F.to_tensor(image)
            
    train_data = MangaDataset(train_path,train_anno,transform=transform)
    valid_data = MangaDataset(valid_path,valid_anno,transform=transform)
    test_data  = MangaDataset(test_path,test_anno,transform=transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return train_loader,valid_loader,test_loader,train_data.coco.cats