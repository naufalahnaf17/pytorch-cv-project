import os
import zipfile
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset,DataLoader

def download_dataset():
    if os.path.isdir("./data") == False:
        command_kaggle = "kaggle datasets download -d tongpython/cat-and-dog -p data/"
        os.system(command_kaggle)

        zip_file_path = "./data/cat-and-dog.zip"
        extract_folder = "./data"

        with zipfile.ZipFile(zip_file_path,'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def raw_data():
    training_path = "./data/training_set/training_set"
    test_path = "./data/test_set/test_set"
    
    def loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image
    
    raw_train = DatasetFolder(
        training_path,
        loader=loader,
        extensions=[".jpg",".jpeg"]
    )
    
    raw_test = DatasetFolder(
        test_path,
        loader=loader,
        extensions=[".jpg",".jpeg"]
    )
    
    return raw_train,raw_test

def get_classes():
    train,_ = raw_data()
    return train.classes

def get_dataloader(processor,batch_size=32):
    train,test = raw_data()

    def transform_image(image):
        encoding = processor(images=image)
        return encoding["pixel_values"][0]

    class CustomDataset(Dataset):
        def __init__(self,data,transform=None):
            super().__init__()
            self.data = data
            self.transform = transform
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self,index):
            image,label = self.data[index]
            if self.transform is not None:
                image = self.transform(image)
                
            return image,label
            
            
    train_data = CustomDataset(data=train,transform=transform_image)
    test_data = CustomDataset(data=test,transform=transform_image)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader,test_loader