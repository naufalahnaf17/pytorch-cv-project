import os
import zipfile
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset,DataLoader,random_split

def download_dataset():
    if os.path.isdir("./data") == False:
        command_kaggle = "kaggle datasets download -d kacpergregorowicz/house-plant-species -p data/"
        os.system(command_kaggle)

        zip_file_path = "./data/house-plant-species.zip"
        extract_folder = "./data"

        with zipfile.ZipFile(zip_file_path,'r') as zip_ref:
            zip_ref.extractall(extract_folder)

def get_classes():
    def loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image

    dataset = DatasetFolder(
        root="./data/house_plant_species",
        extensions=[".jpg",".jpeg"],
        loader=loader
    )

    return dataset.classes

def get_dataloader(processor,batch_size=32):
    def loader(path):
        image = Image.open(path)
        image = image.convert("RGB")
        return image

    dataset = DatasetFolder(
        root="./data/house_plant_species",
        extensions=[".jpg",".jpeg"],
        loader=loader
    )

    def transform_image(image):
        encoding = processor(images=image)
        return encoding["pixel_values"].squeeze(0)

    class CustomDataset(Dataset):
        def __init__(self,data,transform=None):
            self.transform = transform
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self,index):
            image,label = self.data[index]
            if self.transform is not None:
                image = self.transform(image)
                
            return image,label
            
    raw_data = CustomDataset(data=dataset,transform=transform_image)
    train_size = int(0.8 * len(raw_data))
    valid_size = int(len(raw_data) - train_size)

    train_data,test_data = random_split(raw_data,[train_size,valid_size])
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader,dataset.classes