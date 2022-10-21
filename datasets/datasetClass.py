import numpy as np
from torch.utils.data import Dataset
from joblib import load
from sklearn.preprocessing import LabelEncoder as label_encoder
import os
from typing import List
import torchvision.transforms as transforms
from torchvision.io import read_image

class TransformDataset:
    def __init__(
        self,
        image_size:int, 
        normalize_param:List[List],
        aug:bool
    ):
        
        self.aug = aug
        self.image_size = image_size
        self.normalize_param = normalize_param
        
        
    def apply_transformations(self, image_path:str):
        # list_of_transformations =[T.RandomRotation, T.RandomAffine, T.RandomVerticalFlip, T.RandomHorizontalFlip] #Torch Vision transforms

        transforms = self.get_composed_transform()
        image = read_image(image_path)
        #print(image.shape)
        transformed_image = transforms(image)
        
        return transformed_image
        


    def parse_transform(self, transform_type):
        
        method = getattr(transforms, transform_type)
        if transform_type == "RandomSizedCrop" or transform_type == "RandomResizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Scale" or transform_type == "Resize":
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == "Normalize":
            return method(*self.normalize_param)
        else:
            return method()


    def get_composed_transform(self):
        if self.normalize_param is not None:
            if self.aug:
                transform_list = [
                    "ToPILImage",
                    "Resize",
                    "RandomResizedCrop",
                    "RandomHorizontalFlip",
                    "ToTensor",
                    "Normalize",
                ]
            else:
                transform_list = ["ToPILImage", "Resize", "CenterCrop", "ToTensor", "Normalize"]
        else:
            if self.aug:
                transform_list = [
                    "ToPILImage",
                    "Resize",
                    "RandomResizedCrop",
                    "RandomHorizontalFlip",
                    "ToTensor",
                ]
            else:
                transform_list = ["ToPILImage", "Resize", "CenterCrop", "ToTensor"]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class MyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_file,
        phase,
        image_size,
        normalize_param=None
    ):

        images, labels = self.load_datafiles(split_file, phase)
        
        self.labels = labels
        self.images = images
        self.root_dir = root_dir
        
        self.phase = phase
        
        self.image_size = image_size
        
        self.normalize_param = normalize_param
        
        # print("Number of classes", np.unique(labels, return_counts=True))
       
    def load_datafiles(self, filepath, phase):

        # load the pickle file
        
        data = load(filepath)

        images = data[phase + "_images"]
        labels = data[phase + "_labels"]

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path = os.path.join(self.root_dir, self.images[i])
        if self.phase == "test":
            aug = False
        else:
            aug = True
        print(image_path) 
        
        transform_dataset = TransformDataset(self.image_size, self.normalize_param, aug)
        image = transform_dataset.apply_transformations(image_path)
        label = self.labels[i]
        print(label)

        return (image, label)

   
        
        
if __name__ == "__main__":
    test_dataset = MyDataset("/Users/megha/Desktop/Plankton_Domain_Adaptation/data/kaggle",\
        "/Users/megha/Desktop/Plankton_Domain_Adaptation/data/kaggle_dataset.pkl", "train")
