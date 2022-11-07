import argparse
import sys
from turtle import st
import torch
from torch.utils.data import DataLoader
sys.path.append('/home/jwomack30/Plankton_Domain_Adaptation/datasets/')
from datasetClass import MyDataset

def normalize(batch_size, dataset):

    loader = DataLoader(
    dataset, 
    batch_size = batch_size, 
    num_workers=1)

    def batch_mean_and_sd(loader):
        
        channels =1 
        cnt = 0
        fst_moment = torch.empty(channels)
        snd_moment = torch.empty(channels)

        for images, _ in loader:
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                    dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (
                        cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)        
        return mean,std
    
    return batch_mean_and_sd(loader)

def main(args):
    print("here")
    unnormalized_data = MyDataset(root_dir=args.dataset_dir, split_file=args.split_file, phase='train',  image_size=args.image_size, normalize_param=None)
    mean, std = normalize(args.batch_size, unnormalized_data)
    print(mean, std)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalization")

    parser.add_argument(
        "--dataset_dir", type=str, help="Location of directory containing images"
    )
    parser.add_argument("--split_file", type=str, help="Location of pickle file containing training and testing splits")    
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of Image"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Size of Batches"
    ) 
    args = parser.parse_args()
    main(args)
    
