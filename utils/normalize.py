import torch
from torch.utils.data import DataLoader


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

    