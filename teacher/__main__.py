import os
import sys
from typing import Any
from datasets.datasetClass import MyDataset
from utils.normalize import normalize
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pylab as plt
import torchvision.transforms as T
from torch import nn

from models.resnet12 import Resnet12
from models.classifier import Classifier
from datasets import readKaggleData, readWHOIData
from teacher.train import loop_over_all_epochs

import argparse


def define_optimizer(
    model:Any,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    optim_type: str = "ADAM",
):
    if optim_type == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    return optimizer


def run(
    model:Any,
    num_of_epochs: int,
    learning_rate: float,
    weight_decay: float,
    model_save_path: str,
    log_dir: str,
    trainloader,
    valloader,
    testloader,
):

    log_filepath = log_dir + "/" + "cur_log.log"
    initial_sd = model.state_dict()

    train_on_gpu = False
    if torch.cuda.is_available():
        train_on_gpu = True

    optimizer = define_optimizer(model, learning_rate, weight_decay, optim_type="ADAM")
    loss_criteria = nn.CrossEntropyLoss()

    test_acc = 0.0
    test_loss = 0.0

    (epoch_losses, epoch_accs,) = loop_over_all_epochs(
        [trainloader, valloader],
        num_of_epochs,
        model,
        train_on_gpu,
        loss_criteria,
        "train",
        model_save_path,
        optimizer,
        log_filepath,
    )

    train_epoch_losses = epoch_losses[0]
    val_epoch_losses = epoch_losses[1]
    plt.plot(train_epoch_losses)
    plt.plot(val_epoch_losses)

    fig_filepath = log_dir + "/" + "loss.png"
    plt.title("Train & Val Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(fig_filepath)
    plt.clf()
    plt.show()

    # TEST
    # print(model.state_dict)
    # print(torch.load("./teacher/models/teacher_model_best.pkl")["Encoder"].keys())
    # assert 1==0
    model.load_state_dict(torch.load("./teacher/models/teacher_model_best.pkl"))
    (test_loss, test_acc,) = loop_over_all_epochs(
        [testloader], 1, model, train_on_gpu, loss_criteria, "test"
    )

    test_loss += np.asarray(test_loss)[0]
    test_acc += np.asarray(test_acc)[0]

    with open(log_filepath, "a") as f:
        f.write(f"Test Metrics ")
        f.write("\n")
        f.write(f"Test Loss: {np.asarray(test_loss)[0]} Acc: {np.asarray(test_acc)[0]}")
        f.write("\n")

def main(args):
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    log_dir = os.path.join(args.model_save_path, "logs")
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    ## Get dataset splits ###
    if args.dataset == "KaggleData":
        dataset_pkl = os.path.join(args.dataset_dir, "kaggle_dataset.pkl")
        root_dir= os.path.join(args.dataset_dir, "kaggle")    
        mean, std = 0.9016, 0.206
        
    elif args.dataset == "WHOIData":
        dataset_pkl = os.path.join(args.dataset_dir, "whoi_dataset.pkl")
        root_dir= os.path.join(args.dataset_dir, "whoi")    
        
    unnormlaised_data = MyDataset(root_dir=root_dir, split_file=dataset_pkl, phase='train',  image_size=args.image_size, normalize_param=None)
    mean, std = normalize(args.batch_size, unnormlaised_data)
    ## Create datasets ###
    
    
    print("Mean and Std", mean, std)
    
    train_dataset = MyDataset(root_dir=root_dir, split_file=dataset_pkl, phase='train',  image_size=args.image_size, normalize_param=[mean,std])
    val_dataset = MyDataset(root_dir=root_dir, split_file=dataset_pkl, phase='val',  image_size=args.image_size, normalize_param=[mean,std])
    test_dataset = MyDataset(root_dir=root_dir, split_file=dataset_pkl, phase='test',  image_size=args.image_size, normalize_param=[mean,std])
    
    
    ## Create Dataloader ##
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
   
    print("Size of Train Data", len(train_dataloader))
    print("Size of Val Data", len(val_dataloader))
    print("Size of Test Data", len(test_dataloader))
   

    # Create Model ##
    encoder = Resnet12()
    classifier = Classifier(embedding_dimension=512, num_of_classes=args.num_of_classes)
    model = nn.Sequential(encoder, classifier)

    run(
        model,
        args.num_of_epochs,
        args.learning_rate,
        args.weight_decay,
        args.model_save_path,
        log_dir,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Teacher")

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./teacher/models",
        help="directory to save the checkpoints",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch_size for tecaher"
    )

    parser.add_argument(
        "--num_of_classes", type=int, default=118, help="classes in dataset"
    )

    parser.add_argument(
        "--num_of_epochs", type=int, default=1, help="Number of epochs"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Leraning rate"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

    parser.add_argument(
        "--dataset", type=str, default="KaggleData", help="Name of dataset"
    )

    parser.add_argument("--dataset_dir", type=str, default="/home/jwomack30/Plankton_Domain_Adaptation/data", help="Dataset dir")
    
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of Image"
    )

    
    args = parser.parse_args()
    main(args)
