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
from datasets import readKaggleData
from datasets import  readWHOIData
from student_copy.train import loop_over_all_epochs
from torch.nn import functional as F
import argparse
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans




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
    base_trainloader,
    base_valloader,
    base_testloader,
    target_trainloader,
    target_valloader,
    target_testloader,
    args
):

    log_filepath = log_dir + "/" + "cur_log.log"
    initial_sd = model.state_dict()

    train_on_gpu = False
    if torch.cuda.is_available():
        train_on_gpu = True

    optimizer = define_optimizer(model, learning_rate, weight_decay, optim_type="ADAM")
    
    test_acc = 0.0
    test_loss = 0.0

    epoch_losses = loop_over_all_epochs(
        args,
        [base_trainloader, base_valloader],
        [target_trainloader, target_valloader],
        num_of_epochs,
        model,
        train_on_gpu,
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
    model.load_state_dict(torch.load(args.embedding_load_path))
    test_loss = loop_over_all_epochs(
        args, [base_testloader], [target_testloader] , 1, model, train_on_gpu,  "test"
    )

    test_loss += np.asarray(test_loss)[0]
  

    with open(log_filepath, "a") as f:
        f.write(f"Test Metrics ")
        f.write("\n")
        f.write(f"Test Loss: {np.asarray(test_loss)[0]} ")
        f.write("\n")

def pseudolabel_dataset(embedding, clf, dataset, params):
    embedding.eval()
    clf.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                        shuffle=False, drop_last=False)

    # do an inference on the full target dataset
    features_all = []
    for X, _ in loader:
        X = Variable(X).float()
        X = X.cuda()

        with torch.no_grad():
            feature = embedding(X)
            
            # logits = clf(feature)
            # probs = F.softmax(logits, dim=1)
            # probs += 1e-6

        features_all.append(feature)
        
    
    
    representations = torch.cat(features_all, dim=0).cpu()
    
    num_of_classes = len(np.unique(dataset.labels))
    # print("in pseudolabel generation",representations.shape, num_of_classes)
    
    # representations_reshaped = representations.reshape(representations.shape[0], -1)
    pca = PCA(n_components=num_of_classes).fit(representations)
    kmeans = KMeans(init=pca.components_, n_clusters=num_of_classes, n_init=1)
    generated_labels = kmeans.fit_predict(representations)
    
    # print("generated labels", generated_labels.shape)
    # Update the target dataset with the pseudolabel
    if hasattr(dataset, "labels"):
        dataset.labels = generated_labels
    else:
        raise ValueError("No Targets variable found!")
   
    return dataset


def main(args):
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    log_dir = os.path.join(args.model_save_path, "logs")
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    ## Get dataset splits ###
    if args.base_dataset == "KaggleData":
        base_dataset_pkl = os.path.join(args.dataset_dir, "kaggle_dataset.pkl")
        base_root_dir= os.path.join(args.dataset_dir, "kaggle")    
        mean, std = 0.9016, 0.206
        
    elif args.base_dataset == "WHOIData":
        base_dataset_pkl = os.path.join(args.dataset_dir, "whoi_dataset.pkl")
        base_root_dir= os.path.join(args.dataset_dir, "whoi")    
        
    # unnormlaised_data = MyDataset(root_dir=base_root_dir, split_file=base_dataset_pkl, phase='train',  image_size=args.image_size, normalize_param=None)
    # mean, std = normalize(args.batch_size, unnormlaised_data)
    # print("Mean and Std", mean, std)
    ## Create datasets ###

    
    base_train_dataset = MyDataset(root_dir=base_root_dir, split_file=base_dataset_pkl, phase='train',  image_size=args.image_size, normalize_param=[mean,std])
    base_val_dataset = MyDataset(root_dir=base_root_dir, split_file=base_dataset_pkl, phase='val',  image_size=args.image_size, normalize_param=[mean,std])
    base_test_dataset = MyDataset(root_dir=base_root_dir, split_file=base_dataset_pkl, phase='test',  image_size=args.image_size, normalize_param=[mean,std])
    
    
    ## Create Dataloader ##
    base_train_dataloader = DataLoader(base_train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    base_val_dataloader = DataLoader(base_val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    base_test_dataloader = DataLoader(base_test_dataset, args.batch_size, shuffle=False, pin_memory=True)
   
    print("Size of Train Data", len(base_train_dataloader))
    print("Size of Val Data", len(base_val_dataloader))
    print("Size of Test Data", len(base_test_dataloader))
    
    
    # Repeat for the target dataset 
    ## Get dataset splits ###
    if args.target_dataset == "KaggleData":
        target_dataset_pkl = os.path.join(args.dataset_dir, "kaggle_dataset.pkl")
        target_root_dir= os.path.join(args.dataset_dir, "kaggle")  
       
        
    elif args.target_dataset == "WHOIData":
        target_dataset_pkl = os.path.join(args.dataset_dir, "whoi_dataset.pkl")
        target_root_dir= os.path.join(args.dataset_dir, "whoi")   
         
    elif args.target_dataset == "MiniPPlankton":
        target_dataset_pkl = os.path.join(args.dataset_dir, "minipplankton_dataset.pkl")
        target_root_dir= os.path.join(args.dataset_dir, "miniPPlankton")    
        
    elif args.target_dataset == "HB":
        print("hb")
        target_dataset_pkl = os.path.join(args.dataset_dir, "harborBranch_dataset.pkl")
        target_root_dir= os.path.join(args.dataset_dir, "harborBranch")  
        
        
    ## Create datasets ###

    #using mean std of base dataset
    target_train_dataset = MyDataset(root_dir=target_root_dir, split_file=target_dataset_pkl, phase='train',  image_size=args.image_size, normalize_param=[mean,std])
    target_val_dataset = MyDataset(root_dir=target_root_dir, split_file=target_dataset_pkl, phase='val',  image_size=args.image_size, normalize_param=[mean,std])
    target_test_dataset = MyDataset(root_dir=target_root_dir, split_file=target_dataset_pkl, phase='test',  image_size=args.image_size, normalize_param=[mean,std])
   
    
   

    # Create Model ##
   
    encoder = Resnet12(width=1, dropout=0.1)
    classifier = Classifier(embedding_dimension=512, num_of_classes=args.base_num_of_classes)
    model = nn.Sequential(encoder, classifier)
    feature_dim = encoder.output_size
   
    
    # load teacher embeddings
    sd = torch.load(args.embedding_load_path)
    model.load_state_dict(sd)
    model.cuda()
    
    
    #pseudolabel generation
    target_train_dataset = pseudolabel_dataset(encoder, classifier, target_train_dataset, args)
    target_val_dataset = pseudolabel_dataset(encoder, classifier, target_val_dataset, args )
    target_test_dataset = pseudolabel_dataset(encoder, classifier, target_test_dataset, args )
    
     ## Create Dataloader ##
    target_train_dataloader = DataLoader(target_train_dataset, args.batch_size, shuffle=True, pin_memory=True,  drop_last=True)
    target_val_dataloader = DataLoader(target_val_dataset, args.batch_size, shuffle=False, pin_memory=True,  drop_last=True)
    target_test_dataloader = DataLoader(target_test_dataset, args.batch_size, shuffle=False, pin_memory=True , drop_last=True)
   

    print("Size of Train Data", len(target_train_dataloader))
    print("Size of Val Data", len(target_val_dataloader))
    print("Size of Test Data", len(target_test_dataloader))
   
    run(
        model,
        args.num_of_epochs,
        args.learning_rate,
        args.weight_decay,
        args.model_save_path,
        log_dir,
        base_train_dataloader,
        base_val_dataloader,
        base_test_dataloader,
        target_train_dataloader,
        target_val_dataloader,
        target_test_dataloader,
        args
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Teacher")

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./student/models",
        help="directory to save the checkpoints",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch_size for tecaher"
    )
    parser.add_argument('--num_of_target_classes', type=int, default=16, help='Num of target classes')

    parser.add_argument(
        "--base_num_of_classes", type=int, default=118, help="classes in dataset"
    )

    parser.add_argument(
        "--num_of_epochs", type=int, default=150, help="Number of epochs"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Leraning rate"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")

    parser.add_argument(
        "--base_dataset", type=str, default="KaggleData", help="Name of base dataset"
    )
    parser.add_argument(
        "--target_dataset", type=str, default="WHOIData", help="Name of target dataset"
    )

    parser.add_argument("--dataset_dir", type=str, default="/home/jwomack30/Plankton_Domain_Adaptation/data", help="Dataset dir")
    
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of Image"
    )
    parser.add_argument("--embedding_load_path", type=str, \
        default="/home/jwomack30/Plankton_Domain_Adaptation/teacher/models/kaggle/teacher_model_best.pkl", \
        help="Teacher model path")
    
    args = parser.parse_args()
    main(args)
