from typing import List, Any, Tuple
import torch
import copy
from torch.autograd import Variable
from sklearn.metrics import f1_score
from torch import nn
import os
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# TODO: change name
def loop_over_all_epochs(
    args,
    dataloader: List,
    target_dataloader: List,
    num_of_epochs: int,
    net: nn.Module,
    gpu_train: bool,
    phase: str,
    save_model_path: str = None,
    optimizer: Any = None,
    log_filepath: str = "current.log",
    
) -> Tuple[List, List]:
    
    if gpu_train:
        net = net.cuda()

    if phase == "train":

        train_epoch_losses = []
        train_epoch_accs = []
        val_epoch_losses = []
        val_epoch_accs = []
        min_val_loss = 100.0
        for epoch in tqdm(range(num_of_epochs)):

            net.train()

            (
                train_running_loss,
                train_dataset_size,
            ) = loop_over_all_datapoints(
                args,dataloader[0], target_dataloader[0], gpu_train, optimizer, net, "train"
            )

            train_epoch_loss = train_running_loss / train_dataset_size
        
            train_epoch_losses.append(train_epoch_loss)

            

            with open(log_filepath, "a") as f:
                f.write(f"Epoch {epoch}")
                f.write("\n")
                f.write(f" Train Loss: {train_epoch_loss}")
                f.write("\n")
               
            net.eval()

            (
                val_running_loss,
                val_dataset_size,
            ) = loop_over_all_datapoints(
                args, dataloader[1], target_dataloader[1],gpu_train, optimizer, net,  "val"
            )

            val_epoch_loss = val_running_loss / val_dataset_size
           

            ## SAVE THE LAST MODEL
            if epoch == num_of_epochs - 1:
                path = os.path.join(save_model_path, "student_model_last.pkl")
                sd = {}
                sd = copy.deepcopy(net.state_dict())
                torch.save(sd, path)

            ## SAVE THE BEST MODEL
            if val_epoch_loss <= min_val_loss:
                path = os.path.join(save_model_path, "student_model_best.pkl")
                sd = {}
                sd = copy.deepcopy(net.state_dict())
                torch.save(sd, path)
                min_val_loss = val_epoch_loss

            val_epoch_losses.append(val_epoch_loss)
        

            with open(log_filepath, "a") as f:

                f.write(f"Val Loss: {val_epoch_loss} ")
                f.write("\n")

        loss = [train_epoch_losses, val_epoch_losses]


    else:
        net.eval()
        (
            test_loss,
            test_dataset_size,
        ) = loop_over_all_datapoints(
            args, dataloader[0],target_dataloader[0], gpu_train, optimizer, net, "test"
        )

        loss = [test_loss / test_dataset_size]
        

    return loss


# TODO: change name
def loop_over_all_datapoints(
    args, dataloader, target_dataloader, gpu_train, optimizer, net,  phase
):
    running_corrects = 0
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    total_size = 0

    base_loss = nn.CrossEntropyLoss()
    dist_loss = nn.CrossEntropyLoss()
    base_loader_iter = iter(dataloader)
    
    for X, y in tqdm(target_dataloader):
        ## target data points
        X = Variable(X).float()
        y = Variable(y).type(torch.float32)

        if gpu_train:
            X = X.cuda()
            y = y.cuda()
        
        # base data points    
        try:
            X_base, y_base = base_loader_iter.next()
        except StopIteration:
            base_loader_iter = iter(base_dataloader)
            X_base, y_base = base_loader_iter.next()

        X_base = Variable(X_base).float()
        y_base = Variable(y_base).type(torch.LongTensor)  # note

        if gpu_train:
            X_base = X_base.cuda()
            y_base = y_base.cuda()

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            f1 = net[0](X)
            # log_prob_target = F.log_softmax(output_target, dim=1)
            f1 = f1.detach().cpu().numpy()
            # print("features shape", f1.shape)
            pca = PCA(n_components=args.num_of_target_classes).fit(f1)
            # print("args.num_of_target_classes", args.num_of_target_classes)
            kmeans = KMeans(init=pca.components_, n_clusters=args.num_of_target_classes, n_init=1)
            generated_labels = kmeans.fit_predict(f1)
            # print("generated labels", generated_labels.shape)
            y1 = torch.tensor(generated_labels, dtype =torch.float).cuda()
            
            output_base = net(X_base)
            # print("Output",output.shape)
            _, preds = torch.max(output_base, 1)
            print("y", y, "y1", y1)
            loss_on_target = dist_loss(y1, y) 
            loss_on_base = base_loss( output_base, y_base)
            loss = loss_on_base + loss_on_target

            if phase == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * X.size(0)
        total_size += X.size(0)

    return running_loss, total_size
