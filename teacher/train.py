from typing import List, Any, Tuple
import torch
import copy
from torch.autograd import Variable
from sklearn.metrics import f1_score
from torch import nn
import os
from tqdm import tqdm

# TODO: change name
def loop_over_all_epochs(
    dataloader: List,
    num_of_epochs: int,
    net: nn.Module,
    gpu_train: bool,
    loss_criteria: Any,
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
                train_running_corrects,
                train_actual_labels,
                train_pred_labels,
                train_dataset_size,
            ) = loop_over_all_datapoints(
                dataloader[0], gpu_train, optimizer, net, loss_criteria, "train"
            )

            train_epoch_loss = train_running_loss / train_dataset_size
            train_epoch_acc = (
                (train_running_corrects.double() / train_dataset_size)
                .cpu()
                .data.numpy()
            )

            train_epoch_losses.append(train_epoch_loss)

            train_epoch_accs.append(train_epoch_acc)

            with open(log_filepath, "a") as f:
                f.write(f"Epoch {epoch}")
                f.write("\n")
                f.write(f" Train Loss: {train_epoch_loss}")
                f.write("\n")
                f.write(f" Acc: {train_epoch_acc}")
                f.write("\n")
            net.eval()

            (
                val_running_loss,
                val_running_corrects,
                val_actual_labels,
                val_pred_labels,
                val_dataset_size,
            ) = loop_over_all_datapoints(
                dataloader[1], gpu_train, optimizer, net, loss_criteria, "val"
            )

            val_epoch_loss = val_running_loss / val_dataset_size
            val_epoch_acc = (
                (val_running_corrects.double() / val_dataset_size).cpu().data.numpy()
            )

            ## SAVE THE LAST MODEL
            # if epoch == num_of_epochs - 1:
            #     path = os.path.join(save_model_path, "teacher_model_last.pkl")
            #     sd = {}
            #     sd = copy.deepcopy(net.state_dict())
            #     torch.save(sd, path)

            ## SAVE THE BEST MODEL
            # if val_epoch_loss <= min_val_loss:
            #     path = os.path.join(save_model_path, "teacher_model_best.pkl")
            #     sd = {}
            #     sd = copy.deepcopy(net.state_dict())
            #     torch.save(sd, path)
            #     min_val_loss = val_epoch_loss

            val_epoch_losses.append(val_epoch_loss)
            val_epoch_accs.append(val_epoch_acc)

            with open(log_filepath, "a") as f:

                f.write(f"Val Loss: {val_epoch_loss} Acc: {val_epoch_acc}")
                f.write("\n")

        loss = [train_epoch_losses, val_epoch_losses]
        acc = [train_epoch_accs, val_epoch_accs]

    else:
        print("EVAL")
        net.eval()
        (
            test_loss,
            test_corrects,
            test_actual_labels,
            test_pred_labels,
            test_dataset_size,
        ) = loop_over_all_datapoints(
            dataloader[0], gpu_train, optimizer, net, loss_criteria, "test"
        )

        loss = [test_loss / test_dataset_size]
        acc = [(test_corrects.double() / test_dataset_size).cpu().data.numpy()]

    return loss, acc


# TODO: change name
def loop_over_all_datapoints(
    dataloader, gpu_train, optimizer, net, loss_criteria, phase
):
    running_corrects = 0
    running_loss = 0.0
    actual_labels = []
    pred_labels = []
    total_size = 0

    for X, y in tqdm(dataloader):
        X = Variable(X).float()
        y = Variable(y).type(torch.LongTensor)

        if gpu_train:
            X = X.cuda()
            y = y.cuda()

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            print(X.shape)
            output = net(X)
            # print("Output",output.shape)
            _, preds = torch.max(output, 1)
            print("y", y.shape)
            print("output", output.shape)
            loss = loss_criteria( output, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * X.size(0)
        running_corrects += torch.sum(preds == y.data)
        actual_labels.extend(y.cpu().data.numpy())
        pred_labels.extend(preds.cpu().data.numpy())
        total_size += X.size(0)

    return running_loss, running_corrects, actual_labels, pred_labels, total_size
