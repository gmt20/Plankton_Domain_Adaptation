
import torch
import typer
import numpy as np
import os
import matplotlib.pylab as plt
from torch import nn
import sys
from models.resnet12 import Resnet12
from models.classifier import Classifier
from datasets import PMID2019, commons
from teacher.train import loop_over_all_epochs

app = typer.Typer(pretty_exceptions_show_locals=False)


def define_optimizer(
    model:nn.module, learning_rate:float=1e-4, weight_decay: float = 1e-5, optim_type: str = "ADAM"
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
    model: nn.Module,
    num_of_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    model_save_path: str,
    log_dir: str,
    trainloader,
    valloader,
    testloader
):

    log_filepath = log_dir + "/" + "cur_log.log"
    initial_sd = model.state_dict()

    if torch.cuda.is_available():
        train_on_gpu = True
        
        
    optimizer = define_optimizer(model, learning_rate, weight_decay, optim_type="ADAM")
    loss_criteria = nn.CrossEntropyLoss()


    test_acc = 0.0
    test_loss = 0.0
   

    (
        epoch_losses,
        epoch_accs,
        
    ) = loop_over_all_epochs(
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
    # plt.show()

    # TEST

    (
        test_loss,
        test_acc,
    ) = loop_over_all_epochs(
        [testloader], 1,  model,
        train_on_gpu, loss_criteria, "test"
    )

    test_loss += np.asarray(test_loss)[0]
    test_acc += np.asarray(test_acc)[0]

  
    with open(log_filepath, "a") as f:
        

        f.write(f"Test Metrics ")
        f.write(
            f"Test Loss: {np.asarray(test_loss)[0]} Acc: {np.asarray(test_acc)[0]}"
            )
        
    


@app.command()
def main(
    
    num_of_classes: int,
    num_of_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dataset_dir: str,
    model_save_path: str,
    
):
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    log_dir = os.path.join(model_save_path, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    
    ## Create dataset ###
    
    dataset = PMID2019.createDataset(dataset_dir)
    
    
    ## Create dataloader ###
    
    train_dataloader = commons.createDataloader(dataset, batch_size, phase='train')
    val_dataloader = commons.createDataloader(dataset, batch_size, phase='val')
    test_dataloader = commons.createDataloader(dataset, batch_size, phase='test')
    
    ## Apply Transformations ## 
    
    list_of_transformations =[]
    
    transformed_train_dataloader = commons.applyTransformations(train_dataloader, list_of_transformations)
    transformed_val_dataloader = commons.applyTransformations(val_dataloader, list_of_transformations)
    transformed_test_dataloader = commons.applyTransformations(test_dataloader, list_of_transformations)
    
    # Create Model ##
    encoder = Resnet12()
    classifier = Classifier(embedding_dimension=512, num_of_classes=num_of_classes)
    model = nn.Sequential(encoder, classifier)

    run(
        model,
        num_of_epochs,
        learning_rate,
        weight_decay,
        batch_size,
        model_save_path,
        log_dir, 
        transformed_train_dataloader,
        transformed_val_dataloader ,
        transformed_test_dataloader
        
    )


if __name__ == "__main__":
    app()
