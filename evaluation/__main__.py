import argparse
from datasets.datasetClass import FewShotDataset
import os
import torch

import numpy as np
import random
from evaluation.finetune import *


def main(args):
    if args.dataset == "KaggleData":
        dataset_pkl = os.path.join(args.dataset_dir, "kaggle_dataset.pkl")
        root_dir= os.path.join(args.dataset_dir, "kaggle")    
        mean, std = 0.9016, 0.206
        
    elif args.dataset == "WHOIData":
        dataset_pkl = os.path.join(args.dataset_dir, "whoi_dataset.pkl")
        root_dir= os.path.join(args.dataset_dir, "whoi")    
        mean, std = 0.9016, 0.206
        
    
     
    results = {}
    shot_done = []
    
    for shot in args.n_shots:
        print(f"{args.n_way}-way {shot}-shot")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        train_dataset = FewShotDataset(root_dir=root_dir, split_file=dataset_pkl, phase='all',  \
                                                image_size=args.image_size, \
                                                normalize_param=[mean,std], \
                                                n_eposide=args.n_episodes,
                                                n_query=args.n_query, 
                                                n_way=args.n_way,
                                                n_support=shot,  )
  
        ## Create Dataloader ##
        novel_loader = train_dataset.get_data_loader()
        acc_all = finetune(novel_loader, args, n_shot=shot)
        results[shot] = acc_all
        shot_done.append(shot)

        # if params.save_suffix is None:
        #     pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
        #         params.source_dataset + '_' + params.target_dataset + '_' + 
        #         str(params.n_way) + 'way' + '.csv'), index=False)
        # else:
        #     pd.DataFrame(results).to_csv(os.path.join(params.save_dir, 
        #         params.source_dataset + '_' + params.target_dataset + '_' + 
        #         str(params.n_way) + 'way_' + params.save_suffix + '.csv'), index=False)
    print(results)
    return 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finetuning")

    parser.add_argument(
        "--model_load_path",
        type=str,
        default="./teacher/models",
        help="directory to load the checkpoints",
    )


    parser.add_argument(
        "--num_of_classes", type=int, default=22, help="classes in dataset"
    )

    parser.add_argument(
        "--num_of_epochs", type=int, default=100, help="Number of epochs"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Leraning rate"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

    parser.add_argument(
        "--dataset", type=str, default="WHOIData", help="Name of dataset"
    )

    parser.add_argument("--dataset_dir", type=str, \
        default="/home/jwomack30/Plankton_Domain_Adaptation/data", help="Dataset dir")
    
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of Image"
    )
    
    parser.add_argument(
        "--n_way", type=int, default=5, help="Classes for classification"
    )

    parser.add_argument(
        "--n_query", type=int, default=1, help="Number of queries for testing"
    )
    
    parser.add_argument(
        "--n_shots", nargs='+', default=[5,20], help="Number of shots per class"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=600, help="Number of episodes"
    )
    parser.add_argument('--seed', default=78, type=int, help='random seed')
    
    parser.add_argument("--embedding_load_path", type=str, \
        default="/home/jwomack30/Plankton_Domain_Adaptation/teacher/models/teacher_model_best.pkl", \
        help="Teacher model path")
    args = parser.parse_args()
    main(args)
