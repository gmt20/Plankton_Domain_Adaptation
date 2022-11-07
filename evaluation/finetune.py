import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import os
from models.resnet12 import Resnet12
from tqdm import tqdm
import copy 

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def finetune(novel_loader, params, n_shot): 

    print("Loading Model: ", params.embedding_load_path)
    

    pretrained_model_template = Resnet12()
    feature_dim = pretrained_model_template.output_size
   
    
    if not  params.random_initialize:
        print("loading weights")
        sd = torch.load(params.embedding_load_path)
        enc_sd = {}
        for key in sd.keys():
            if key.startswith('0'):
                enc_key = key[2:]
                enc_sd[enc_key] = sd[key]
        
        pretrained_model_template.load_state_dict(enc_sd)

    n_query = params.n_query
    n_way = params.n_way
    n_support = n_shot

    acc_all = []

    for i, (x, y) in tqdm(enumerate(novel_loader)):

        # print(y)
        pretrained_model = copy.deepcopy(pretrained_model_template)
        classifier = Classifier(feature_dim, params.n_way)

        pretrained_model.cuda()
        classifier.cuda()

        ###############################################################################################
        x = x.cuda()
        x_var = x

        assert len(np.unique(y)) == n_way
    
        batch_size = 5
        support_size = n_way * n_support 
       
        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()

        # split into support and query
        x_b_i = x_var[:, n_support:,: ,: ,:].contiguous().view(n_way*n_query, *x.size()[2:]).cuda() 
        x_a_i = x_var[:, :n_support,: ,: ,:].contiguous().view(n_way*n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)

        if params.random_initialize:
            pretrained_model.train()
        else: 
            pretrained_model.eval()
        
            with torch.no_grad():
                f_a_i = pretrained_model(x_a_i)
      

         ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        

        ###############################################################################################
        total_epoch = 100
        
        classifier.train()

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
               
                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                y_batch = y_a_i[selected_id]

                if  params.random_initialize:
                    output = pretrained_model(x_a_i[selected_id])
                else:   
                    output = f_a_i[selected_id]
               
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()


        pretrained_model.eval()
        classifier.eval()

        with torch.no_grad():
            output = pretrained_model(x_b_i)
            scores = classifier(output)
       
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        # print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))

        if (i+1) % 100 == 0:
            acc_all_np = np.asarray(acc_all)
            acc_mean = np.mean(acc_all_np)
            acc_std = np.std(acc_all_np)
            print('Test Acc (%d episodes) = %4.2f%% +- %4.2f%%' %
                (len(acc_all),  acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all))))
        
        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %
          (len(acc_all),  acc_mean, 1.96 * acc_std/np.sqrt(len(acc_all))))

    return acc_all
