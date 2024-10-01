import torch  
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from mamba_ssm import Mamba 
from tqdm import tqdm
from sst2Dataset import sst2Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import math

if __name__ == '__main__':
    # 1. 数据加载和预处理  
    #MODEL_PATH = "/home/ubuntu/test/bert-base-uncased"
    MODEL_PATH = '/mnt/c/Users/86152/PycharmProjects/pythonProject/uncased_L-12_H-768_A-12'
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device('cuda')
    model_bert = BertModel.from_pretrained(MODEL_PATH).to(device)

    #设置超参数
    d_model = 768
    d_state = 2
    batch_size = 64
    lr = 1e-3
    max_length = 256
    d_expand = 2

    tb_writer = SummaryWriter(log_dir="logs/test{}_lr{}ml_{}ds_{}_de_{}_show20".format("SST2", lr, max_length, d_state, d_expand))
    # 加载SST2数据集  
    sst2data_train = sst2Dataset(device, max_length, True, tokenizer, batch_size, d_model)
    sst2data_test = sst2Dataset(device, max_length, False, tokenizer, batch_size, d_model)

    train_loader = DataLoader(sst2data_train.data_iter, batch_size, shuffle=True, collate_fn=sst2data_train.collate_batch)  
    test_loader = DataLoader(sst2data_test.data_iter, batch_size, shuffle=False, collate_fn=sst2data_test.collate_batch)  
  
    # 2. 初始化模型  
    model =  Mamba(d_model=d_model, 
                    expand = d_expand,
                    d_state=d_state,
                    device=device).to(device) # 二分类问题

    # 3. 定义优化器
    epochs = 50
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    flatten_layer = torch.nn.Flatten().to(device)
    linear_2 = torch.nn.Linear(max_length*d_model, 2).to(device)
    params_to_optimize = list(model.parameters()) + list(linear_2.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=lr, eps=1e-8, weight_decay=5e-4)
    lf_pg = lambda epoch: math.exp(0-epoch/epochs)/math.exp(epoch/epochs) 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_pg)
    
    # 4. 训练模型  
    max_acc = 0
    for epoch in range(epochs):  
        loss_mean = torch.cuda.FloatTensor([0])
        #loss_mean = torch.tensor([0], dtype=torch.float32, device='cuda')
        scheduler.step()
        right_num , train_acc_num = 0, 0
        model.train()
        linear_2.train()
        for i, (input_data, label) in tqdm(enumerate(train_loader)):
            inputdata = model_bert(input_data['input_ids'], attention_mask=input_data['attention_mask'])
            outputs = model(inputdata.last_hidden_state)
            outputs = F.softmax(linear_2(flatten_layer(outputs)), dim=-1) # .sigmoid()
            label_ts = torch.zeros(outputs.size()[0], 2).to(device)
            for i, idx in enumerate(label):  
                label_ts[i, idx] = 1 
            loss = criterion(outputs, label_ts)
            optimizer.zero_grad()
            loss.backward()   #  retain_graph=True
            optimizer.step() 
            loss_mean += loss

            label_list = torch.tensor(label, dtype=torch.int64).to(device).float()
            predd = torch.argmax(outputs, dim=1)
            train_acc_num += torch.eq(predd, label_list).sum().item()

        model.eval()
        with torch.no_grad():
            for i, (input_data, label) in tqdm(enumerate(test_loader)):
                inputdata = model_bert(input_data['input_ids'], attention_mask=input_data['attention_mask'])
                outputs = model(inputdata.last_hidden_state)# .mean(dim=2).mean(dim=1)
                outputs = linear_2(flatten_layer(outputs))
                pred = torch.argmax(outputs, dim=1)
                label_list = torch.tensor(label, dtype=torch.int64).to(device).float()
                right_num += torch.eq(pred, label_list).sum().item()
        print("Epoch: {} ".format(epoch), 
                "loss: {} ".format(loss_mean.mean().item()),
                "train_acc: {} all{} ".format(train_acc_num/len(sst2data_train), len(sst2data_train)),
                "acc: {} all{} ".format(right_num/len(sst2data_test), len(sst2data_test)) )
        tags = ["train_loss", "train_acc", "acc", "lr"] # scalar_value=val, global_step=i
        tb_writer.add_scalar(tags[0], scalar_value = loss_mean.mean(), global_step=epoch)
        tb_writer.add_scalar(tags[1], scalar_value = train_acc_num/len(sst2data_train), global_step=epoch)
        tb_writer.add_scalar(tags[2], scalar_value = right_num/len(sst2data_test), global_step=epoch)
        tb_writer.add_scalar(tags[3], scalar_value = optimizer.param_groups[0]["lr"], global_step=epoch)
        if max_acc < right_num/len(sst2data_train):
            max_acc = right_num/len(sst2data_train)
    print(max_acc)
        
    
        