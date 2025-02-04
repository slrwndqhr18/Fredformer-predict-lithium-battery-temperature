from torch import nn, optim

def Get_optimizer():
    return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

def Get_criterion():
    criterion = nn.MSELoss()
    #criterion2 = SupConLoss(temperature=0.01)
    #criterion2 = swavloss(self.device)
    criterion2 = NTXentLoss(self.device,self.args.batch_size,temperature=0.07,use_cosine_similarity=False)#self.args.enc_in
    return criterion,criterion2