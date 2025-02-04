import torch 
import time
from model.lstm import LSTM
from model.Fredformer import Fredformer
from model.matrics import metric
from torch import nn, optim, autograd
import math
import numpy as np
import os

class AIExecutor:
    def __init__(self,_name, _options, _model, _optim):
        self.__saveParamPath = "./SV_PARAM"
        self.__saveLoadPath = _options["savedLoadPath"]
        self.__modelName =_name
        self.Model:Fredformer = _model
        self.Optimizer = _optim
        
        self.batchSize = _options["batch_size"]
        self.epochSize = _options["train_epochs"]

        self.schedularStep = _options["schedular_step"]
        self.schedularGamma = _options["schedular_gamma"]
        
        if torch.cuda.is_available():
            #cuda:0
            self.Device = torch.device("cuda")
            print("Executing model - Using GPU")
        else:
            self.Device = torch.device("cpu")
            print("Executing model - GPU is not available -> Using CPU")    
    def Get_param_nums(self):
        print("Print param nums:")
        temp = 0
        for parameter in self.Model.parameters():
            if len(parameter.shape) == 2:
                temp+=parameter.shape[0] * parameter.shape[1]
            else:
                temp+=parameter.shape[0]
        print(temp)

    def __save_params(self):
        now = time
        now = now.strftime('%Y_%m_%d_%H:%M:%S')
        torch.save(self.Model,"{}/{}_{}.pt".format(self.__saveParamPath,self.__modelName, now))
        torch.save(self.Model.state_dict(),"{}/{}_state_{}.pt".format(self.__saveParamPath,self.__modelName, now))
        print("\tㄴModel saved", now)
    
    def _run_model(self, _input, _target):
        _input = _input.to(self.Device)
        _target = _target.to(self.Device)
        return self.Model(_input), _target

    def Train_model(self, _dataLoader):
        creterion = nn.MSELoss()
        print("Step[Train model]")
        start = time.time()
        math.factorial(100000)
        scheduler = optim.lr_scheduler.StepLR(self.Optimizer, step_size=self.schedularStep, gamma=self.schedularGamma)
        autograd.set_detect_anomaly(True)
        for e in range(self.epochSize): #Epoch
            print("\tㄴ progress - Epoch: {}/{}".format(e, self.epochSize))
            self.Model.train()
            with torch.set_grad_enabled(True):
                i = 0
                trainLoss = []
                for inputs, targets in _dataLoader:
                    self.Optimizer.zero_grad()
                    try:
                        outputs, targets = self._run_model(inputs, targets)
                    except Exception as e:
                        print(e)
                        os._exit(0)
                    loss = creterion(outputs, targets)
                    #if self.weight_decay > 0:
                    #    loss = loss + 0.01 * self.reg_loss(self.model)
                    if i % 5000 == 0:
                        print("\t\t\tㄴ{0}th Loss: {1:.7f}".format(i, loss.item()))
                    # if torch.any(torch.isnan(loss.item())):
                    #     torch.any(torch.isnan(weight)) # weight에 NaN 존재 여부
                    #     model.layer.grad # layer의 gradient
                    if np.isnan(loss.item()):
                        print("\t\t\tㄴBreak!!! {0}th Loss is NaN {1}".format(i, loss.item()))
                        break
                    trainLoss.append(loss.item())
                    loss.backward()
                    self.Optimizer.step()
                    #Early break ========================
                    if i == 45001:
                       break
                    i+=1
                print("\t\tEpoch {0}th Loss avg: {1:.7f}".format(e, np.average(trainLoss)))
                scheduler.step()
        autograd.set_detect_anomaly(False)
        end = time.time()
        print(f"\tㄴfinished: {end - start:.5f} sec")
        self.__save_params()
    
    def Test_model(self, _dataLoader):
        print("Step[Test model]")
        loadFilePath = self.__saveLoadPath
        self.Model.load_state_dict(torch.load(loadFilePath))
        print("\tㄴload model -",loadFilePath)

        preds = []
        trues = []
        self.Model.eval()
        with torch.no_grad():
            i = 0
            for inputs, targets in _dataLoader:
                outputs, targets = self._run_model(inputs, targets)
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy() 
                # print(outputs.flatten().flatten().shape, targets.flatten().flatten().shape)
                # return
                preds.append(outputs)
                trues.append(targets)
                if i %1000 == 0:
                    print("\t\t\tㄴ{0}th".format(i))
                if i == 5000:
                    break
                i+=1

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print(preds.flatten().shape, trues.flatten().shape)
        #print(preds[0][0], trues[0][0])
        #return preds, trues
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('\tㄴTest result: mse={}, mae={}, rse={}'.format(mse, mae, rse))
        return preds, trues
        


class RunFredformer(AIExecutor):
    def __init__(self, _hyperparams):
        model = Fredformer(_hyperparams)
        optimizer = optim.Adam(model.parameters(), lr=_hyperparams["learning_rate"])
        super().__init__("Fredformer",_hyperparams, model, optimizer)
        self.predLen = _hyperparams["pred_len"]
        self.targetLen = _hyperparams["target_len"]

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     #criterion2 = SupConLoss(temperature=0.01)
    #     #criterion2 = swavloss(self.device)
    #     criterion2 = NTXentLoss(self.Device, self.batchSize,temperature=0.07,use_cosine_similarity=False)#self.args.enc_in
    #     return criterion,criterion2
    
    def _run_model(self, _input, _target):
        batch_x = _input.float().to(self.Device)
        batch_y = _target.float().to(self.Device)
        # decoder input
        if torch.any(torch.isnan(batch_x)):
            print(batch_x)
            print("Nan detected at input data(batch_x):", batch_x.shape)
        outputs = self.Model(batch_x)
        f_dim = -1 #if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.predLen:, f_dim:]
        batch_y = batch_y[:, -self.predLen:, f_dim:].to(self.Device)
        return outputs, batch_y
    
class RunLSTM(AIExecutor):
    #model = apply_lora_to_model(model, lora_config)
    def __init__(self, _hyperparams):
        model = LSTM()
        optimizer = optim.Adam(model.parameters(), lr=_hyperparams["learning_rate"])
        super().__init__("LSTM",_hyperparams, model, optimizer)
        self.predLen = _hyperparams["pred_len"]
        self.targetLen = _hyperparams["target_len"]

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     #criterion2 = SupConLoss(temperature=0.01)
    #     #criterion2 = swavloss(self.device)
    #     criterion2 = NTXentLoss(self.Device, self.batchSize,temperature=0.07,use_cosine_similarity=False)#self.args.enc_in
    #     return criterion,criterion2
    
    def _run_model(self, _input, _target):
        batch_x = _input
        batch_y = _target
        batch_x = batch_x.float().to(self.Device)
        batch_y = batch_y.float().to(self.Device)
        if torch.any(torch.isnan(batch_x)):
            print(batch_x)
            print("Nan detected at input data(batch_x):", batch_x.shape)
        outputs = self.Model(batch_x)
        return outputs, batch_y
    
