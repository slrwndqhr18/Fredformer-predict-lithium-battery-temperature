from handleConfig import Get_config
from handlePreprocess import Run_preprocess
from handleDataset import Make_dataloader
from makeGraph import Make_loss_fig, Make_data_fig
from model.Executor import RunFredformer, RunLSTM
import polars as pl
import os

if __name__ == "__main__":
    print("start")
    paramSet = Get_config()
    __PROCESS_MODE = paramSet["executeMode"]
    dataSets = Run_preprocess(__PROCESS_MODE, paramSet["preprocess"])
    dataLoader = Make_dataloader(__PROCESS_MODE, dataSets, paramSet["datasetOpts"])
    model = RunFredformer(paramSet["hyperParams"])
    #model = RunLSTM(paramSet["hyperParams"])
    model.Get_param_nums()
    #if __PROCESS_MODE == "show_dataset":
    Make_data_fig(dataLoader)
    os._exit(0)
    if __PROCESS_MODE == "train":
        model.Train_model(dataLoader)
        os._exit(0)
    elif __PROCESS_MODE == "test":
        o, t = model.Test_model(dataLoader)
        print(o[:,-1,:].shape, t[:,-1,:].shape)
        pl.DataFrame({
            "out": o[:,-1,:].flatten(),
            "target": t[:,-1,:].flatten()
        }).write_csv("./result.csv")
    print("Step[Make fig]")
    Make_loss_fig()
    print("end")
    