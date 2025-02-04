from  torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
class BigDataset(Dataset):
    def __init__(self, _exeMode, _dset: pl.DataFrame, _inputLen, _targetLen, _predLen):
        super().__init__()
        
        #filename,Time,Temperature_measured,target_nxt,target_rng_max,Voltage_measured,Current_measured,Current_charge,Voltage_charge,ambient_temperature
        if _exeMode == "show_dataset":
            _exeMode = "show_dataset"
        if _exeMode == "train":
            self.__set_train_dataset(_dset, _exeMode)
        else:
            self.__set_test_dataset(_dset)
        self.input_len = _inputLen
        self.target_len = _targetLen
        self.pred_len = _predLen
    def __set_test_dataset(self, _dset):
        self.__TARGET = _dset.select(["target_rng_max"]).to_numpy()
        self.__DATASET = _dset.select([
            "Temperature_measured",
            "Voltage_measured",
            "Current_measured",
            "Current_charge",
            "Voltage_charge",
            "ambient_temperature"
        ]).to_numpy()
        self.__PE_DATA = _dset.select(["Time"]).to_numpy()
    def __set_train_dataset(self, _dset, _exeMode):
        self.__TARGET = self.__preprocess(_exeMode, _dset.select(["target_rng_max"]), 0)
        self.__DATASET = self.__preprocess(_exeMode, _dset.select([
            "Temperature_measured",
            "Voltage_measured",
            "Current_measured",
            "Current_charge",
            "Voltage_charge",
            "ambient_temperature"
        ]), 1)
        self.__PE_DATA = self.__preprocess(_exeMode, _dset.select(["Time"]), 2)
    def __preprocess(self,_exeMode, _dset: pl.DataFrame, _dType):
        print("\t\tㄴ",_dset.shape)
        scaler = StandardScaler()
        if _exeMode == "train":
            scaler.fit(_dset.to_numpy())
        else:
            # 임시코드 부분 =======================================
            trainData = pl.read_csv("./DATA/_TRAIN.csv")
            if _dType == 0:
                trainData = trainData.select(["target_rng_max"])
            elif _dType == 1:
                trainData = trainData.select([
                    "Temperature_measured",
                    "Voltage_measured",
                    "Current_measured",
                    "Current_charge",
                    "Voltage_charge",
                    "ambient_temperature"
                ])
            elif _dType == 2:
                trainData = trainData.select(["Time"])
            else: return
            print("\t\tㄴ",trainData.shape)
            scaler.fit(trainData.to_numpy())
            # =================================================
        _dset = scaler.transform(_dset.to_numpy())
        return _dset

    def __getitem__(self, index): #for Fredformer
        seqStart = index
        seqEnd = seqStart + self.input_len
        _inputData = self.__DATASET[seqStart: seqEnd]
        _targetData = self.__TARGET[seqStart: seqEnd] # seqStart : seqEnd
        return _inputData, _targetData
    def __getitem__2(self, index): #for LSTM
        seqStart = index
        seqEnd = seqStart + self.input_len
        _inputData = self.__DATASET[seqStart:seqEnd]
        _targetData = self.__TARGET[seqEnd-1]
        #return np.array([_inputData]), np.array([_targetData])
        return np.array(_inputData), np.array(_targetData)
    
    def __len__(self):
        return len(self.__DATASET)

def Make_dataloader(_exeMode, _dSet: pl.DataFrame, _opts):
    #rawDataset = random_split(_dset, [50000,10000])
    #Split은 전처리 단계서 이미 수행
    return DataLoader(
        BigDataset(_exeMode, _dSet, _opts["input_len"], _opts["target_len"], _opts["pred_len"]),
        batch_size=_opts["batch_size"], #수정필요
        num_workers=_opts["loader_workers"],
        shuffle=False,
        drop_last=True
    )
    # TestSet = DataLoader(
    #     BigDataset(_testSet),
    #     batch_size=1000, #수정필요
    #     num_workers=10,
    #     shuffle=False
    #     )
    
    #원본은 return data_set, data_loader
