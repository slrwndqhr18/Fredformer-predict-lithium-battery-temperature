import polars as pl
import os
from numpy import trunc
#from handleCQT import Conv_to_CQT_img
#from handleMultiPs import Exec_in_parallel

class Preprocess:
    def __init__(self, _rawdPath, _metadPath, _outputPath,
                 _trainSetPath, _testSetPath, _predSetPath):
        self.__OUTPUT_FILE_PATH = _outputPath
        self.__srcDataPath = _rawdPath
        self.__metadataPath = _metadPath
        self.__trainSetPath = _trainSetPath
        self.__testSetPath = _testSetPath
        self.__predSetPath = _predSetPath
    
    def __make_target_set(self, _dset: pl.DataFrame):
        tgtColNm = "Temperature_measured"
        _dset = _dset.sort(["filename", "Time"])
        _dset = _dset.with_columns(
            Time=pl.col("Time").round(3),
            Temperature_measured=pl.col("Temperature_measured").round(4),
            Voltage_measured=pl.col("Voltage_measured").round(4),
            Current_measured=pl.col("Current_measured").round(4),
            Current_charge=pl.col("Current_charge").round(4),
            Voltage_charge=pl.col("Voltage_charge").round(4),
            ambient_temperature=pl.col("ambient_temperature").round(4),
        ).with_columns(
            t_range1=pl.col(tgtColNm).shift(1),
            t_range2=pl.col(tgtColNm).shift(2),
            t_range3=pl.col(tgtColNm).shift(3),
            t_range4=pl.col(tgtColNm).shift(4),
        ).with_columns(
            target_nxt = pl.when(pl.col("filename") == pl.col("filename").shift(-1))
            .then(pl.col(tgtColNm).shift(-1)).otherwise(pl.lit(-1)),
            target_rng_max = pl.when(pl.col("filename") == pl.col("filename").shift(-4))
            .then(pl.max_horizontal("Temperature_measured", "t_range1","t_range2","t_range3","t_range4")).otherwise(pl.lit(-1)),
        ).select(
            ["filename", "Time", "Temperature_measured", "target_nxt","target_rng_max", "Voltage_measured","Current_measured","Current_charge","Voltage_charge","ambient_temperature"]
        )
        return _dset
    
    def __get_all_data_from_dir(self, _metad):
        fileList = _metad.select("filename").to_series().to_list()
        #os.listdir(self.__srcDataPath)
        print("\tㄴTotal data files:",len(fileList))
        
        startFile = fileList.pop()
        dset = pl.read_csv(self.__srcDataPath + "/" + startFile).with_columns(filename=pl.lit(startFile))
        totalDataLen = dset.shape[0] #7376834
        
        for f in fileList:
            temp = pl.read_csv(self.__srcDataPath + "/" + f).with_columns(filename=pl.lit(f))
            totalDataLen += temp.shape[0]
            dset = dset.extend(temp)
        del fileList
        print("\tㄴTot dataset size:",totalDataLen,dset.shape[0]==totalDataLen)
        return dset.join(_metad, "filename", "left", join_nulls=False)

    def __get_const_var_from_metad(self):
        print("\tㄴReading metadata -",self.__metadataPath)
        metadata = pl.read_csv(self.__metadataPath).filter(pl.col("type") == pl.lit("charge"))
        return metadata.select(["filename", "ambient_temperature"]) #"type"

    def __split_datasets(self, _dset: pl.DataFrame):
        print("\tㄴsplit data set, Total dataset", _dset.shape[0])
        predSet = _dset.filter(pl.col("target_rng_max")==pl.lit(-1))
        _dset = _dset.filter(pl.col("target_rng_max")!=pl.lit(-1))
        labList = _dset.select("filename").unique(keep="first").to_series().to_list()
        splitIdx = int(len(labList)*0.9)
        trainList = labList[:splitIdx]
        testList = labList[splitIdx:]
        trainSet = _dset.filter(pl.col("filename").is_in(trainList))
        testSet = _dset.filter(pl.col("filename").is_in(testList))
        print("\tㄴSplit data to follows: Train={}, Test={}, Pred={}, Tot={}".format(trainSet.shape[0], testSet.shape[0], predSet.shape[0], trainSet.shape[0]+testSet.shape[0]+predSet.shape[0]))
        trainSet.write_csv(self.__trainSetPath)
        testSet.write_csv(self.__testSetPath)
        predSet.write_csv(self.__predSetPath)
        return trainSet, testSet, predSet

    def __preprocess_raw_data(self) -> pl.DataFrame:
        metadata = self.__get_const_var_from_metad()
        dset = self.__get_all_data_from_dir(metadata)
        dset = self.__make_target_set(dset)
        print("\tㄴBefore drop NA:", dset.shape, end=" -> ")
        dset = dset.drop_nulls()
        print("After drop NA:", dset.shape)
        dset.write_csv(self.__OUTPUT_FILE_PATH)
        return dset
    
    def __get_only_datasets(self, _exeMode):
        fPath = self.__predSetPath
        if _exeMode == "train":
            fPath = self.__trainSetPath
        elif _exeMode == "test":
            fPath = self.__testSetPath
        return pl.read_csv(fPath)


    def __call__(self, _exeMode, _forcedRun = False):
        print("Step[Run preprocess]")
        if not _forcedRun and os.path.exists(self.__trainSetPath):
            return self.__get_only_datasets(_exeMode)
        elif not _forcedRun and os.path.exists(self.__OUTPUT_FILE_PATH):
            dset = pl.read_csv(self.__OUTPUT_FILE_PATH)
            trainSet, testSet, predSet = self.__split_datasets(dset)
        else:
            dset = self.__preprocess_raw_data()
            trainSet, testSet, predSet = self.__split_datasets(dset)
            print("\tData Size: Train={} | Test={} | Pred={}".format(trainSet.shape, testSet.shape, predSet.shape))
            if _exeMode == "train":
                return trainSet
            elif _exeMode == "test":
                return testSet
            else:
                return predSet

def Run_preprocess(_exeMode, _prepConfigParams) -> pl.DataFrame:
    print("Run preprocess:")
    return Preprocess(
        _prepConfigParams["inputPath"],
        _prepConfigParams["metadPath"],
        _prepConfigParams["outputPath"],
        _prepConfigParams["trainSetPath"],
        _prepConfigParams["testSetPath"],
        _prepConfigParams["predictSetPath"],
        )(_exeMode, _prepConfigParams["forced"])
