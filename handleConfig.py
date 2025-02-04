import yaml

def Get_config(_configFilePath = "./CONFIG.yaml"):
    with open(_configFilePath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Share params between components
        config["hyperParams"]["batch_size"] = config["datasetOpts"]["batch_size"]
        config["hyperParams"]["input_len"] = config["datasetOpts"]["input_len"]
        config["hyperParams"]["pred_len"] = config["datasetOpts"]["pred_len"]
        config["hyperParams"]["target_len"] = config["datasetOpts"]["target_len"]
        config["hyperParams"]["learning_rate"] = config["datasetOpts"]["learning_rate"]
        config["hyperParams"]["train_epochs"] = config["datasetOpts"]["train_epochs"]
        # ===============================

        return config