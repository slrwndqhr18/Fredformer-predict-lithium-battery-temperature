# Fredformer predict lithium battery temperature
Transformer based time series prediction

SRC Project Name: Predicting battery overcharging on EV car 

<img width="937" alt="image" src="https://github.com/user-attachments/assets/cffdbee9-c6b6-420e-b8de-a7e18804f1dd" />


Fredformer reference [by Xihao Piao]</br>
GitHub: https://github.com/chenzRG/Fredformer</br>
Archive: https://arxiv.org/abs/2406.09009</br>

Data reference [NASA Battery Dataset]</br>
src: https://ieee-dataport.org/documents/nasa-lithium-ion-battery-dataset</br>

------------------------------
## About Code


The original code was so difficult and some part had error so I changed it lighter and simmple to use it.
Components of model (in directory /model/Component), most of them are copied from original sorce, I changed some error part.
So the logic is same.

|battery temperature (deg)|percentage|
|------|---|
| /model | Every codes related to ML model |
| /model/Component | Codes that used inside the ML model.</br>Its a component of ML model like attention alg or FN layer. |
| handleConfig.py | load CONFIG.yaml and setup every parameters |
| handleDataset.py | By using PyTorch Dataloader, revise and format preprocessed data into PyTorch data structure |
| handlePreprocess | Load raw dataset and execure preprocessing |
| makeGraph | Just making graph to see the result. Nothing important in here. |

- main.py is the entry point of process. Just run main.py
- /model/Executor.py is the entry point of ML model. It is  a class to load params and define model train / test / run process.
- /model/Component, In this project, all of the codes are from original codes of Fredformer [https://github.com/chenzRG/Fredformer]
I chainged the original codes because there's an error and I had to handle those to run the model.



## About the test parameters

<img width="228" alt="image" src="https://github.com/user-attachments/assets/f3004158-7987-4290-9775-dc3f329537a3" />


[Fig.0 data distribution]
|battery temperature (deg)|percentage|
|------|---|
|10 ~ 20|56%|
|0 ~ 10|29.2%|
|20 ~ 30|12.9%|
|30+|0.9%|

Learning rate=0.001->0.0001  I changed with scheduler </br>
batch=128 </br>
sequence=98</br>
Total epoch=8</br>
Parameter count=(Fred: 139,281, LSTM: 139,179)


## Result

<img width="511" alt="image" src="https://github.com/user-attachments/assets/00deb74d-b45c-4d10-8246-0dec9aff7b36" />


[Fig1. Comparing with LSTM and Fredformer]

<img width="639" alt="image" src="https://github.com/user-attachments/assets/0418d954-b49d-4cbf-b3c1-7ec4d5cb885d" />


[Fig2. Average loss of each data. Left - LSTM, Right - Fredformer]</br>
I compared with LSTM, representative model of time series prediction.</br>
Transformer model was not used very much at time series field at least when I was doing my project. And I found this paper and do it at my capstone project. According to the test result, mostly LSTM shows error 4 times higher then Fredformer.
