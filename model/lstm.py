import torch
# 설정값
# Param = 4*((input_shape_size +1) * ouput_node + output_node^2)6
class LSTM(torch.nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_dim = 106
        self.seq_len = 96
        self.input_dim = 6
        self.output_dim = 1
        self.layers = 2
        
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.layers,dropout=0.5,batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_dim, bias = True) 
        
    # 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    # 예측을 위한 함수
    def forward(self, x):
        # Initialize hidden state with zeros
        #print(x.shape)
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layers,  x.size(0), self.hidden_dim).requires_grad_()
        x, _status = self.lstm(x,(h0.detach(), c0.detach()))
        x = self.relu(x)
        x = self.fc(x[:, -1,:])
        #print(x.shape)
        return x