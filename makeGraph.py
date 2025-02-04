import matplotlib.pyplot as plt
import numpy as np
import polars as pl
plt.rcParams['font.family']='AppleGothic'
plt.rcParams['axes.unicode_minus']=False
plt.rc('font', size=20)

# =================================================
def Make_data_fig(_dataLoader):
    i = 0
    print("Yay")
    for inputs, _ in _dataLoader:
        if i == 45001:
            print(inputs[-1][-1])
            print(inputs.shape)
            break
        i+=1
    print("Tot i == {}".format(i))


# ==================================================
def Make_loss_fig():
    xSize = 30
    a=pl.read_csv("./result.csv")
    a=a.with_columns(
        comp=pl.col("target")-pl.col("out"),
        temp=pl.when(pl.col("target")>pl.lit(29)).then(pl.lit(30))
        .when(pl.col("target")>pl.lit(19)).then(pl.lit(20))
        .when(pl.col("target")>pl.lit(9)).then(pl.lit(10)).otherwise(pl.lit(0))
    ).with_columns(
        comp=pl.col("comp").floor()
    )
    print(
        a.filter(pl.col("temp")==pl.lit(0)).shape,
        a.filter(pl.col("temp")==pl.lit(10)).shape,
        a.filter(pl.col("temp")==pl.lit(20)).shape,
        a.filter(pl.col("temp")==pl.lit(30)).shape
        )
    X = np.arange(0,xSize,1)
    plt.plot(X, a.filter(pl.col("temp")==pl.lit(0)).select("comp").sample(n=xSize).to_numpy(), label="T 0 ~ 10")
    plt.plot(X, a.filter(pl.col("temp")==pl.lit(10)).select("comp").sample(n=xSize).to_numpy(), label="T 10 ~ 20")
    plt.plot(X, a.filter(pl.col("temp")==pl.lit(20)).select("comp").sample(n=xSize).to_numpy(), label="T 20 ~ 30")
    plt.plot(X, a.filter(pl.col("temp")==pl.lit(30)).select("comp").sample(n=xSize).to_numpy(), label="T > 30")
    plt.ylabel("오차")
    plt.xlabel("ith input(Test데이터셋)")
    plt.legend()
    plt.title("epoch avg loss")
    plt.show()


def _make_epoch_fig():
    X = np.arange(1,9,1)
    plt.plot(X,[0.06509, 0.06278, 0.05894,0.06132,0.04202,0.01776,0.01962, 0.0148])
    plt.vlines(4, 0.01, 0.1, color='orange', linestyle='solid', linewidth=3,label="descrease learning rate")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.title("Epoch별 평균 loss값")
    plt.show()

def _make_batch_loss_fig():
    batch1Loss = [
         0.0255759,
                         0.0015966,
                         0.0022946,
                         0.0013359,
                         0.0100467,
                         0.0002479,
                         0.0070915,
                         0.0003797,
                         0.0266116,
                         0.0828771,
                         0.0017274,
                         0.0369463,
                         0.0107079,
                         0.0054091,
                         0.0193767,
                         0.0281048,
                         0.0008856,
                         0.0085697,
                         0.0014604,
                         0.2897934,
                         0.0158387,
                         0.0004783,
                         0.0024640,
                         0.0017885,
                         0.0019879,
                         0.0070398,
                         0.0055429,
                         0.0175998,
                         0.0009202,
                         0.0666681,
                         0.0629135,
                         0.0043979,
                         0.0008716,
                         0.0007237,
                         0.0007001,
                         0.0158006,
                         0.0022124,
                         0.0057751,
                         0.0004717,
                         0.0026407,
                         0.0026991,
                         0.0024800,
                         0.0653314,
                         0.0020631,
                         0.0191421,
                         0.0022651,
                         ]
    batch8Loss = [
        0.0071868,
                         0.0023342,
                         0.0020846,
                         0.0015122,
                         0.0162819,
                         0.0042735,
                         0.0336714,
                         0.0021953,
                         0.0332523,
                         0.0097424,
                         0.0035148,
                         0.0096622,
                         0.0014845,
                         0.0011514,
                         0.0047661,
                         0.0187038,
                         0.0016543,
                         0.0069542,
                         0.0062950,
                         0.1483995,
                         0.0036191,
                         0.0152003,
                         0.0030097,
                         0.0024754,
                         0.0099176,
                         0.0040134,
                         0.0052673,
                         0.0070228,
                         0.0017051,
                         0.0025867,
                         0.0031567,
                         0.0009837,
                         0.0004018,
                         0.0015459,
                         0.0005759,
                         0.0006593,
                         0.0038658,
                         0.0036285,
                         0.0005058,
                         0.0036112,
                         0.0043506,
                         0.0029309,
                         0.0082670,
                         0.0025746,
                         0.0242316,
                         0.0007916,
    ]
    X = np.arange(0, 46000, 1000)
    plt.ylim((0.00000001,0.35))
    plt.plot(X, batch1Loss, label="epoch 1 loss")
    plt.plot(X, batch8Loss, label="epoch 8 loss")
    plt.plot(X, [0.0650]*len(batch1Loss), label="avg epoch 1")
    plt.plot(X, [0.0148]*len(batch8Loss), label="avg epoch 8")
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.legend()
    plt.title("Batch진행에 따른 loss값 (Epoch=1,8)")
    plt.show()

if __name__ == "__main__":
    _make_batch_loss_fig()
    #_make_epoch_fig()
    #Make_loss_fig()