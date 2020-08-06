#coding=utf-8
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





if __name__=="__main__":

    date_parser=lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
    #读取数据
    path1=".//mcp.xlsx"
    df_mcp=pd.read_excel(path1,header=2,usecols=2,index_col=0)
    df_mcp.index=pd.DatetimeIndex(df_mcp.index)
    df_mcp.index=df_mcp.index.round("H")

    #数据预处理------------------
    #缺失值处理
    print("处理前缺失值数：", df_mcp.isnull().sum().sum())
    df_mcp.fillna(axis=0,inplace=True,method="ffill")
    print("处理后缺失值数：",df_mcp.isnull().sum().sum())

    #分解为训练集、测试集  交叉验证？
    train_ydata=df_mcp.loc["2019-07-01 00:00":"2019-08-31 00:00","SYS"].astype(np.float)
    test_ydata=df_mcp.loc["2019-08-31 00:00":"2019-09-07 00:00","SYS"].astype(np.float)
    #print(train_ydata.tail())
    #ARIMA   参数确定

    #差分


    train_ydata.plot(figsize=(12,6),label="real")
    diff1=train_ydata.diff(1)
    diff2=train_ydata.diff(2)
    diff2.plot(figsize=(12,6),label="diff-2")
    diff1.plot(figsize=(12,6),label="diff-1")
    plt.title("diff data")
    plt.legend()
    plt.show()

    diff1.dropna(inplace=True)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(diff1.values.squeeze(),lags=40, ax=ax1)#自相关图  决定滑动阶数
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(diff1.values.squeeze(), lags=40,ax=ax2)#偏相关图   决定自回归阶数   squeeze()变为单维度 行列//
    plt.show()


    p=6#自回归阶数
    q=0#滑动阶数
    d=1#差分阶数
    model_arima=ARIMA(train_ydata,order=(p,d,q))
    arima=model_arima.fit(disp=-1)#disp>0 输出过程
    print(arima.summary())
    #模型检验
    arima.fittedvalues.plot( label="arima")
    diff1.plot(label="train")
    plt.legend()
    plt.show()


    # resid=arima.resid#残差偏自相关图
    # fig2 = plt.figure(figsize=(12,6))
    # ax3=fig2.add_subplot(211)
    # fig2 =plot_pacf(resid.values.squeeze(),lag=40)
    # ax4=fig2.add_subplot(212)
    # fig2 =plot_acf(resid.values.squeeze(),lag=40)
    # plt.title("residual_acf & _pacf")
    # plt.show()


    #模型预测
    y_arima_prediction_cumsum = arima.predict("2019-07-01 01:00 ", "2019-09-07 00:00 ", dynamic=False).cumsum()

    diff1.cumsum().plot(label="diff1cumsum")
    arima.fittedvalues.cumsum().plot(label="fitt cumsum")
    plt.legend()
    plt.show()
    y_arima_prediction= y_arima_prediction_cumsum+pd.Series(train_ydata[0],index=pd.DatetimeIndex(df_mcp.index))
    plt.plot(y_arima_prediction,"r-",label="forecast")
    plt.plot(df_mcp["SYS"],"b-",label="real")
    plt.title("ARIMA")
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()







