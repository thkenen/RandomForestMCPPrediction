import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import  matplotlib.pyplot as plt
from  sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import time
if __name__=="__main__":

    date_parser=lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
    #读取数据
    path1=".//mcp.xlsx"
    path2=".//volumes.xlsx"
    df_mcp=pd.read_excel(path1,header=2,usecols=2,index_col=0)
    df_volumes=pd.read_excel(path2, header=2,index_col=0,usecols=2)

    # RF  随机森林回归  以负荷为参数

    # 数据预处理

    mcp_14 = df_mcp.loc["2019-01-01 00:00":"2019-08-17 00:00", "SYS"].astype(np.float64).values
    mcp_13 = df_mcp.loc["2019-01-02 00:00":"2019-08-18 00:00", "SYS"].astype(np.float64).values
    mcp_12 = df_mcp.loc["2019-01-03 00:00":"2019-08-19 00:00", "SYS"].astype(np.float64).values
    mcp_11 = df_mcp.loc["2019-01-04 00:00":"2019-08-20 00:00", "SYS"].astype(np.float64).values
    mcp_10 = df_mcp.loc["2019-01-05 00:00":"2019-08-21 00:00", "SYS"].astype(np.float64).values
    mcp_9 = df_mcp.loc["2019-01-06 00:00":"2019-08-22 00:00", "SYS"].astype(np.float64).values
    mcp_8 = df_mcp.loc["2019-01-07 00:00":"2019-08-23 00:00", "SYS"].astype(np.float64).values
    mcp_7= df_mcp.loc["2019-01-08 00:00":"2019-08-24 00:00","SYS"].astype(np.float64).values
    mcp_6= df_mcp.loc["2019-01-09 00:00":"2019-08-25 00:00", "SYS"].astype(np.float64).values
    mcp_5= df_mcp.loc["2019-01-10 00:00":"2019-08-26 00:00", "SYS"].astype(np.float64).values
    mcp_4= df_mcp.loc["2019-01-11 00:00":"2019-08-27 00:00", "SYS"].astype(np.float64).values
    mcp_3= df_mcp.loc["2019-01-12 00:00":"2019-08-28 00:00", "SYS"].astype(np.float64).values
    mcp_2 = df_mcp.loc["2019-01-13 00:00":"2019-08-29 00:00", "SYS"].astype(np.float64).values
    mcp_1 = df_mcp.loc["2019-01-14 00:00":"2019-08-30 00:00", "SYS"].astype(np.float64).values
    mcp_0=df_mcp.loc["2019-01-15 00:00":"2019-08-31 00:00", "SYS"].astype(np.float64).values
    volumes=df_volumes.loc["2019-01-15 00:00":"2019-08-31 00:00", "Turnover at system price"].astype(np.float64).values

    mcp_15_test = df_mcp.loc["2019-08-16 00:00":"2019-08-23 00:00", "SYS"].astype(np.float64).values
    mcp_14_test = df_mcp.loc["2019-08-17 00:00":"2019-08-24 00:00", "SYS"].astype(np.float64).values
    mcp_13_test = df_mcp.loc["2019-08-18 00:00":"2019-08-25 00:00", "SYS"].astype(np.float64).values
    mcp_12_test = df_mcp.loc["2019-08-19 00:00":"2019-08-26 00:00", "SYS"].astype(np.float64).values
    mcp_11_test = df_mcp.loc["2019-08-20 00:00":"2019-08-27 00:00", "SYS"].astype(np.float64).values
    mcp_10_test = df_mcp.loc["2019-08-21 00:00":"2019-08-28 00:00", "SYS"].astype(np.float64).values
    mcp_9_test = df_mcp.loc["2019-08-22 00:00":"2019-08-29 00:00", "SYS"].astype(np.float64).values
    mcp_8_test = df_mcp.loc["2019-08-23 00:00":"2019-08-30 00:00", "SYS"].astype(np.float64).values
    mcp_7_test= df_mcp.loc["2019-08-24 00:00":"2019-08-31 00:00","SYS"].astype(np.float64).values
    mcp_6_test= df_mcp.loc["2019-08-25 00:00":"2019-09-01 00:00", "SYS"].astype(np.float64).values
    mcp_5_test= df_mcp.loc["2019-08-26 00:00":"2019-09-02 00:00", "SYS"].astype(np.float64).values
    mcp_4_test= df_mcp.loc["2019-08-27 00:00":"2019-09-03 00:00", "SYS"].astype(np.float64).values
    mcp_3_test= df_mcp.loc["2019-08-28 00:00":"2019-09-04 00:00", "SYS"].astype(np.float64).values
    mcp_2_test = df_mcp.loc["2019-08-29 00:00":"2019-09-05 00:00", "SYS"].astype(np.float64).values
    mcp_1_test = df_mcp.loc["2019-08-30 00:00":"2019-09-06 00:00", "SYS"].astype(np.float64).values
    mcp_0_test=df_mcp.loc["2019-08-31 00:00":"2019-09-07 00:00", "SYS"].astype(np.float64).values
    volumes_test=df_volumes.loc["2019-08-31 00:00":"2019-09-07 00:00", "Turnover at system price"].astype(np.float64).values


    data=pd.DataFrame({
        # "mcp_14": pd.Series(mcp_14),
        # "mcp_13": pd.Series(mcp_13),
        # "mcp_12": pd.Series(mcp_12),
        # "mcp_11": pd.Series(mcp_11),
        # "mcp_10": pd.Series(mcp_10),
        # "mcp_9": pd.Series(mcp_9),
        # "mcp_8": pd.Series(mcp_8),
        "mcp_7":pd.Series(mcp_7),
        # "mcp_6": pd.Series(mcp_6),
        # "mcp_5": pd.Series(mcp_5),
        # "mcp_4": pd.Series(mcp_4),
        "mcp_3": pd.Series(mcp_3),
        "mcp_2": pd.Series(mcp_2),
        "mcp_1": pd.Series(mcp_1),
        "volumes":pd.Series(volumes),
        "mcp_0": pd.Series(mcp_0),
    })

    data2=pd.DataFrame({
        # "mcp_14_test": pd.Series(mcp_14_test),
        # "mcp_13_test": pd.Series(mcp_13_test),
        # "mcp_12_test": pd.Series(mcp_12_test),
        # "mcp_11_test": pd.Series(mcp_11_test),
        # "mcp_10_test": pd.Series(mcp_10_test),
        # "mcp_9_test": pd.Series(mcp_9_test),
        # "mcp_8_test": pd.Series(mcp_8_test),
        "mcp_7_test":pd.Series(mcp_7_test),
        # "mcp_6_test": pd.Series(mcp_6_test),
        # "mcp_5_test": pd.Series(mcp_5_test),
        # "mcp_4_test": pd.Series(mcp_4_test),
        "mcp_3_test": pd.Series(mcp_3_test),
        "mcp_2_test": pd.Series(mcp_2_test),
        "mcp_1_test": pd.Series(mcp_1_test),
        "volumes_test":pd.Series(volumes_test),
        "mcp_0_test": pd.Series(mcp_0_test),
    })
    print("缺失值个数",data.isnull().sum().sum())#获取数据中null值个数
    normolize_data=(data-np.mean(data))/np.std(data)
    normolize_data2=(data2-np.mean(data))/np.std(data2)


    #input()

    data.fillna(axis=0, inplace=True,method="ffill" )#去除含nall的行

    print("填充后缺失值个数",data.isnull().sum().sum())  # 获取数据中null值个数


    train_x=data.iloc[:,:-1].copy()
    train_y=data["mcp_0"].copy()

    test_x=data2.iloc[:,:-1].copy()
    test_y=data2["mcp_0_test"].copy()
    train_x_scale = preprocessing.scale(train_x)
    test_x_scale = preprocessing.scale(test_x)
    #
    scorelist=[None]*50
    mselist=[None]*50
    for i in range(50):
        mlp=MLPRegressor(solver="adam",learning_rate="adaptive", activation="identity",learning_rate_init=0.0001,max_iter=20000,alpha=0.0001,hidden_layer_sizes=(12,))
    # param_Grid={
    #     'alpha':[1e-2,1e-3,1e-4,1e-5],
    #     "hidden_layer_sizes":[(3,),(6,),(9,),(12,),(15,)],
    # }
    # mlpCV=GridSearchCV(mlp,param_grid=param_Grid)
    # mlpCV.fit(train_x_scale,train_y)
    # print("best param:",mlpCV.best_params_)
    # print("bset score:", mlpCV.best_score_)
        start=time.time()
        mlp.fit(train_x,train_y)
        end=time.time()
        print("fit time:",(end-start))
        print("train score: ", mlp.score(train_x, train_y))
        print("test score:",mlp.score(test_x, test_y))
        scorelist[i]=mlp.score(test_x, test_y)
        mselist[i]=mean_squared_error(test_y,mlp.predict(test_x))
        # pd.DataFrame({"real": test_y, "predict": mlp.predict(test_x)}).to_excel(".//mlppredict.xlsx")
        #
        # plt.plot(mlp.predict(test_x), label="mlp predict")
        # test_y.plot(label="real")
        # plt.legend()
        # plt.show()
    pd.DataFrame({"score": pd.Series(scorelist),"mse":pd.Series(mselist)}).to_excel(".//mlpscorexxx.xls")


