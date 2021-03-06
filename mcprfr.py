#coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV   #网格搜索调参

from sklearn.model_selection import train_test_split



if __name__=="__main__":

	date_parser=lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
	#读取数据
	path1=".//mcp.xlsx"
	path2=".//volumes.xlsx"
	df_mcp=pd.read_excel(path1,header=2,usecols=2,index_col=0)
	df_volumes=pd.read_excel(path2, header=2,index_col=0,usecols=2)
	df_mcp.fillna(method="ffill",inplace=True)
	df_volumes.fillna(method="ffill", inplace=True)

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
	volumes_14 = df_volumes.loc["2019-01-01 00:00":"2019-08-17 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_13 = df_volumes.loc["2019-01-02 00:00":"2019-08-18 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_12 = df_volumes.loc["2019-01-03 00:00":"2019-08-19 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_11 = df_volumes.loc["2019-01-04 00:00":"2019-08-20 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_10 = df_volumes.loc["2019-01-05 00:00":"2019-08-21 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_9 = df_volumes.loc["2019-01-06 00:00":"2019-08-22 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_8 = df_volumes.loc["2019-01-07 00:00":"2019-08-23 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_7 = df_volumes.loc["2019-01-08 00:00":"2019-08-24 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_6 = df_volumes.loc["2019-01-09 00:00":"2019-08-25 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_5 = df_volumes.loc["2019-01-10 00:00":"2019-08-26 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_4 = df_volumes.loc["2019-01-11 00:00":"2019-08-27 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_3 = df_volumes.loc["2019-01-12 00:00":"2019-08-28 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_2 = df_volumes.loc["2019-01-13 00:00":"2019-08-29 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_1 = df_volumes.loc["2019-01-14 00:00":"2019-08-30 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_0=df_volumes.loc["2019-01-15 00:00":"2019-08-31 00:00", "Turnover at system price"].astype(np.float64).values
	train_time=df_mcp.loc["2019-01-15 00:00":"2019-08-31 00:00", "SYS"].index.values


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
	volumes_14_test = df_volumes.loc["2019-08-17 00:00":"2019-08-24 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_13_test = df_volumes.loc["2019-08-18 00:00":"2019-08-25 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_12_test = df_volumes.loc["2019-08-19 00:00":"2019-08-26 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_11_test = df_volumes.loc["2019-08-20 00:00":"2019-08-27 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_10_test = df_volumes.loc["2019-08-21 00:00":"2019-08-28 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_9_test = df_volumes.loc["2019-08-22 00:00":"2019-08-29 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_8_test = df_volumes.loc["2019-08-23 00:00":"2019-08-30 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_7_test = df_volumes.loc["2019-08-24 00:00":"2019-08-31 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_6_test = df_volumes.loc["2019-08-25 00:00":"2019-09-01 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_5_test = df_volumes.loc["2019-08-26 00:00":"2019-09-02 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_4_test = df_volumes.loc["2019-08-27 00:00":"2019-09-03 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_3_test = df_volumes.loc["2019-08-28 00:00":"2019-09-04 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_2_test = df_volumes.loc["2019-08-29 00:00":"2019-09-05 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_1_test = df_volumes.loc["2019-08-30 00:00":"2019-09-06 00:00", "Turnover at system price"].astype(np.float64).values
	volumes_0_test=df_volumes.loc["2019-08-31 00:00":"2019-09-07 00:00", "Turnover at system price"].astype(np.float64).values
	test_time=df_mcp.loc["2019-08-31 00:00":"2019-09-07 00:00", "SYS"].index.values

	data=pd.DataFrame({
		"mcp_14": pd.Series(mcp_14),
		"mcp_13": pd.Series(mcp_13),
		"mcp_12": pd.Series(mcp_12),
		"mcp_11": pd.Series(mcp_11),
		"mcp_10": pd.Series(mcp_10),
		"mcp_9": pd.Series(mcp_9),
		"mcp_8": pd.Series(mcp_8),
		"mcp_7":pd.Series(mcp_7),
		"mcp_6": pd.Series(mcp_6),
		"mcp_5": pd.Series(mcp_5),
		"mcp_4": pd.Series(mcp_4),
		"mcp_3": pd.Series(mcp_3),
		"mcp_2": pd.Series(mcp_2),
		"mcp_1": pd.Series(mcp_1),
		"volumes_14":pd.Series(volumes_14),
		"volumes_13":pd.Series(volumes_13),
		"volumes_12":pd.Series(volumes_12),
		"volumes_11":pd.Series(volumes_11),
		"volumes_10":pd.Series(volumes_10),
		"volumes_9":pd.Series(volumes_9),
		"volumes_8":pd.Series(volumes_8),
		"volumes_7":pd.Series(volumes_7),
		"volumes_6":pd.Series(volumes_6),
		"volumes_5":pd.Series(volumes_5),
		"volumes_4":pd.Series(volumes_4),
		"volumes_3":pd.Series(volumes_3),
		"volumes_2":pd.Series(volumes_2),
		"volumes_1":pd.Series(volumes_1),
		"volumes_0":pd.Series(volumes_0),
		"mcp_0": pd.Series(mcp_0),
	})

	data2=pd.DataFrame({
		"mcp_14_test": pd.Series(mcp_14_test),
		"mcp_13_test": pd.Series(mcp_13_test),
		"mcp_12_test": pd.Series(mcp_12_test),
		"mcp_11_test": pd.Series(mcp_11_test),
		"mcp_10_test": pd.Series(mcp_10_test),
		"mcp_9_test": pd.Series(mcp_9_test),
		"mcp_8_test": pd.Series(mcp_8_test),
		"mcp_7_test":pd.Series(mcp_7_test),
		"mcp_6_test": pd.Series(mcp_6_test),
		"mcp_5_test": pd.Series(mcp_5_test),
		"mcp_4_test": pd.Series(mcp_4_test),
		"mcp_3_test": pd.Series(mcp_3_test),
		"mcp_2_test": pd.Series(mcp_2_test),
		"mcp_1_test": pd.Series(mcp_1_test),
		"volumes_14_test":pd.Series(volumes_14_test),
		"volumes_13_test":pd.Series(volumes_13_test),
		"volumes_12_test":pd.Series(volumes_12_test),
		"volumes_11_test":pd.Series(volumes_11_test),
		"volumes_10_test":pd.Series(volumes_10_test),
		"volumes_9_test":pd.Series(volumes_9_test),
		"volumes_8_test":pd.Series(volumes_8_test),
		"volumes_7_test":pd.Series(volumes_7_test),
		"volumes_6_test":pd.Series(volumes_6_test),
		"volumes_5_test":pd.Series(volumes_5_test),
		"volumes_4_test":pd.Series(volumes_4_test),
		"volumes_3_test":pd.Series(volumes_3_test),
		"volumes_2_test":pd.Series(volumes_2_test),
		"volumes_1_test":pd.Series(volumes_1_test),
		"volumes_0_test":pd.Series(volumes_0_test),
		"mcp_0_test": pd.Series(mcp_0_test),
	})
	normolize_data=(data-np.mean(data))/np.std(data)
	normolize_data2=(data2-np.mean(data))/np.std(data2)

	print("缺失值个数",data.isnull().sum().sum())#获取数据中null值个数
	#input()
	data.fillna(axis=0, inplace=True,method="ffill" )#插补nall的行
	print("填充后缺失值个数",data.isnull().sum().sum())  # 获取数据中null值个数
	train_x=data.iloc[:,:-1].copy()
	train_y=data["mcp_0"].copy()
	test_x=data2.iloc[:,:-1].copy()
	test_y=data2["mcp_0_test"].copy()
	train_x,test_x,train_y,test_y=trax_data=train_test_split(data.loc[:,["mcp_7","mcp_6","mcp_5","mcp_4","mcp_3","mcp_2","mcp_1","volumes"]],data["mcp_0"],test_size=0.3,random_state=0)

	# print(train_x.shape,test_y.shape)
	# print(test_y.head())
	# print(data2.isnull().sum().sum())

#随机森林模型

	# model_rfr = RandomForestRegressor()
	# param_grid={
	#     "max_features":range(2,29),
	#     "n_estimators":range(10,50),
	#     "max_depth":range(3,10),
	# }
	# modelCv=GridSearchCV(model_rfr,param_grid=param_grid,n_jobs=1,cv=3)  #n_job=1  并行数

	# modelCv.fit(train_x,train_y)

	# print("cv result:" ,modelCv.cv_results_)
	# pd.DataFrame(modelCv.cv_results_).to_excel(".//test.xls")
	# print("best parameters: ", modelCv.best_params_)
	# print("best score:%f" % modelCv.best_score_)

	# input()
	
	#predict_y=modelCv.predict(test_x)

	
	scorelist=[None]*50
	mselist=[None]*50
	timelist=[None]*50
	ooblist=[None]*50
	for i in range(50):
	
		model_rfr2 = RandomForestRegressor(n_estimators=26,max_features=9,max_depth=4,oob_score=True)
		fit_start=time.time()
		model_rfr2.fit(train_x,train_y)
		fit_over=time.time()
		print("fit time: %f" %(fit_over-fit_start))
		#print(model_rfr2.estimators_)
		print("oob score %f"  %model_rfr2.oob_score_)
		#print("feature importance:",model_rfr2.feature_importances_)
		predict_y1=model_rfr2.predict(train_x)
		predict_y2 = model_rfr2.predict(test_x)




	#得分



		print("train score: %f" %model_rfr2.score(train_x,train_y))   #即R^2
		print("test score: %f" % model_rfr2.score(test_x, test_y))
		mselist[i]=mean_squared_error(test_y,predict_y2)
		scorelist[i]=model_rfr2.score(test_x, test_y)
		timelist[i]=fit_over-fit_start
		ooblist[i]=model_rfr2.oob_score_

	#绘图
		plt.plot(train_time,train_y,"r-",label="real(train)")
		plt.plot(train_time,predict_y1,"b-",label="RFR")
		plt.title("RFR train " ,fontsize=20)
		plt.xlabel("Time",fontsize=17)
		plt.ylabel("MCP/(EUR/MWh)",fontsize=17)
		plt.xticks(fontsize=16)
		plt.yticks(fontsize=16)
		plt.legend(loc='upper right',fontsize=19)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		ax = plt.gca()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
		
		plt.show()
		
		
		plt.plot(test_time,test_y,"r-",label="real(test)")
		plt.plot(test_time,predict_y2,"b-",label="RFR")
		plt.xlabel("Time", fontsize=17)
		plt.yticks(fontsize=16)
		plt.xticks(rotation=25,fontsize=16)
		plt.ylabel("MCP/(EUR/MWh)",fontsize=17)
		plt.title("RFR test",fontsize=20)
		plt.legend(loc='upper right',fontsize=19)
		plt.rcParams["figure.figsize"]=(16,15)
		plt.rcParams['xtick.direction'] = 'in'
		plt.rcParams['ytick.direction'] = 'in'
		ax = plt.gca()
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
		plt.show()
		pd.DataFrame({"time":train_time,"real": train_y, "predict": model_rfr2.predict(train_x)}).to_excel(".//rfrtrain"+str(i)+".xlsx")
		pd.DataFrame({"time":test_time ,"real": test_y, "predict": model_rfr2.predict(test_x)}).to_excel(".//rfrtest"+str(i)+".xlsx")

	pd.DataFrame({"score": pd.Series(scorelist),"mse":pd.Series(mselist),"fittime":pd.Series(timelist),"oobscore":pd.Series(ooblist)}).to_excel(".//rfrscorelist1.xls")
