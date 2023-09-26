from sklearn import datasets
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris() # iris是dictionary
# print(iris.keys())
# print(iris.values())
# print(iris.items()) 回傳所有鍵值組合

x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target'])
iris_new = pd.concat([x,y], axis=1)
# iris.shape是形狀(150,5)，size是大小75)

color = ['r', 'b', 'g']
for i in range(3):
    plt.scatter(iris_new[iris_new['target']==i]['sepal length (cm)'], iris_new[iris_new['target']==i]['petal length (cm)'], c=color[i], label=iris['target_names'][i])
plt.xlabel("sepal length (cm)")
plt.ylabel("petal length (cm)")
plt.legend()
plt.show() # 要分三組畫圖，分批上色

target_names_dic = pd.Series(iris['target_names']).to_dict()
target_class = {0:1, 1:-1}
iris_new["target_names"] = iris_new['target'].map(target_names_dic)
iris_new["target_class"] = iris_new['target'].map(target_class)
iris_data = iris_new[iris_new["target_class"].notna()].iloc[:,[0,2,6]]

def ac_func(z):
    if z>0:
        return 1
    else:
        return -1

learning_rate = 0.01
w = np.array([10.,10.,10.])
error = 1
iterator = 0
while error != 0:
    error = 0
    for i in range(len(iris_data)): # 每個data去找
        x,y = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2])), np.array(iris_data.iloc[i])[2] # 其中np.concatenate()目的是把x的資料都加上x0=1，變成x0,x1,x2去更新w0,w1,w2三個權重，其中w0即ml自己要學出來的threshold值，可以想成有一個input一直都是1去乘上這個權重
        if np.sign(np.dot(w,x)) != y: # 預測失敗時要更新w的方向
            print("iterator: "+str(iterator))
            iterator += 1
            error += 1
            sns.lmplot(iris_data, x='sepal length (cm)', y='petal length (cm)', fit_reg=False, hue ='target_class') # lmplot會畫出散布圖及一條迴歸線，hue則用於分組，seabron是建構在matplotlib之上的api，可以更簡潔的畫圖，比如說這一行就可以自動畫出iris_data依據target分組的散布圖了
            
            # 前一個Decision boundary 的法向量
            if w[1] != 0:
                x_last_decision_boundary = np.linspace(0,w[1])
                y_last_decision_boundary = (w[2]/w[1])*x_last_decision_boundary
                plt.plot(x_last_decision_boundary, y_last_decision_boundary,'c--') # C表青色(cyan), --表虛線
            w += y*x            
            print("x: " + str(x))            
            print("w: " + str(w))
            # x向量 
            x_vector = np.linspace(0,x[1]) # 在一定範圍內均勻地灑點
            y_vector = (x[2]/x[1])*x_vector
            plt.plot(x_vector, y_vector,'b')
            # Decision boundary 的方向向量
            x_decision_boundary = np.linspace(-0.5,7)
            y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
            plt.plot(x_decision_boundary, y_decision_boundary,'r')
            # Decision boundary 的法向量
            x_decision_boundary_normal_vector = np.linspace(0,w[1])
            y_decision_boundary_normal_vector = (w[2]/w[1])*x_decision_boundary_normal_vector
            plt.plot(x_decision_boundary_normal_vector, y_decision_boundary_normal_vector,'g')
            plt.xlim(-0.5,7.5)
            plt.ylim(5,-3)
            plt.show()