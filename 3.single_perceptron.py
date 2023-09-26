import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target'])
iris_new = pd.concat([x,y], axis=1)
target_names_dic = pd.Series(iris['target_names']).to_dict()
target_class = {0:1, 1:-1}
iris_new["target_names"] = iris_new['target'].map(target_names_dic)
iris_new["target_class"] = iris_new['target'].map(target_class)
iris_data = iris_new[iris_new["target_class"].notna()].iloc[:,[0,2,6]]

x=iris_data.iloc[:, :2].values # 每個item就是一列，總共是list
y=iris_data.iloc[:,2].values
for i,j in zip(x,y):
    print(i,j)

class Perceptron:
    def __init__(self, eta=1):
        self.eta = eta

    def train(self, X, y):
        self.w = np.zeros(1 + X.shape[1]) # 要多加一個threshold的維度
        self.errors = [1] # 一開始設1，while才能執行
        iterator = 0
        counts = []
        while self.errors[-1] != 0: # 直到所有數值的error都是0才可以停下來
            error = 0 # 計算每一輪error的總數
            iterator += 1  # 計算跑幾次輪
            count = 0 # 計算跑幾次迴圈
            for xi, target in zip(X, y): # zip(X, y)會回傳一個iterable object，把兩個list對應的item放入一個tuple中
                update = self.eta * (target - self.predict(xi))
                count += 1
                if update != 0:
                    sns.lmplot(iris_data, x='sepal length (cm)', y='petal length (cm)', fit_reg=False, hue ='target_class')
                    if self.w[1] != 0:
                        x_lastw = np.linspace(0,self.w[1])
                        y_lastw = (self.w[2]/self.w[1]) * x_lastw # self.w[2]/self.w[1])即斜率 y=ax
                        plt.plot(x_lastw, y_lastw,'c--')
                        
                        x_last_decision_boundary = np.linspace(-2,7) # 畫出x1
                        y_last_decision_boundary = (-self.w[1]/self.w[2])*x_decision_boundary - (self.w[0]/self.w[2]) # 畫出x2，依公式w0*x0+w1*x1+w2*x2=0，故x2=(-w0*x0-w1*x1)/w2
                        plt.plot(x_last_decision_boundary, y_last_decision_boundary,'r--')
                        
                    self.w[1:] +=  update * xi
                    self.w[0] +=  update # 因為x0永遠是1，所以要w0要和w1,w2分開算
                    print("x: " + str(xi))            
                    print("w: " + str(self.w))
            
                    # 畫x向量
                    x_vector = np.linspace(0,xi[0]) # 在一定範圍內均勻地灑點
                    y_vector = (xi[1]/xi[0])*x_vector
                    plt.plot(x_vector, y_vector,'b')
            
                    # 畫邊界向量
                    x_decision_boundary = np.linspace(-2,8) # 畫出x1
                    y_decision_boundary = (-self.w[1]/self.w[2])*x_decision_boundary - (self.w[0]/self.w[2]) # 畫出x2，依公式w0*x0+w1*x1+w2*x2=0，故x2=(-w0*x0-w1*x1)/w2
                    plt.plot(x_decision_boundary, y_decision_boundary,'r')
                    
                    # 畫更新後的法向量
                    x_w = np.linspace(0,self.w[1])
                    y_w = (self.w[2]/self.w[1]) * x_w # self.w[2]/self.w[1])即斜率 y=ax
                    plt.plot(x_w, y_w,'g')
                    ax=plt.gca()
                    ax.set_aspect(1) # 讓xy軸固定長度
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    plt.xlim(-2,8)
                    plt.ylim(-2,6)
                    plt.show()
                    error += 1
            
            counts.append(count)
            self.errors.append(error)
        return self.errors, iterator, counts

    def z(self, X):
        return np.dot(X, self.w[1:]) + self.w[0] # z即weighted sum

    def predict(self, X):
        return np.where(self.z(X) > 0.0, 1, -1) # np.where(condition, x, y) 對condition中的ndarray做替換，符合條件者為x，不符條件者為y

pp = Perceptron(eta=1)
print(pp.train(x,y)) # 總共錯10次，在第六個

print('Weights:', pp.w)
plot_decision_regions(x, y.astype("int"), clf=pp, legend=2, colors='yellow,red')
plt.title('Perceptron')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.show()

plt.plot(range(1, len(pp.errors)), pp.errors[1:], marker='o')
plt.xlabel('Iterations')
plt.ylabel('errors')
plt.show()

# 參考資料:
# https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
# https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-2%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E6%84%9F%E7%9F%A5%E5%99%A8-perceptron-%E4%BB%8B%E7%B4%B9-84d8b809f866
# https://www.youtube.com/watch?v=1xnUlrgJJGo&list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf&index=12&ab_channel=Hsuan-TienLin
