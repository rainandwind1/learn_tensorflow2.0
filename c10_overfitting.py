from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
N_SAMPLES = 1000
TEST_SIZE = 0.2
x,y = make_moons(n_samples=N_SAMPLES,noise = 0.25,random_state= 100)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = TEST_SIZE,random_state = 42)

def make_plot(x,y,plot_name,XX = None,YY=None,preds = None):
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([x_min,x_max])
    axes.set_ylim([y_min,y_max])
    axes.set(xlabel="$x_1$",ylabel = "$x_2$")
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 0.08,cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys",vmin=0, vmax=.6)
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    plt.mscatter(x[:, 0], x[:, 1], c=y.ravel(), s=20,cmap=plt.cm.Spectral, edgecolors='none', m=markers))
make_moons(x,y,None)


for n in range(5): # 构建5 种不同层数的网络
    model = Sequential()# 创建容器
    # 创建第一层
    model.add(Dense(8, input_dim=2,activation='relu'))
    for _ in range(n): # 添加n 层，共n+2 层
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # 创建最末层
    model.compile(loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy']) # 模型装配与训练
    history = model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
    # 绘制不同层数的网络决策边界曲线
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "网络层数({})".format(n)
    file = "网络容量%f.png"%(2+n*1)
    make_plot(X_train, y_train, title, file, XX, YY, preds)








