import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers, Sequential, losses, optimizers, datasets
tf.enable_eager_execution()


def preprocess(labels, images):
	#把numpy数据转为Tensor
	labels = tf.cast(labels, dtype=tf.int32)
	# labels 转为one_hot编码
	labels = tf.one_hot(labels, depth=10)
	# 顺手归一化
	images = tf.cast(images, dtype=tf.float32) / 255
	return labels, images

(x,y),(x_test,y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y,axis = 1)
y_test = tf.squeeze(y_test,axis = 1)
# print(x.shape,y.shape,x_test.shape,y_test.shape)
#构建训练集对象
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
#构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)
#从训练集中采样一个batch并观察
sample = next(iter(train_db))
# print('sample:',sample[0].shape,sample[1].shape,tf.reduce_mean(sample[0]),tf.reduce_mean(sample[0]))

conv_layers = [
    
    # Conv-Conv-Pooling 单元 1
    layers.Conv2D(64,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),

    # Conv-Conv-Pooling 单元 2
    layers.Conv2D(128,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),

    # Conv-Conv-Pooling 单元 3
    layers.Conv2D(256,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(256,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),

    # Conv-Conv-Pooling 单元 4
    layers.Conv2D(512,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),

    # Conv-Conv-Pooling 单元 5
    layers.Conv2D(512,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size = [3,3],padding = 'same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size = [2,2],strides = 2,padding = 'same'),


]


conv_net = Sequential(conv_layers)

fc_net = Sequential([
    layers.Dense(256,activation = 'relu'),
    layers.Dense(128,activation = 'relu'),
    layers.Dense(10,activation = None)
    ]
)

conv_net.build(input_shape = [4,32,32,3])
fc_net.build(input_shape = [4,512])
# conv_net.summary()
# fc_net.summary()

variables = conv_net.trainable_variables + fc_net.trainable_variables
grads = tape.gradient(loss,variables)
optimizers.apply_gradients(zip(grads,variables))






