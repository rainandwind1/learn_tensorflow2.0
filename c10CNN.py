import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets
# tf.enable_eager_execution()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu,True)
#         logical_gpus = tf.config.experimental.list_logival_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# model = Sequential([
#     layers.Dense(256, activation = 'relu'),
#     layers.Dense(256, activation = 'relu'),
#     layers.Dense(256, activation = 'relu'),
#     layers.Dense(10)
# ])

# model.build(input_shape=(4,784))
# model.summary()

# x = tf.random.normal([2,5,5,3])
# w = tf.random.normal([3,3,3,4])
# output = tf.nn.conv2d(x,w,strides=[1,3,3,1],padding='SAME')
# print(output.shape)

# x = tf.random.normal([2,5,5,3])
# layer = layers.Conv2D(4,kernel_size = (3,4),strides = (1,1),padding = 'SAME')#layers.conv2d(4,kernal_size = (3,4,3),strides = (1,1),padding = 'SAME')
# o = layer(x)
# print("kernel:",layer.kernel.shape)


network = Sequential([
    layers.Conv2D(6,kernel_size = 3,strides = 1),
    layers.MaxPooling2D(pool_size = 2,strides = 2),
    layers.ReLU(),
    layers.Conv2D(16,kernel_size = 3,strides = 1),
    layers.MaxPooling2D(pool_size = 2,strides = 2),
    layers.ReLU(),
    layers.Flatten(),
    
    layers.Dense(120,activation = 'relu'),
    layers.Dense(84,activation = 'relu'),
    layers.Dense(10)

])
network.build(input_shape = (4,28,28,1))
network.summary()

from tensorflow.keras import losses,optimizers
criteon = losses.CategoricalCrossentropy(from_logits=True)

# 训练阶段
for i in range(epoch):
    with tf.GradientTape() as tape:
        x = tf.expand_dims(x,axis = 3)
        out = network(x)
        y_onehot = tf.one_hot(y,depth=10)
        loss = criteon(y_onehot,out)

    grads = tape.gradient(loss,network.trainable_variables)
    optimizers.apply_gradients(zip(grads,network.trainable_variables))

# 测试阶段
correct, total = 0,0
for x,y in db_test:
    x = tf.expand_dims(x,axis=3)
    out = network(x)
    pred = tf.arg_max(out,axis = -1)
    y = tf.cast(y,tf.int64)
    correct += float(tf.reduce_sum(tf.cast(tf.equal(pred,y),tf.float32))) 
    total += x.shape[0]

    print('test acc',correct/total)









