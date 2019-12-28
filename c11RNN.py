import  tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential,losses
import  os

tf.enable_eager_execution()

x = tf.range(10)
x = tf.random.shuffle(x)
net = layers.Embedding(10,4)
o = net(x)
# print(o)

# cell = layers.SimpleRNNCell(3)
# cell.build(input_shape = (None,4))
#print(cell.trainable_variables)
h0 = [tf.zeros([4,64])]
x = tf.random.normal([4,80,100])
xt = x[:,0,:]
cell = layers.SimpleRNNCell(64)
out,h1 = cell(xt,h0)
# print(out.shape,h1[0].shape)

# print(id(out),id(h1[0]))


h = h0
for xt in tf.unstack(x,axis=1):
    out,h = cell(xt,h)
out = out 



