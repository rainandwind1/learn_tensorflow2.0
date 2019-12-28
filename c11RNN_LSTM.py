import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,Model,optimizers,losses
tf.enable_eager_execution()


# model = Sequential(
#     layers.SimpleRNN(64),
#     layers.SimpleRNN(64),
#     layers.Dense(2,activation = 'relu')
# )

batchsz = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# print(x_train.shape,len(x_train[0]),y_train.shape)
# print(x_test.shape,len(x_test[0]),y_test.shape)
word_index = keras.datasets.imdb.get_word_index()
# for k,v in word_index.items():
#     print(k,v)
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["UNK"] = 2
word_index["UNUSED"] = 3

reverse_word_index = dict([value,key] for (key,value) in word_index.items())
def decode_review(test):
    return ' '.join([reverse_word_index.get(i,'?') for i in test])

# print(decode_review(x_train[2]))
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_review_len)

# 包裹成数据对象
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(batchsz,drop_remainder = True)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batchsz,drop_remainder=True)
# print('x_trian shape:',x_train.shape,tf.reduce_max(y_train),tf.reduce_min(y_train))
# print('x_test shape',x_test.shape)


# 网络模型
class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN,self).__init__()
        # [b,64]构建cell初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]
        self.state1 = [tf.zeros([batchsz,units]),tf.zeros([batchsz,units])]
        # 词向量编码[b,80] => [b,80,100]
        self.embedding = layers.Embedding(total_words ,embedding_len,input_length = max_review_len)
        # 网络结构
        self.rnn_cell0 = layers.LSTMNCell(units,dropout = 0.5)
        self.rnn_cell1 = layers.LSTMCell(units,dropout = 0.5)
        self.outlayer = layers.Dense(1)

    def call(self,inputs,training = True):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x,axis = 1):
            out0,state0 = self.rnn_cell0(word,state0,training)
            out1,state1 = self.rnn_cell1(out0,state1,training)
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob



# 训练与测试
def main():
    units = 64
    epochs = 20
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),loss=losses.BinaryCrossentropy(),metrics=['accuracy'])
    model.fit(db_train,epochs = epochs,validation_data=db_test)
    model.evaluate(db_test)

if __name__ == "__main__":
    main()


          