import numpy as np 
import tensorflow as tf 



def generate_batch(X,Y,n_examples, batch_size):

    for batch_i in range(n_examples // batch_size):

        start = batch_i*batch_size

        end = start + batch_size

        batch_xs = X[start:end]

        batch_ys = Y[start:end]

        yield batch_xs, batch_ys # 生成每一个batch



class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['i', 'j', 'V_ij']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)

        self.num_users = df['i'].max() + 1
        self.num_products = df['j'].max() + 1

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
return df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)





def get_nnmf(num_usr, num_pro, rank=25):
    i = tf.placeholder(dtype=tf.int32, shape=[None])
    j = tf.placeholder(dtype=tf.int32, shape=[None])
    usr_matrix = tf.Variable(tf.truncated_normal(num_usr, rank))
    pro_matrix = tf.Variable(tf.truncated_normal(num_pro, rank))
    usr_bias = tf.Variable(tf.truncated_normal([num_usr]))
    pro_bias = tf.Variable(tf.truncated_normal([num_pro]))
    global_mean = tf.Variable(0.0)
    usr_i = tf.gather(usr_matrix, i)
    pro_j = tf.gather(pro_matrix, j)
    usr_bias = tf.gather(usr_bias, i)
    pro_bias = tf.gather(pro_bias, j)
    interaction = tf.reduce_sum(tf.multiply(usr_i * pro_j), 1)
    preds = global_mean + usr_bias + pro_bias + interaction
    return preds

def train_nnmf(num_usr, num_pro, rank=25):
    V_ij = tf.placeholder(dtype=tf.float32, shape=[None])
    output = get_nnmf(num_usr, num_pro, rank)
    rmse = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, V_ij)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:

            for batch_xs,batch_ys in generatebatch(X,Y,Y.shape[0],batch_size): # 每个周期进行MBGD算法

                sess.run(train_step,feed_dict={tf_X:batch_xs,tf_Y:batch_ys})


            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.5:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break
            step += 1




