import numpy as np 
import tensorflow as tf 
import pandas as  pd
import os
from sklearn.cross_validation import train_test_split 

def generate_batch(I, J, V_IJ, batch_size):
    n_examples = I.shape[0]
    for batch_i in range(n_examples // batch_size):

        start = batch_i*batch_size

        end = start + batch_size

        batch_I = I[start:end]

        batch_J = J[start:end]

        batch_V_IJ = V_IJ[start:end]

        yield batch_I, batch_J, batch_V_IJ # 生成每一个batch

def load_data(data_dir):
    data_cols = ['i', 'j', 'V_ij']
    data = np.array([np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols])
    data = data.T
    df = pd.DataFrame(data,columns=data_cols)
    data_cols = ['i', 'j']
    target_cols = ['V_ij']
    data_df = df[data_cols]
    target_df = df[target_cols]
    train_data, test_data, train_target, test_target = train_test_split(data_df,  
                                                   target_df,  
                                                   test_size = 0.2,  
                                                   random_state = 0) 
    num_usr = df['i'].max()
    num_pro = df['j'].max() 


    print(data_df.head())
    print(target_df.head())
    print(df['i'].max())
    print(df['j'].max())
    print(df[df['i'] == 1])
    print(df.shape)

    return train_data, test_data, train_target, test_target, num_usr, num_pro


def get_nnmf(num_usr, num_pro, rank, i, j, V_ij):
    usr_matrix = tf.Variable(tf.truncated_normal([num_usr, rank]))
    pro_matrix = tf.Variable(tf.truncated_normal([num_pro, rank]))
    usr_bias = tf.Variable(tf.truncated_normal([num_usr]))
    pro_bias = tf.Variable(tf.truncated_normal([num_pro]))
    global_mean = tf.Variable(0.0)
    usr_i = tf.gather(usr_matrix, i)
    pro_j = tf.gather(pro_matrix, j)
    usr_bias = tf.gather(usr_bias, i)
    pro_bias = tf.gather(pro_bias, j)
    interaction = tf.reduce_sum(tf.multiply(usr_i , pro_j), 1)
    preds = global_mean + usr_bias + pro_bias + interaction
    return preds

def main(path):
    train_data, test_data, train_target, test_target, num_usr, num_pro = load_data(path)
    train_I = np.array(train_data['i'])
    train_J = np.array(train_data['j'])
    train_VIJ = np.array(train_target)
    test_I = np.array(test_data['i'])
    test_J = np.array(test_data['j'])
    test_VIJ = np.array(test_target)  
    test_I = np.reshape(test_I, (-1,1))
    test_J = np.reshape(test_J, (-1,1))
    test_VIJ = np.reshape(test_VIJ, (-1,1))    
    i = tf.placeholder(dtype=tf.int32, shape=[None,1])
    j = tf.placeholder(dtype=tf.int32, shape=[None,1])
    V_ij = tf.placeholder(dtype=tf.float32, shape=[None,1])
    rank = 25
    batch_size = 4096
    output = get_nnmf(num_usr+1, num_pro+1, rank, i, j, V_ij)
    rmse = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, V_ij)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(rmse)
    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            for batch_I, batch_J, batch_VIJ in generate_batch(train_I, train_J, train_VIJ, batch_size): 
                batch_I = np.reshape(batch_I, (-1,1))
                batch_J = np.reshape(batch_J, (-1,1))
                batch_VIJ = np.reshape(batch_VIJ, (-1,1))
                _, rmse_eval = sess.run([optimizer, rmse],feed_dict={i:batch_I, j:batch_J, V_ij:batch_VIJ})
                print ("step: %s, rmse: %s" %(step, rmse_eval))
            # 每100 step计算一次准确率
            
            if step % 100 == 0:
                rmse_eval = sess.run(rmse, feed_dict={i:test_I, j:test_J, V_ij:test_VIJ})
                print ("testing!!!!!!!!! step: %s, rmse: %s" %(step, rmse_eval))
                # 如果准确率大于50%,保存模型,完成训练
                if rmse_eval < 100:
                    saver.save(sess, "./model/nnmf.model")
                    break
            
            step += 1


def test():
    load_data("./data")

if __name__ == "__main__":
    main("./data")












