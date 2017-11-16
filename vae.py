import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data 


def xavier_init(size):
    return tf.truncated_normal(shape=size)
class VariationalAutoEncoder:
    """ A deep variational autoencoder"""
    def __init__(self, input_dim, epoch=50, loss='cross-entropy',
                lr=0.005, batch_size=100, print_step=50):
        """ fixed structure """
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr 
        self.loss = loss
        #self.activations = activations
        self.epoch = epoch
        #self.dims = dims
        #self.depth = len(dims)
        #self.n_z = z_dim
        self.input_dim = input_dim
    
    def process(self, X_in):
        rec = {
            "w1" : tf.Variable(xavier_init([784, 300])),
            "b1" : tf.Variable(tf.zeros(shape=[300])),
            "w_z_mean" : tf.Variable(xavier_init([300,50])),
            "b_z_mean" : tf.Variable(tf.zeros(shape=[50])),
            "w_z_log_sigma" : tf.Variable(xavier_init([300,50])),
            "b_z_log_sigma" : tf.Variable(tf.zeros(shape=[50]))
        }
        h1 = tf.nn.sigmoid(tf.matmul(X_in, rec["w1"]) + rec['b1'])
        z_mean = tf.matmul(h1, rec["w_z_mean"]) + rec['b_z_mean']
        z_log_sigma_sq = tf.matmul(h1, rec['w_z_log_sigma']) + rec['b_z_log_sigma']
        eps = tf.random_normal((self.batch_size, 50), 0 , 1, dtype=tf.float32)
        z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps
        gen = {
            "w2" : tf.Variable(xavier_init([50,300])),
            "b2" : tf.Variable(tf.zeros(shape=[300])),
            "w_x" : tf.Variable(xavier_init([300, 784])),
            "b_x" : tf.Variable(xavier_init([784]))
        }
        h1 = tf.nn.sigmoid(tf.matmul(z, gen["w2"]) + gen["b2"])
        x_recon = tf.nn.sigmoid(tf.matmul(h1, gen["w_x"]) + gen["b_x"])
        latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_sigma_sq)
            - z_log_sigma_sq - 1, 1))
        gen_loss = -tf.reduce_mean(tf.reduce_sum(X_in * tf.log(tf.maximum(x_recon, 1e-10)) 
            + (1-X_in) * tf.log(tf.maximum(1 - x_recon, 1e-10)),1))
        #loss = latent_loss + gen_loss
        return latent_loss, gen_loss, x_recon, z, z_mean, z_log_sigma_sq

    def run(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None,self.input_dim])
        #loss = self.process(x)
        latent_loss , gen_loss, x_recon, z, z_mean, z_log_sigma_sq = self.process(x)
        loss = latent_loss + gen_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        saver = tf.train.Saver()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        n_samples = mnist.train.num_examples
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch):
                total_batch = int (n_samples / self.batch_size)
                for i in range(total_batch):
                    batch_x, _ = mnist.train.next_batch(self.batch_size)
                    _, d = sess.run([optimizer, loss], feed_dict = {x:batch_x})
                    print("epoch: %s , batch: %s , loss: %s" %(epoch, i, d))
            saver.save(sess, "model/model.ckpt")
    def check(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None,self.input_dim])
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        latent_loss , gen_loss, x_recon, z, z_mean, z_log_sigma_sq = self.process(x)
        saver = tf.train.Saver()
        check_point_file = "model/model.ckpt"
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)
            print("Model restored.")
            x_sample, _ = mnist.test.next_batch(self.batch_size)

            x_eval,z_vals,z_mean_val,z_log_sigma_sq_val = sess.run([x_recon, z, z_mean, z_log_sigma_sq], feed_dict={x: x_sample})
            print len(x_eval)
            plt.figure(figsize=(8, 12))
            for i in range(5):
                plt.subplot(5, 3, 3*i + 1)
                plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1,  interpolation='none',cmap=plt.get_cmap('gray'))
                plt.title("Test input")
                #plt.show()
                
                #plt.colorbar()
                plt.subplot(5, 3, 3*i + 2)
                plt.scatter(z_vals[:,0],z_vals[:,1], c='gray', alpha=0.5)
                plt.scatter(z_mean_val[i,0],z_mean_val[i,1], c='green', s=64, alpha=0.5)
                plt.scatter(z_vals[i,0],z_vals[i,1], c='blue', s=16, alpha=0.5)
            
                plt.xlim((-3,3))
                plt.ylim((-3,3))
                plt.title("Latent Space")
                
                plt.subplot(5, 3, 3*i + 3)
                plt.imshow(x_eval[i].reshape(28, 28), vmin=0, vmax=1, interpolation='none',cmap=plt.get_cmap('gray'))
                plt.title("Reconstruction")
            plt.show()

            #plt.colorbar()
            #plt.tight_layout()



def train():
    vae = VariationalAutoEncoder(784)
    vae.run()

def test():
    vae = VariationalAutoEncoder(784)
    vae.check()



if __name__ == "__main__":
    #train()
    test()






        




            

