import sys, os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *

# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
_NUM_IMAGES = {
    'train': 100000,
    'validation': 10000,
    'test': 10000
}
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 100 #10000 #TODO
start_step = 11800
#root = '/home/fnaser/DropboxMIT/Miniplaces/alexnet-10000-default-TensorBoard/'
root = './'
path_save = root + 'alexnet/alexnet'
start_from = path_save + '-' + str(start_step)
logs_path = path_save + '/logs/'
start_eval = True
batch_evaluation = 25

print(path_save)
print(start_from)
#sys.exit()

def alexnet(x, keep_dropout):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros(96)),
        'bc2': tf.Variable(tf.zeros(256)),
        'bc3': tf.Variable(tf.zeros(384)),
        'bc4': tf.Variable(tf.zeros(256)),
        'bc5': tf.Variable(tf.zeros(256)),

        'bf6': tf.Variable(tf.zeros(4096)),
        'bf7': tf.Variable(tf.zeros(4096)),
        'bo': tf.Variable(tf.zeros(100))
    }

    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.add(tf.matmul(fc6, weights['wf6']), biases['bf6'])
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.add(tf.matmul(fc6, weights['wf7']), biases['bf7'])
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }
opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }
loader_test = DataLoaderDisk(**opt_data_test)
loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits = alexnet(x, keep_dropout)

# Define loss and optimizer
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
with tf.name_scope('SGD'):
    train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
_, top5 = tf.nn.top_k(logits,k=5)
with tf.name_scope('Accuracy1'):
    accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
with tf.name_scope('Accuracy5'):
    accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# TensorBoard
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy1", accuracy1)
tf.summary.scalar("accuracy5", accuracy5)
merged_summary_op = tf.summary.merge_all()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

val_txt = open(start_from + "-test.txt", "w")

def inference(sess, batch_size, loader):
    images_batch, labels_batch, _ = loader.next_batch(batch_size)
    
    l, acc1, acc5, summary = sess.run([loss, accuracy1, accuracy5, merged_summary_op],
                             feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})

    print("Loss= " + \
          "{:.6f}".format(l) + ", Accuracy Top1 = " + \
          "{:.4f}".format(acc1) + ", Top5 = " + \
          "{:.4f}".format(acc5))

    return l, acc1, acc5

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        #TODO either restore to train or eval
        print('Restore session.')
        saver.restore(sess, start_from)
        
        # Evaluate on the whole validation set
        print('*****************************************')
        print('\nEvaluation on the whole validation set on the last model:')
        loader_val.reset()
        iterations = _NUM_IMAGES['validation']//batch_evaluation
        l_total, acc1_total, acc5_total = 0.0, 0.0, 0.0
        for i in range(iterations):
            l, acc1, acc5 = inference(sess,batch_evaluation, loader_val)
            l_total += l
            acc1_total += acc1
            acc5_total += acc5
        acc1_total /= iterations
        acc5_total /= iterations
        print('*****************************************')
        print('RESULT FOR VALIDATION SET')
        print("Loss= " +
              "{:.6f}".format(l_total) + ", Accuracy Top1 = " +
              "{:.4f}".format(acc1_total) + ", Top5 = " +
              "{:.4f}".format(acc5_total))

        # Evaluate on the whole training set
        print('*****************************************')
        print('\nEvaluation on the whole validation set on the last model:')
        loader_val.reset()
        iterations = _NUM_IMAGES['train']//batch_evaluation
        l_total, acc1_total, acc5_total = 0.0, 0.0, 0.0
        for i in range(iterations):
            l, acc1, acc5 = inference(sess,batch_evaluation, loader_train)
            l_total += l
            acc1_total += acc1
            acc5_total += acc5
        acc1_total /= iterations
        acc5_total /= iterations
        print('*****************************************')
        print('RESULT FOR TRAINING SET')
        print("Loss= " +
              "{:.6f}".format(l_total) + ", Accuracy Top1 = " +
              "{:.4f}".format(acc1_total) + ", Top5 = " +
              "{:.4f}".format(acc5_total))
        print('*****************************************')

        if start_eval:
            print('Start Eval.')
            batch_size = 1
            num_batch = loader_test.size() // batch_size
            loader_test.reset()
        
            for i in range(0,num_batch):
            
                images_batch, labels_batch, file_name = loader_test.next_batch_eval(batch_size, i)
                t5 = sess.run([top5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
            
                res_str = file_name[18:] + ""
                for t in t5:
                    for j in range(0,5):
                        res_str += " " + str(t[0][j])
                    
                    val_txt.write(res_str + '\n')
            
                if i % 500 == 0:        
                    print(res_str)
    
            val_txt.close()
            print("Done")
            sys.exit()
        else:
            step = start_step
            print('Restored Session.')

    else:
        sess.run(init)
        step = 0
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch, _ = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val, _ = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))
        
        # Run optimization op (backprop)
        _, summary = sess.run([train_optimizer, merged_summary_op], feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})
        summary_writer.add_summary(summary, step)
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch, _ = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.2f}".format(acc1) + ", Top5 = " + \
              "{:.2f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
