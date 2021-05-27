from __future__ import print_function
import tensorflow as tf
import os

import argparse
import pathlib
import datetime
import numpy as np  # for debugging
from tensorflow.keras import backend as K

from eval_tf import eval
from data_tf import COVIDxDataset
from model import build_UNet2D_4L, build_resnet_attn_model
from load_data import loadDataJSRTSingle
from utils.tensorboard import heatmap_overlay_summary_op, scalar_summary,log_tensorboard_images

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_keras_collections(graph, keras_model):
    """
    Creates missing collections in a tf.Graph using keras model attributes
    Args:
        graph (tf.Graph): Tensorflow graph with missing collections
        keras_model (keras.Model): Keras model with desired attributes
    """
    if hasattr(keras_model, 'metrics'):
        for metric in keras_model.metrics:
            for update_op in metric.updates:
                graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
            for weight in metric._non_trainable_weights:
                graph.add_to_collection(tf.GraphKeys.METRIC_VARIABLES, weight)
                graph.add_to_collection(tf.GraphKeys.LOCAL_VARIABLES, weight)
    else:
        print('skipped adding variables from metrics')

    for update_op in keras_model.updates:
        graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    # Clear default trainable collection before adding tensors
    graph.clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for trainable_layer in keras_model.trainable_weights:
        graph.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, trainable_layer)


parser = argparse.ArgumentParser(description='COVID-Net Training Script')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--bs', default=16, type=int, help='Batch size')
parser.add_argument('--col_name', nargs='+', default=["folder_name", "img_path", "class"])
parser.add_argument('--target_name', type=str, default="class")
parser.add_argument('--weightspath', default='/home/hossein.aboutalebi/data/urgent_sev/0.85', type=str, help='Path to output folder')
# parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='2021-05-21#18-16-44.464148COVIDNet-lr8e-05_27',
                    type=str, help='Name of model ckpts')
parser.add_argument('--trainfile', default='labels/train_pnemunia.txt', type=str, help='Path to train file')
parser.add_argument('--cuda_n', type=str, default="0", help='cuda number')
parser.add_argument('--testfile', default='labels/test_pnemunia.txt', type=str, help='Path to test file')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of folder to store training checkpoints')
parser.add_argument('--datadir', default='/home/hossein.aboutalebi/data/pneumonia/images/', type=str,
                    help='Path to data folder')
parser.add_argument('--in_sem', default=0, type=int,
                    help='initial_itrs until training semantic')
parser.add_argument('--covid_weight', default=1, type=float, help='Class weighting for covid')
parser.add_argument('--covid_percent', default=0.5, type=float, help='Percentage of covid samples in batch')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_2/Softmax:0', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logged_images', default='labels/logged_p.txt', type=str,
                    help='Name of output tensor from graph')
parser.add_argument('--logit_tensorname', default='norm_dense_2/MatMul:0', type=str,
                    help='Name of logit tensor for loss')
parser.add_argument('--label_tensorname', default='norm_dense_1_target:0', type=str,
                    help='Name of label tensor for loss')
parser.add_argument('--load_weight', action='store_true',
                    help='default False')
parser.add_argument('--resnet_type', default='resnet1', type=str,
                    help='type of resnet arch. Values can be: resnet0_M, resnet0_R, resnet1, resnet2')
parser.add_argument('--training_tensorname', default='keras_learning_phase:0', type=str,
                    help='Name of training placeholder tensor')
parser.add_argument('--is_severity_model', action='store_true',
                    help='Add flag if training COVIDNet CXR-S model')


height_semantic = 256  # do not change unless train a new semantic model
width_semantic = 256
switcher = 3

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_n

# Parameters
learning_rate = args.lr
batch_size = args.bs
test_batch_size = 20
display_step = 1    # evaluation interval in epochs
log_interval = 100  # image and loss log interval in steps (batches)
class_weights = [1., args.covid_weight]

# Make output paths
current_time = (str(datetime.datetime.now()).replace(" ", "#")).replace(":", "-")
outputPath = './output/' + current_time
runID = args.name + '-lr' + str(learning_rate)
runPath = outputPath + runID
# path_images_train=os.path.join(runPath,"images/train")
# path_images_test=os.path.join(runPath,"images/test")
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
# pathlib.Path(path_images_train).mkdir(parents=True, exist_ok=True)
# pathlib.Path(path_images_test).mkdir(parents=True, exist_ok=True)

print('Output: ' + runPath)

# Load list of test files
# testfiles_frame = pd.read_csv(args.testfile, delimiter=" ", names=args.col_name).values
with open(args.testfile) as f:
    testfiles = f.readlines()

# Get image file names to log throughout training
with open(args.logged_images) as f:
    log_images = f.readlines()

# Get stack of images to log
log_positive, log_negative = [], []
for i in range(len(log_images)):
    line = log_images[i].split()
    # image = process_image_file(os.path.join(args.datadir, 'test', line[1]), 0.08, args.input_size)
    # image = image.astype('float32') / 255.0
    sem_image = loadDataJSRTSingle(os.path.join(args.datadir, 'test', line[1]), (width_semantic, width_semantic))
    if line[2] == 'positive':
        log_positive.append(sem_image)
    elif line[2] == 'negative':
        log_negative.append(sem_image)
log_positive, log_negative = np.array(log_positive), np.array(log_negative)

dataset = COVIDxDataset(
    args.datadir, num_classes=2, image_size=args.input_size,
    sem_image_size=width_semantic, class_weights=class_weights)

with tf.Session() as sess:
    K.set_session(sess)
    # First we load the semantic model:
    model_semantic = build_UNet2D_4L((height_semantic, width_semantic, 1))
    labels_tensor = tf.placeholder(tf.float32)
    sample_weights = tf.placeholder(tf.float32)

    resnet_50 = build_resnet_attn_model(name=args.resnet_type, classes=2, model_semantic=model_semantic)
    training_ph = K.learning_phase()
    model_main = resnet_50.call(input_shape=(args.input_size, args.input_size, 3), training=training_ph)

    image_tensor = model_main.input[0]  # The model.input is a tuple of (input_2:0, and input_1:0)
    semantic_image_tensor = model_semantic.input

    graph = tf.get_default_graph()
    pred_tensor = model_main.output
    saver = tf.train.Saver(max_to_keep=1)

    logit_tensor = graph.get_tensor_by_name('final_output/BiasAdd:0')

    # Define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_tensor, labels=labels_tensor)*sample_weights)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

    # Initialize update ops collection
    init_keras_collections(graph, model_main)
    print('length with model_main: ', len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
    # init_keras_collections(graph, model_semantic)
    # print('length with model_semantic: ', len(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

    # Create train ops
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_vars_all=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:-1]
    # train_vars_resnet = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "^((?!sem).)*$")
    # train_vars_sem = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sem*")
    # accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in train_vars_resnet]
    # zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    with tf.control_dependencies(extra_ops):
        # gvs = optimizer.compute_gradients(loss_op, train_vars_resnet)
        train_op_resnet = optimizer.minimize(loss_op, var_list=train_vars_all)
        if args.resnet_type[:7] != 'resnet0':
            train_op_sem = optimizer.minimize(loss_op, var_list=train_vars_all)
        # print('Train vars resnet: ', len(train_vars_resnet))
        # print('Train vars semantic: ', len(train_vars_sem))
        # accum_ops = [accum_vars[j].assign_add(gv[0]) for j, gv in enumerate(gvs)]
        # train_step_bacth = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    # Run the initializer
    sess.run(tf.global_variables_initializer())

    # Make summary ops and writer
    loss_summary = tf.summary.scalar('train/loss', loss_op)
    image_summary = heatmap_overlay_summary_op(
        'train/semantic', model_semantic.input, model_semantic.output, max_outputs=5)
    test_image_summary_pos = heatmap_overlay_summary_op(
        'test/semantic/positive', model_semantic.input, model_semantic.output, max_outputs=len(log_images))
    test_image_summary_neg = heatmap_overlay_summary_op(
        'test/semantic/negative', model_semantic.input, model_semantic.output, max_outputs=len(log_images))
    summary_op = tf.summary.merge([loss_summary, image_summary])
    summary_writer = tf.summary.FileWriter(os.path.join(runPath, 'events'), graph)

    # Load weights
    if args.load_weight:
        saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    else:
        model_semantic.load_weights("./model/trained_model.hdf5")
    # saver.restore(sess, tf.train.latest_checkpoint(args.weightspath))

    # Save base model and run baseline eval
    saver.save(sess, os.path.join(runPath, 'model'))
    print('Saved baseline checkpoint')
    summary_pos, summary_neg = log_tensorboard_images(sess, K,test_image_summary_pos, semantic_image_tensor, log_positive,
                                                      test_image_summary_neg, log_negative)
    summary_writer.add_summary(summary_pos, 0)
    summary_writer.add_summary(summary_neg, 0)
    print("Finished tensorboard baseline")
    metrics = eval(
        sess, dataset, args.testfile, test_batch_size, image_tensor, semantic_image_tensor,
        pred_tensor, dataset.class_map)
    summary_writer.add_summary(scalar_summary(metrics, 'val/'), 0)

    # Training cycle
    print('Training started')
    train_dataset, count, batch_size = dataset.train_dataset(args.trainfile, batch_size)
    data_next = train_dataset.make_one_shot_iterator().get_next()
    total_batch = int(np.ceil(count/batch_size))
    progbar = tf.keras.utils.Progbar(total_batch)

    for epoch in range(args.epochs):
        # Select train op depending on training stage
        if epoch < args.in_sem or epoch % switcher != 0 or args.resnet_type[:7] == 'resnet0':
            train_op = train_op_resnet
        else:
            train_op = train_op_sem

        # Log images and semantic output
        summary_pos, summary_neg = log_tensorboard_images(
            sess, K, test_image_summary_pos, semantic_image_tensor,
            log_positive, test_image_summary_neg, log_negative)
        summary_writer.add_summary(summary_pos, epoch)
        summary_writer.add_summary(summary_neg, epoch)

        for i in range(total_batch):
            # Get batch of data
            data = sess.run(data_next)
            batch_x = data['image']
            batch_sem_x = data['sem_image']
            batch_y = data['label']
            weights = data['weight']
            feed_dict = {
                image_tensor: batch_x,
                semantic_image_tensor: batch_sem_x,
                labels_tensor: batch_y,
                sample_weights: weights,
                K.learning_phase(): 1}
            total_steps = epoch*total_batch + i

            if not (total_steps % log_interval):
                # if (i % 4 == 0):
                #     sess.run(train_step_bacth, feed_dict=feed_dict)
                #     sess.run(zero_ops)
                # run summary op for batch
                _, pred, semantic_output, summary = sess.run(
                    (train_op, pred_tensor, model_semantic.output, summary_op),
                    feed_dict=feed_dict)
                summary_writer.add_summary(summary, total_steps)
            else:  # run without summary op
                # if (i % 4 == 0):
                #     sess.run(train_step_bacth, feed_dict=feed_dict)
                #     sess.run(zero_ops)
                _, pred, semantic_output = sess.run((train_op, pred_tensor, model_semantic.output),
                                                    feed_dict=feed_dict)
            progbar.update(i + 1)

        if epoch % display_step == 0:
            # Print minibatch loss and lr
            # semantic_output=model_semantic(batch_x.astype('float32')).eval(session=sess)
            # pred = model_main((batch_x.astype('float32'),semantic_output)).eval(session=sess)
            loss = sess.run(loss_op, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1), "Minibatch loss=", "{:.9f}".format(loss))
            print("lr: {},  batch_size: {}".format(str(args.lr),str(args.bs)))

            # Run eval and log results to tensorboard
            metrics = eval(sess, dataset, args.testfile, test_batch_size, image_tensor,
                           semantic_image_tensor, pred_tensor, dataset.class_map)
            summary_writer.add_summary(scalar_summary(metrics, 'val/'), (epoch + 1)*total_batch)
            # model_main.save_weights(runPath+"_"+str(epoch))
            print('Output: ' + runPath+"_"+str(epoch))
            print('Saving checkpoint at epoch {}'.format(epoch + 1))

print("Optimization Finished!")
