# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SketchRNN training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import model as sketch_p2s_model
import utils
import numpy as np
import random
import six
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'sketch_data_dir',
    'datasets',
    'The directory of sketch data of the dataset.')
tf.app.flags.DEFINE_string(
    'photo_data_dir',
    'datasets/QMUL/shoes/photos',
    'The directory of photo data of the dataset.')
tf.app.flags.DEFINE_string(
    'log_root', 'outputs/log',
    'Directory to store tensorboard.')
tf.app.flags.DEFINE_string(
    'snapshot_root', 'outputs/snapshot',
    'Directory to store model checkpoints.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')


def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_dataset(sketch_data_dir, photo_data_dir, model_params, inference_mode=False):
    """Loads the .npz file, and splits the set into train/test."""

    # normalizes the x and y columns using the training set.
    # applies same scaling factor to test set.

    if isinstance(model_params.data_set, list):
        datasets = model_params.data_set
    else:
        datasets = [model_params.data_set]

    train_strokes = None
    test_strokes = None
    train_image_paths = []
    test_image_paths = []

    for dataset in datasets:
        if model_params.data_type == 'QMUL':
            train_data_filepath = os.path.join(sketch_data_dir, dataset, 'train_svg_sim_spa_png.h5')
            test_data_filepath = os.path.join(sketch_data_dir, dataset, 'test_svg_sim_spa_png.h5')

            train_data_dict = utils.load_hdf5(train_data_filepath)
            test_data_dict = utils.load_hdf5(test_data_filepath)

            train_sketch_data = utils.reassemble_data(
                train_data_dict['image_data'],
                train_data_dict['data_offset'])  # list of [N_sketches], each [N_points, 4]
            train_photo_names = train_data_dict['image_base_name']  # [N_sketches, 1], byte
            train_photo_paths = [os.path.join(photo_data_dir, train_photo_names[i, 0].decode() + '.png')
                                 for i in range(train_photo_names.shape[0])]  # [N_sketches], str
            test_sketch_data = utils.reassemble_data(
                test_data_dict['image_data'],
                test_data_dict['data_offset'])  # list of [N_sketches], each [N_points, 4]
            test_photo_names = test_data_dict['image_base_name']  # [N_sketches, 1], byte
            test_photo_paths = [os.path.join(photo_data_dir, test_photo_names[i, 0].decode() + '.png')
                                for i in range(test_photo_names.shape[0])]  # [N_sketches], str

            # transfer stroke-4 to stroke-3
            train_sketch_data = utils.to_normal_strokes_4to3(train_sketch_data)
            test_sketch_data = utils.to_normal_strokes_4to3(test_sketch_data)  # [N_sketches,], each with [N_points, 3]

            if train_strokes is None:
                train_strokes = train_sketch_data
                test_strokes = test_sketch_data
            else:
                train_strokes = np.concatenate((train_strokes, train_sketch_data))
                test_strokes = np.concatenate((test_strokes, test_sketch_data))

        elif model_params.data_type == 'QuickDraw':
            data_filepath = os.path.join(sketch_data_dir, dataset, 'npz', 'sketchrnn_' + dataset + '.npz')
            if six.PY3:
                data = np.load(data_filepath, encoding='latin1')
            else:
                data = np.load(data_filepath)

            if train_strokes is None:
                train_strokes = data['train']  # [N_sketches,], each with [N_points, 3]
                test_strokes = data['test']
            else:
                train_strokes = np.concatenate((train_strokes, data['train']))
                test_strokes = np.concatenate((test_strokes, data['test']))

            train_photo_paths = [os.path.join(sketch_data_dir, dataset, 'png', 'train',
                                              str(model_params.image_size) + 'x' + str(model_params.image_size),
                                              str(im_idx) + '.png') for im_idx in range(len(data['train']))]
            test_photo_paths = [os.path.join(sketch_data_dir, dataset, 'png', 'test',
                                             str(model_params.image_size) + 'x' + str(model_params.image_size),
                                             str(im_idx) + '.png') for im_idx in range(len(data['test']))]
        else:
            raise Exception('Unknown data type:', model_params.data_type)

        print('Loaded {}/{} from {} {}'.format(len(train_photo_paths), len(test_photo_paths),
                                               model_params.data_type, dataset))
        train_image_paths += train_photo_paths
        test_image_paths += test_photo_paths

    all_strokes = np.concatenate((train_strokes, test_strokes))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset combined: {} ({}/{}), avg len {}'.format(
        len(all_strokes), len(train_strokes), len(test_strokes), int(avg_len)))
    assert len(train_image_paths) == len(train_strokes)
    assert len(test_image_paths) == len(test_strokes)

    # calculate the max strokes we need.
    max_seq_len = utils.get_max_len(all_strokes)

    # overwrite the hps with this calculation.
    model_params.max_seq_len = max_seq_len
    print('model_params.max_seq_len %i.' % model_params.max_seq_len)

    eval_model_params = sketch_p2s_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 1

    if inference_mode:
        eval_model_params.batch_size = 1
        eval_model_params.is_training = 0

    sample_model_params = sketch_p2s_model.copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    train_set = utils.DataLoader(
        train_strokes,
        train_image_paths,
        model_params.image_size,
        model_params.image_size,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)

    normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    train_set.normalize(normalizing_scale_factor)

    # valid_set = utils.DataLoader(
    #     valid_strokes,
    #     eval_model_params.batch_size,
    #     max_seq_length=eval_model_params.max_seq_len,
    #     random_scale_factor=0.0,
    #     augment_stroke_prob=0.0)
    # valid_set.normalize(normalizing_scale_factor)

    test_set = utils.DataLoader(
        test_strokes,
        test_image_paths,
        model_params.image_size,
        model_params.image_size,
        eval_model_params.batch_size,
        max_seq_length=eval_model_params.max_seq_len,
        random_scale_factor=0.0,
        augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    print('normalizing_scale_factor %4.4f.' % normalizing_scale_factor)

    result = [
        train_set, None, test_set, model_params, eval_model_params,
        sample_model_params
    ]
    return result


def evaluate_model(sess, model, data_set):
    """Returns the average weighted cost, reconstruction cost and KL cost."""
    total_cost = 0.0
    total_r_cost = 0.0
    total_kl_cost = 0.0
    for batch in range(data_set.num_batches):
        unused_orig_x, point_x, point_l, img_x, img_paths = data_set.get_batch(batch)

        feed = {
            model.input_sketch: point_x,
            model.input_photo: img_x,
            model.sequence_lengths: point_l
        }

        cost, r_cost, kl_cost = sess.run([model.cost, model.r_cost_sum, model.kl_cost_sum], feed)
        total_cost += cost
        total_r_cost += r_cost
        total_kl_cost += kl_cost

    total_cost /= data_set.num_batches
    total_r_cost /= data_set.num_batches
    total_kl_cost /= data_set.num_batches
    return total_cost, total_r_cost, total_kl_cost


def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('Loading model %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, saver, model_save_path, global_step):
    checkpoint_path = os.path.join(model_save_path, 'p2s')
    print('saving model %s.' % checkpoint_path)
    print('global_step %i.' % global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def create_summary(summary_writer, summ_map, step):
    for summ_key in summ_map:
        summ_value = summ_map[summ_key]
        summ = tf.summary.Summary()
        summ.value.add(tag=summ_key, simple_value=float(summ_value))
        summary_writer.add_summary(summ, step)
    summary_writer.flush()


def train(sess, train_model, eval_model, train_set, test_set):
    # Setup summary writer.
    summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    print('-' * 100)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i' % (var.name, str(var.get_shape()), num_param))
    print('Total trainable variables %i.' % count_t_vars)
    print('-' * 100)

    # setup eval stats
    best_eval_cost = 100000000.0  # set a large init value

    # main train loop

    hps = train_model.hps
    start = time.time()

    # create saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    for _ in range(hps.num_steps):
        step = sess.run(train_model.global_step)

        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                              hps.decay_rate ** step + hps.min_learning_rate)
        curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                          hps.kl_decay_rate ** step)

        _, point_x, point_l, img_x, img_paths = train_set.random_batch()

        feed = {
            train_model.input_sketch: point_x,
            train_model.input_photo: img_x,
            train_model.sequence_lengths: point_l,
            train_model.lr: curr_learning_rate,
            train_model.kl_weight: curr_kl_weight
        }

        (train_cost, r_cost, kl_cost,
         kl_p2s, kl_s2p, kl_p2p, kl_s2s,
         r_p2s, r_s2p, r_p2p, r_s2s,
         _, _,
         train_step, _) = \
            sess.run([
                train_model.cost, train_model.r_cost_sum, train_model.kl_cost_sum,
                train_model.kl_cost_p2s, train_model.kl_cost_s2p, train_model.kl_cost_p2p, train_model.kl_cost_s2s,
                train_model.r_cost_p2s, train_model.r_cost_s2p, train_model.r_cost_p2p, train_model.r_cost_s2s,
                train_model.final_state_p2s, train_model.final_state_s2s,
                train_model.global_step, train_model.train_op
            ], feed)

        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            train_summary_map = {
                'Train_Cost': train_cost, 'Train_R_Cost': r_cost, 'Train_KL_Cost': kl_cost,
                'Train_p2s_KL_Cost': kl_p2s, 'Train_p2s_R_Cost': r_p2s,
                'Train_s2p_KL_Cost': kl_s2p, 'Train_s2p_R_Cost': r_s2p,
                'Train_p2p_KL_Cost': kl_p2p, 'Train_p2p_R_Cost': r_p2p,
                'Train_s2s_KL_Cost': kl_s2s, 'Train_s2s_R_Cost': r_s2s,
                'Learning_Rate': curr_learning_rate, 'KL_Weight': curr_kl_weight,
                'Time_Taken_Train': time_taken
            }
            create_summary(summary_writer, train_summary_map, train_step)

            output_format = ('step: %d, lr: %.6f, klw: %0.4f, cost: %.4f, '
                             'recon: %.4f, kl: %.4f, train_time_taken: %.4f')
            output_values = (step, curr_learning_rate, curr_kl_weight, train_cost,
                             r_cost, kl_cost, time_taken)
            output_log = output_format % output_values
            print(output_log)
            start = time.time()

        if step % hps.save_every == 0 and step > 0:
            eval_cost, eval_r_cost, eval_kl_cost = evaluate_model(sess, eval_model, test_set)

            end = time.time()
            time_taken_eval = end - start
            start = time.time()

            eval_summary_map = {
                'Eval_Cost': eval_cost, 'Eval_R_Cost': eval_r_cost, 'Eval_KL_Cost': eval_kl_cost,
            }
            create_summary(summary_writer, eval_summary_map, train_step)

            output_format = ('best_eval_cost: %0.4f, eval_cost: %.4f, eval_recon: '
                             '%.4f, eval_kl: %.4f, eval_time_taken: %.4f')
            output_values = (min(best_eval_cost, eval_cost), eval_cost,
                             eval_r_cost, eval_kl_cost, time_taken_eval)
            output_log = output_format % output_values
            print(output_log)

            save_model(sess, saver, FLAGS.snapshot_root, step)

            if eval_cost < best_eval_cost:
                best_eval_cost = eval_cost
                best_eval_summary_map = {
                    'Eval_Cost_Best': best_eval_cost
                }
                create_summary(summary_writer, best_eval_summary_map, train_step)


def trainer(model_params):
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('Loading data files.')
    print('-' * 100)
    datasets = load_dataset(os.path.join(FLAGS.sketch_data_dir, model_params.data_type),
                            FLAGS.photo_data_dir, model_params)

    train_set = datasets[0]
    unused_valid_set = datasets[1]
    test_set = datasets[2]
    train_model_params = datasets[3]
    eval_model_params = datasets[4]
    unused_sample_model_params = datasets[5]

    reset_graph()
    train_model = sketch_p2s_model.Model(train_model_params)
    eval_model = sketch_p2s_model.Model(eval_model_params, reuse=True)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    if FLAGS.resume_training:
        load_checkpoint(sess, FLAGS.snapshot_root)

    # Write config file to json file.
    os.makedirs(FLAGS.log_root, exist_ok=True)
    os.makedirs(FLAGS.snapshot_root, exist_ok=True)
    with tf.gfile.Open(
            os.path.join(FLAGS.snapshot_root, 'model_config.json'), 'w') as f:
        json.dump(train_model_params.values(), f, indent=True)

    train(sess, train_model, eval_model, train_set, test_set)


def main():
    """Load model params, save config file and start trainer."""
    model_params = sketch_p2s_model.get_default_hparams()
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams)
    trainer(model_params)


if __name__ == '__main__':
    main()
