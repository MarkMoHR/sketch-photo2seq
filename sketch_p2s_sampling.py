import numpy as np
import time
import random
import pickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import range
import svgwrite
from cairosvg import svg2png

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import model as sketch_p2s_model
import utils
from sketch_p2s_train import load_dataset, reset_graph, load_checkpoint, FLAGS


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def draw_strokes(data, svg_filename, factor=0.2, padding=50, make_png=True):
    """
    little function that displays vector images and saves them to .svg
    :param data:
    :param factor:
    :param svg_filename:
    :return:
    """
    min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
    dims = (padding + max_x - min_x, padding + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = int(padding / 2) - min_x
    abs_y = int(padding / 2) - min_y
    p = "M%s, %s " % (abs_x, abs_y)
    # use lowcase for relative position
    command = "m"
    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + ", " + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()

    if make_png:
        png_filename = svg_filename[:-4] + '.png'
        svg2png(bytestring=dwg.tostring(), write_to=png_filename)

    return dims, dwg.tostring()


def make_grid_svg(s_list, grid_space=20.0, grid_space_x=20.0):
    """
    generate a 2D grid of many vector drawings
    :param s_list:
    :param grid_space:
    :param grid_space_x:
    :return:
    """

    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def load_env_compatible(sketch_data_dir, photo_data_dir, model_base_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_p2s_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_base_dir, model_params.data_type, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    return load_dataset(os.path.join(sketch_data_dir, model_params.data_type), photo_data_dir,
                        model_params, inference_mode=True)


def sample(sess, model, pix_h, seq_len=250, temperature=1.0, greedy_mode=False):
    """Samples a sequence from a pre-trained model."""

    def adjust_temp(pi_pdf, temp):
        pi_pdf = np.log(pi_pdf) / temp
        pi_pdf -= pi_pdf.max()
        pi_pdf = np.exp(pi_pdf)
        pi_pdf /= pi_pdf.sum()
        return pi_pdf

    def get_pi_idx(x, pdf, temp=1.0, greedy=False):
        """Samples from a pdf, optionally greedily."""
        if greedy:
            return np.argmax(pdf)
        pdf = adjust_temp(np.copy(pdf), temp)
        accumulate = 0
        for i in range(0, pdf.size):
            accumulate += pdf[i]
            if accumulate >= x:
                return i
        print('Error with sampling ensemble.')
        return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
        if greedy:
            return mu1, mu2
        mean = [mu1, mu2]
        s1 *= temp * temp
        s2 *= temp * temp
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # S0: [0, 0, 1, 0, 0]

    prev_state = sess.run(model.initial_state_p2s, feed_dict={model.pix_h: pix_h})

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []

    greedy = greedy_mode
    temp = temperature

    for i in range(seq_len):
        feed = {
            model.input_x: prev_x,
            model.sequence_lengths: [1],
            model.initial_state_p2s: prev_state,
            model.pix_h: pix_h
        }

        gmm_coef, next_state = sess.run([model.gmm_output_p2s, model.final_state_p2s], feed_dict=feed)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = gmm_coef
        # top 6 param: [1, 20], o_pen: [1, 3], next_state: [1, 1024]

        idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

        idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)

        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                              o_sigma1[0][idx], o_sigma2[0][idx],
                                              o_corr[0][idx], np.sqrt(temp), greedy)

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [
            o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
            o_pen[0]
        ]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        prev_x[0][0] = np.array(
            [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
        prev_state = next_state

    # strokes in stroke-5 format, strokes in stroke-3 format
    return utils.to_normal_strokes(strokes)


def sampling_conditional(sketch_data_dir, photo_data_dir, sampling_base_dir, model_base_dir):
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = \
        load_env_compatible(sketch_data_dir, photo_data_dir, model_base_dir)
    model_dir = os.path.join(model_base_dir, sample_hps_model.data_type)

    # construct the sketch-rnn model here:
    reset_graph()
    model = sketch_p2s_model.Model(hps_model)
    eval_model = sketch_p2s_model.Model(eval_hps_model, reuse=True)
    sampling_model = sketch_p2s_model.Model(sample_hps_model, reuse=True)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)

    for _ in range(20):
        rand_idx = random.randint(0, test_set.num_batches - 1)
        orig_x, unused_point_x, unused_point_l, img_x, img_paths = test_set.get_batch(rand_idx)

        img_path = img_paths[0]
        img_name = img_path[img_path.rfind('/') + 1:-4]
        sub_sampling_dir = os.path.join(sampling_base_dir, sample_hps_model.data_type, img_name)
        os.makedirs(sub_sampling_dir, exist_ok=True)
        print('rand_idx', rand_idx, 'stroke.shape', orig_x[0].shape, img_paths)

        ori_img = img_x[0].astype(np.uint8)
        ori_img_png = Image.fromarray(ori_img, 'RGB')
        ori_img_png.save(os.path.join(sub_sampling_dir, 'photo_gt.png'), 'PNG')
        draw_strokes(orig_x[0], os.path.join(sub_sampling_dir, 'sketch_gt.svg'))

        # encode the image
        common_pix_h = sess.run(sampling_model.pix_h, feed_dict={sampling_model.input_photo: img_x})

        # decoding for sampling
        strokes_out = sample(sess, sampling_model, common_pix_h,
                             eval_model.hps.max_seq_len, temperature=0.1)  # in stroke-3 format
        draw_strokes(strokes_out, os.path.join(sub_sampling_dir, 'sketch_pred.svg'))

        # Create generated grid at various temperatures from 0.1 to 1.0
        stroke_list = []
        for i in range(10):
            for j in range(1):
                print(i, j)
                stroke_list.append([sample(sess, sampling_model, common_pix_h,
                                           eval_model.hps.max_seq_len, temperature=0.1), [j, i]])
        stroke_grid = make_grid_svg(stroke_list)
        draw_strokes(stroke_grid, os.path.join(sub_sampling_dir, 'sketch_pred_multi.svg'))


def main():
    sampling_base_dir = 'outputs/sampling'

    # set numpy output to something sensible
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    sampling_conditional(FLAGS.sketch_data_dir, FLAGS.photo_data_dir, sampling_base_dir, FLAGS.snapshot_root)


if __name__ == '__main__':
    main()
