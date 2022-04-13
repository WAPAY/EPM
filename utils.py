# -*- coding: utf-8 -*-
# /usr/bin/python3

import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.INFO)

def save_hparams(path, hp):
    if not os.path.exists(path): os.makedirs(path)
    with open(os.path.join(path, "hparams"), 'w') as fout:
        for k, v in vars(hp).items():
            fout.write(str(k) + ':' + str(v) + '\n')


def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *= shp[d]
        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params, flush=True)

    trainable_params, num_trainable_params = [], 0
    for v in tf.trainable_variables():
        trainable_params.append("{}==={}".format(v.name, v.shape))
        num_trainable_params += _get_size(v.shape)
    print("num_trainable_params: ", num_trainable_params, flush=True)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
        fout.write("-------------------------------\n")
        fout.write("-------------------------------\n")
        fout.write("-------------------------------\n")
        fout.write("Trainable: \n")
        fout.write("\n".join(trainable_params))
    logging.info("Variables info has been saved.")


