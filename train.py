# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
# sys.path.append('../../../')
from model import Transformer
from tqdm import tqdm
from data_load_combination import get_batch
# from data_load_topjudge import get_batch
from utils import save_hparams, save_variable_specs
import os
import math
import logging
import config

import constant
from metric import my_metric, NER_evaluation, NER_evaluation_new
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

hp = config.parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpu_num)

logging.basicConfig(level=logging.INFO)

logging.info("# Prepare train batches")
train_batches, num_train_batches = get_batch(hp.train_event, hp.train_non_event, hp.batch_size, shuffle=True)
test_batches, num_test_batches = get_batch(hp.test_event, hp.test_non_event, hp.test_batch_size, shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
token_ids, segment_ids, role, law, accu, term, flag, token_len = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(token_ids, segment_ids, role, law, accu, term, flag, token_len)
predict_law, predict_accu, predict_term, predict_role = m.test(token_ids, segment_ids, role, law, accu, term, flag, token_len)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        save_hparams(hp.logdir, hp)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        save_variable_specs(os.path.join(hp.logdir, "specs"))
        # ---------
        var = tf.trainable_variables()
        var_to_restore = [val for val in var if "embedding" in val.name or "encoder" in val.name]
        saver_restore = tf.train.Saver(var_to_restore)
        saver_restore.restore(sess, hp.pretraining_dir)

        # ---------
        # var = tf.trainable_variables()
        # var_to_restore = [val for val in var if "Embedding-" in val.name
        #                   or "Transformer-" in val.name
        #                   or "classification_law" in val.name
        #                   or "classification_accu" in val.name
        #                   or "classification_term" in val.name]
        # saver_restore = tf.train.Saver(var_to_restore)
        # # saver_restore.restore(sess, '../log_prediction_small/EPM_wo_event/small_all_0.3_32batch/training/model_E05-15085')
        # saver_restore.restore(sess,
        #                       '../log_prediction_large/EPM_wo_event/large_all_0.3_64batch/training/model_E15-372645')


    else:
        saver.restore(sess, ckpt)


    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    
    best_mean_f1 = 0
    
    
    for i in tqdm(range(_gs, total_steps + 1)):
        _, _gs, _summary, _loss = sess.run([train_op, global_step, train_summaries, loss])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)
           
        
        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))

#             logging.info("# save models")
#             model_output = "model_E%02d" % (epoch)
#             ckpt_name = os.path.join(hp.logdir, model_output)
#             saver.save(sess, ckpt_name, global_step=_gs)
#             logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            sess.run(test_init_op)

            if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)

            with open(hp.testdir + '/results_E' + str(epoch), 'w', encoding='utf8') as f_out:

                true_list_law = []
                predict_list_law = []

                true_list_accu = []
                predict_list_accu = []

                true_list_term = []
                predict_list_term = []

                true_list_role = []
                predict_list_role = []

                print(num_test_batches)
                for i in tqdm(range(num_test_batches)):
                    _predict_law, _true_law, _predict_accu, _true_accu, _predict_term, _true_term, _predict_role, _true_role, _flag = \
                        sess.run([predict_law, law, predict_accu, accu, predict_term, term, predict_role, role, flag])  # (N, )

                    for k in range(len(_true_law)):

                            true_list_law.append(_true_law[k])
                            predict_list_law.append(_predict_law[k])

                            true_list_accu.append(_true_accu[k])
                            predict_list_accu.append(_predict_accu[k])

                            true_list_term.append(_true_term[k])
                            predict_list_term.append(_predict_term[k])


                            true_list_role.append(_true_role[k])
                            predict_list_role.append(_predict_role[k])

                a_law, r_law, p_law, f1_law = my_metric(true_list_law, predict_list_law)
                a_accu, r_accu, p_accu, f1_accu = my_metric(true_list_accu, predict_list_accu)
                a_term, r_term, p_term, f1_term = my_metric(true_list_term, predict_list_term)

                f_out.write(str(a_law) + '\n')
                f_out.write(str(p_law) + '\n')
                f_out.write(str(r_law) + '\n')
                f_out.write(str(f1_law) + '\n')
                f_out.write('\n')

                f_out.write(str(a_accu) + '\n')
                f_out.write(str(p_accu) + '\n')
                f_out.write(str(r_accu) + '\n')
                f_out.write(str(f1_accu) + '\n')
                f_out.write('\n')

                f_out.write(str(a_term) + '\n')
                f_out.write(str(p_term) + '\n')
                f_out.write(str(r_term) + '\n')
                f_out.write(str(f1_term) + '\n')
                f_out.write('\n')

                role_metrics = NER_evaluation_new(trues=true_list_role, preds=predict_list_role)
                for me in role_metrics:
                    f_out.write(str(me) + '\n')
                f_out.write('\n')
                
                current_mean_f1 = f1_law
                if hp.bias_task == 'accu':
                    current_mean_f1 = f1_accu
                if hp.bias_task == 'term':
                    current_mean_f1 = f1_term
                
                if current_mean_f1 > best_mean_f1:
                    best_mean_f1 = current_mean_f1
                    logging.info("# save models")
                    model_output = "model_E%02d" % (epoch)
                    ckpt_name = os.path.join(hp.logdir, model_output)
                    saver.save(sess, ckpt_name, global_step=_gs)

                    with open(hp.testdir + '/Best', 'w', encoding='utf8') as f_best:
                        f_best.write(str(a_law) + '\n')
                        f_best.write(str(p_law) + '\n')
                        f_best.write(str(r_law) + '\n')
                        f_best.write(str(f1_law) + '\n')
                        f_best.write('\n')

                        f_best.write(str(a_accu) + '\n')
                        f_best.write(str(p_accu) + '\n')
                        f_best.write(str(r_accu) + '\n')
                        f_best.write(str(f1_accu) + '\n')
                        f_best.write('\n')

                        f_best.write(str(a_term) + '\n')
                        f_best.write(str(p_term) + '\n')
                        f_best.write(str(r_term) + '\n')
                        f_best.write(str(f1_term) + '\n')
                        f_best.write('\n')

                        for me in role_metrics:
                            f_best.write(str(me) + '\n')
                        f_best.write('\n')



            logging.info("# fall back to train model")
            sess.run(train_init_op)
    summary_writer.close()




logging.info("Done")