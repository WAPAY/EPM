# -*- coding: utf-8 -*-

import os

os.environ['TF_KERAS'] = "1"
from bert4keras.backend import keras, K
import tensorflow as tf
import config
import bert4keras
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from module import label_smoothing, noam_scheme, get_token_embeddings_word, dag_lstm, get_lawformer
from bert4keras.layers import ConditionalRandomField
import logging
from keras.models import Model
from bert4keras.tokenizers import Tokenizer
import constant
from bert4keras.snippets import sequence_padding
import numpy as np
import law_accu_term_constraint

logging.basicConfig(level=logging.INFO)

# bert配置
config_path = constant.config_path
checkpoint_path = constant.checkpoint_path
dict_path = constant.dict_path
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class Transformer:

    def __init__(self, hp):
        self.hp = hp

        # self.transformer = build_transformer_model(
        #     config_path,
        #     checkpoint_path,
        #     model='albert',
        #     # model='nezha',
        # )

        self.transformer = get_lawformer('../../keras_lawformer')

        # self.word_embedding = get_token_embeddings_word()

        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE):
            self.trans = tf.get_variable(
                "transitions",
                shape=[constant.len_sub, constant.len_sub],
                # shape=[2, 2],
                initializer=tf.contrib.layers.xavier_initializer())

        # ------
        # matching
        with tf.variable_scope('law_contents', reuse=tf.AUTO_REUSE):
            self.law_contents_embeddings = tf.get_variable(name='xxx', initializer=np.load('law_contents.npy'), trainable=True)
        #
        # # law_accu, law_term constraint
        self.law_accu = tf.constant(law_accu_term_constraint.law_accu, dtype=tf.float32)
        self.law_term = tf.constant(law_accu_term_constraint.law_term, dtype=tf.float32)

        # self.SUP_SUB_MATRIX = tf.constant(constant.SUP_SUB_MATRIX, dtype=tf.float32)

    def encoder(self, token_ids, segment_ids, token_len):
        # memory = self.transformer([token_ids, segment_ids])
        mask_ids = tf.to_int32(tf.sequence_mask(token_len))
        memory = self.transformer([token_ids, segment_ids, mask_ids])


        # memory = tf.nn.embedding_lookup(self.word_embedding, token_ids)

        # with tf.variable_scope('rnnnn', reuse=tf.AUTO_REUSE):
        #     memory, _ = tf.compat.v1.nn.dynamic_rnn(
        #         tf.nn.rnn_cell.LSTMCell(128),
        #         memory,
        #         sequence_length=token_len,
        #         dtype=tf.float32)

        return memory

    # def trigger_module(self, memory):
    #     with tf.variable_scope('trigger', reuse=tf.AUTO_REUSE):
    #         trigger_sup_weights = tf.layers.dense(memory, constant.len_trigger_sup) # [N, T, len_sup]
    #         trigger_sup_weights = tf.nn.softmax(trigger_sup_weights, dim=-2)
    #         contexts = tf.matmul(tf.transpose(memory, [0, 2, 1]), trigger_sup_weights) # [N, dim, len_sup]
    #         # contexts = tf.matmul(contexts, self.SUP_SUB_MATRIX) # [N, dim, len_sub])
    #         # contexts = tf.transpose(contexts, [0, 2, 1]) # [N, len_sub, dim]
    #         # memory_expand = tf.tile(tf.expand_dims(memory, axis=2), [1, 1, constant.len_event_sub, 1]) # [N, T, len_sub, dim]
    #         # contexts_expand = tf.tile(tf.expand_dims(contexts, axis=1), [1, memory.get_shape().as_list()[1], 1, 1]) # [N, T, len_sub, dim]
    #         # concat = tf.concat((memory_expand, contexts_expand), axis=-1)
    #         # logits = tf.layers.dense(concat, 1, use_bias=False) # [N, T, len_sub, 1]
    #         # logits = tf.squeeze(logits, axis=-1)
    #         orientended = tf.reduce_mean(contexts, axis=-1)  # [N, dim]
    #         orientended_expand = tf.tile(tf.expand_dims(orientended, axis=1),
    #                                      [1, tf.shape(memory)[1], 1])  # [N, T, dim]
    #         concat = (memory + orientended_expand) / 2
    #         logits = tf.layers.dense(concat, constant.len_sub)
    #
    #     return logits

    def role_module(self, memory):
        with tf.variable_scope('role', reuse=tf.AUTO_REUSE):
            role_sup_weights = tf.layers.dense(memory, constant.len_sup)  # [N, T, len_sup]
            role_sup_weights = tf.nn.softmax(role_sup_weights, dim=-2)
            contexts = tf.matmul(tf.transpose(memory, [0, 2, 1]), role_sup_weights)  # [N, dim, len_sup]
            # contexts = tf.matmul(contexts, self.SUP_SUB_MATRIX) # [N, dim, len_sub])
            # contexts = tf.transpose(contexts, [0, 2, 1]) # [N, len_sub, dim]
            # memory_expand = tf.tile(tf.expand_dims(memory, axis=2), [1, 1, constant.len_sub, 1]) # [N, T, len_sub, dim]
            # contexts_expand = tf.tile(tf.expand_dims(contexts, axis=1), [1, tf.shape(memory)[1], 1, 1]) # [N, T, len_sub, dim]
            # concat = (memory_expand + contexts_expand) / 2
            # logits = tf.layers.dense(concat, 1, use_bias=False) # [N, T, len_sub, 1]
            # logits = tf.squeeze(logits, axis=-1)

            orientended = tf.reduce_mean(contexts, axis=-1)  # [N, dim]
            orientended_expand = tf.tile(tf.expand_dims(orientended, axis=1),
                                         [1, tf.shape(memory)[1], 1])  # [N, T, dim]
            concat = (memory + orientended_expand) / 2
            logits = tf.layers.dense(concat, constant.len_sub)

            # logits = tf.layers.dense(memory, constant.len_sub)

        return logits

    def legal_predict(self, memory, logits_role, token_len, flag):
        # -----------------------
        # role_indices = tf.argmax(logits_role, axis=-1)  # [N, T]

        # role_indices, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)

        # target_indices = role_indices
        # target_indices_mask = tf.cast(tf.not_equal(target_indices, 0), dtype=tf.float32)

        # target_words_embeddings = tf.multiply(tf.expand_dims(target_indices_mask, axis=-1), memory)
        # target_words_embeddings = tf.reduce_max(target_words_embeddings, axis=1)

        # with tf.variable_scope('classification_law', reuse=tf.AUTO_REUSE):
        #     logtis_law = tf.layers.dense(target_words_embeddings, constant.len_law)

        # with tf.variable_scope('classification_accu', reuse=tf.AUTO_REUSE):
        #     logtis_accu = tf.layers.dense(target_words_embeddings, constant.len_accu)

        # with tf.variable_scope('classification_term', reuse=tf.AUTO_REUSE):
        #     logtis_term = tf.layers.dense(target_words_embeddings, constant.len_term)

        # --- lstm dag
        # y1, y2, y3 = dag_lstm(target_words_embeddings)
        #
        # with tf.variable_scope('classification_law', reuse=tf.AUTO_REUSE):
        #     logtis_law = tf.layers.dense(y1, constant.len_law)
        #
        # with tf.variable_scope('classification_accu', reuse=tf.AUTO_REUSE):
        #     logtis_accu = tf.layers.dense(y2, constant.len_accu)
        #
        # with tf.variable_scope('classification_term', reuse=tf.AUTO_REUSE):
        #     logtis_term = tf.layers.dense(y3, constant.len_term)

        # --------------------------
        #         with tf.variable_scope('classification_law', reuse=tf.AUTO_REUSE):
        #             logtis_law = tf.layers.dense(tf.reduce_max(memory, axis=1), constant.len_law)

        #         with tf.variable_scope('classification_accu', reuse=tf.AUTO_REUSE):
        #             logtis_accu = tf.layers.dense(tf.reduce_max(memory, axis=1), constant.len_accu)

        #         with tf.variable_scope('classification_term', reuse=tf.AUTO_REUSE):
        #             logtis_term = tf.layers.dense(tf.reduce_max(memory, axis=1), constant.len_term)

        # matching + event

        # # role_indices = tf.argmax(logits_role, axis=-1)  # [N, T]
        role_indices, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)
        target_indices = role_indices
        target_indices_mask = tf.cast(tf.not_equal(target_indices, 0), dtype=tf.float32)
        # target_indices_mask = tf.where(tf.cast(target_indices, dtype=tf.float32) > 0.5,
        #                                tf.ones_like(target_indices), tf.zeros_like(target_indices))
        #
        # target_indices_mask = tf.cast(target_indices_mask, dtype=tf.float32)
        target_words_embeddings = tf.multiply(tf.expand_dims(target_indices_mask, axis=-1), memory)
        target_words_embeddings = tf.reduce_max(target_words_embeddings, axis=1) 
        
        x_weight = tf.nn.softmax(tf.matmul(target_words_embeddings, tf.transpose(self.law_contents_embeddings, [1, 0])), axis=-1) # N, 101
        
        law_context = tf.matmul(x_weight, self.law_contents_embeddings) 
        
        total_context = tf.concat([law_context, target_words_embeddings], axis=-1)
        
        with tf.variable_scope('classification_law', reuse=tf.AUTO_REUSE):
            logtis_law = tf.layers.dense(total_context, constant.len_law)
        
        with tf.variable_scope('classification_accu', reuse=tf.AUTO_REUSE):
            logtis_accu = tf.layers.dense(total_context, constant.len_accu)
        
        with tf.variable_scope('classification_term', reuse=tf.AUTO_REUSE):
            # logtis_term = tf.layers.dense(total_context, constant.SUP_TERM)
            logtis_term = tf.layers.dense(total_context, constant.len_term)

        return logtis_law, logtis_accu, logtis_term

    def train(self, token_ids, segment_ids, role_labels, law, accu, term, flag, token_len):
        memory = self.encoder(token_ids, segment_ids, token_len)

        logits_role = self.role_module(memory)

        logits_law, logits_accu, logits_term = self.legal_predict(memory, logits_role, token_len, flag)

        # true_law = label_smoothing(tf.one_hot(law, depth=constant.len_law))
        # true_accu = label_smoothing(tf.one_hot(accu, depth=constant.len_accu))
        # true_term = label_smoothing(tf.one_hot(term, depth=constant.len_term))

        true_law = tf.one_hot(law, depth=constant.len_law)
        true_accu = tf.one_hot(accu, depth=constant.len_accu)
        true_term = tf.one_hot(term, depth=constant.len_term)

        # loss_law = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_law, labels=true_law))
        # loss_accu = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_accu, labels=true_accu))
        # loss_term = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_term, labels=true_term))

        # -----------
        # law-accu law-term constraint
        law_indexs = tf.one_hot(tf.argmax(logits_law, axis=-1), depth=101, axis=-1)
        accu_dis = tf.matmul(law_indexs, self.law_accu) # N, 117
        term_dis = tf.matmul(law_indexs, self.law_term) # N, 11
        
        logits_accu_argmax = tf.argmax(logits_accu, axis=-1)
        logits_term_argmax = tf.argmax(logits_term, axis=-1)
        
        accu_softmax = tf.nn.softmax(logits_accu, axis=-1)
        accu_softmax_mask = tf.where(tf.cast(accu_dis, dtype=tf.bool), accu_softmax, tf.ones_like(accu_softmax))
        loss_accu_mask = tf.reduce_sum(true_accu * tf.log(accu_softmax_mask), axis=-1)
        loss_accu_original = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_accu, labels=true_accu)
        
        term_softmax = tf.nn.softmax(logits_term, axis=-1)
        term_softmax_mask = tf.where(tf.cast(term_dis, dtype=tf.bool), term_softmax, tf.ones_like(term_softmax))
        loss_term_mask = tf.reduce_sum(true_term * tf.log(term_softmax_mask), axis=-1)
        loss_term_original = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_term, labels=true_term)
        
        loss_accu = tf.reduce_mean(
            tf.where(tf.equal(tf.cast(logits_accu_argmax, tf.int32), accu), loss_accu_mask, loss_accu_original))
        
        loss_term = tf.reduce_mean(
            tf.where(tf.equal(tf.cast(logits_term_argmax, tf.int32), term), loss_term_mask, loss_term_original))
        
        loss_law = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_law, labels=true_law))

        # with tf.variable_scope('law_weight', reuse=tf.AUTO_REUSE):
        #     law_weight = tf.get_variable(name='law_weight', shape=1, initializer=tf.random_uniform_initializer(maxval=2, minval=0))
        #     law_weight = tf.reduce_sum(law_weight)
        # with tf.variable_scope('accu_weight', reuse=tf.AUTO_REUSE):
        #     accu_weight = tf.get_variable(name='accu_weight', shape=1, initializer=tf.random_uniform_initializer(maxval=2, minval=0))
        #     accu_weight = tf.reduce_sum(accu_weight)
        # with tf.variable_scope('term_weight', reuse=tf.AUTO_REUSE):
        #     term_weight = tf.get_variable(name='term_weight', shape=1, initializer=tf.random_uniform_initializer(maxval=2, minval=0))
        #     term_weight = tf.reduce_sum(term_weight)
        # loss_legal = loss_law / tf.math.square(law_weight) + tf.math.log(tf.math.square(law_weight)) + \
        # loss_accu / tf.math.square(accu_weight) + tf.math.log(tf.math.square(accu_weight)) + \
        # loss_term / tf.math.square(term_weight) + tf.math.log(tf.math.square(term_weight))

        # role loss
        ## cross entropy
        # true_role = label_smoothing(tf.one_hot(role_labels, depth=constant.len_sub, axis=-1))
        # # true_role = tf.multiply(true_role, tf.constant(constant.role_label_weights, dtype=tf.float32))
        # loss_role = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_role, labels=true_role)
        # loss_role = tf.reduce_mean(loss_role, axis=-1)
        # loss_role = tf.multiply(loss_role, tf.cast(flag, tf.float32))
        # loss_role = tf.reduce_sum(loss_role) / (tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=logits_role,
            tag_indices=role_labels,
            transition_params=self.trans,
            sequence_lengths=token_len)
        loss_role = tf.multiply(-log_likelihood, tf.cast(flag, tf.float32))
        loss_role = tf.reduce_sum(loss_role) / (tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

        # golden_event_constraint
        # ------------
        # logtis_law_golden, logtis_accu_golden, logtis_term_golden, \
        # logtis_law_fact, logtis_accu_fact, logtis_term_fact = self.event_constraint(memory, role_labels)
        #
        # ES_law = tf.losses.absolute_difference(tf.nn.softmax(logtis_law_fact, axis=-1), tf.nn.softmax(logits_law, axis=-1)) \
        #          + tf.losses.absolute_difference(tf.nn.softmax(logtis_law_golden, axis=-1), tf.nn.softmax(logits_law, axis=-1))
        #
        # ES_accu = tf.losses.absolute_difference(tf.nn.softmax(logtis_accu_fact, axis=-1),
        #                                        tf.nn.softmax(logits_accu, axis=-1)) \
        #          + tf.losses.absolute_difference(tf.nn.softmax(logtis_accu_golden, axis=-1),
        #                                          tf.nn.softmax(logits_accu, axis=-1))
        #
        # ES_term = tf.losses.absolute_difference(tf.nn.softmax(logtis_term_fact, axis=-1),
        #                                        tf.nn.softmax(logits_term, axis=-1)) \
        #          + tf.losses.absolute_difference(tf.nn.softmax(logtis_term_golden, axis=-1),
        #                                          tf.nn.softmax(logits_term, axis=-1))
        # ES = ES_law + ES_accu + ES_term

        # specific role type constraint
        # ------------

        trigger_constraint = tf.reduce_max(tf.reduce_max(logits_role[:, :, 1:14], axis=-1), axis=-1)\
                             - tf.reduce_sum(tf.reduce_sum(logits_role[:, :, 1:14], axis=-1), axis=-1) \
        + tf.math.abs(1- tf.reduce_max(tf.reduce_max(logits_role[:, :, 1:14], axis=-1), axis=-1))
        TS = tf.reduce_mean(trigger_constraint)  + tf.math.abs(1- tf.reduce_max(tf.reduce_max(logits_role[:, :, 14], axis=-1), axis=-1))

        # arg_role = tf.argmax(logits_role, axis=-1)  # (N, T)
        #
        # # multiple trigger
        # target_role_mask_1 = tf.where(arg_role < 1, tf.ones_like(arg_role), tf.zeros_like(arg_role))
        # target_role_mask_2 = tf.where(arg_role > 13, tf.ones_like(arg_role), tf.zeros_like(arg_role))
        #
        # trigger_constraint = tf.reduce_sum(tf.ones_like(arg_role), axis=-1) \
        #                      - tf.reduce_sum(tf.ones_like(target_role_mask_1), axis=-1) \
        #                      - tf.reduce_sum(tf.ones_like(target_role_mask_2), axis=-1)
        #
        # # single trigger
        # # trigger_constraint = tf.reduce_sum(tf.where( tf.equal(arg_role,1), tf.ones_like(arg_role), tf.zeros_like(arg_role)), axis=-1)
        #
        # TS = tf.losses.absolute_difference(tf.tile(tf.constant([1]), [tf.shape(trigger_constraint)[0]]),
        #                                    trigger_constraint)

        # --------
        # Trigger and role type constraint
        # arg_role = tf.argmax(logits_role, axis=-1)
        arg_role, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)
        # trigger_5_constraint = tf.reduce_sum(
        #     tf.where(tf.equal(arg_role, 5), tf.ones_like(arg_role), tf.zeros_like(arg_role)), axis=-1)
        # role_5_constraint = tf.reduce_sum(
        #     tf.where(tf.equal(arg_role, 29), tf.ones_like(arg_role), tf.zeros_like(arg_role)), axis=-1)
        # CS_5 = tf.losses.absolute_difference(trigger_5_constraint, role_5_constraint)

        CS_5 = tf.math.abs(1 - tf.reduce_max(tf.reduce_max(logits_role[:, :, 29], axis=-1), axis=-1)) * \
               tf.reduce_sum( tf.where(tf.equal(arg_role, 5), tf.ones_like(arg_role), tf.zeros_like(arg_role)), axis=-1)

        # law_accu, law_term constraint

        # law_indexs = tf.one_hot(tf.argmax(logits_law, axis=-1), depth=101, axis=-1)
        # accu_indexs = tf.one_hot(tf.argmax(logits_accu, axis=-1), depth=117, axis=-1)
        # term_indexs = tf.one_hot(tf.argmax(logits_term, axis=-1), depth=11, axis=-1)
        #
        # accu_error = tf.multiply(1 - tf.matmul(law_indexs, self.law_accu), accu_indexs) # N, 117
        # term_error = tf.multiply(1 - tf.matmul(law_indexs, self.law_term), term_indexs) # N, 11
        #
        # accu_error_loss = tf.reduce_mean(tf.reduce_sum(accu_error, axis=-1))
        # term_error_loss = tf.reduce_mean(tf.reduce_sum(term_error, axis=-1))

        #         total_loss = loss_legal + loss_role/3 + TS/3 + CS_5/3

        loss_legal = loss_law * self.hp.law_weight \
                     + loss_accu * self.hp.accu_weight \
                     + loss_term * self.hp.term_weight

        total_loss = loss_legal \
                     + loss_role * self.hp.role_weight \
                     + TS * self.hp.CSTR1_weight \
                     + CS_5 * self.hp.CSTR2_weight

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        # lr = 0.001
        optimizer = tf.train.AdamOptimizer(lr)

        # trainable_vars = tf.trainable_variables()
        # var_list = [t for t in trainable_vars if not (t.name.startswith(u'Embedding') or t.name.startswith(u'Transformer'))]
        # train_op = optimizer.minimize(total_loss, global_step=global_step, var_list=var_list)

        train_op = optimizer.minimize(total_loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("loss_legal", loss_legal)
        tf.summary.scalar("loss_law", loss_law)
        tf.summary.scalar("loss_accu", loss_accu)
        tf.summary.scalar("loss_term", loss_term)
        tf.summary.scalar("loss_role", loss_role)

        summaries = tf.summary.merge_all()

        return total_loss, train_op, global_step, summaries

    def test(self, token_ids, segment_ids, role_labels, law, accu, term, flag, token_len):
        memory = self.encoder(token_ids, segment_ids, token_len)
        logits_role = self.role_module(memory)
        # predict_role = tf.argmax(logits_role, axis=-1)

        predict_role, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)

        logits_law, logits_accu, logits_term = self.legal_predict(memory, logits_role, token_len, flag)
        predict_law = tf.argmax(logits_law, axis=-1)
        predict_accu = tf.argmax(logits_accu, axis=-1)
        predict_term = tf.argmax(logits_term, axis=-1)

        predict_law_one_hot = tf.one_hot(predict_law, depth=101, axis=-1)
        predict_accu = tf.argmax(
            tf.multiply(tf.matmul(predict_law_one_hot, self.law_accu), tf.nn.softmax(logits_accu, axis=-1)), axis=-1)
        predict_term = tf.argmax(
            tf.multiply(tf.matmul(predict_law_one_hot, self.law_term), tf.nn.softmax(logits_term, axis=-1)), axis=-1)

        return predict_law, predict_accu, predict_term, predict_role

    def generate_logits(self, token_ids, segment_ids, role_labels, law, accu, term, flag, token_len):
        memory = self.encoder(token_ids, segment_ids, token_len)
        logits_role = self.role_module(memory)
        logits_law, logits_accu, logits_term = self.legal_predict(memory, logits_role, flag)

        return logits_law, logits_accu, logits_term
