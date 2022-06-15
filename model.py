# -*- coding: utf-8 -*-
import os
os.environ['TF_KERAS'] = "1"
import tensorflow as tf
import config
from bert4keras.models import build_transformer_model
from module import label_smoothing, noam_scheme
import logging
import constant
import numpy as np
import law_accu_term_constraint
logging.basicConfig(level=logging.INFO)


class Transformer:

    def __init__(self, hp):

        self.hp = hp

        self.transformer = build_transformer_model(
            config_path='tf_xsbert/config.json',
            model='bert',
        )

        with tf.variable_scope('crf_layer', reuse=tf.AUTO_REUSE):
            self.trans = tf.get_variable(
                "transitions",
                shape=[constant.len_sub, constant.len_sub],
                # shape=[2, 2],
                initializer=tf.contrib.layers.xavier_initializer())

        # ------
        # matching
        with tf.variable_scope('law_contents', reuse=tf.AUTO_REUSE):
            self.law_contents_embeddings = tf.get_variable(name='law_content_embedding', initializer=np.load('data/law_contents.npy'), trainable=True)

        # law_accu, law_term constraint
        self.law_accu = tf.constant(law_accu_term_constraint.law_accu, dtype=tf.float32)
        self.law_term = tf.constant(law_accu_term_constraint.law_term, dtype=tf.float32)

        self.trigger_role = tf.constant(constant.TRIGGER_ROLE_MATRIX, dtype=tf.float32)

        with tf.variable_scope('sub_embeddings', reuse=tf.AUTO_REUSE):
            self.sub_embeddings = tf.get_variable(
                "sub_embeddings",
                shape=[constant.len_sub, 768],
                initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('law_content_attention', reuse=tf.AUTO_REUSE):
            self.law_content_attention = tf.get_variable(
                "law_content_attention_matrix",
                shape=[768, 768],
                initializer=tf.contrib.layers.xavier_initializer())

        # self.SUP_SUB_MATRIX = tf.constant(constant.SUP_SUB_MATRIX, dtype=tf.float32)

    def encoder(self, token_ids, segment_ids, token_len):
        memory = self.transformer([token_ids, segment_ids])
        return memory

    def role_module(self, memory):
        with tf.variable_scope('role', reuse=tf.AUTO_REUSE):
            role_sup_weights = tf.layers.dense(memory, constant.len_sup) # [N, T, len_sup]
            role_sup_weights = tf.nn.softmax(role_sup_weights, dim=-2)
            contexts = tf.matmul(tf.transpose(memory, [0, 2, 1]), role_sup_weights) # [N, dim, len_sup]
            orientended = tf.reduce_mean(contexts, axis=-1) # [N, dim]
            orientended_expand = tf.tile(tf.expand_dims(orientended, axis=1), [1, tf.shape(memory)[1], 1])  # [N, T, dim]
            concat = (memory + orientended_expand) / 2
            logits = tf.nn.softmax(tf.matmul(concat, tf.transpose(self.sub_embeddings)), axis=-1)

        return logits


    def legal_predict(self, memory, logits_role, flag, token_len):

        # -----------------------
        if self.hp.train_event != 'None':
            # role_indices = tf.argmax(logits_role, axis=-1)  # [N, T]
            role_indices, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)

            target_indices = role_indices
            target_indices_mask = tf.cast(tf.not_equal(target_indices, 0), dtype=tf.float32)
            target_words_embeddings = tf.multiply(tf.expand_dims(target_indices_mask, axis=-1), memory)
            target_words_sub_embeddings = tf.nn.embedding_lookup(self.sub_embeddings, target_indices)
            # target_words_embeddings = tf.concat([target_words_embeddings, target_words_sub_embeddings], axis=-1)
            target_words_embeddings = (target_words_embeddings + target_words_sub_embeddings) / 2
            target_words_embeddings = tf.reduce_max(target_words_embeddings, axis=1)
        else:
            target_words_embeddings = memory
            target_words_embeddings = tf.reduce_max(target_words_embeddings, axis=1)


        x_weight = tf.nn.softmax(tf.matmul(tf.matmul(target_words_embeddings, self.law_content_attention), tf.transpose(self.law_contents_embeddings, [1, 0])),
                                 axis=-1)  # N, 101

        law_context = tf.matmul(x_weight, self.law_contents_embeddings)

        total_context = tf.concat([law_context, target_words_embeddings], axis=-1)
        
        with tf.variable_scope('classification_law', reuse=tf.AUTO_REUSE):
            logtis_law = tf.layers.dense(total_context, constant.len_law)

        with tf.variable_scope('classification_accu', reuse=tf.AUTO_REUSE):
            logtis_accu = tf.layers.dense(total_context, constant.len_accu)

        with tf.variable_scope('classification_term', reuse=tf.AUTO_REUSE):
            logtis_term = tf.layers.dense(total_context, constant.len_term)

        return logtis_law, logtis_accu, logtis_term


    def train(self, token_ids, segment_ids, role_labels, law, accu, term, flag, token_len):

        memory = self.encoder(token_ids, segment_ids, token_len)

        logits_role = self.role_module(memory)

        logits_law, logits_accu, logits_term = self.legal_predict(memory, logits_role, flag, token_len)

        true_law = label_smoothing(tf.one_hot(law, depth=constant.len_law))
        true_accu = label_smoothing(tf.one_hot(accu, depth=constant.len_accu))
        true_term = label_smoothing(tf.one_hot(term, depth=constant.len_term))

#         true_law = tf.one_hot(law, depth=constant.len_law)
#         true_accu = tf.one_hot(accu, depth=constant.len_accu)
#         true_term = tf.one_hot(term, depth=constant.len_term)

        #-----------
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

        logits_law_argmax = tf.argmax(logits_law, axis=-1)
        loss_accu = tf.reduce_mean(
            tf.where(tf.equal(tf.cast(logits_law_argmax, tf.int32), law), loss_accu_mask, loss_accu_original))

        loss_term = tf.reduce_mean(
            tf.where(tf.equal(tf.cast(logits_law_argmax, tf.int32), law), loss_term_mask, loss_term_original))

        loss_law = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_law, labels=true_law))

        loss_legal = loss_law * self.hp.law_weight \
                     + loss_accu * self.hp.accu_weight \
                     + loss_term * self.hp.term_weight

        if self.hp.train_event != 'None':
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=logits_role,
                tag_indices=role_labels,
                transition_params=self.trans,
                sequence_lengths=token_len)
            loss_role = tf.multiply(-log_likelihood, tf.cast(flag, tf.float32))
            loss_role = tf.reduce_sum(loss_role) / (tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

            # ------------
 
            criminal_prob = tf.squeeze(logits_role[:, :, 14:15], axis=-1)  # [N, T]
            criminal_prob = tf.reduce_max(criminal_prob, axis=-1)  # N
            absolute_loss_1 = tf.multiply(1 - criminal_prob, tf.cast(flag, tf.float32))
            absolute_loss_1 = tf.reduce_sum(absolute_loss_1) / (
                        tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

            trigger_prob = logits_role[:, :, 1:14]  # [N, T, 13]
            sum_all = tf.reduce_sum(tf.reduce_sum(trigger_prob, axis=-1), axis=-1)  # N
            max_all = tf.reduce_max(tf.reduce_max(trigger_prob, axis=-1), axis=-1)  # N
            absolute_loss_2 = sum_all - max_all + (1 - max_all)
            absolute_loss_2 = tf.multiply(absolute_loss_2, tf.cast(flag, tf.float32))
            absolute_loss_2 = tf.reduce_sum(absolute_loss_2) / (
                    tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

            # --------
            trigger_type = 1 + tf.argmax(tf.reduce_max(trigger_prob, axis=1), axis=-1)  # N
            selected_role = tf.matmul(tf.one_hot(trigger_type, depth=constant.len_sub), self.trigger_role)  # N, len_sub
            pos = tf.tile(tf.expand_dims(selected_role, axis=1), [1, tf.shape(trigger_prob)[1], 1])  # N, T, len_sub
            neg = 1 - pos
            pos = tf.multiply(pos, logits_role)
            neg = tf.multiply(neg, logits_role)
            sum_all_2 = tf.reduce_sum(tf.reduce_sum(neg, axis=-1), axis=-1)  # N
            max_all_2 = tf.reduce_max(tf.reduce_max(pos, axis=-1), axis=-1)  # N
            loss_3 = (1 - max_all_2) + sum_all_2
            loss_3 = tf.multiply(loss_3, tf.cast(flag, tf.float32))
            loss_3 = tf.reduce_sum(loss_3) / (
                    tf.cast(tf.reduce_sum(flag), dtype=tf.float32) + constant.INF)

            total_loss = loss_legal \
                         + loss_role * self.hp.role_weight \
                         + (absolute_loss_1 + absolute_loss_2 + loss_3) / 3 * self.hp.CSTR_weight

        else:

            total_loss = loss_legal

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("loss_law", loss_law)
        tf.summary.scalar("loss_accu", loss_accu)
        tf.summary.scalar("loss_term", loss_term)
        # tf.summary.scalar("loss_role", loss_role)
        # tf.summary.scalar("loss_absolute1", absolute_loss_1)
        # tf.summary.scalar("loss_absolute2", absolute_loss_2)
        # tf.summary.scalar("loss_trigger_role", loss_3)

        summaries = tf.summary.merge_all()

        return total_loss, train_op, global_step, summaries

    def test(self, token_ids, segment_ids, role_labels, law, accu, term, flag, token_len):
        memory = self.encoder(token_ids, segment_ids, token_len)
        logits_role = self.role_module(memory)
        
        if self.hp.train_event != 'None':
            predict_role, _ = tf.contrib.crf.crf_decode(logits_role, self.trans, token_len)
        else:
            predict_role = tf.argmax(logits_role, axis=-1)
        
        logits_law, logits_accu, logits_term = self.legal_predict(memory, logits_role, flag, token_len)
        predict_law = tf.argmax(logits_law, axis=-1)

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
