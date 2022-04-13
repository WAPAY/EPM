'''
pretraining之后, 再用event的信息。
或者两者的数据一起训练
'''

import json
import constant
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf


config_path = constant.config_path
checkpoint_path = constant.checkpoint_path
dict_path = constant.dict_path
tokenizer = Tokenizer(dict_path, do_lower_case=True)

id2law = dict(enumerate(constant.law))
law2id = {j: i for i, j in id2law.items()}

id2accu = dict(enumerate(constant.accu))
accu2id = {j: i for i, j in id2accu.items()}

id2term = dict(enumerate(constant.term))
term2id = {j: i for i, j in id2term.items()}



def load_data(fpath, type="event"):

    D_law = []
    D_accu = []
    D_term = []

    D_role = []  # 单条数据是list  [(文本片段，"keyword"), (文本片段, "O")]
    D_flag = []

    if fpath == b"None":
        return D_law, D_accu, D_term, D_role, D_flag

    samples = []
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f:
            all_data = json.loads(line.strip())
            samples.append(all_data)



    for sample in samples:

        fact_cut = sample['fact_cut'].replace(' ', '')
        # 加载summary

        if type == 'event':
            D_law.append(law2id[sample['law']])
            D_accu.append(accu2id[sample['accu']])
            D_term.append(term2id[sample['term']])
            D_flag.append(1)
        else:

            D_law.append(sample['law'])
            D_accu.append(sample['accu'])
            D_term.append(sample['term'])
            D_flag.append(0)



        if 'keywords_index' in sample.keys():
            index = sample['keywords_index']
            role = sample['role']

            current = []

            start_end = {}
            start_role = {}

            for i in range(len(index)):
                start_end[index[i][0]] = index[i][1]

                #single trigger ----------
                # start_role[index[i][0]] = role[i]

                # #multiple trigger ---------
                if role[i] == 'Trigger':
                    start_role[index[i][0]] = role[i] + "-" + str(D_law[-1])
                else:
                    start_role[index[i][0]] = role[i]

            temp_str = ''

            i = 0
            while i < len(fact_cut):
                if i not in start_end.keys():
                    temp_str += fact_cut[i]
                    i += 1
                    continue
                else:
                    if temp_str != '':
                        current.append([temp_str, 'O'])
                        temp_str = ''
                    current.append([fact_cut[i:start_end[i]], start_role[i]])
                    i = start_end[i]
            if temp_str != '':
                current.append([temp_str, 'O'])
            D_role.append(current)

        else:
            D_role.append([[fact_cut, 'O']])

    return D_law, D_accu, D_term, D_role, D_flag


labels_sub = constant.SUB_TYPES
id2label = dict(enumerate(labels_sub))
label2id = {j: i for i, j in id2label.items()}
# num_labels = len(labels_sub) * 2 + 1
num_labels = len(labels_sub)

def generator_fn(fpath_event, fpath_non_event):

    D_law_event, D_accu_event, D_term_event, D_role_event, D_flag_event = load_data(fpath_event)
    D_law_non_event, D_accu_non_event, D_term_non_event, D_role_non_event, D_flag_non_event = load_data(fpath_non_event, type='non_event')

    D_law = D_law_event + D_law_non_event
    D_accu = D_accu_event + D_accu_non_event
    D_term = D_term_event + D_term_non_event
    D_role = D_role_event + D_role_non_event
    D_flag = D_flag_event + D_flag_non_event


    for txt_list, law, accu, term, flag in zip(D_role, D_law, D_accu, D_term, D_flag):

        token_ids, labels = [tokenizer._token_start_id], [0]



        for w, l in txt_list:



            w_token_ids = tokenizer.encode(w)[0][1:-1]
            if len(token_ids) + len(w_token_ids) < constant.max_len:
                token_ids += w_token_ids
                if len(w_token_ids) == 0: continue
                if l == 'O':
                    labels += [0] * len(w_token_ids)
                else:
                    # -- BIO
                    # B = label2id[l] * 2 + 1
                    # I = label2id[l] * 2 + 2
                    # labels += [B] + [I] * (len(w_token_ids) - 1)
                    # --
                    labels += [label2id[l]] + [label2id[l]] * (len(w_token_ids) - 1)
            
                    # --- binary
                    # labels += [1] + [1] * (len(w_token_ids) - 1)
            
            else:
                w_token_ids = w_token_ids[:constant.max_len - len(token_ids) - 1]
                token_ids += w_token_ids
                if len(w_token_ids) == 0: continue
                if l == 'O':
                    labels += [0] * len(w_token_ids)
                else:
                    # B = label2id[l] * 2 + 1
                    # I = label2id[l] * 2 + 2
                    # labels += [B] + [I] * (len(w_token_ids) - 1)
                    labels += [label2id[l]] + [label2id[l]] * (len(w_token_ids) - 1)
                    # labels += [1] + [1] * (len(w_token_ids) - 1)
                break
            # -----------------
            # 只用事件的信息
#             w_token_ids = tokenizer.encode(w)[0][1:-1]
#             if l != 'O':

#                 token_ids += w_token_ids
#                 labels += [label2id[l]] + [label2id[l]] * (len(w_token_ids) - 1)

        # 关键词的token_ids和labels和segment_ids
        token_ids += [tokenizer._token_end_id]
        labels += [0]
        segment_ids = [0] * len(token_ids)


        yield (token_ids, segment_ids, labels, law, accu, term, flag, len(token_ids))



def input_fn(fpath_event, fpath_non_event, batch_size, shuffle):
    shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]),
              tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))

    types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)

    # paddings = ((0, 0, 0))
    pad_id = tokenizer._token_pad_id
    paddings = (pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, pad_id)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=[fpath_event, fpath_non_event])


    if shuffle:
        dataset = dataset.shuffle(32 * batch_size)

    # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(2)
    dataset = dataset.repeat()

    return dataset




def get_batch(fpath_event, fpath_non_event, batch_size, shuffle=False):

    count = 0
    if fpath_event != 'None':
        with open(fpath_event, 'r', encoding='utf8') as f:
            count += len(f.readlines())
    if fpath_non_event != 'None':
        with open(fpath_non_event, 'r', encoding='utf8') as f:
            count += len(f.readlines())

    batches = input_fn(fpath_event, fpath_non_event, batch_size, shuffle)

    num_batches = count // batch_size + int(count % batch_size != 0)

    return batches, num_batches


if __name__ == '__main__':


    import config
    hp = config.parser.parse_args()

    train_batches, num_batches = get_batch(hp.train_event, hp.train_non_event, 1, shuffle=True)

    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)

    token_ids, segment_ids, labels, law, accu, term, flag, token_len = iter.get_next()

    train_init_op = iter.make_initializer(train_batches)

    with tf.Session() as sess:
        sess.run(train_init_op)
        for _ in range(2):
            print(sess.run([token_ids, labels, law, accu, term, flag, token_len]))




