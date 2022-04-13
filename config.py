import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-train_event', default='data_train.txt')
parser.add_argument('-train_non_event', default='../data/train_cs_shuffle.json')
parser.add_argument('-batch_size', default=16, type=int)
parser.add_argument('-num_epochs', default=20, type=int)
parser.add_argument('-logdir', default='../log_prediction/combination_1/training')

parser.add_argument('-gpu_num', default=0, type=int)

parser.add_argument('-pretraining_dir', default='../../legal_bert/legal_bert')

parser.add_argument('-test_event', default='data_test.txt')
parser.add_argument('-test_non_event', default='../data/test_cs.json')
parser.add_argument('-test_batch_size', default=32, type=int)
parser.add_argument('-testdir', default='../log_prediction/combination_1/testing')

parser.add_argument('-lr', default=0.0001, type=float)
parser.add_argument('-warmup_steps', default=3000, type=int)

parser.add_argument('-bias_task', default='law')

parser.add_argument('-law_weight', default=0.5, type=float)
parser.add_argument('-accu_weight', default=0.5, type=float)
parser.add_argument('-term_weight', default=0.4, type=float)
parser.add_argument('-role_weight', default=0.1, type=float)
parser.add_argument('-CSTR1_weight', default=0.2, type=float)
parser.add_argument('-CSTR2_weight', default=0.2, type=float)



