# trigger类型
TRIGGER_SUP_TYPES = ['Property', 'Human_rights', 'Public_order', 'Public_health', 'Environment', 'Drug']
# TRIGGER_SUB_TYPES = [
#                    ['Trigger-Robbery', 'Trigger-Larceny', 'Trigger-Fraud', 'Trigger-Racketeering'],
#                    ['Trigger-Intentional_injury', 'Trigger-Rape', 'Trigger-Kidnapping'],
#                    ['Trigger-Official_duty', 'Trigger-Forgery', 'Trigger-Gambling'],
#                    ['Trigger-Doctoring'],
#                    ['Trigger-Wildlife', 'Trigger-Logging'],
#                    ['Trigger-Drug_possession']]
# TRIGGER_SUB_TYPES = [
#                    ['Trigger-Robbery', 'Trigger-Larceny', 'Trigger-Fraud', 'Trigger-Racketeering'],
#                    ['Trigger-Intentional_injury', 'Trigger-Rape', 'Trigger-Kidnapping'],
#                    ['Trigger-Official_duty', 'Trigger-Forgery', 'Trigger-Gambling'],
#                    ['Trigger-Doctoring'],
#                    ['Trigger-Wildlife', 'Trigger-Logging'],
#                    ['Trigger-Drug_possession', 'Trigger-Drug_plant']]

TRIGGER_SUB_TYPES = [
                   ['Trigger-32', 'Trigger-9', 'Trigger-23'],
                   ['Trigger-34', 'Trigger-31', 'Trigger-78'],
                   ['Trigger-99', 'Trigger-76', 'Trigger-47'],
                   ['Trigger-0'],
                   ['Trigger-6', 'Trigger-88'],
                   ['Trigger-39']]



# TRIGGER_SUP_TYPES = ['Trigger']
# TRIGGER_SUB_TYPES = [['Trigger']]
# 角色类型
ROLE_SUP_TYPES = ['Party', 'State', 'Object', 'Attribute']
ROLE_SUB_TYPES = [
                  ['Criminal', 'Victim', 'Officer'],
                  ['Qualified', 'Intention', 'Method'],
                  ['Property', 'Instrument', 'Animal', 'Plant', 'Drug', 'Gambling_device', 'License'],
                  ['Quantity', 'Injury']]

# ROLE_SUB_TYPES = [
#                   ['Criminal', 'Victim', 'Officer'],
#                   ['Qualified', 'Intention', 'Method'],
#                   ['Property', 'Instrument', 'Animal', 'Plant', 'Drug', 'Drug_Plant', 'Gambling_device', 'License'],
#                   ['Quantity', 'Injury']]


SUP_TYPES = ['None'] + TRIGGER_SUP_TYPES + ROLE_SUP_TYPES
SUB_TYPES = ['None'] + [j for i in TRIGGER_SUB_TYPES for j in i] + [j for i in ROLE_SUB_TYPES for j in i]



len_trigger_sup = len(TRIGGER_SUP_TYPES)
len_trigger_sub = len([j for i in TRIGGER_SUB_TYPES for j in i])


len_role_sup = len(ROLE_SUP_TYPES)
len_role_sub = len([j for i in ROLE_SUB_TYPES for j in i])


len_sup = len_trigger_sup + len_role_sup + 1
len_sub = len_trigger_sub + len_role_sub + 1




# role_label_weights = [0.05] + [1]*(len_sub-1)

SUP_SUB_MATRIX = [[0 for i in range(len(SUB_TYPES))] for j in range(len(SUP_TYPES))]
SUP_SUB_MATRIX[0][0] = 1.0



count = 0
for i in range(len(TRIGGER_SUP_TYPES)):
    for j in range(len(TRIGGER_SUB_TYPES[i])):
        SUP_SUB_MATRIX[1+i][1+count] = 1.0
        count += 1

count = 0
for i in range(len(ROLE_SUP_TYPES)):
    for j in range(len(ROLE_SUB_TYPES[i])):
        SUP_SUB_MATRIX[1+i+len_trigger_sup][1+count+len_trigger_sub] = 1.0
        count += 1


# bert 路径
# config_path = '../../nezha_wwm_base/bert_config.json'
# checkpoint_path = '../../nezha_wwm_base/model.ckpt-691689'
# dict_path = '../../nezha_wwm_base/vocab.txt'


config_path = '../PLM/albert/albert_config_small_google.json'
checkpoint_path = '../PLM/albert/albert_model.ckpt'
dict_path = '../PLM/albert/vocab.txt'


# embedding_dim = 768
embedding_dim = 384
max_len = 512
INF = 1e-10


law_selected = [264, 234, 266, 236, 303, 336, 239, 274, 277, 341, 280, 345, 348]
# accu = ['开设赌场', '盗窃', '非法行医', '诈骗', '非法狩猎', '伪造公司、企业、事业单位、人民团体印章', '非法持有毒品', '绑架', '故意伤害', '合同诈骗', '伪造、变造、买卖国家机关公文、证件、印章', '滥伐林木', '非法收购、运输盗伐、滥伐的林木', '信用卡诈骗', '走私、贩卖、运输、制造毒品', '盗伐林木', '赌博', '强奸', '非法进行节育手术', '敲诈勒索', '非法猎捕、杀害珍贵、濒危野生动物', '妨害公务', '伪造、变造居民身份证']
# term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

law = []
accu = []
term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
with open('../data/new_law.txt', 'r', encoding='utf8') as f:
    for line in f:
        law.append(int(line.strip()))
with open('../data/new_accu.txt', 'r', encoding='utf8') as f:
    for line in f:
        accu.append(line.strip())

# with open('../data/large_law.txt', 'r', encoding='utf8') as f:
#     for line in f:
#         law.append(int(line.strip()))
# with open('../data/large_accu.txt', 'r', encoding='utf8') as f:
#     for line in f:
#         accu.append(line.strip())


# accu = ['敲诈勒索', '非法持有毒品', '走私、贩卖、运输、制造毒品', '妨害公务', '滥伐林木', '赌博', '故意伤害', '信用卡诈骗', '合同诈骗', '诈骗', '强奸', '开设赌场']
# law = [264, 234, 266, 236, 303, 274, 277, 345, 348]
# term = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]

len_law = len(law)
len_accu = len(accu)
len_term = len(term)

# SUP_TERM = 0
# with open('term_condition.txt', 'r', encoding='utf8') as f:
#     for line in f:
#         SUP_TERM += int(line.strip())


# law_contents = []
# with open('law_keyword.txt', 'r', encoding='utf8') as f:
#     for line in f:
#         law_contents.append(line.strip().split('||')[1])

