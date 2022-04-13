from sklearn import metrics
import constant
import numpy as np

def my_metric(y_true, y_predict):
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_predict)
    r = metrics.recall_score(y_true=y_true, y_pred=y_predict, average='macro')
    p = metrics.precision_score(y_true=y_true, y_pred=y_predict, average='macro')
    f = metrics.f1_score(y_true=y_true, y_pred=y_predict, average='macro')

    return accuracy, r, p, f

labels = constant.SUB_TYPES
id2label = dict(enumerate(labels))

def convert_to_token(ids, id2token):
    result = []
    current_id = None
    for i, id in enumerate(ids):
        if id == 0:
            result.append('O')
        elif id != 0 and current_id == None:
            result.append('B-' + id2token[id])
            current_id = id
        elif id != 0 and current_id != None:
            if id == current_id:
                result.append('I-' + id2token[id])
            else:
                result.append('B-' + id2token[id])
                current_id = id
    return result


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append((ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append((ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append((ent_type, start_offset, len(tokens) - 1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags=None):
    if tags != None:
        true_named_entities = [ent for ent in true_named_entities if ent[0] in tags]
        pred_named_entities = [ent for ent in pred_named_entities if ent[0] in tags]
    TP = 0.0
    TP_loc = 0.0

    for pred in pred_named_entities:
        if pred in true_named_entities:
            TP += 1

    for pred in pred_named_entities:
        for true in true_named_entities:
            if pred[1] == true[1] and pred[2] == true[2]:
                TP_loc += 1

    INF = 1e-8

    p = TP / (len(pred_named_entities) + INF)
    r = TP / (len(true_named_entities) + INF)
    f1 = 2 * p * r / (p + r + INF)

    p_loc = TP_loc / (len(pred_named_entities) + INF)
    r_loc = TP_loc / (len(true_named_entities) + INF)
    f1_loc = 2 * p_loc * r_loc / (p_loc + r_loc + INF)

    return (p, r, f1, p_loc, r_loc, f1_loc)


def NER_evaluation(trues, preds, tags=None):

    results = []
    for i in range(len(trues)):
        true = convert_to_token(trues[i], id2label)
        pred = convert_to_token(preds[i], id2label)


        true = collect_named_entities(true)
        pred = collect_named_entities(pred)

        print(true)

        results.append(compute_metrics(true, pred, tags=tags))
    if len(results) == 0:
        return 0
    return np.mean(results, axis=0)

def NER_specific_type(trues, preds, type):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for i in range(len(trues)):
        true = convert_to_token(trues[i], id2label)
        pred = convert_to_token(preds[i], id2label)

        true = set(collect_named_entities(true))
        pred = set(collect_named_entities(pred))

        R = set([entity for entity in pred if type in entity[0]])
        T = set([entity for entity in true if type in entity[0]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    print(type)
    print(f1)
    print(precision)
    print(recall)
    print()

def NER_evaluation_new(trues, preds, tags=None):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X_trigger, Y_trigger, Z_trigger = 1e-10, 1e-10, 1e-10
    for i in range(len(trues)):
        true = convert_to_token(trues[i], id2label)
        pred = convert_to_token(preds[i], id2label)

        true = set(collect_named_entities(true))
        pred = set(collect_named_entities(pred))
        print(pred)
        R = pred
        T = true
        X += len(R & T)
        Y += len(R)
        Z += len(T)

        R_trigger = set([entity for entity in pred if "Trigger" in entity[0]])
        T_trigger = set([entity for entity in true if "Trigger" in entity[0]])
        X_trigger += len(R_trigger & T_trigger)
        Y_trigger += len(R_trigger)
        Z_trigger += len(T_trigger)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    f1_trigger, precision_trigger, recall_trigger = 2 * X_trigger / (Y_trigger + Z_trigger), X_trigger / Y_trigger, X_trigger / Z_trigger
    return (f1, precision, recall, f1_trigger, precision_trigger, recall_trigger)

if __name__ == '__main__':
    NER_evaluation([[0, 0, 1, 1, 2, 2, 0]], [[0, 4, 4, 4, 1, 1, 0, 0, 0]])