import numpy as np
from scipy.stats import rankdata
from utils.Constants import PAD, EOS
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
		This function computes the average prescision at k between two lists of
		items.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(y_prob, y_true, k):
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y_true]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def hits_k(y_prob, y_true, k):
    acc = []
    for p_, y_ in zip(y_prob, y_true):
        top_k = np.argsort(p_)[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return sum(acc) / len(acc)

def mean_rank(y_prob, y):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y):
        ranks += [n_classes - rankdata(p_, method='max')[y_]]
    return sum(ranks) / float(len(ranks))

def MRR(y_prob, y_true):
    ranks = []
    n_classes = y_prob.shape[1]
    for p_, y_ in zip(y_prob, y_true):
        ranks += [1 / (n_classes - rankdata(p_, method='max')[y_] + 1)]
    sum_ranks = sum(ranks)
    if type(sum_ranks) == 'list':
        sum_ranks = sum_ranks[0]
    return sum_ranks / float(len(ranks))

def compute_metrics(y_prob, y_true, k_list=[10,50,100]):
    """
    y_prob: (#samples, #users), y_true: (#samples,)
    """
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    
    y_prob_simp, y_true_simp = [], []
    for i in range(y_true.shape[0]): # predict counts
        if y_true[i]!=PAD and y_true[i]!=EOS:
            y_prob_simp.append(y_prob[i])
            y_true_simp.append(y_true[i])
    y_prob_simp = np.array(y_prob_simp)
    y_true_simp = np.array(y_true_simp)
    
    scores = {}

    # # Calculate F1
    # y_pred_simp = y_prob_simp.argmax(axis=1)
    # _, _, f1, _ = precision_recall_fscore_support(y_true_simp, y_pred_simp, average="micro")
    # scores['F1'] = f1

    # # Calculate AUC
    # # y_true_onehot = np.eye(y_prob.shape[1])[y_true_simp]
    # y_prob_norm = np.exp(y_prob_simp) / np.sum(np.exp(y_prob_simp), axis=1, keepdims=True)
    # auc = roc_auc_score(y_true_simp, y_prob_norm, multi_class='ovr')
    # scores['AUC'] = auc

    # Calculate Rank-Series Metrics, i.e. MRR, Hits@N, Map@N
    # n_users = 4973
    # y_pred_f1 = np.zeros(shape=(n_users), dtype=np.int32)
    # y_pred_f1[y_prob_simp.argmax(axis=1)] = 1
    # y_true_f1 = np.zeros(shape=(n_users), dtype=np.int32)
    # y_true_f1[y_true_simp] = 1
    # scores['prec'] = precision_score(y_true_f1, y_pred_f1, average='binary')
    # scores['rec'] = recall_score(y_true_f1, y_pred_f1, average='binary')
    # scores['F1'] = f1_score(y_true_f1, y_pred_f1, average='binary')
    scores['MRR'] = MRR(y_prob_simp, y_true_simp)
    for k in k_list:
        scores['hits@' + str(k)] = hits_k(y_prob_simp, y_true_simp, k=k)
        scores['map@' + str(k)] = mapk(y_prob_simp, y_true_simp, k=k)
    return scores
