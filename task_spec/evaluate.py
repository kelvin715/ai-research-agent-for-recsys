"""KuaiRand-Pure validation metric used only for candidate checkpoint selection.

The trusted evaluator independently imports the frozen starter-kit implementation and
remains authoritative for Agent decisions. Test labels are absent from the sandbox.
"""
import collections
import math


def auc(labels, scores):
    """Mann-Whitney U with tie correction."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(label for _, label in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)


def ndcg_at_k(labels, k):
    discounts = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** target) - 1) / discounts[i]
              for i, target in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** target) - 1) / discounts[i]
               for i, target in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate(user_ids, labels, scores, k=5):
    by_user = collections.defaultdict(list)
    for user_id, label, score in zip(user_ids, labels, scores):
        by_user[user_id].append((score, label))
    gnum = gden = 0.0
    ndcg = []
    for rows in by_user.values():
        rows.sort(key=lambda item: -item[0])
        user_labels = [label for _, label in rows]
        npos = sum(user_labels)
        if 0 < npos < len(user_labels):
            gnum += npos * auc(user_labels, [score for score, _ in rows])
            gden += npos
        ndcg.append(ndcg_at_k(user_labels, k))
    gauc = gnum / gden if gden else 0.5
    mean_ndcg = sum(ndcg) / len(ndcg) if ndcg else 0.0
    return {'GAUC': gauc, f'nDCG@{k}': mean_ndcg,
            'primary': (gauc + mean_ndcg) / 2.0,
            'users': len(by_user), 'rows': len(labels)}
