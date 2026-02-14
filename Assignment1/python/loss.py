import mydl_cpp


def cross_entropy_loss(logits, target_label):
    """
    logits: Tensor of shape (1, C)
    target_label: int
    """

    C = logits.shape[1]

    # Create one-hot target (1, C)
    target = mydl_cpp.Tensor([1, C], False)

    for j in range(C):
        value = 1.0 if j == target_label else 0.0
        target.set_item(j, value)

    # Use fused softmax + cross entropy
    loss = mydl_cpp.softmax_cross_entropy(
        logits,
        target
    )

    return loss


def accuracy(logits, target_label):
    """
    logits: (1, C)
    """

    C = logits.shape[1]

    # manual softmax (no gradient needed)
    max_val = -1e9
    for j in range(C):
        val = logits.get_item(j)
        if val > max_val:
            max_val = val

    exp_vals = []
    sum_exp = 0.0

    for j in range(C):
        val = logits.get_item(j)
        e = pow(2.718281828, val - max_val)
        exp_vals.append(e)
        sum_exp += e

    probs = [e / sum_exp for e in exp_vals]

    pred = 0
    best = probs[0]

    for j in range(1, C):
        if probs[j] > best:
            best = probs[j]
            pred = j

    return 1 if pred == target_label else 0
