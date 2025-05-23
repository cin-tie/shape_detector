import numpy as np

def cross_entropy_loss(logits, true_labels):
    # logits shape: (batch_size, num_classes)
    # true_labels shape: (batch_size,), integer классы
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    batch_size = logits.shape[0]
    correct_logprobs = -np.log(probs[range(batch_size), true_labels] + 1e-9)
    loss = np.sum(correct_logprobs) / batch_size

    grad = probs.copy()
    grad[range(batch_size), true_labels] -= 1
    grad /= batch_size
    return loss, grad
