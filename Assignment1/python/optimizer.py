class SGD:
    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad:
                p.zero_grad()

    def step(self):
        for p in self.parameters:

            if not p.requires_grad:
                continue

            n = p.numel()

            for i in range(n):

                value = p.get_item(i)
                grad = p.get_grad(i)

                # ----- L2 Regularization -----
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * value

                new_value = value - self.lr * grad
                p.set_item(i, new_value)
