#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor add(const Tensor&, const Tensor&);
Tensor matmul(const Tensor&, const Tensor&);
Tensor relu(const Tensor&);
Tensor flatten(const Tensor&);
Tensor linear(const Tensor&, const Tensor&, const Tensor&);
Tensor softmax_cross_entropy(const Tensor&, const Tensor&);
Tensor sigmoid(const Tensor&);

#endif
