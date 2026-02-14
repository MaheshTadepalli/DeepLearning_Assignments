#ifndef CONV_H
#define CONV_H

#include "tensor.h"

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int stride,
    int padding
);

Tensor maxpool2d(
    const Tensor& input,
    int pool_h,
    int pool_w,
    int stride
);

#endif
