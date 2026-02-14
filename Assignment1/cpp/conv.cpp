#include "conv.h"
#include <algorithm>

/* =========================
   Convolution 2D
   ========================= */

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int stride,
    int padding
) {

    int H = input.shape[0];
    int W = input.shape[1];
    int D = input.shape[2];

    int F = weight.shape[0];
    int K = weight.shape[3];

    int H_out = (H - F + 2 * padding) / stride + 1;
    int W_out = (W - F + 2 * padding) / stride + 1;

    Tensor out({H_out, W_out, K},
               input.requires_grad ||
               weight.requires_grad ||
               bias.requires_grad);

    // -------- FORWARD --------
    for (int i = 0; i < H_out; i++)
        for (int j = 0; j < W_out; j++)
            for (int k = 0; k < K; k++) {

                float sum = 0.0f;

                for (int u = 0; u < F; u++)
                    for (int v = 0; v < F; v++)
                        for (int d = 0; d < D; d++) {

                            int in_i =
                                i*stride + u - padding;
                            int in_j =
                                j*stride + v - padding;

                            if (in_i >= 0 && in_i < H &&
                                in_j >= 0 && in_j < W) {

                                int input_idx =
                                    (in_i*W + in_j)*D + d;

                                int w_idx =
                                    ((u*F + v)*D + d)*K + k;

                                sum += (*input.data)[input_idx] *
                                       (*weight.data)[w_idx];
                            }
                        }

                sum += (*bias.data)[k];

                (*out.data)[(i*W_out + j)*K + k] = sum;
            }

    // -------- BACKWARD --------
    if (out.requires_grad) {

        Tensor input_copy  = input;
        Tensor weight_copy = weight;
        Tensor bias_copy   = bias;

        out.backward_fn =
        [input_copy, weight_copy, bias_copy,
         H, W, D, F, K,
         H_out, W_out,
         stride, padding]
        (Tensor& self) mutable {

            // Allocate grads if needed
            if (input_copy.requires_grad &&
                input_copy.grad->empty())
                input_copy.grad->assign(
                    input_copy.numel(), 0.0f);

            if (weight_copy.requires_grad &&
                weight_copy.grad->empty())
                weight_copy.grad->assign(
                    weight_copy.numel(), 0.0f);

            if (bias_copy.requires_grad &&
                bias_copy.grad->empty())
                bias_copy.grad->assign(
                    bias_copy.numel(), 0.0f);

            for (int i = 0; i < H_out; i++)
                for (int j = 0; j < W_out; j++)
                    for (int k = 0; k < K; k++) {

                        float grad_val =
                            (*self.grad)[(i*W_out + j)*K + k];

                        // ---- Bias grad ----
                        if (bias_copy.requires_grad)
                            (*bias_copy.grad)[k] += grad_val;

                        for (int u = 0; u < F; u++)
                            for (int v = 0; v < F; v++)
                                for (int d = 0; d < D; d++) {

                                    int in_i =
                                        i*stride + u - padding;
                                    int in_j =
                                        j*stride + v - padding;

                                    if (in_i >= 0 && in_i < H &&
                                        in_j >= 0 && in_j < W) {

                                        int input_idx =
                                            (in_i*W + in_j)*D + d;

                                        int w_idx =
                                            ((u*F + v)*D + d)*K + k;

                                        // ---- Weight grad ----
                                        if (weight_copy.requires_grad)
                                            (*weight_copy.grad)[w_idx] +=
                                                (*input_copy.data)[input_idx] *
                                                grad_val;

                                        // ---- Input grad ----
                                        if (input_copy.requires_grad)
                                            (*input_copy.grad)[input_idx] +=
                                                (*weight_copy.data)[w_idx] *
                                                grad_val;
                                    }
                                }
                    }

            if (input_copy.backward_fn)
                input_copy.backward_fn(input_copy);

            if (weight_copy.backward_fn)
                weight_copy.backward_fn(weight_copy);

            if (bias_copy.backward_fn)
                bias_copy.backward_fn(bias_copy);
        };
    }

    return out;
}


/* =========================
   Max Pool 2D
   ========================= */

Tensor maxpool2d(
    const Tensor& input,
    int pool_h,
    int pool_w,
    int stride
) {

    int H = input.shape[0];
    int W = input.shape[1];
    int D = input.shape[2];

    int H_out = (H - pool_h) / stride + 1;
    int W_out = (W - pool_w) / stride + 1;

    Tensor out({H_out, W_out, D},
               input.requires_grad);

    std::vector<int> max_index(out.numel());

    // -------- FORWARD --------
    for (int i = 0; i < H_out; i++)
        for (int j = 0; j < W_out; j++)
            for (int d = 0; d < D; d++) {

                float maxv = -1e9f;
                int best = -1;

                for (int ph = 0; ph < pool_h; ph++)
                    for (int pw = 0; pw < pool_w; pw++) {

                        int h = i*stride + ph;
                        int w = j*stride + pw;

                        int idx =
                            (h*W + w)*D + d;

                        if ((*input.data)[idx] > maxv) {
                            maxv = (*input.data)[idx];
                            best = idx;
                        }
                    }

                int out_idx =
                    (i*W_out + j)*D + d;

                (*out.data)[out_idx] = maxv;
                max_index[out_idx] = best;
            }

    // -------- BACKWARD --------
    if (out.requires_grad) {

        Tensor input_copy = input;

        out.backward_fn =
        [input_copy, max_index]
        (Tensor& self) mutable {

            if (input_copy.requires_grad &&
                input_copy.grad->empty())
                input_copy.grad->assign(
                    input_copy.numel(), 0.0f);

            for (int i = 0; i < self.numel(); i++)
                (*input_copy.grad)[max_index[i]] +=
                    (*self.grad)[i];

            if (input_copy.backward_fn)
                input_copy.backward_fn(input_copy);
        };
    }

    return out;
}
