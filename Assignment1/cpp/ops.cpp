#include "ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

/* =========================
   ADD
   ========================= */

Tensor add(const Tensor& A, const Tensor& B) {

    Tensor out(A.shape,
               A.requires_grad || B.requires_grad);

    for (int i = 0; i < A.numel(); i++)
        (*out.data)[i] =
            (*A.data)[i] + (*B.data)[i];

    if (out.requires_grad) {

        Tensor A_copy = A;
        Tensor B_copy = B;

        out.backward_fn =
        [A_copy, B_copy](Tensor& self) mutable {

            if (A_copy.requires_grad) {
                if (A_copy.grad->empty())
                    A_copy.grad->assign(A_copy.numel(), 0.0f);

                for (int i = 0; i < self.numel(); i++)
                    (*A_copy.grad)[i] += (*self.grad)[i];
            }

            if (B_copy.requires_grad) {
                if (B_copy.grad->empty())
                    B_copy.grad->assign(B_copy.numel(), 0.0f);

                for (int i = 0; i < self.numel(); i++)
                    (*B_copy.grad)[i] += (*self.grad)[i];
            }

            if (A_copy.backward_fn)
                A_copy.backward_fn(A_copy);

            if (B_copy.backward_fn)
                B_copy.backward_fn(B_copy);
        };
    }

    return out;
}


/* =========================
   MATMUL
   ========================= */

Tensor matmul(const Tensor& A, const Tensor& B) {

    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    Tensor out({M, N},
               A.requires_grad || B.requires_grad);

    // -------- FORWARD --------
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++)
                sum += (*A.data)[i*K + k] *
                       (*B.data)[k*N + j];

            (*out.data)[i*N + j] = sum;
        }

    // -------- BACKWARD --------
    if (out.requires_grad) {

        Tensor A_copy = A;
        Tensor B_copy = B;

        out.backward_fn =
        [A_copy, B_copy, M, K, N]
        (Tensor& self) mutable {

            // ---- A grad ----
            if (A_copy.requires_grad) {

                if (A_copy.grad->empty())
                    A_copy.grad->assign(
                        A_copy.numel(), 0.0f);

                for (int i = 0; i < M; i++)
                    for (int k = 0; k < K; k++) {

                        float sum = 0.0f;

                        for (int j = 0; j < N; j++)
                            sum += (*self.grad)[i*N + j] *
                                   (*B_copy.data)[k*N + j];

                        (*A_copy.grad)[i*K + k] += sum;
                    }
            }

            // ---- B grad ----
            if (B_copy.requires_grad) {

                if (B_copy.grad->empty())
                    B_copy.grad->assign(
                        B_copy.numel(), 0.0f);

                for (int k = 0; k < K; k++)
                    for (int j = 0; j < N; j++) {

                        float sum = 0.0f;

                        for (int i = 0; i < M; i++)
                            sum += (*A_copy.data)[i*K + k] *
                                   (*self.grad)[i*N + j];

                        (*B_copy.grad)[k*N + j] += sum;
                    }
            }

            if (A_copy.backward_fn)
                A_copy.backward_fn(A_copy);

            if (B_copy.backward_fn)
                B_copy.backward_fn(B_copy);
        };
    }

    return out;
}




/* =========================
   RELU
   ========================= */

Tensor relu(const Tensor& x) {

    Tensor out(x.shape, x.requires_grad);

    for (int i = 0; i < x.numel(); i++)
        (*out.data)[i] =
            (*x.data)[i] > 0 ?
            (*x.data)[i] : 0.0f;

    if (out.requires_grad) {

        Tensor x_copy = x;

        out.backward_fn =
        [x_copy](Tensor& self) mutable {

            if (x_copy.requires_grad) {

                if (x_copy.grad->empty())
                    x_copy.grad->assign(x_copy.numel(), 0.0f);

                for (int i = 0; i < x_copy.numel(); i++)
                    if ((*x_copy.data)[i] > 0)
                        (*x_copy.grad)[i] +=
                            (*self.grad)[i];
            }

            if (x_copy.backward_fn)
                x_copy.backward_fn(x_copy);
        };
    }

    return out;
}


/* =========================
   FLATTEN
   ========================= */

Tensor flatten(const Tensor& x) {

    int total = x.numel();

    Tensor out({1, total}, x.requires_grad);

    // -------- FORWARD --------
    for (int i = 0; i < total; i++)
        (*out.data)[i] = (*x.data)[i];

    // -------- BACKWARD --------
    if (out.requires_grad) {

        Tensor x_copy = x;

        out.backward_fn =
        [x_copy](Tensor& self) mutable {

            if (x_copy.grad->empty())
                x_copy.grad->assign(
                    x_copy.numel(), 0.0f);

            for (int i = 0; i < x_copy.numel(); i++)
                (*x_copy.grad)[i] +=
                    (*self.grad)[i];

            if (x_copy.backward_fn)
                x_copy.backward_fn(x_copy);
        };
    }

    return out;
}




/* =========================
   LINEAR
   ========================= */

Tensor linear(const Tensor& x,
              const Tensor& weight,
              const Tensor& bias) {

    Tensor out = matmul(x, weight);

    for (int j = 0; j < out.shape[1]; j++)
        (*out.data)[j] += (*bias.data)[j];

    if (out.requires_grad) {

        Tensor bias_copy = bias;
        auto prev_backward = out.backward_fn;

        out.backward_fn =
        [bias_copy, prev_backward]
        (Tensor& self) mutable {

            if (bias_copy.requires_grad) {

                if (bias_copy.grad->empty())
                    bias_copy.grad->assign(bias_copy.numel(), 0.0f);

                for (int j = 0; j < bias_copy.numel(); j++)
                    (*bias_copy.grad)[j] +=
                        (*self.grad)[j];
            }

            if (prev_backward)
                prev_backward(self);
        };
    }

    return out;
}


/* =========================
   SOFTMAX + CE
   ========================= */

Tensor softmax_cross_entropy(
    const Tensor& logits,
    const Tensor& target) {

    int N = logits.shape[0];
    int C = logits.shape[1];

    Tensor out({1}, logits.requires_grad);

    std::vector<float> softmax_vals(N*C);

    (*out.data)[0] = 0.0f;

    for (int i = 0; i < N; i++) {

        float maxv = -1e9f;

        for (int j = 0; j < C; j++)
            maxv = std::max(maxv,
                            (*logits.data)[i*C + j]);

        float sum = 0.0f;

        for (int j = 0; j < C; j++) {
            softmax_vals[i*C + j] =
                std::exp((*logits.data)[i*C + j] - maxv);
            sum += softmax_vals[i*C + j];
        }

        for (int j = 0; j < C; j++) {

            softmax_vals[i*C + j] /= sum;

            if ((*target.data)[i*C + j] > 0)
                (*out.data)[0] -=
                    std::log(softmax_vals[i*C + j] + 1e-9f);
        }
    }

    (*out.data)[0] /= N;

    Tensor logits_copy = logits;

    out.backward_fn =
    [logits_copy, softmax_vals, target, N, C]
    (Tensor& self) mutable {

        if (logits_copy.grad->empty())
            logits_copy.grad->assign(logits_copy.numel(), 0.0f);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < C; j++)
                (*logits_copy.grad)[i*C + j] +=
                    (softmax_vals[i*C + j]
                     - (*target.data)[i*C + j]) / N;

        if (logits_copy.backward_fn)
            logits_copy.backward_fn(logits_copy);
    };

    return out;
}


/* =========================
   SIGMOID
   ========================= */

Tensor sigmoid(const Tensor& x) {

    Tensor out(x.shape, x.requires_grad);

    for (int i = 0; i < x.numel(); i++)
        (*out.data)[i] =
            1.0f /
            (1.0f + std::exp(-(*x.data)[i]));

    if (out.requires_grad) {

        Tensor x_copy = x;

        out.backward_fn =
        [x_copy](Tensor& self) mutable {

            if (x_copy.requires_grad) {

                if (x_copy.grad->empty())
                    x_copy.grad->assign(x_copy.numel(), 0.0f);

                for (int i = 0; i < x_copy.numel(); i++) {

                    float s =
                        1.0f /
                        (1.0f + std::exp(-(*x_copy.data)[i]));

                    (*x_copy.grad)[i] +=
                        (*self.grad)[i] *
                        s * (1.0f - s);
                }
            }

            if (x_copy.backward_fn)
                x_copy.backward_fn(x_copy);
        };
    }

    return out;
}
