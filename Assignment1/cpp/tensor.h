#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cassert>

struct Tensor {

    // Shared storage (very important)
    std::shared_ptr<std::vector<float>> data;
    std::shared_ptr<std::vector<float>> grad;

    std::vector<int> shape;

    bool requires_grad;

    std::function<void(Tensor&)> backward_fn;

    Tensor() : requires_grad(false) {}

    Tensor(const std::vector<int>& shape_,
           bool requires_grad_ = false)
        : shape(shape_),
          requires_grad(requires_grad_) {

        int n = numel();

        data = std::make_shared<std::vector<float>>(n, 0.0f);

        if (requires_grad)
            grad = std::make_shared<std::vector<float>>(n, 0.0f);
        else
            grad = std::make_shared<std::vector<float>>();
    }

    int numel() const {
        if (shape.empty()) return 0;

        return std::accumulate(
            shape.begin(),
            shape.end(),
            1,
            std::multiplies<int>());
    }

    void zero_grad() {
        if (!requires_grad) return;
        std::fill(grad->begin(),
                  grad->end(),
                  0.0f);
    }

    void backward() {
        assert(requires_grad);

        if (grad->empty())
            grad->assign(numel(), 0.0f);

        (*grad)[0] = 1.0f;

        if (backward_fn)
            backward_fn(*this);

        backward_fn = nullptr; // break graph
    }
};

#endif
