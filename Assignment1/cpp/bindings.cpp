#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "ops.h"
#include "conv.h"

namespace py = pybind11;

PYBIND11_MODULE(mydl_cpp, m) {

    m.doc() = "Minimal Deep Learning Framework (C++ backend)";

    /* =========================
       Tensor
       ========================= */

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, bool>())

        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad)

        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward)
        .def("numel", &Tensor::numel)

        // ---- DATA ACCESS ----
        .def("set_item",
             [](Tensor &t, int idx, float value) {
                 (*t.data)[idx] = value;
             })

        .def("get_item",
             [](Tensor &t, int idx) {
                 return (*t.data)[idx];
             })

        // ---- GRAD ACCESS ----
        .def("set_grad",
             [](Tensor &t, int idx, float value) {
                 if (t.grad->empty())
                     t.grad->assign(t.numel(), 0.0f);
                 (*t.grad)[idx] = value;
             })

        .def("get_grad",
             [](Tensor &t, int idx) {
                 return (*t.grad)[idx];
             });


    /* =========================
       Core Ops
       ========================= */

    m.def("matmul", &matmul);
    m.def("add", &add);
    m.def("flatten", &flatten);
    m.def("linear", &linear);

    m.def("relu", &relu);
    m.def("sigmoid", &sigmoid);


    /* =========================
       CNN Ops
       ========================= */

    m.def("conv2d", &conv2d,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"));

    m.def("maxpool2d", &maxpool2d,
          py::arg("input"),
          py::arg("pool_h"),
          py::arg("pool_w"),
          py::arg("stride"));


    /* =========================
       Loss
       ========================= */

    m.def("softmax_cross_entropy",
          &softmax_cross_entropy,
          py::arg("logits"),
          py::arg("target"));
}
