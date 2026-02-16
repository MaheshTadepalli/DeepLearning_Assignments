import random
import mydl_cpp


class SimpleCNN:

    def __init__(self, num_classes=10):

        self.num_classes = num_classes

        # -------- Conv1 --------
        self.conv1_weight = mydl_cpp.Tensor(
            [3, 3, 3, 64], True
        )
        self.conv1_bias = mydl_cpp.Tensor(
            [64], True
        )

        # -------- Conv2 --------
        self.conv2_weight = mydl_cpp.Tensor(
            [3, 3, 64, 64], True
        )
        self.conv2_bias = mydl_cpp.Tensor(
            [64], True
        )

        # -------- Fully Connected --------
        self.fc_weight = mydl_cpp.Tensor(
            [3 * 3 * 64, num_classes], True
        )
        self.fc_bias = mydl_cpp.Tensor(
            [num_classes], True
        )

        for p in self.parameters():
            self._init_tensor(p)

    def _init_tensor(self, tensor):
        for i in range(tensor.numel()):
            tensor.set_item(i,
                            random.uniform(-0.05, 0.05))

    def forward(self, x):

        # Conv1
        x = mydl_cpp.conv2d(
            x,
            self.conv1_weight,
            self.conv1_bias,
            stride=1,
            padding=0
        )
        x = mydl_cpp.relu(x)
        x = mydl_cpp.maxpool2d(x, 2, 2, 2)

        # Conv2
        x = mydl_cpp.conv2d(
            x,
            self.conv2_weight,
            self.conv2_bias,
            stride=1,
            padding=0
        )
        x = mydl_cpp.relu(x)

        # Two pools
        x = mydl_cpp.maxpool2d(x, 2, 2, 2)
        x = mydl_cpp.maxpool2d(x, 2, 2, 2)

        # Flatten
        x = mydl_cpp.flatten(x)

        # FC
        x = mydl_cpp.linear(
            x,
            self.fc_weight,
            self.fc_bias
        )

        return x

    def parameters(self):
        return [
            self.conv1_weight,
            self.conv1_bias,
            self.conv2_weight,
            self.conv2_bias,
            self.fc_weight,
            self.fc_bias
        ]

    def count_parameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total

    def compute_macs(self):

        conv1 = 30 * 30 * 16 * (3 * 3 * 3)
        conv2 = 13 * 13 * 32 * (3 * 3 * 16)
        fc    = 3 * 3 * 32 * self.num_classes

        return conv1 + conv2 + fc

    def compute_flops(self):
        return 2 * self.compute_macs()
