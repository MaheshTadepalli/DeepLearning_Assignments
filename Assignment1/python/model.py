import random
import mydl_cpp


class SimpleCNN:

    def __init__(self, num_classes=10):

        # -------- Architecture --------
        # Input: 32x32x3
        # Conv: 3x3, 3 -> 8
        # Output: 30x30x8
        # Pool: 2x2
        # Output: 15x15x8
        # Flatten: 1800
        # FC: 1800 -> num_classes

        self.num_classes = num_classes

        # ----- Conv Layer -----
        self.conv_weight = mydl_cpp.Tensor(
            [3, 3, 3, 8], True
        )
        self.conv_bias = mydl_cpp.Tensor(
            [8], True
        )

        # Initialize small random weights
        self._init_tensor(self.conv_weight)
        self._init_tensor(self.conv_bias)

        # ----- Fully Connected -----
        self.fc_weight = mydl_cpp.Tensor(
            [15 * 15 * 8, num_classes], True
        )
        self.fc_bias = mydl_cpp.Tensor(
            [num_classes], True
        )

        self._init_tensor(self.fc_weight)
        self._init_tensor(self.fc_bias)


    def _init_tensor(self, tensor):
        for i in range(tensor.numel()):
            tensor.set_item(i,
                            random.uniform(-0.05, 0.05))


    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x):

        x = mydl_cpp.conv2d(
            x,
            self.conv_weight,
            self.conv_bias,
            stride=1,
            padding=0
        )

        x = mydl_cpp.relu(x)

        x = mydl_cpp.maxpool2d(
            x,
            pool_h=2,
            pool_w=2,
            stride=2
        )

        x = mydl_cpp.flatten(x)

        x = mydl_cpp.linear(
            x,
            self.fc_weight,
            self.fc_bias
        )

        return x


    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    def parameters(self):
        return [
            self.conv_weight,
            self.conv_bias,
            self.fc_weight,
            self.fc_bias
        ]


    # -------------------------------------------------
    # Parameter Count
    # -------------------------------------------------
    def count_parameters(self):

        total = 0

        for p in self.parameters():
            total += p.numel()

        return total


    # -------------------------------------------------
    # MACs
    # -------------------------------------------------
    def compute_macs(self):

        # Conv MAC
        # 30x30x8 * (3x3x3)
        conv_macs = (
            30 * 30 * 8 *
            (3 * 3 * 3)
        )

        # FC MAC
        fc_macs = (
            15 * 15 * 8 *
            self.num_classes
        )

        return conv_macs + fc_macs


    # -------------------------------------------------
    # FLOPs
    # -------------------------------------------------
    def compute_flops(self):
        return 2 * self.compute_macs()
