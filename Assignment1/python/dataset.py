import os
import time
import cv2
import mydl_cpp


class ImageDataset:
    """
    Image folder dataset loader.

    Structure:
        root/
            class0/
            class1/
            ...

    Images:
        PNG only
        Resized to 32x32
        Stored in memory once
    """

    def __init__(self, root_dir, image_size=32):
        self.root_dir = root_dir
        self.image_size = image_size

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self.images = []

        self.load_time = 0.0

        self._load_dataset()


    def _load_dataset(self):

        start = time.time()

        class_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        for idx, cls in enumerate(class_names):
            self.class_to_idx[cls] = idx
        self.num_classes = len(self.class_to_idx)

        for cls in class_names:
            cls_path = os.path.join(self.root_dir, cls)
            label = self.class_to_idx[cls]

            for fname in os.listdir(cls_path):
                if fname.lower().endswith(".png"):

                    path = os.path.join(cls_path, fname)

                    self.image_paths.append(path)
                    self.labels.append(label)

        # -------- PRELOAD INTO MEMORY --------
        for path in self.image_paths:
            tensor = self._load_image_as_tensor(path)
            self.images.append(tensor)

        end = time.time()

        self.load_time = end - start


    def __len__(self):
        return len(self.images)


    def get_load_time(self):
        return self.load_time


    def _load_image_as_tensor(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(
                f"Failed to load image: {img_path}"
            )

        img = cv2.resize(
            img,
            (self.image_size, self.image_size)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        H, W, C = img.shape

        tensor = mydl_cpp.Tensor(
            [H, W, C],
            False
        )

        idx = 0
        for i in range(H):
            for j in range(W):
                for c in range(C):
                    tensor.set_item(
                        idx,
                        float(img[i, j, c]) / 255.0
                    )
                    idx += 1

        return tensor


    def get_batch(self, batch_size, start_idx):

        end_idx = min(
            start_idx + batch_size,
            len(self)
        )

        images = self.images[start_idx:end_idx]
        labels = self.labels[start_idx:end_idx]

        return images, labels

    def get_batch_by_indices(self, indices):

        images = []
        labels = []

        for idx in indices:
            images.append(self.images[idx])
            labels.append(self.labels[idx])

        return images, labels

