import torch
from torch.utils.data import TensorDataset as TorchTensorDataset, DataLoader
from torchnet.dataset import TensorDataset as TorchnetTensorDataset, ResampleDataset
from torchvision import datasets, transforms

class convert_tensor:
    def __call__(self, image):
        # Convert image to tensor 
        image = transforms.ToTensor()(image)

        return image

def load_data_from_path(path, batch_size=10000, shuffle=True):
    # Use the custom transform
    custom_transform = convert_tensor()

    # Load MNIST datasets with the custom transform
    t1 = DataLoader(datasets.MNIST(path, train=True, download=False, transform=custom_transform), batch_size=batch_size, shuffle=shuffle)
    s1 = DataLoader(datasets.MNIST(path, train=False, download=False, transform=custom_transform), batch_size=batch_size, shuffle=shuffle)

    # Load SVHN datasets with the custom transform
    t2 = DataLoader(datasets.SVHN(path, split='train', download=False, transform=custom_transform), batch_size=batch_size, shuffle=shuffle)
    s2 = DataLoader(datasets.SVHN(path, split='test', download=False, transform=custom_transform), batch_size=batch_size, shuffle=shuffle)

    # Get transformed indices
    t_mnist = torch.load(path + '/train-ms-mnist-idx.pt', weights_only=True)
    t_svhn = torch.load(path + '/train-ms-svhn-idx.pt', weights_only=True)
    s_mnist = torch.load(path + '/test-ms-mnist-idx.pt', weights_only=True)
    s_svhn = torch.load(path + '/test-ms-svhn-idx.pt', weights_only=True)

    # Create resampled datasets
    train_mnist_svhn = TorchnetTensorDataset([
        ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
        ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))
    ])
    test_mnist_svhn = TorchnetTensorDataset([
        ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
        ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))
    ])

    return train_mnist_svhn, test_mnist_svhn


def load_svhn_mnist(data_path, train_batches = 450, val_batches = 90, test_batches = 150, batch_size=10000, shuffle=True):
    train_mnist_svhn, test_mnist_svhn = load_data_from_path(data_path, batch_size, shuffle=shuffle)
    mega_train_loader = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle)
    mega_test_loader = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle)
    train_data_x_list = []
    train_data_y_list = []
    train_labels_list = []

    val_data_x_list = []
    val_data_y_list = []
    val_labels_list = []

    rgb_to_gray = torch.tensor([0.2989, 0.5870, 0.1140])
    rgb_to_gray = rgb_to_gray.view(1, 3, 1)

    b=0
    for (mnist_data, labels_mnist), (svhn_data, labels_svhn) in mega_train_loader:
            b+=1
            reshaped_images = svhn_data.view(batch_size, 3, 1024)
            svhn_data = torch.sum(reshaped_images * rgb_to_gray, dim=1).view(batch_size, 1, 32,32)

            train_data_x_list.extend(svhn_data)
            train_data_y_list.extend(mnist_data)
            train_labels_list.extend(labels_mnist)

            if b == train_batches :
                    break

    test_data_x_list = []
    test_data_y_list = []
    test_labels_list = []

    b=0
    for (mnist_data, labels_mnist), (svhn_data, labels_svhn) in mega_test_loader:
            b+=1
            reshaped_images = svhn_data.view(batch_size, 3, 1024)
            svhn_data = torch.sum(reshaped_images * rgb_to_gray, dim=1).view(batch_size, 1, 32,32)

            if b <= test_batches : 
                    test_data_x_list.extend(svhn_data)
                    test_data_y_list.extend(mnist_data)
                    test_labels_list.extend(labels_mnist)
            elif b > test_batches and b <= test_batches + val_batches : 
                    val_data_x_list.extend(svhn_data)
                    val_data_y_list.extend(mnist_data)
                    val_labels_list.extend(labels_mnist)
            else : 
                    break 

    train_data_x = torch.stack(train_data_x_list)
    train_data_y = torch.stack(train_data_y_list)
    train_labels = torch.stack(train_labels_list)
    val_data_x = torch.stack(val_data_x_list)
    val_data_y = torch.stack(val_data_y_list)
    val_labels = torch.stack(val_labels_list)
    test_data_x = torch.stack(test_data_x_list)
    test_data_y = torch.stack(test_data_y_list)
    test_labels = torch.stack(test_labels_list)

    train_dataset = TorchTensorDataset(train_data_x, train_data_y, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    val_dataset = TorchTensorDataset(val_data_x, val_data_y, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    test_dataset = TorchTensorDataset(test_data_x, test_data_y, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return train_loader, val_loader, test_loader