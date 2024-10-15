import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10, STL10


def get_dataset(args):
    """
    Returns vanilla CIFAR10/STL10 dataset (modified with train test splitting)
    """
    transform = transforms.Compose(
        [
            # transforms.Resize(32 if args.dataset == "cifar10" else 64),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if hasattr(args, "eval"):
        if args.eval:
            transform = transform_test

    if args.dataset == "cifar10":
        train_dataset = CIFAR10(
            args.data_path,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = CIFAR10(
            args.data_path,
            train=False,
            download=True,
            transform=transform_test,
        )

    elif args.dataset == "stl10":
        # for STL10 use both train and test sets due to its small size
        # the original implementation is buggy
        # data augmentation should not be used for test set
        dataset_00 = STL10(
            args.data_path,
            split="train",
            download=True,
            transform=transform,
        )
        dataset_01 = STL10(
            args.data_path,
            split="train",
            download=True,
            transform=transform_test,
        )
        dataset_10 = STL10(
            args.data_path,
            split="test",
            download=True,
            transform=transform,
        )
        dataset_11 = STL10(
            args.data_path,
            split="test",
            download=True,
            transform=transform_test,
        )
        split = torch.randperm(13000, generator=torch.Generator().manual_seed(1234))
        train_split, test_split = split[:12000], split[12000:]
        train_split0, train_split1 = train_split[train_split < 5000], train_split[train_split >= 5000] - 5000
        test_split0, test_split1 = test_split[test_split < 5000], test_split[test_split >= 5000] - 5000
        train_dataset = ConcatDataset([Subset(dataset_00, train_split0), Subset(dataset_10, train_split1)])
        test_dataset = ConcatDataset([Subset(dataset_01, test_split0), Subset(dataset_11, test_split1)])

        # dataset = ConcatDataset([train_dataset, test_dataset])
        # train_dataset, test_dataset = torch.utils.data.random_split(
        #     dataset, [12000, 1000], generator=torch.Generator().manual_seed(1234)
        # )

    shuffle = False if args.eval else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )

    return train_loader, test_loader


def get_dataset_image_folder(args):
    transform = transforms.Compose(
        [
            # transforms.Resize(32 if args.dataset == "cifar10" else 64),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(args.data_path, transform=transform)
    print(len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "stl10"], help="Dataset type")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to dataset"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--freeze_layers",
        type=bool,
        default=False,
        help="Freeze the convolution layers or not",
    )
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = torchvision.models.resnet34(pretrained=True)

    # Freeze all layers except the final layer
    if args.freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    torch.nn.init.xavier_uniform_(model.fc.weight)

    if args.eval:
        try:
            state_dict = torch.load(f"{args.dataset}_resnet34.pth", map_location="cpu")
        except FileNotFoundError:
            exit(1)
        model.load_state_dict(state_dict)
        model.requires_grad_(False).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, test_loader = get_dataset(args)

    if args.eval:
        from sklearn.metrics import confusion_matrix

        total = 0
        correct = 0
        y_true = []
        y_pred = []
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())
        print(
            "Accuracy on the train set: %.2f %%"
            % (100 * correct / total)
        )
        print(
            "Confusion matrix on the train set: %s"
            % repr(confusion_matrix(y_true, y_pred, labels=list(range(10))))
        )
        total = 0
        correct = 0
        y_true = []
        y_pred = []
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())
        print(
            "Accuracy on the test set: %.2f %%"
            % (100 * correct / total)
        )
        print(
            "Confusion matrix on the test set: %s"
            % repr(confusion_matrix(y_true, y_pred, labels=list(range(10))))
        )
        exit(0)

    criterion = nn.CrossEntropyLoss()
    weight_decay = 5e-4

    # params_1x are the parameters of the network body, i.e., of all layers except the FC layers
    params_1x = [
        param for name, param in model.named_parameters() if "fc" not in str(name)
    ]
    optimizer = torch.optim.Adam(
        [{"params": params_1x}, {"params": model.fc.parameters(), "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=weight_decay,
    )

    # Train the model
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and labels
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print the accuracy on the test set
        print(
            "Accuracy on the test set after epoch %d: %d %%"
            % (epoch + 1, 100 * correct / total)
        )

    print("Finished fine-tuning")
    torch.save(model.state_dict(), f"{args.dataset}_resnet34.pth")

# CUDA_VISIBLE_DEVICES="1" python train_classifier.py --dataset cifar10 --eval
# Accuracy on the train set: 99.96 %
# Confusion matrix on the train set: array([[5000,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [   0, 4998,    0,    0,    0,    0,    0,    0,    0,    2],
#        [   1,    0, 4997,    0,    0,    0,    1,    0,    0,    1],
#        [   0,    0,    1, 4991,    1,    7,    0,    0,    0,    0],
#        [   0,    0,    1,    1, 4996,    0,    1,    1,    0,    0],
#        [   0,    0,    0,    2,    0, 4998,    0,    0,    0,    0],
#        [   0,    0,    0,    0,    0,    0, 5000,    0,    0,    0],
#        [   0,    0,    0,    0,    1,    0,    0, 4999,    0,    0],
#        [   0,    0,    0,    0,    0,    0,    0,    0, 5000,    0],
#        [   0,    0,    0,    0,    0,    0,    0,    0,    0, 5000]])
# Accuracy on the test set: 95.03 %
# Confusion matrix on the test set: array([[966,   1,   6,   3,   0,   0,   2,   3,  14,   5],
#        [  4, 971,   0,   0,   0,   0,   1,   1,   4,  19],
#        [  8,   0, 937,  18,  15,   8,   7,   4,   2,   1],
#        [  3,   1,  13, 883,  15,  62,   8,   8,   3,   4],
#        [  2,   0,   5,  10, 948,   7,   8,  20,   0,   0],
#        [  0,   0,   4,  52,  10, 925,   2,   6,   0,   1],
#        [  1,   0,   6,   9,   0,   3, 981,   0,   0,   0],
#        [  5,   0,   3,   7,  16,  13,   2, 951,   0,   3],
#        [ 16,   1,   1,   3,   0,   0,   1,   0, 973,   5],
#        [  4,  22,   1,   1,   0,   0,   0,   0,   4, 968]])
# Test Recall of Class 0: 96.60%

# CUDA_VISIBLE_DEVICES="1" python train_classifier.py --dataset stl10 --eval
# Accuracy on the train set: 100.00 %
# Confusion matrix on the train set: array([[1192,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#        [   0, 1195,    0,    0,    0,    0,    0,    0,    0,    0],
#        [   0,    0, 1203,    0,    0,    0,    0,    0,    0,    0],
#        [   0,    0,    0, 1187,    0,    0,    0,    0,    0,    0],
#        [   0,    0,    0,    0, 1193,    0,    0,    0,    0,    0],
#        [   0,    0,    0,    0,    0, 1217,    0,    0,    0,    0],
#        [   0,    0,    0,    0,    0,    0, 1226,    0,    0,    0],
#        [   0,    0,    0,    0,    0,    0,    0, 1194,    0,    0],
#        [   0,    0,    0,    0,    0,    0,    0,    0, 1188,    0],
#        [   0,    0,    0,    0,    0,    0,    0,    0,    0, 1205]])
# Accuracy on the test set: 96.20 %
# Confusion matrix on the test set: array([[106,   0,   1,   0,   0,   0,   0,   0,   0,   1],
#        [  0, 101,   0,   1,   1,   1,   0,   1,   0,   0],
#        [  0,   1,  94,   0,   0,   0,   0,   0,   0,   2],
#        [  0,   1,   0, 106,   0,   4,   0,   2,   0,   0],
#        [  0,   0,   0,   3, 104,   0,   0,   0,   0,   0],
#        [  0,   0,   0,   4,   1,  76,   2,   0,   0,   0],
#        [  0,   0,   1,   0,   4,   1,  68,   0,   0,   0],
#        [  0,   0,   0,   1,   0,   0,   0, 105,   0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0, 112,   0],
#        [  3,   0,   1,   0,   0,   0,   0,   0,   1,  90]])
# Test Recall of Class 0: 106 / 108 = 98.15%

# model = [nn.Conv2d(12000, 1, 1, 1, 0).to(f"cuda:{i}") for i in [0, 1, 2, 4]]
# while True:
#     for m, xx in zip(model, x):
#         _ = m(xx)