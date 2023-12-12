from HSDTP import HSDTP
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as t_f
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

# NOTES:
# -need a positive act func if using Hebbian BCM
# -theta activation for latent inhibition works best with larger batch (200) and layer_sizes ([[784, 1680, 1280, 10]])

dataset = "MNIST"

backwards_training = True
# flag whether to train backwards weights

semi_supervised = False
# flag whether to perform semi_supervised training

parallel = True
# flag whether backwards weights trained alternating or parallel

hebbian = True
# flag whether hebbian updating is performed

one_shot = False
# flag whether to make training set one shot

error_driven = False
# flag whether error-drive (DTP) updating is performed

validation_split = True
# flag whether to split training into training and validation sets

theta = True
# flag whether to use theta activation and threshold updates for latent inhibition

state_dict = None # torch.load("weights/MNIST_pretrain/scarce_10_uniform_lr_0")
# preloaded state_dict, None if none

model_name = "ft_cond_expectation" # None # "uniform_01_lr_001" # "first_moment_001_uniform_000001" # "uniform_1_01"

weight_savename = f"{dataset}_pretrain/{model_name}"
# weights_savename, None if no weights saved

config = {"device": "cpu",
          "batch_size": 200,
          "layer_sizes": [784, 1880, 516, 10],
          "weight_initialization": torch.nn.init.uniform_,
          "forward_activation": torch.nn.Sigmoid(),
          "backward_activation": torch.nn.Sigmoid(),
          "output_activation": torch.nn.Softmax(dim=0),
          "optimizer": torch.optim.Adam,
          "weight_decay": .0001,
          "state_dict": state_dict,
          "forward_loss": torch.nn.MSELoss(),
          "backward_loss": torch.nn.MSELoss(),
          "task_loss": torch.nn.CrossEntropyLoss(),
          "lr_error_driven": .0001,
          "lr_hebbian": .001,
          "lr_threshold": .01, # learning rate of floating threshold
          "corruption": .001,  # amount of noise in training backward weights
          "starting_threshold": .25,  # starting floating threshold
          "scarcity": .1,  # portion of neurons above membrane threshold that fire for a given input
          "theta": theta,
          }

batch_size = config["batch_size"]
hsdtp = HSDTP(config)

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, _ = self.dataset[index]  # ignore label
        return data,

    def __len__(self):
        return len(self.dataset)


class SemiSupervisedDataset(Dataset):
    def __init__(self, dataset, labels, n_labeled):
        self.n_labeled = n_labeled
        self.n_classes = 10
        self.n_samples = len(dataset)

        # Calculate the number of samples per class
        self.n_samples_per_class = self.n_labeled // self.n_classes
        self.data = []
        self.targets = []

        # First, let's make sure classes are balanced
        indices = np.where(np.array(labels) == 0)[0].tolist()
        np.random.shuffle(indices)
        self.data.extend(dataset[i].float() for i in indices[:self.n_samples_per_class])  # Convert to Float tensor
        self.targets.extend([0] * self.n_samples_per_class)

        for i in range(1, self.n_classes):
            indices = np.where(np.array(labels) == i)[0].tolist()
            np.random.shuffle(indices)
            self.data.extend(dataset[i].float() for i in indices[:self.n_samples_per_class])  # Convert to Float tensor
            self.targets.extend([i] * self.n_samples_per_class)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)

def load_dataset(dataset_name):
    if dataset_name == 'MNIST':
        transform = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

        root = './data'
        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)

    else:
        raise ValueError('Invalid dataset name. Choose from "MNIST" or "CIFAR".')

    if semi_supervised:
        n_labeled = len(train_dataset) // 10
        targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.train_labels
        labeled_dataset = SemiSupervisedDataset(train_dataset.data, targets, n_labeled)

        all_indices = set(range(len(train_dataset)))
        labeled_indices = set(range(len(labeled_dataset)))
        unlabeled_indices = list(all_indices - labeled_indices)

        unlabeled_dataset = torch.utils.data.Subset(train_dataset, unlabeled_indices)
        unlabeled_dataset = UnlabeledDataset(unlabeled_dataset)

        # For the training set, use the labeled dataset
        train_dataset = labeled_dataset

    if validation_split:
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
    else:
        train_size = len(train_dataset)
        val_size = 0

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if semi_supervised:
        # Also return the unlabeled dataset and its DataLoader
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
        return train_dataset, train_loader, val_loader, unlabeled_dataset, unlabeled_loader

    return train_dataset, train_loader, val_loader

if semi_supervised:
    train_dataset, train_loader, val_loader, unlabeled_dataset, unlabeled_loader = load_dataset(dataset)
else:
    train_dataset, train_loader, val_loader = load_dataset(dataset)

print(f"loaded {dataset}, beginning training")
if model_name:
    with open(f"data/performance/{model_name}", "a") as file:
        file.write(f"{model_name} training accuracy")

epoch = 0
train_accuracies = []
val_accuracies = []

for epoch in range(501):

    if semi_supervised and hebbian:

        for image in tqdm(unlabeled_loader):
            image = image[0].flatten(start_dim=1).T.squeeze().to(config["device"])
            label_h = hsdtp.forward(image)
            hsdtp.apply_BCM_update()

    if backwards_training and not parallel: # weights are trained alternating
        for step in range(60000 // config["batch_size"]):
           hsdtp.train_backward_weights()

    training_losses = []
    batch = 0
    for image, label in tqdm(train_loader):
        batch += 1

        # Perform your training operations here
        image = image.flatten(start_dim=1).T.squeeze().to(config["device"])
        label = t_f.one_hot(label, num_classes=10).T.float().squeeze()
        label_h = hsdtp.forward(image)

        if error_driven:
            hsdtp.compute_targets(label)
            # avg_pre = torch.mean(hsdtp.forward_weights[0])
            loss = hsdtp.train_forward_weights()
            # avg_post = torch.mean(hsdtp.forward_weights[0])
            # print(f"change from error_driven: {torch.abs(avg_post - avg_pre)}")

        if backwards_training and parallel:
            hsdtp.train_backward_weights()

        if not semi_supervised and hebbian:
            # avg_pre = torch.mean(hsdtp.forward_weights[0])
            hsdtp.apply_BCM_update()
            # avg_post = torch.mean(hsdtp.forward_weights[0])
            # print(f"change from hebbian: {torch.abs(avg_post - avg_pre)}")

    with torch.no_grad():

        val_correct = 0
        val_total = 0
        train_correct = 0
        train_total = 0

        for image, label in train_loader:
            image = image.flatten(start_dim=1).T
            label = t_f.one_hot(label, num_classes=10).t().float().squeeze()
            label_h = hsdtp.forward(image)
            predicted = torch.argmax(label_h.data, 0)
            labels = torch.argmax(label, 0)
            train_total += label.size(1) if batch_size > 1 else 1
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)
        print(f"train_accuracy: {train_accuracy}")

        if model_name:
            with open(f"data/performance/{model_name}", "a") as file:
                file.write(f"\n{round(train_accuracy, 3)}")

        if validation_split:

            for image, label in val_loader:
                image = image.flatten(start_dim=1).T
                label = t_f.one_hot(label, num_classes=10).t().float().squeeze()
                label_h = hsdtp.forward(image)
                predicted = torch.argmax(label_h.data, 0)
                labels = torch.argmax(label, 0)
                val_total += label.size(1) if batch_size > 1 else 1
                val_correct += (predicted == labels).sum().item()

            val_accuracy = val_correct / val_total
            val_accuracies.append(val_accuracy)
            print(f"val_accuracy: {val_accuracy}")

    if epoch % 1 == 0 and weight_savename is not None:
        torch.save(hsdtp.state_dict, f"weights/{weight_savename}_{epoch}")

plt.plot(range(len(train_accuracies)), train_accuracies, label="Train Accuracy")
plt.plot(range(len(val_accuracies)), val_accuracies, label="Val Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("SDTP MNIST Eval")
plt.legend(loc="best")
plt.show()
