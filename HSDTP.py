# Hebbian Simplified Difference Target Propogation

import torch.nn

"""
Notes: 
    -how do we handle batch training with the membrane potential/floating_threshold?

example config:
config = {"device": "cpu",
          "batch_size": 50,
          "layer_sizes": [784, 258, 16, 10],
          "forward_activation": torch.nn.Tanh(),
          "backward_activation": torch.nn.Tanh(),
          "output_activation": torch.nn.Softmax(dim=0),
          "optimizer": torch.optim.Adam,
          "lr_error_driven": .0001,
          "lr_hebbian": .0001,
          "forward_loss": torch.nn.MSELoss(),
          "backward_loss": torch.nn.MSELoss(),
          "task_loss": torch.nn.CrossEntropyLoss(),
          "forward_weights": None,
          "backward_weights": None,
          "corruption": .001,
          "starting_threshold": .25
          }  
"""

class HSDTP (torch.nn.Module):

    def __init__(self, config):
        super(HSDTP, self).__init__()
        self.config = config
        self.build_module()

    def build_module(self):

        layer_sizes = self.config["layer_sizes"]
        self.layers = torch.nn.ModuleList()
        for l in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[l], layer_sizes[l+1], bias=False))

        self.activations = [torch.zeros(size) for size in layer_sizes]
        self.targets = [torch.zeros(size) for size in layer_sizes]

        if self.config["state_dict"] and self.config["state_dict"]["forward_weights"]:
            self.forward_weights = self.config["state_dict"]["forward_weights"]
        else:
            self.forward_weights = [self.config["weight_initialization"]
                                    (layer.weight.data.to(self.config["device"]),  a=-.1, b=.1)
                                    for layer in self.layers]

        if self.config["state_dict"] and self.config["state_dict"]["backward_weights"]:
            self.backward_weights = self.config["state_dict"]["backward_weights"]
        else:
            self.backward_weights = [torch.randn(forward_weight.shape).to(self.config["device"])
                                     for forward_weight in self.forward_weights]
            self.backward_weights = [backward_weight.t().detach() for backward_weight in self.backward_weights]

        if self.config["state_dict"] and self.config["state_dict"]["floating_threshold"]:
            self.floating_threshold = self.config["state_dict"]["floating_threshold"]
        else:
            self.floating_threshold = [torch.ones(layer_shape).to(self.config["device"]) *
                                       self.config["starting_threshold"]
                                       for layer_shape in self.config["layer_sizes"][1:]]

        if self.config["state_dict"] and self.config["state_dict"]["membrane_potentials"]:
            self.membrane_potentials = self.config["state_dict"]["membrane_potentials"]
        else:
            self.membrane_potentials = [torch.zeros(layer_shape).to(self.config["device"])
                                       for layer_shape in self.config["layer_sizes"][1:]]

        self.state_dict = {
            'forward_weights': self.forward_weights,
            'backward_weights': self.backward_weights,
            'floating_threshold': self.floating_threshold,
            'membrane_potentials': self.membrane_potentials
        }

        self.forward_optimizer = self.config["optimizer"](params=self.forward_weights, lr=self.config["lr_error_driven"])
        self.backward_optimizer = self.config["optimizer"](params=self.backward_weights, lr=self.config["lr_error_driven"])

    def forward(self, x):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                self.activations[i] = x
                x = torch.matmul(self.forward_weights[i], x)
                if i < len(self.layers) - 1:
                    x = self.config["forward_activation"](x)

                if self.config["theta"]:

                    """
                    activation_mean = torch.mean(x)
                    threshold_mean = torch.mean(self.floating_threshold[i])
                    latent_inhibition = activation_mean - threshold_mean
                    self.membrane_potentials[i] = self.floating_threshold[i] + latent_inhibition

                    # this expands the potentials in batch dim for batch training
                    batched_potentials = self.membrane_potentials[i].unsqueeze(1)
                    batched_potentials = batched_potentials.repeat(1, self.config["batch_size"])

                    activation = x - batched_potentials
                    x = torch.where(activation > 0, x, torch.zeros_like(x))
                    """

                    activation_mean = torch.mean(x)
                    threshold_mean = torch.mean(self.floating_threshold[i])

                    # latent inhibition not needed with Kwta
                    # latent_inhibition = activation_mean - threshold_mean

                    self.membrane_potentials[i] = self.floating_threshold[i] # + latent_inhibition

                    # this expands the potentials in batch dim for batch training
                    batched_potentials = self.membrane_potentials[i].unsqueeze(1)
                    batched_potentials = batched_potentials.repeat(1, self.config["batch_size"])

                    activation = x - batched_potentials

                    # Calculate the number of neurons to activate based on scarcity
                    k = int(self.config["scarcity"] * activation.shape[0])

                    # Only retain top-k activations
                    topk_activations, _ = torch.topk(activation, k, dim=0)

                    # Create a mask of the same shape as activation, with True values for the indices of top-k activations
                    mask = activation >= topk_activations[-1]
                    x = torch.where(mask, x, torch.zeros_like(x))

                    # print(x)

            self.activations[-1] = self.config["output_activation"](x)

            if self.config["theta"]:
                self.update_thresholds()

            return self.activations[-1]

    def update_thresholds(self):

        # this computes the floating threshold as P(fires)
        """
        for i in range(len(self.floating_threshold)):
            fired = self.activations[i+1].gt(0).float().detach()
            batched_threshold = self.floating_threshold[i].unsqueeze(1)
            batched_threshold = batched_threshold.repeat(1, self.config["batch_size"])
            threshold = torch.sum(batched_threshold * (1 - self.config["lr_threshold"])
                                  + fired * self.config["lr_threshold"], dim=1)/self.config["batch_size"]
            self.floating_threshold[i] = threshold
        """

        """
        # this computes the floating threshold as E[a]
        for i in range(len(self.floating_threshold)):
            fired = self.activations[i+1].gt(0).float().detach()
            batched_threshold = self.floating_threshold[i].unsqueeze(1)
            batched_threshold = batched_threshold.repeat(1, self.config["batch_size"])
            threshold = torch.sum(batched_threshold * (1 - self.config["lr_threshold"])
                                  + fired * self.activations[i+1] * self.config["lr_threshold"], dim=1)/self.config["batch_size"]
            self.floating_threshold[i] = threshold

        # this computes the floating threshold as E[a^2]
        for i in range(len(self.floating_threshold)):
            fired = self.activations[i+1].gt(0).float().detach()
            batched_threshold = self.floating_threshold[i].unsqueeze(1)
            batched_threshold = batched_threshold.repeat(1, self.config["batch_size"])
            threshold = torch.sum(batched_threshold * (1 - self.config["lr_threshold"])
                                  + fired * torch.square(self.activations[i+1]) * self.config["lr_threshold"], dim=1)/self.config["batch_size"]
            self.floating_threshold[i] = threshold
        """

        # this computes the floating threshold as E[a|fired]
        for i in range(len(self.floating_threshold)):
            fired = self.activations[i+1].gt(0).float().detach()
            not_fired = (self.activations[i+1] == 0).float().detach()
            batched_threshold = self.floating_threshold[i].unsqueeze(1)
            batched_threshold = batched_threshold.repeat(1, self.config["batch_size"])
            not_fired_threshold = torch.sum(not_fired * batched_threshold)/self.config["batch_size"]
            fired_threshold = torch.sum(fired * ((1 - self.config["lr_threshold"]) * batched_threshold +
                                                 self.config["lr_threshold"] * self.activations[i+1]), dim=1)/self.config["batch_size"]
            self.floating_threshold[i] = not_fired_threshold + fired_threshold

        """
        # this computes the floating threshold as E[a^2|fired]
        for i in range(len(self.floating_threshold)):
            fired = self.activations[i+1].gt(0).float().detach()
            not_fired = (self.activations[i+1] == 0).float().detach()
            batched_threshold = self.floating_threshold[i].unsqueeze(1)
            batched_threshold = batched_threshold.repeat(1, self.config["batch_size"])
            not_fired_threshold = torch.sum(not_fired * batched_threshold, dim=1)/self.config["batch_size"]
            fired_threshold = torch.sum(fired * ((1 - self.config["lr_threshold"]) * batched_threshold +
                                                 self.config["lr_threshold"] * torch.square(self.activations[i+1])), dim=1)/self.config["batch_size"]
            self.floating_threshold[i] = not_fired_threshold + fired_threshold
        """

    def compute_targets(self, target):

        with torch.no_grad():
            curr_target = target
            self.targets = [curr_target]
            for l in range(len(self.layers), 1, -1): # don't compute targets for input
                target = torch.matmul(self.backward_weights[l - 1], curr_target).squeeze()
                prediction = torch.matmul(self.backward_weights[l - 1], self.activations[l])
                curr_target = self.activations[l - 1] + \
                              self.config["backward_activation"](target) - \
                              self.config["backward_activation"](prediction)
                self.targets.append(curr_target)
            self.targets.reverse()

    def train_backward_weights(self):
        self.set_backward_grad(True)
        for l in range(len(self.activations) - 1):
            self.backward_optimizer.zero_grad()
            corrupt_activation = self.activations[l] + torch.randn_like(self.activations[l]) * self.config["corruption"]
            forward_error = self.config["forward_activation"](
                    torch.matmul(self.forward_weights[l], corrupt_activation))
            backward_error = self.config["backward_activation"](
                    torch.matmul(self.backward_weights[l], forward_error))
            loss = self.config["backward_loss"](backward_error, self.activations[l].detach())
            loss.backward()
            self.backward_optimizer.step()

    def train_forward_weights(self):

        # have to forward through the model again
        # (needs to be a better way of doing this)
        self.set_forward_grad(True)
        for i in range(0, len(self.activations)-1):
            self.forward_optimizer.zero_grad()
            x = self.activations[i]
            x = torch.matmul(self.forward_weights[i], x)
            if i < len(self.layers) - 1:
                x = self.config["forward_activation"](x)
                loss = self.config["forward_loss"](x, self.targets[i].detach())
            else:
                x = self.config["output_activation"](x)
                loss = self.config["task_loss"](x, self.targets[i].detach())
            loss.backward()

            self.forward_optimizer.step()

        return loss.detach()

    def apply_BCM_update(self):

        self.set_forward_grad(False)
        for i in range(0, len(self.activations)-2): # not training on last layer

            # batch implementation slows it down?
            """
            batched_potentials = self.membrane_potentials[i].reshape((self.membrane_potentials[i].shape[0], 1))
            batched_potentials = batched_potentials.expand(-1, self.config["batch_size"])
            homeostasis = self.activations[i+1] - batched_potentials
            dW = torch.bmm(torch.mul(self.activations[i+1], homeostasis).unsqueeze(1).permute(2, 0, 1),
                               self.activations[i].unsqueeze(0).permute(2, 0, 1))
            dW = dW * torch.bmm(self.activations[i+1].unsqueeze(1).permute(2, 0, 1),
                                self.activations[i].unsqueeze(0).permute(2, 0, 1))
            self.forward_weights[i] += self.config["lr_hebbian"] * torch.sum(dW, dim=0) / self.config["batch_size"]
            """

            for j in range(self.config["batch_size"]):
                homeostasis = (self.activations[i+1][:,j] - torch.square(self.membrane_potentials[i])) #  - torch.quantile(self.activations[i+1][:,j], .85) #
                dW = torch.matmul(torch.mul(self.activations[i+1][:,j], homeostasis).unsqueeze(1), self.activations[i][:,j].unsqueeze(0))

                """
                positive_activations = torch.sum(dW > 0)
                negative_activations = torch.sum(dW < 0)
                zero_activations = torch.sum(dW == 0)

                print(f"Number of positive changes: {positive_activations}")
                print(f"Number of negative changes: {negative_activations}")
                print(f"Number of zero changes: {zero_activations}")
                """

                dW -= self.config["weight_decay"] * self.forward_weights[i]
                self.forward_weights[i] += self.config["lr_hebbian"] * dW / self.config["batch_size"]

    def set_backward_grad(self, requires_grad):
        for backward_weights in self.backward_weights:
            backward_weights.requires_grad = requires_grad

    def set_forward_grad(self, requires_grad):
        for forward_weights in self.forward_weights:
            forward_weights.requires_grad = requires_grad
