import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_size, n_transformations, n_outputs):
        super(Model, self).__init__()

        self.input_size = input_size
        self.n_transformations = n_transformations
        self.n_outputs = n_outputs
        self.fully_connected_multiplier = 128
        self.number_of_filters = 40

        self.linear_input = 4 * self.number_of_filters * self.input_size * self.input_size // (4 * 4 * 4)
        self.linear_output = self.fully_connected_multiplier * self.number_of_filters

        self.branch_cnn = nn.Sequential(
            nn.Conv2d(3,self.number_of_filters,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(self.number_of_filters, 2 * self.number_of_filters, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(2 * self.number_of_filters,4 * self.number_of_filters, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.branch_fcn = nn.Linear(self.linear_input,self.linear_output)

        self.maxpooling = nn.MaxPool1d(kernel_size=self.n_transformations)
        self.feature_size = self.fully_connected_multiplier * self.number_of_filters

        self.dropout = nn.Dropout()
        self.linear = nn.Linear(self.feature_size, self.n_outputs)
        self.logsoftmax = nn.LogSoftmax()

    def forward_branch(self, x):

        x = self.branch_cnn(x)
        x = x.view(-1,self.linear_input)
        x = self.branch_fcn(x)

        return x

    def forward(self, x):

        im_features = []
        for i in range(self.n_transformations):
            im_features.append(self.forward_branch(x[:, :, :, :, i]))
        im_features = torch.stack(im_features, dim=2)

        max_features = self.maxpooling(im_features)
        max_features = max_features.reshape(-1, self.feature_size)

        # classify
        features = self.dropout(max_features)
        features = self.linear(features)

        return self.logsoftmax(features)
