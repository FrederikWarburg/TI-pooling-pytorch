import tools
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from rot_mnist12K_model import Model
import os

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

def evaluate_model(model, test_data_loader, number_of_test_chunks, TEST_CHUNK_SIZE, criterion):
    correct_pred, observations, running_loss = 0.0, 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for chunk_index in range(number_of_test_chunks):
            chunk_x, chunk_y = test_data_loader.next_batch(TEST_CHUNK_SIZE)
            _, target = np.where(chunk_y == 1)

            chunk_x = torch.tensor(chunk_x, dtype=torch.float32).to(device)
            chunk_y = torch.tensor(target, dtype=torch.long).to(device)

            pred = model(chunk_x)
            loss = criterion(pred, chunk_y)

            correct_pred += torch.sum(torch.argmax(pred, dim=1) == chunk_y)
            observations += len(chunk_y)
            running_loss += loss.item()

        test_accuracy = float(correct_pred) / float(observations)

        print("testing accuracy %g, testing loss %g" % (test_accuracy, running_loss))

    return test_accuracy, running_loss




TRAIN_FILENAME = '../mnist_all_rotation_normalized_float_train_valid.amat'
TEST_FILENAME = '../mnist_all_rotation_normalized_float_test.amat'

LOADED_SIZE = 28
DESIRED_SIZE = 32
# model constants
NUMBER_OF_CLASSES = 10
NUMBER_OF_TRANSFORMATIONS = 24
# optimization constants
BATCH_SIZE = 64
TEST_CHUNK_SIZE = 1627
ADAM_LEARNING_RATE = 1e-4
PRINTING_INTERVAL = 10
MAX_EPOCHS = 40
# set seeds
np.random.seed(100)
torch.manual_seed(100)
# results path
result_folder = 'results/'
filename = 'results'

train_data_loader = tools.DataLoader(TRAIN_FILENAME,
                                     NUMBER_OF_CLASSES,
                                     NUMBER_OF_TRANSFORMATIONS,
                                     LOADED_SIZE,
                                     DESIRED_SIZE)
test_data_loader = tools.DataLoader(TEST_FILENAME,
                                    NUMBER_OF_CLASSES,
                                    NUMBER_OF_TRANSFORMATIONS,
                                    LOADED_SIZE,
                                    DESIRED_SIZE)

test_size = test_data_loader.all()[1].shape[0]
assert test_size % TEST_CHUNK_SIZE == 0

number_of_test_chunks = test_size // TEST_CHUNK_SIZE

INPUT_SIZE = 32  # image size
OUTPUT_SIZE = 10  # digit from 0..9

model = Model(INPUT_SIZE,
              NUMBER_OF_TRANSFORMATIONS,
              OUTPUT_SIZE)

model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=ADAM_LEARNING_RATE)

correct_pred, observations, running_loss = 0.0, 0.0, 0.0

train_stats = pd.DataFrame(columns=['epoch', 'loss', 'acc'])
test_stats = pd.DataFrame(columns=['epoch', 'loss', 'acc'])

while True:
    batch_x, batch_y = train_data_loader.next_batch(BATCH_SIZE)

    _, target = np.where(batch_y == 1)

    batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
    batch_y = torch.tensor(target, dtype=torch.long).to(device)

    if train_data_loader.is_new_epoch():

        train_accuracy = float(correct_pred) / float(observations)

        print("completed_epochs %d, training accuracy %g, training loss %g" %
              (train_data_loader.get_completed_epochs(), train_accuracy, running_loss))
        train_stats = train_stats.append(pd.DataFrame([[train_data_loader.get_completed_epochs(), running_loss, train_accuracy]],columns=['epoch','loss','acc']))

        correct_pred, observations, running_loss = 0.0, 0.0, 0.0
        if (train_data_loader.get_completed_epochs() % PRINTING_INTERVAL == 0):

            test_accuracy, test_running_loss = evaluate_model(model, test_data_loader, number_of_test_chunks, TEST_CHUNK_SIZE, criterion)
            test_stats = test_stats.append(
                pd.DataFrame([[train_data_loader.get_completed_epochs(),
                               test_running_loss,
                               test_accuracy]], columns=['epoch','loss','acc']))

        if train_data_loader.get_completed_epochs() == MAX_EPOCHS:
            train_stats.to_csv(result_folder + filename + '_train.csv')
            test_stats.to_csv(result_folder + filename + '_test.csv')
            torch.save(model.state_dict(), result_folder + filename + '_model.pt')
            torch.cuda.empty_cache()
            break

    model.train()
    optimizer.zero_grad()
    pred = model(batch_x)
    loss = criterion(pred, batch_y)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    correct_pred += torch.sum(torch.argmax(pred, dim=1) == batch_y)
    observations += len(batch_y)
