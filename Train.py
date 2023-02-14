from Classes import protein_dataset
from Data_prepration import dataset
from sklearn.model_selection import train_test_split
import torch, optuna
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim




def split(dataset):
    X = [tup[0] for tup in dataset]
    y = [tup[1] for tup in dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    return [X.index(i) for i in X_train], [X.index(i) for i in X_test]


def train_and_evaluate(param, model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.MSELoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in train_dataloader:
            train_label = train_label.to(device)
            train_input = train_input.to(device)

            output = model(train_input.float())

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return total_loss_train



def build_model(params):
    in_features = 1024

    return nn.Sequential(

        nn.Linear(in_features, params['n_unit']),
        nn.LeakyReLU(),
        nn.Linear(params['n_unit'], 768),

    )



def objective(trial):
    params = {
        # 'EPOCHS': trial.suggest_int("n_unit", 10, 200,[10]),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'n_unit': trial.suggest_int("n_unit", 10, 100)
    }

    model = build_model(params)

    loss = train_and_evaluate(params, model)

    return loss







# number of the test/train random subsets to produce
dataset_series = 5
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
EPOCHS = 100




for i in range(0, dataset_series):
    train_indx, test_indx = split(dataset)


    # partition test and train ids
    partition = {}
    partition["train"] = ['id-' + str(indx) for indx in train_indx]
    partition["test"] = ['id-' + str(indx) for indx in test_indx]

    # build dataset
    training_set = protein_dataset(partition['test'])
    test_set = protein_dataset(partition['test'])

    # generators
    train_dataloader = DataLoader(training_set, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())