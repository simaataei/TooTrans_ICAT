import torch
from Classes import Protein_encoder, Label_encoder
import pandas as pd
from sklearn.model_selection import train_test_split



def read_data(Dataset):
    # Read the dataset from train, test, val

    dataset = []
    with open('./Data/' + Dataset) as f:
        next(f)
        data = f.readlines()
        for d in data:
            d = d.split(',')
            dataset.append((d[0], int(d[1].strip('\n'))))



    return dataset

def encode(model, dataset, label):
    # encode sequences using frozen model
    sequence_output = []
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            sequence_output.append(model(sample).tolist())


    # save encodings as tensors
    for seq in sequence_output:
        id = "id-" + str(int(sequence_output.index(seq)))
        if label:
            filename = "./Data/Data_encoded_pt" + id + "_label.pt"
        else:
            filename = "./Data/Data_encoded_pt" + id + ".pt"
        seq = torch.tensor(seq)
        torch.save(seq, filename)

    return sequence_output


def split(dataset):
    X = [tup[0] for tup in dataset]
    y = [tup[1] for tup in dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    return [X.index(i) for i in X_train], [X.index(i) for i in X_test]


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")




# read data
Dataset = 'inorganic_swissprot_ident100_t10'
dataset = read_data(Dataset)


# load PLM
model_protein = Protein_encoder().to(device)


# encode sequences using PLM ProtBERT
sequence_output = encode(model_protein, [tup[0] for tup in dataset], False)


# read label descriptions
CHEBI_df = pd.read_csv('./Data/term_name_description_df_inorganic_swissprot_ident100_t10.txt')
term_name = CHEBI_df.set_index('CHEBI term').to_dict()['name']
term_description = CHEBI_df['definition'].tolist()
term_chebi = CHEBI_df['CHEBI term'].tolist()

# load LM SciBERT
model_label = Label_encoder().to(device)

# encode labels
label_encoding = encode(model_label, term_description, True)



# make datasets


# number of the test/train random subsets to produce
dataset_series = 5
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

for i in range(0, dataset_series):
    train_indx, test_indx = split(dataset)


    # partition test and train ids
    partition = {}
    partition["train"] = ['id-' + str(indx) for indx in train_indx]
    partition["test"] = ['id-' + str(indx) for indx in test_indx]

    # generators
    training_set = Dataset(partition['train'])
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    test_set = Dataset(partition['test'])
    test_generator = torch.utils.data.DataLoader(test_set, **params)