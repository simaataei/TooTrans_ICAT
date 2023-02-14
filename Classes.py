import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer, AutoConfig




use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Protein_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
    def forward(self,input):
        input=self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].to(device))
        cls_rep = bert_rep.last_hidden_state[0][0]
        return cls_rep





class Label_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
        self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    def forward(self,input):
        input = self.tokenizer(input, return_tensors="pt", truncation = True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].to(device))
        cls_rep = bert_rep.last_hidden_state[0][0]
        return cls_rep


class protein_dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs):
        self.aa = 1
        #Initialization
        #self.labels = list_labels
        self.list_IDs = list_IDs


    def __len__(self):
        # Denotes the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        for name in self.list_IDs:
            if name == 'id-' + str(index):
                ID = name

        # Load data and get label
        X = torch.load('./Data/Data_encoded_pt/' + ID + '.pt')
        y = torch.load('./data/Data/Data_encoded_pt' + ID + '_label.pt')

        return X


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.relu = nn.LeakyReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.relu(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out