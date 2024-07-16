import sys

sys.path.append('/home/michieletto/hmp_utils')
if '../' not in sys.path:
    sys.path.append('../')

import json
import glob
import os
import random

from hmp_utils.motion.body_models_constants import SmplConstants
from hmp_utils.motion.mocap_processor import MocapLoader
from hmp_utils.motion.kinematics import ForwardKinematics
from hmp_utils.visualize.stickman_animation import pose_animation, CameraOrientation, AnimationMode

import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import pandas as pd
import wandb
import re
import time

wandb.login()


directory = '/home/michieletto/datasets/AMASS_H/KIT'
file_paths = glob.glob(directory + '/**/*.npz', recursive=True)

file_paths = sorted(file_paths)
print(len(file_paths))

occur_lab = []
deleted = []
for i,file in enumerate(file_paths):
    file_name = os.path.basename(file)    # ottiene il nome del file

    # Controlla se la stringa inizia con una lettera
    if re.match(r'^[a-zA-Z]', file_name):
        # Estrarre la parte della stringa fino al primo numero
        match = re.match(r'^[^\d]+', file_name)
        if match:
            result = match.group()

        # Sostituire gli underscore con spazi vuoti
        result = result.replace('_', ' ')
        result = re.sub(r'(?<!^)(?=[A-Z])', ' ', result)

        occur_lab.append(result)
    else:
        deleted.append(i)

print(len(occur_lab))

for i in sorted(deleted, reverse=True):
    del file_paths[i]

print(len(file_paths))

# -- load mocap --
mocap_loader=MocapLoader(body_model_type=SmplConstants.BODY_MODEL_TYPE,
                         keypoint_ids_to_load=SmplConstants.KEYPOINTS,
                         target_framerate = 25)

motions = []
motions = [mocap_loader.load_mocap(path) for path in file_paths]     # lista delle classi di tutti i mocap

# -- compute joint positions --
kine = ForwardKinematics(SmplConstants.OFFSETS, SmplConstants.PARENTS)
kine.set_body_model_type(SmplConstants.BODY_MODEL_TYPE)

frame_vocab = []
deleted_paths = []
frames_to_jump = 17
cont = 0
random.seed(42)
for i in range(len(motions)):
    if motions[i] is not None:
        if len(motions[i].rots) >= 75 + frames_to_jump:
            rand_x = random.randint(0, len(motions[i].rots) - 75)
            frame_vocab.append(motions[i].rots[rand_x:rand_x + 75])
            assert len(frame_vocab[-1]) == 75

            frame_vocab[-1] = [[angle for k_point in frame for angle in k_point] for frame in frame_vocab[-1]]
            assert len(frame_vocab[-1][0]) == 72
        else:
            frame_vocab.append([])
            deleted_paths.append(i)
            cont +=1
    else:
        frame_vocab.append([])
        deleted_paths.append(i)
        cont += 1


sum = 0
for i in range(len(frame_vocab)):
    sum += len(frame_vocab[i])

print(f"Numero di mocap non validi: {cont}")
print(f"Il numero totale di frame è: {sum}")

dataset = {
    'pose': [],
    'description': [],
}

for i in range(len(occur_lab)):
    if i not in deleted_paths:
        dataset["pose"].append(frame_vocab[i])
        dataset["description"].append(occur_lab[i])

dataset = pd.DataFrame(dataset)
dataset_copy = dataset.copy()

print(len(dataset['pose']), len(dataset['description']))

# Funzione per eliminare una colonna da ogni matrice
def elimina_colonna(matrice, indexes_to_exclude):

    return [np.delete(row, indexes_to_exclude) for row in matrice]

indexes_to_exclude = [0,1,2, 30,31,32, 33,34,35, 66,67,68, 69,70,71] # Indici corrispondenti ai keypoint 0,10,11,22,23
# Applica la funzione a ogni elemento del DataFrame
dataset["pose"] = dataset["pose"].apply(elimina_colonna, indexes_to_exclude = indexes_to_exclude)


# Set the model name
MODEL_NAME = 'bert-base-cased'

# Build a BERT based tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Store length of each review
token_lens = []

# Iterate through the content slide
for txt in dataset["description"]:
    tokens = tokenizer.encode(txt, max_length=512)  #Capire la scelta del numero
    token_lens.append(len(tokens))

MAX_LEN = 11

class GPLabelDataset(Dataset):
    # Constructor Function
    def __init__(self, pose, labs, tokenizer, max_len):
        self.pose = pose
        self.labs = labs
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Length magic method
    def __len__(self):
        return len(self.labs)

    # get item magic method
    def __getitem__(self, item):
        lab = str(self.labs[item])
        pose = self.pose.iloc[item]
        
        # Encoded format to be returned
        encoding = self.tokenizer.encode_plus(
            lab,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'pose': pose,
            'raw_label': lab,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
    
def get_collate_fn():
    def collate_fn(batch):
        # La batch si presenta come una lista di batch_size dizionari, ognuno dei quali ha 4 chiavi: pose, raw_label, 
        # input_id e attention_mask. La posa inoltre è una matrice 75x57. Voglio quindi convertire 
        # la batch che sia un dizionario di 4 chiavi in ognuna delle quali ci sono tutte le informazioni raccolte insieme.
        reformed_batch = {
            'pose': [],
            'raw_label': [],
            'input_ids': [],
            'attention_mask': []
        }
        
        for i in range(len(batch)):
            reformed_batch["pose"].append(batch[i]["pose"])
            reformed_batch["raw_label"].append(batch[i]["raw_label"])
            reformed_batch["input_ids"].append(batch[i]["input_ids"].tolist())
            reformed_batch["attention_mask"].append(batch[i]["attention_mask"].tolist())

        
        reformed_batch["input_ids"] = torch.tensor(reformed_batch["input_ids"])
        reformed_batch["attention_mask"] = torch.tensor(reformed_batch["attention_mask"])

        reformed_batch["pose"] = [[row[i] for row in reformed_batch["pose"]] for i in range(len(reformed_batch["pose"][0]))]
        reformed_batch["pose"] = np.array(reformed_batch["pose"])
        reformed_batch["pose"] = torch.tensor(reformed_batch["pose"])
        reformed_batch["pose"] = reformed_batch["pose"].float()
        
        return reformed_batch

    return collate_fn

def get_data_loader(dataset, batch_size, max_len, shuffle=True):
    dataset = GPLabelDataset(
        pose = dataset.pose,
        labs=dataset.description.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    collate_fn = get_collate_fn()

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    
    return data_loader


# Build the Sentiment Classifier class
class ActionClassifier(nn.Module): 

    # Constructor class
    def __init__(self):
        super(ActionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.5)

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False
        )
        #  Add a dropout layer
        hidden = self.drop(pooled_output)
        return hidden
    
class Encoder(nn.Module):
    def __init__(self, dense_vector_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(dense_vector_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        
        # src = [src length, batch size, dense_vector_dim]
        outputs, (hidden, cell) = self.rnn(src)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, dense_vector_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = dense_vector_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(dense_vector_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, dense_vector_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size, dense_vector_dim]
        # hidden = [n layers * n directions, batch size, 2*hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, 2*hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size, dense_vector_dim]

        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        
        input = input.squeeze(0)
        prediction = input + prediction
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, actionclassifier, encoder, decoder, device):
        super().__init__()
        self.actionclassifier = actionclassifier
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.linear = nn.Linear(768, decoder.hidden_dim - encoder.hidden_dim)
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, input_ids, attention_mask, teacher_forcing_ratio):
        # src = [src length, batch size, dense_vector_dim]
        # trg = [trg length, batch size, dense_vector_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]

        concat_hidden = self.actionclassifier(input_ids, attention_mask)
        # concat_hidden = [batch_size, 768]
        concat_hidden = self.linear(concat_hidden)
        # concat_hidden = [batch_size, encoder_dim]
        concat_hidden = concat_hidden.unsqueeze(0).expand(hidden.size(0), concat_hidden.size(0), concat_hidden.size(1))

        hidden = torch.cat((concat_hidden, hidden), dim=-1).to(self.device)
        # hidden = [n layers * n directions, batch size, 2*hidden dim]
        # cell = [n layers * n directions, batch size, 2*hidden dim]
        concat_cell = torch.zeros(self.encoder.n_layers, hidden.shape[1], concat_hidden.shape[-1]).to(self.device)
        cell = torch.cat((concat_cell, cell), dim=2).to(self.device)

        input = src[-1, :]
        # input = [batch size, output_dim]
        for t in range(trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # outputs = [trg_lenght, batch_size, dense_vector_dim]
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            # top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else output
            # input = [batch size]
        
        return outputs

# Random seed for reproducibilty
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

batch_size = 64
# Number of iterations
EPOCHS = 300

dense_vector_encoder_dim = 57
dense_vector_decoder_dim = 57
hidden_size_encoder = [512, 720, 1024]
hidden_size_decoder = [512, 720, 1024]
bert_size = [64, 32, 16]
lr = [0.002, 0.003, 0.005]
n_layers = [3, 4, 5]
encoder_dropout = 0
decoder_dropout = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn(model, data_loader, optimizer, clip, teacher_forcing_ratio, scheduler, device):
    model.train()
    epoch_loss = 0
    num_mocap = 0
    for batch in data_loader:
        src_lenght = 50
        trg_lenght = 25

        src = batch["pose"][:src_lenght].to(device)
        trg = batch["pose"][-trg_lenght:].to(device)
        # src = [src length, batch size, dense_vector_dim]
        # trg = [trg length, batch size, dense_vector_dim]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, input_ids, attention_mask, teacher_forcing_ratio)
        # output = [trg length, batch size, dense_vector_dim]
        output_dim = output.shape[-1]
        output = output[:].view(-1, output_dim)
        # output = [trg length * batch size, dense_vector_dim]
        trg = trg[:].view(-1, output_dim)
        # trg = [trg length * batch size, dense_vector_dim]
        loss = torch.square(output - trg)
        loss = torch.sum(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        num_mocap += src.shape[1]

    return epoch_loss / num_mocap

def evaluate_fn(model, data_loader, device):
    model.eval()
    epoch_loss = 0
    num_mocap = 0
    with torch.no_grad():
        for batch in data_loader:
            src_lenght = 50
            trg_lenght = 25

            src = batch["pose"][:src_lenght].to(device)
            trg = batch["pose"][-trg_lenght:].to(device)
            # src = [src length, batch size, dense_vector_dim]
            # trg = [trg length, batch size, dense_vector_dim]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(src, trg, input_ids, attention_mask, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[:].view(-1, output_dim)
            # output = [trg length * batch size, trg vocab size]
            trg = trg[:].view(-1, output_dim)
            # trg = [trg length * batch size]
            loss = torch.square(output - trg)
            loss = torch.sum(loss)
            epoch_loss += loss.item()
            num_mocap += src.shape[1]

    return epoch_loss / num_mocap


clip = 1.0
teacher_forcing_ratio = 0

best_valid_loss = float("inf")
history = {
    "train_loss": [],
    "valid_loss": [],
    "losses": []
}

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=RANDOM_SEED)

train_data_loader = get_data_loader(train_data, batch_size, MAX_LEN)
valid_data_loader = get_data_loader(valid_data, batch_size, MAX_LEN)
test_data_loader = get_data_loader(test_data, batch_size, MAX_LEN)

all_seeds = [434345, 434345, 232, 875434, 3232356, 42, 645332, 67789, 3426, 2354]

for i in range(len(hidden_size_encoder)):
    for j in range(len(lr)):
        for k in range(len(n_layers)):
            for l in range(len(bert_size)):
                # dataset = dataset.sample(frac=1).reset_index(drop=True)
                print("Parametri utilizzati:")
                print("Hidden size encoder:", hidden_size_encoder[i])
                print("Learning rate:", lr[j])
                print("Num layers:", n_layers[k])
                print(f"Bert size: {bert_size[l]}")

                run = wandb.init(
                    # Set the project where this run will be logged
                    project="amass_bert_size_test_plot",

                    name=f"{hidden_size_encoder[i]}_{lr[j]}_{n_layers[k]}_{bert_size[l]}",
                    # Track hyperparameters and run metadata
                    config={
                        "hidden_size_encoder": hidden_size_encoder[i],
                        "learning_rate": lr[j],
                        "num_layers": n_layers[k],
                        "Bert size:": bert_size[l],
                    },
                )

                torch.manual_seed(RANDOM_SEED)

                actionclassifier = ActionClassifier()

                encoder = Encoder(
                    dense_vector_encoder_dim,
                    hidden_size_encoder[i],
                    n_layers[k],
                    encoder_dropout,
                )

                decoder = Decoder(
                    dense_vector_decoder_dim,
                    hidden_size_decoder[i]+bert_size[l],
                    n_layers[k],
                    decoder_dropout,
                )

                model = Seq2Seq(actionclassifier, encoder, decoder, device).to(device)

                optimizer = optim.Adam(model.parameters(), lr[j])

                total_steps = len(train_data_loader) * EPOCHS

                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )
                
                for epoch in tqdm.tqdm(range(EPOCHS)):
                    train_loss = train_fn(
                        model,
                        train_data_loader,
                        optimizer,
                        clip,
                        teacher_forcing_ratio,
                        scheduler,
                        device,
                    )
                    history["train_loss"].append(train_loss)

                    valid_loss = evaluate_fn(
                        model,
                        valid_data_loader,
                        device,
                    )
                    history["valid_loss"].append(valid_loss)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        # torch.save(model.state_dict(), "amass_file_name_descriptions.pt")
                    # print(f"\tTrain Loss: {train_loss:7.3f}")
                    # print(f"\tValid Loss: {valid_loss:7.3f}")
                
                    wandb.log({"validation error": valid_loss, "train error": train_loss})

                run.finish()

                history["losses"].append(best_valid_loss)
                print(best_valid_loss)
                best_valid_loss = float("inf")