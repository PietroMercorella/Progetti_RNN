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
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import pandas as pd
import wandb
import itertools
import datetime
import time

wandb.login()

dataset = []

# Definisci il percorso completo del file JSON
percorso_file_train = '/home/michieletto/datasets/BABEL_DATASET/babel_v1.0_release/train.json'
percorso_file_val   = '/home/michieletto/datasets/BABEL_DATASET/babel_v1.0_release/val.json'
percorso_file_test  = '/home/michieletto/datasets/BABEL_DATASET/babel_v1.0_release/test.json'

dataset.append(percorso_file_train)
dataset.append(percorso_file_val)
dataset.append(percorso_file_test)

data = {}           # Dizionario contenente l'intero dataset  

# Apre il file JSON in modalità lettura
for i,percorso_file in enumerate(dataset):
    with open(percorso_file, 'r') as f:
        # Carica i dati JSON
        data.update(json.load(f))


occur_lab = {
    'label': [],              
    'feat_p': []
    }

no_lab = 0

folders=[]
# Creazione del dataset contenente tutte le raw_label dei frame e passaggi cartelle
for i, k in enumerate(data.keys()):
    cont=0
    label = data[k]["frame_ann"]
    feat_p = data[k]["feat_p"]

    parts = feat_p.split(os.sep)
    if parts[0] not in folders:
        folders.append(parts[0])

    if label is not None:
        for j in range(len(label["labels"])):
            raw_label = label["labels"][j]["raw_label"]
            if raw_label is None:
                cont+=1
        if cont == len(label["labels"]):
            no_lab += 1
            continue

        occur_lab["label"].append(label)
        occur_lab["feat_p"].append(feat_p)
    else:
        no_lab += 1
            
print(len(folders), folders)
print("Dati totali: ", len(data))
print("Dati con label: ", len(occur_lab["label"]))
print("Dati senza label:", no_lab)
print("num label, paths:", len(occur_lab["label"]), len(occur_lab["feat_p"]))



directory = '/home/michieletto/datasets/AMASS_H/ACCAD'
file_paths = glob.glob(directory + '/**/*.npz', recursive=True) # ottiene tutti i file nelle diverse cartelle contenenti i mocap in una stringa

directory = '/home/michieletto/datasets/AMASS_H/KIT'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/BMLrub'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/CMU'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/MPIHDM05'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/EyesJapanDataset'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/EKUT'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/MPImosh'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/TCDhandMocap'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/DFaust67'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/MPILimits'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/SFU'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/TotalCapture'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/HumanEva'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/SSMsynced'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/BMLmovi'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

directory = '/home/michieletto/datasets/AMASS_H/Transitionsmocap'
file_paths += glob.glob(directory + '/**/*.npz', recursive=True)

file_paths = sorted(file_paths)
print(len(file_paths))

deleted_paths = []
for i in range(len(file_paths)):
    file_name = os.path.basename(file_paths[i])
    folder_path = os.path.dirname(file_paths[i])   # ottiene il percorso fino alla cartella in cui è contenuto il file
    last_folder_name = os.path.basename(folder_path)     # ottiene il nome dell'ultima cartella
    last_folder_and_file_name = os.path.join('/', last_folder_name, file_name)     # unisce i due nomi
    
    if file_name == "shape.npz":
        deleted_paths.append(i)
    else:
        for j in range(len(occur_lab["feat_p"])):
            if last_folder_and_file_name not in occur_lab["feat_p"][j] and j == len(occur_lab["feat_p"])-1:
                deleted_paths.append(i)
            elif last_folder_and_file_name in occur_lab["feat_p"][j]:
                break


for i in sorted(deleted_paths, reverse=True):
    del file_paths[i]

print(len(file_paths))

# -- load mocap --
mocap_loader=MocapLoader(body_model_type=SmplConstants.BODY_MODEL_TYPE,
                         keypoint_ids_to_load=SmplConstants.KEYPOINTS,
                         target_framerate = 25)

motions = []
motions = [mocap_loader.load_mocap(path) for path in file_paths] 


frame_vocab = []
deleted_paths = []

cont = 0
total_frames = 75
frames_to_jump = 10
window_frames = 50
for i in range(len(motions)):
    if motions[i] is not None:
        motions[i].rots = motions[i].rots[frames_to_jump:] # elimina i primi frame
        if len(motions[i].rots) >= total_frames:
            sample = []
            valid_mocap_frames = len(motions[i].rots)
            for j in range(int((valid_mocap_frames - 25)/window_frames)):
                sample.append(motions[i].rots[window_frames*j:window_frames*j + total_frames])
                assert len(sample[-1]) == 75

                sample[-1] = [[angle for k_point in frame for angle in k_point] for frame in sample[-1]]
                assert len(sample[-1][0]) == 72

            frame_vocab.append(sample)
        else:
            frame_vocab.append([])
            deleted_paths.append(i)
            cont +=1
    else:
        frame_vocab.append([])
        deleted_paths.append(i)
        cont += 1

print(f"Numero di mocap non validi: {cont}")

sum = 0
for i in range(len(frame_vocab)):
    sum += len(frame_vocab[i])
    
print(f"Il numero totale di mocap è: {sum}")

dataset = {
    'pose': [],
    'description': [],
}

for i in range(len(file_paths)):
    if i not in deleted_paths:
        filepath = file_paths[i]
        file_name = os.path.basename(filepath)    # ottiene il nome del file
        folder_path = os.path.dirname(filepath)   # ottiene il percorso fino alla cartella in cui è contenuto il file
        last_folder_name = os.path.basename(folder_path)     # ottiene il nome dell'ultima cartella
        last_folder_and_file_name = os.path.join('/', last_folder_name, file_name)     # unisce i due nomi
        
        for j in range(len(occur_lab["feat_p"])):
            if last_folder_and_file_name in occur_lab["feat_p"][j]:
                dataset["pose"].append(frame_vocab[i])
                dataset["description"].append(occur_lab["label"][j])

print(len(dataset['pose']), len(dataset['description']))

sum = 0
for i in range(len(dataset['pose'])):
    sum += len(dataset['pose'][i])
    
print(f"Il numero di mocap del dataset è: {sum}")





descriptions = []
min_window = 5
framerate = 25
for i in range(len(dataset["description"])):
    frames_descriptions = []
    labels = dataset["description"][i]["labels"]
    
    # crea la lista di tutte le label annotation trovate con più di 5 frame
    for j in range(len(labels)):
        annotation_list = []
        start_t = dataset["description"][i]["labels"][j]["start_t"]
        start_t = int(start_t*framerate)
        end_t = dataset["description"][i]["labels"][j]["end_t"]
        end_t = int(end_t*framerate)

        if end_t < start_t:
            end_t, start_t = start_t, end_t

        if start_t == end_t:
            continue

        annotation_list.append(start_t)
        annotation_list.append(end_t)

        raw_label = dataset["description"][i]["labels"][j]["raw_label"]
        if raw_label is not None:
            annotation_list.append(raw_label)
        else:
            # non entra mai qui, ma nel dubbio lo lascio
            annotation_list.append("unknown")

        frames_descriptions.append(annotation_list)
    
    # ordina la lista in ordeine crescente rispetto al frame di inizio
    frames_descriptions = sorted(frames_descriptions, key=lambda x: x[0])

    # se il frame più alto non corrisponde al massimo frame del mocap, lo si fa corrispondere
    end_frames = [row[1] for row in frames_descriptions]
    max_value = max(end_frames)
    indices = [index for index, value in enumerate(end_frames) if value == max_value]
    if max_value < (len(dataset["pose"][i])*window_frames) + frames_to_jump + framerate:
        for idx in indices:
            frames_descriptions[idx][1] = (len(dataset["pose"][i])*window_frames) + frames_to_jump + framerate

    # corregge un paio di errori del dataset e si assicura un paio di condizioni
    max_frame = 0
    for l in range(len(frames_descriptions)):
        if l > 0:
            if frames_descriptions[l][1] == frames_descriptions[l-1][1]:
                frames_descriptions[l-1][1] = frames_descriptions[l][0]

        if max_frame < frames_descriptions[l][1]:
            max_frame = frames_descriptions[l][1]

        # assert frames_descriptions[l][0] >= frames_to_jump
        assert frames_descriptions[l][0] < frames_descriptions[l][1]
    
    # creazione matrice delle label
    matrix = [["" for _ in range(max_frame)] for _ in range(len(frames_descriptions))]

    for m in range(len(frames_descriptions)):
        for col in range(frames_descriptions[m][0], frames_descriptions[m][1]):
            matrix[m][col] = frames_descriptions[m][2]
    
    num_columns = len(matrix[0])
    # Verifica quando si ha un cambio di colonna
    flag_column = [False] * num_columns
    
    for col in range(1, num_columns):
        column = [row[col] for row in matrix]
        prev_column = [row[col-1] for row in matrix]
        if column == prev_column:
            flag_column[col] = False
        else:
            flag_column[col] = True
    
    flag_column[frames_to_jump] = True
    start_frame = frames_to_jump
    end_frame = start_frame + total_frames
    mocap_descriptions = []
    while (end_frame <= num_columns):
        sentence = ""
        for col in range(start_frame, end_frame):
            if any(flag_column[col+1:col+min_window+1]) or end_frame - col <= min_window:
                result = True
            else:
                result = False
            if flag_column[col] == True and not result:
                phrase = ""
                for row in matrix:
                    if row[col] != "":
                        if phrase == "":
                            phrase = row[col]
                        else:
                            phrase = phrase + " and "+ row[col]
                if phrase == "":
                    phrase = "unknown"
                sentence = sentence + phrase + " then "
        if end_frame < num_columns - 1:
            flag_column[start_frame + window_frames + 1] = True
        if sentence[-6:] == " then ":
            sentence = sentence[:-6]

        start_frame += window_frames
        end_frame = start_frame + total_frames

        if sentence != "":
            mocap_descriptions.append(sentence)

    assert len(mocap_descriptions) == len(dataset["pose"][i])    
    descriptions.append(mocap_descriptions)

dataset["description"] = descriptions


print(len(dataset["description"]), len(dataset["pose"]))

dataset["description"] = list(itertools.chain.from_iterable(dataset["description"]))
dataset["pose"] = list(itertools.chain.from_iterable(dataset["pose"]))

print(len(dataset["description"]), len(dataset["pose"]))

dataset_copy = dataset.copy()

from sentence_transformers import SentenceTransformer

# sentence_model = SentenceTransformer('KennethEnevoldsen/dfm-sentence-encoder-small-v1')
# print("Kenneth")

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
print("MiniLM-L12")

for param in sentence_model.parameters():
    param.requires_grad = False

embeddings = sentence_model.encode(dataset["description"])

data_set = {
    "pose": dataset["pose"],
    "description": list(embeddings),
}

data_set = pd.DataFrame(data_set)

# Funzione per eliminare una colonna da ogni matrice
def elimina_colonna(matrice, indexes_to_exclude):
    return [np.delete(row, indexes_to_exclude) for row in matrice]

indexes_to_exclude = [0,1,2, 30,31,32, 33,34,35, 66,67,68, 69,70,71] # Indici corrispondenti ai keypoint 0,10,11,22,23
# Applica la funzione a ogni elemento del DataFrame
data_set["pose"] = data_set["pose"].apply(elimina_colonna, indexes_to_exclude = indexes_to_exclude)


class CustomDataset(Dataset):
    def __init__(self, pose, desc):
        self.pose = pose
        self.desc = desc

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):
        return self.pose[idx], self.desc[idx]

def get_data_loader(ds, batch_size, shuffle=True):
    list_of_poses = [torch.tensor(matrix) for matrix in ds["pose"]]
    pose = torch.stack(list_of_poses)

    list_of_desc = [torch.tensor(raw) for raw in ds["description"]]
    desc = torch.stack(list_of_desc)
    print(pose.shape, desc.shape)
    class_set = CustomDataset(pose, desc)

    data_loader = DataLoader(
        dataset=class_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    
    return data_loader



    
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
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.linear = nn.Linear(384, decoder.hidden_dim - encoder.hidden_dim)
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, desc, teacher_forcing_ratio):
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

        # desc = [batch_size, 384]
        concat_hidden = self.linear(desc)
        # concat_hidden = desc
        # concat_hidden = [batch_size, size_dim]
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
EPOCHS = 20

dense_vector_encoder_dim = 57
dense_vector_decoder_dim = 57
hidden_size_encoder = [1024]
hidden_size_decoder = [1024]
desc_size = [192]
lr = [0.001]
n_layers = [4]
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

        batch[0] = batch[0].permute(1, 0, 2).float()
        src = batch[0][:src_lenght].to(device)
        trg = batch[0][-trg_lenght:].to(device)
        # src = [src length, batch size, dense_vector_dim]
        # trg = [trg length, batch size, dense_vector_dim]
        desc = batch[1].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, desc, teacher_forcing_ratio)
        # output = [trg length, batch size, dense_vector_dim]
        output_dim = output.shape[-1]
        output = output[:].view(-1, output_dim)
        # output = [trg length * batch size, dense_vector_dim]
        trg = trg.reshape(-1, output_dim)
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

            batch[0] = batch[0].permute(1, 0, 2).float()
            src = batch[0][:src_lenght].to(device)
            trg = batch[0][-trg_lenght:].to(device)
            # src = [src length, batch size, dense_vector_dim]
            # trg = [trg length, batch size, dense_vector_dim]
            desc = batch[1].to(device)

            output = model(src, trg, desc, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[:].view(-1, output_dim)
            # output = [trg length * batch size, trg vocab size]
            trg = trg.reshape(-1, output_dim)
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

train_data, test_data = train_test_split(data_set, test_size=0.2, random_state=RANDOM_SEED)
valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=RANDOM_SEED)

train_data_loader = get_data_loader(train_data, batch_size)
valid_data_loader = get_data_loader(valid_data, batch_size)
test_data_loader = get_data_loader(test_data, batch_size)

# batches = []
# for batch in test_data_loader:
#     batches.append(batch)

# torch.save(batches, "/home/michieletto/Progetti_per_tesi/Progetto_RNN/Parametri_modello/test_data_loader_frame_description.pt")

all_seeds = [434345, 434345, 232, 875434, 3232356, 42, 645332, 67789, 3426, 2354]

now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d")

for i in range(len(hidden_size_encoder)):
    for j in range(len(lr)):
        for k in range(len(n_layers)):
            for l in range(len(desc_size)):
                print("Parametri utilizzati:")
                print("Hidden size encoder:", hidden_size_encoder[i])
                print("Learning rate:", lr[j])
                print("Num layers:", n_layers[k])
                print(f"Desc size: {desc_size[l]}")

                run = wandb.init(
                    # Set the project where this run will be logged
                    project="amass_babel_MiniLM-L12_frame_annotation",

                    name=f"{hidden_size_encoder[i]}_{lr[j]}_{n_layers[k]}_{desc_size[l]}",
                    # Track hyperparameters and run metadata
                    config={
                        "hidden_size_encoder": hidden_size_encoder[i],
                        "learning_rate": lr[j],
                        "num_layers": n_layers[k],
                        "desc size:": desc_size[l],
                    },
                )

                torch.manual_seed(RANDOM_SEED)

                encoder = Encoder(
                    dense_vector_encoder_dim,
                    hidden_size_encoder[i],
                    n_layers[k],
                    encoder_dropout,
                )

                decoder = Decoder(
                    dense_vector_decoder_dim,
                    hidden_size_decoder[i]+desc_size[l],
                    n_layers[k],
                    decoder_dropout,
                )

                model = Seq2Seq(encoder, decoder, device).to(device)

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
                        torch.save(model.state_dict(), f"/home/michieletto/Progetti_per_tesi/Progetti_RNN_git/Parametri_modello/amass_babel_frame_ann_MiniLM_{date_string}_{hidden_size_encoder[i]}_{lr[j]}_{n_layers[k]}_{desc_size[l]}.pt")
                    # print(f"\tTrain Loss: {train_loss:7.3f}")
                    # print(f"\tValid Loss: {valid_loss:7.3f}")
                
                    wandb.log({"validation error": valid_loss, "train error": train_loss})

                run.finish()

                history["losses"].append(best_valid_loss)
                # if best_valid_loss == min(history["losses"]):
                #     torch.save(model.state_dict(), "/home/michieletto/Progetti_per_tesi/Progetto_RNN/Parametri_modello/amass_babel_frame_annotations.pt")
                print(best_valid_loss)
                best_valid_loss = float("inf")

quit()

from hmp_utils import EvaluationProtocol

bodyConstants = SmplConstants

fk_processor = ForwardKinematics(bodyConstants.OFFSETS, bodyConstants.PARENTS)
fk_processor.set_body_model_type(bodyConstants.BODY_MODEL_TYPE)
fk_processor.normalize_joints(reference_joint=bodyConstants.JOINT_TO_NORMALIZE)

metrics_dict, metric_processor_dict = EvaluationProtocol.initialize_metrics_dicts(fk_processor=fk_processor)

def test_fn(model, data_loader, zero_velocity, device):
    model.eval()
    epoch_loss = 0
    num_mocap = 0
    first = True
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src_lenght = 50
            trg_lenght = 25

            batch[0] = batch[0].permute(1, 0, 2).float()
            src = batch[0][:src_lenght].to(device)
            trg = batch[0][-trg_lenght:].to(device)
            # src = [src length, batch size, dense_vector_dim]
            # trg = [trg length, batch size, dense_vector_dim]
            desc = batch[1].to(device)

            output = model(src, trg, desc, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            # output_dim = output.shape[-1]
            # output = output[:].view(-1, output_dim)
            # # output = [trg length * batch size, trg vocab size]
            # trg = trg.reshape(-1, output_dim)
            
            loss = torch.square(output - trg)
            loss = torch.sum(loss)
            epoch_loss += loss.item()
            num_mocap += src.shape[1]

            #ZERO VELOCITY
            if zero_velocity:
                for idx in range(output.shape[0]):
                    output[idx, :, :] = src[-1, :, :]

            if first:
                targets = trg
                predictions = output
                first = False
            else:
                targets = torch.cat((targets, trg), dim=1)
                predictions = torch.cat((predictions, output), dim=1)

    return epoch_loss / num_mocap, predictions, targets

model.load_state_dict(torch.load("/home/michieletto/Progetti_per_tesi/Progetto_RNN/Parametri_modello/amass_babel_frame_ann_MiniLM_16_07_2004_1024_0.001_4_192.pt"))


zero_vel = [False, True]
for zero_velocity in zero_vel:
    metrics = {"amass": {}}
    test_loss, predictions, targets = test_fn(model, test_data_loader, zero_velocity, device)
    print(targets[1][1])
    predictions = predictions.cpu()
    targets = targets.cpu()

    print(f"Test Loss: {test_loss:.3f}")
    test_loss = [0]*25

    targets = targets.permute(1, 0, 2).float()
    predictions = predictions.permute(1, 0, 2).float()

    pred = torch.zeros(predictions.shape[0], predictions.shape[1], 72)
    trg = torch.zeros(targets.shape[0], targets.shape[1], 72)

    mask = torch.ones(72, dtype=bool)
    mask[indexes_to_exclude] = False

    pred[:, :, mask] = predictions
    trg[:, :, mask] = targets

    predictions = pred
    targets = trg

    targets = targets.reshape(targets.shape[0], targets.shape[1], -1, 3)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], -1, 3)

    EvaluationProtocol.compute_all_metrics(
        predictions=predictions.cpu(),
        targets=targets.cpu(),
        framerate=25,
        target_seq_len=25,
        default_error=test_loss,
        bodyConstants=bodyConstants,
        fk_processor=fk_processor,
        metrics_res_by_action_dict=metrics,
        action="amass",
        metrics_dict=metrics_dict,
        metric_processor_dict=metric_processor_dict,
        zero_velocity=zero_velocity,
    )

    if not zero_velocity:
        EvaluationProtocol.save_all_metrics(
            metrics_res_by_action_dict=metrics,
            metrics_dict=metrics_dict,
            framerate=25,
            csv_path="/home/michieletto/Progetti_per_tesi/Progetto_RNN/Risultati_test_csv/amass_babel_frame_ann_MiniML",
            print_table=False,
        )
    else:
        EvaluationProtocol.save_all_metrics(
            metrics_res_by_action_dict=metrics,
            metrics_dict=metrics_dict,
            framerate=25,
            csv_path="/home/michieletto/Progetti_per_tesi/Progetto_RNN/Risultati_test_csv/amass_babel_frame_ann_MiniML_zero_velocity",
            print_table=False,
        )
