import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adam import Adam
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(".."))

from Actionsrecognition.Models import *
from Visualizer import plot_graphs, plot_confusion_metrix
from sklearn.metrics import precision_recall_fscore_support


save_folder = 'saved/ori2'
device = 'cuda'
epochs = 30
batch_size = 32

data_files = ['/home/moore/school/Human-Falling-Detect-Tracks/Data/Le2i.pkl']
class_names = ['Standing', 'Walking', 'Sitting', 'Fall',
               'Stand up', 'Sit down', 'Falling']
num_class = len(class_names)

def load_dataset(data_files, batch_size, split_size=0):
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(np.argmax(y_valid, axis=1), dtype=torch.long))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(np.argmax(labels, axis=1), dtype=torch.long))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader

def count_correct_batch(y_pred, y_true):
    pred_class = y_pred.argmax(1)
    correct = (pred_class == y_true).sum().item()
    total = y_true.size(0)
    return correct, total


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

def build_motion(pts):
    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
    pts = pts[:, :, 1:, :]
    return pts, mot

if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    os.makedirs(save_folder, exist_ok=True)

    train_loader, valid_loader = load_dataset(data_files[0:1], batch_size, 0.2)
    train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader.dataset]),
                                   batch_size, shuffle=True)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    del train_loader

    graph_args = {'strategy': 'spatial'}
    USE_LITE_MODEL = False

    if USE_LITE_MODEL:
        from Actionsrecognition.Models import StreamSpatialTemporalGraphLite
        model = StreamSpatialTemporalGraphLite(3, graph_args, num_class).to(device)
        save_name = 'tsstg-lite.pth'
    else:
        from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
        save_name = 'tsstg-full.pth'

    print(f"✅ 使用模型：{'Lite' if USE_LITE_MODEL else 'Full'}，參數總數：{sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=0.001)
    # 根據你的數據量分布，手動計算 class weights（倒數比例）
    class_counts = torch.tensor([7227, 8484, 4852, 4243, 2060, 982, 2099], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize 總和為 num_class

    losser = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            model = set_training(model, phase == 'train')
            run_loss, run_accu = 0.0, 0.0
            total_samples = 0
            y_preds_epoch = []
            y_trues_epoch = [] #for F1-score

            with tqdm(dataloader[phase], desc=phase) as iterator:

                for pts, lbs in iterator:
                    pts, mot = build_motion(pts)
                    pts, mot, lbs = pts.to(device), mot.to(device), lbs.to(device)

                    out = model((pts, mot))
                    loss = losser(out, lbs)

                    y_preds_epoch.extend(out.argmax(1).detach().cpu().numpy()) 
                    y_trues_epoch.extend(lbs.detach().cpu().numpy()) #for F1-score

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    run_loss += loss.item()
                    correct, total = count_correct_batch(out.detach().cpu(), lbs.detach().cpu())
                    run_accu += correct
                    total_samples += total


                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(loss.item(), run_accu / total_samples))

            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / total_samples)

            precision, recall, f1, _ = precision_recall_fscore_support(
                     y_trues_epoch, y_preds_epoch, average='macro', zero_division=0)
            print(f"  → {phase.title()} Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}") #F1-score

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss: {:.4f}, accu: {:.4f}'.format(
            loss_list['train'][-1], accu_list['train'][-1], loss_list['valid'][-1], accu_list['valid'][-1]))

        torch.save(model.state_dict(), os.path.join(save_folder, save_name))

        plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(loss_list['train'][-1], loss_list['valid'][-1]),
                    'Loss', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'loss_graph.png'))
        plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                    'Last Train: {:.2f}, Valid: {:.2f}'.format(accu_list['train'][-1], accu_list['valid'][-1]),
                    'Accu', xlim=[0, epochs],
                    save=os.path.join(save_folder, 'accu_graph.png'))

    model.load_state_dict(torch.load(os.path.join(save_folder, save_name)))

    # EVALUATION.
    model = set_training(model, False)
    data_file = data_files[0]
    eval_loader, _ = load_dataset([data_file], 32)

    print('Evaluation.')
    run_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_preds = []
    y_trues = []

    with tqdm(eval_loader, desc='eval') as iterator:
        for pts, lbs in iterator:
            pts, mot = build_motion(pts)
            pts, mot, lbs = pts.to(device), mot.to(device), lbs.to(device)

            out = model((pts, mot))
            loss = losser(out, lbs)
            run_loss += loss.item()

            correct, total = count_correct_batch(out.detach().cpu(), lbs.detach().cpu())
            total_correct += correct
            total_samples += total

            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.detach().cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(loss.item(), correct / total))

    run_loss = run_loss / len(iterator)
    run_accu = total_correct / total_samples


    plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
        os.path.basename(data_file), run_loss, run_accu
    ), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
        os.path.basename(data_file).split('.')[0])))

    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))