import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import RobertaModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook

import tqdm
import time as time
from datetime import datetime
import random
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

score_columns = [ 
    'openness', 
    'conscientiousness', 
    'extraversion', 
    'agreeableness',
    'neuroticism']

def score_boxplots(data, filename):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data[score_columns])
    plt.title('Boxplots for Personality Traits')
    plt.ylabel('Scores')
    plt.xlabel('Traits')
    plt.show()

def highest_score_histogram(data, filename='reddit_score_histograms.png'):
    data['highest_trait'] = data[score_columns].idxmax(axis=1)
    plt.figure(figsize=(8, 4))
    sns.countplot(x='highest_trait', data=data, order=score_columns)
    plt.title('Histogram of Highest Scoring Personality Traits')
    plt.ylabel('Frequency')
    plt.xlabel('Traits')
    #plt.savefig(f'images/"{filename}.png')
    plt.show()

def text_length_histogram(data, filename='reddit_text_length_histograms.png'):
    text_lengths = [len(text.split()) for text in data['text']]
    plt.figure(figsize=(8, 4))
    sns.histplot(text_lengths, bins=50, kde=True)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Number of Words')
    max_text_length = max(text_lengths)
    print(f"Maximum length of text: {max_text_length}")
    #plt.savefig(f'images/"{filename}.png')
    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value. Code credit:CS7643 A2"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def batch_correl(predictions, target_scores):
    predictions_flat = predictions.view(predictions.size(0), -1)
    target_scores_flat = target_scores.view(target_scores.size(0), -1)
    mean_predictions = torch.mean(predictions_flat, dim=1, keepdim=True)
    mean_target_scores = torch.mean(target_scores_flat, dim=1, keepdim=True)
    numerator = torch.sum((predictions_flat - mean_predictions) * (target_scores_flat - mean_target_scores), dim=1)
    denominator = (torch.sqrt(torch.sum((predictions_flat - mean_predictions) ** 2, dim=1) * 
        torch.sum((target_scores_flat - mean_target_scores) ** 2, dim=1) + 1e-8))
    batch_correlations = numerator / denominator
    return torch.mean(batch_correlations).item()


def learn(model, dataloader, optimizer, criterion, epoch, from_text=True): # , loss_meter, iter_meter):

    start = time.time()
    loss_meter = AverageMeter()
    correl_meter = AverageMeter()
    iter_meter = AverageMeter()
    total_loss = 0.0
    model.train()

    for i, data in enumerate(dataloader):
        
        target_scores = data['scores']
        if from_text:
            input_tokens = data['input_ids']
            attention_mask = data['attention_mask']
            predictions = model(input_tokens, attention_mask=attention_mask)
        else:
            encodings = data['encodings']
            predictions = model(encodings)
        mse_loss = criterion(predictions, target_scores)
        correl = batch_correl(predictions, target_scores)
        loss = mse_loss - 0.0 * correl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(mse_loss.item(), predictions.size(0))
        correl_meter.update(correl, predictions.size(0))
        iter_meter.update(time.time()-start)
        total_loss += loss.item()

        if i % 50 == 0 or i == len(dataloader):
            print(f'Epoch: [{epoch}][{i}/{len(dataloader)}]\t'
                  f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                  f'Correl {correl_meter.val:.3f} ({correl_meter.avg:.3f})\t'
                  f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t')

    return loss_meter.val, loss_meter.avg


def evaluate(model, dataloader, criterion, epoch, from_text):

    loss_meter = AverageMeter()
    correl_meter = AverageMeter()
    iter_meter = AverageMeter()    
    accuracies = []
    correls = []
    model.eval()

    with torch.no_grad():  
        for i, data in enumerate(dataloader):
            start = time.time()

            target_scores = data['scores']
            if from_text:
                input_tokens = data['input_ids']
                attention_mask = data['attention_mask']
                predictions = model(input_tokens, attention_mask=attention_mask)
            else:
                encodings = data['encodings']
                predictions = model(encodings)
            mse_loss = criterion(predictions, target_scores)
            correl = batch_correl(predictions, target_scores)
            #loss = mse_loss - 0.0 * correl

            loss_meter.update(mse_loss.item(), predictions.size(0))
            correl_meter.update(correl, predictions.size(0))
            iter_meter.update(time.time()-start)

            # compute accuracy
            target_domtrait = torch.argmax(target_scores, dim=1)
            predicted_domtrait = torch.argmax(predictions, dim=1)
            equivalencies = (predicted_domtrait == target_domtrait).sum()
            accuracy = equivalencies.item() / len(target_domtrait)
            accuracies.append(accuracy)

            # Compute Spearman correlation for each row
            #predictions_ = predictions.cpu().numpy()
            #target_scores_ = target_scores.cpu().numpy()
            #correls_ = np.array([
            #    spearmanr(predictions_[j, :], target_scores_[j, :])[0] 
            #    for j in range(predictions_.shape[0])])
            #correl = np.mean(correls_)
            #correls.append(correl)

            if i % 10 == 0 or i == len(dataloader):
                print(f'Val Epoch: [{epoch}][{i}/{len(dataloader)}]\t'
                    f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                    f'Correl {correl_meter.val:.3f} ({correl_meter.avg:.3f})\t'
                    f'Accuracy {accuracy:.3f}\t'
                    f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t')

    accuracy = sum(accuracies) / len(accuracies)
    #correl = sum(correls) / len(correls)
    return loss_meter.val, loss_meter.avg, accuracy, correl_meter.avg


def train(epochs, model, train_loader, valid_loader, test_loader, optimizer, scheduler, criterion, from_text=True):
    
    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_losses = [0.0] * epochs
    valid_losses = [0.0] * epochs

    for epoch in range(epochs):
        
        print("")
        print(f"Training Epoch {epoch + 1}/{epochs}")
        train_loss, \
        avg_train_loss = learn(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            epoch=epoch,
            from_text=from_text)
        print(f"Training Loss: {avg_train_loss:.4f}")

        print("")
        print(f"Validating Epoch {epoch + 1}/{epochs}")
        valid_loss, \
        avg_valid_loss, \
        valid_accuracy, \
        valid_correl = evaluate(
            model=model, 
            dataloader=valid_loader,
            criterion=criterion,
            epoch=epoch,
            from_text=from_text)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        print(f"Validation Correlation: {valid_correl:.4f}")

        print("")
        print("Saving model...")
        model.save(timestamp)
        
        #scheduler.step(train_loss)
        train_losses[epoch] = avg_train_loss
        valid_losses[epoch] = avg_valid_loss

    print("")
    print(f"Completed in {(time.time()-start):.3f}")

    if epochs > 1:
        print("")
        print("Plotting learning curves")
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"images/learning_curves_{timestamp}.png") 

    print("")
    print(f"Predicting on test set")
    test_loss, \
    avg_test_loss, \
    test_accuracy, \
    test_correl = evaluate(
        model=model, 
        dataloader=test_loader,
        criterion=criterion,
        epoch=epoch,
        from_text=from_text)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Correlation: {test_correl:.4f}")