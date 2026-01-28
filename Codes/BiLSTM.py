import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

class BiLSTM(nn.Module):
    def __init__(
        self,
        n_mfcc=40,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_classes=6
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)

        feat = out.mean(dim=1)

        logits = self.classifier(feat)
        return logits

    def trainWithData(self, optimizer, trainLoader, device, epochs):
        self.train()

        losses = []
        accuracies = []
        for epoch in range(epochs):

            correct = 0
            total_loss = 0.0
            sample_count = 0
            for x, y in trainLoader:
                x, y = x.to(device), y.to(device)

                logits = self(x)
                loss = F.cross_entropy(logits, y)  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += (logits.argmax(dim=1) == y).sum().item()
                total_loss += loss.item()
                sample_count += y.size(0)

            print(f"Epoch {epoch+1} | loss: {total_loss / len(trainLoader):.4f} | accuracy: {correct / sample_count:.4f}")
            losses.append(total_loss / len(trainLoader))
            accuracies.append(correct / sample_count)
            
        return losses, accuracies

    def trainFully(self, optimizer, trainLoader, valLoader, device, epochs, patience=5):
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.trainWithData(optimizer, trainLoader, device, 1)
            train_losses.append(train_loss[0])
            train_accuracies.append(train_acc[0])

            val_loss, val_acc = self.evaluateData(valLoader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1} | Train Loss: {train_loss[0]:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = self.state_dict().copy()
            else:
                epochs_without_improvement += 1
                print(f"EarlyStopping counter: {epochs_without_improvement} out of {patience}")

            if epochs_without_improvement >= patience:
                print(f"!!! Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f} !!!")
                self.load_state_dict(best_model_state)
                break

        return train_losses, train_accuracies, val_losses, val_accuracies

    
    def evaluateData(self, validationLoader, device):
        self.eval()

        total_loss = 0.0
        correct = 0
        sample_count = 0
        with torch.no_grad():
            for x, y in validationLoader:
                x, y = x.to(device), y.to(device)

                logits = self(x)
                loss = F.cross_entropy(logits, y)

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                sample_count += y.size(0)

        print(f"Validation loss: {total_loss / len(validationLoader):.4f}")
        print(f"Validation accuracy: {correct / sample_count:.4f}")
        return total_loss / len(validationLoader), correct / sample_count

    def evaluateDetailed(self, validationLoader, device):
        self.eval()
        all_preds = []
        all_labels = []
        emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

        with torch.no_grad():
            for x, y in validationLoader:
                x, y = x.to(device), y.to(device)
                logits = self(x)
                
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 1. Calculate the Confusion Matrix
        # Rows = Actual Classes, Columns = Predicted Classes
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(emotions)))
        
        print("\n--- Detailed Per-Class Analysis ---")
        
        stats_summary = []
        for i, emotion in enumerate(emotions):
            total_actual = np.sum(all_labels == i)
            total_predicted = np.sum(all_preds == i)
            correct_predictions = cm[i, i]
            
            # Success Rate (Recall)
            success_rate = (correct_predictions / total_actual) * 100 if total_actual > 0 else 0
            
            print(f"\nResults for {emotion}:")
            print(f"  Total Actual Files: {total_actual}")
            print(f"  Total times model guessed {emotion}: {total_predicted}")
            print(f"  Correct guesses: {correct_predictions} ({success_rate:.2f}% success rate)")
            
            # Breakdown of wrong guesses when the model THOUGHT it was this emotion
            # (This helps with your bar chart: "What classes were actually inside the 1500 Anger guesses?")
            print(f"  Breakdown of the {total_predicted} '{emotion}' guesses:")
            for j, other_emotion in enumerate(emotions):
                count = cm[j, i] # Look down the column for this predicted class
                if count > 0:
                    status = "CORRECT" if i == j else "WRONG"
                    print(f"    - {count} were actually {other_emotion} ({status})")
            
            stats_summary.append({
                "emotion": emotion,
                "success_rate": success_rate,
                "total_predicted": total_predicted,
                "confusion_row": cm[i, :], # For Success Rate Graph
                "confusion_col": cm[:, i]  # For Guess Breakdown Graph
            })

        return stats_summary