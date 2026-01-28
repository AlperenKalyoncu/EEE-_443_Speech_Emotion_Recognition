import torch
import random
import data as data
from torch.utils.data import DataLoader

EPOCHS = [3, 4]
FOLD_TRIES = [2, 3]

def getFold(folds, original_folds, fold):
    train_folds = []
    for j in range(len(folds)):
        if j != fold % len(folds):
            train_folds += folds[j]
    return train_folds, original_folds[fold % len(folds)]

def tryModel(model_class, device, trainLoader, validationLoader, n_mfccs, config, epochs):
    try:
        model_params = config.copy()
        lr = model_params.pop('lr')
        model_params.pop('batch_size', None)

        trialModel = model_class(n_mfcc=n_mfccs, num_classes=6, **model_params).to(device)
        optimizer = torch.optim.Adam(trialModel.parameters(), lr=lr)

        # Training
        total_loss, training_accuracy = trialModel.trainWithData(optimizer, trainLoader, device, epochs)
        val_loss, val_accuracy = trialModel.evaluateData(validationLoader, device)
        
        return val_accuracy

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"| SKIPPING CONFIG: Out of VRAM |")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
        else:
            raise e

def tune_model(model, device, folds, original_folds, mfccs, labels, param_grid, 
               epochs=EPOCHS, randomN=8, foldTries=FOLD_TRIES):
    
    results = []
    keys = list(param_grid.keys())
    
    for _ in range(randomN):
        current_indices = {k: random.randint(0, len(param_grid[k]) - 1) for k in keys}
        current_config = {k: param_grid[k][current_indices[k]] for k in keys}

        print(f"Random Training with config: {current_config}")

        per_fold_accuracies = []
        for fold in range(foldTries[0]):
            train_idx, val_idx = getFold(folds, original_folds, fold)
            
            loader_args = {'batch_size': current_config['batch_size'], 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
            trainLoader = DataLoader(data.MFCCDataset(mfccs, labels, train_idx), **loader_args)
            
            loader_args['shuffle'] = False
            validationLoader = DataLoader(data.MFCCDataset(mfccs, labels, val_idx), **loader_args)

            acc = tryModel(model, device, trainLoader, validationLoader, mfccs[0].shape[1], current_config, epochs[0])
            per_fold_accuracies.append(acc)

        avg_accuracy = sum(per_fold_accuracies) / len(per_fold_accuracies)
        print("Average accuracy: ", avg_accuracy)
        
        res_entry = {"val_accuracy": avg_accuracy, "per_fold_accuracies": per_fold_accuracies, "actual_config": current_config}
        res_entry.update(current_indices)
        results.append(res_entry)

    best_rand = max(results, key=lambda x: x["val_accuracy"])
    best_rand_indices = {k: best_rand[k] for k in keys}
    print(f"Best random config indices: {best_rand_indices}")
    
    nudge_keys = keys[:3] 

    for i in range(-1, 2): 
        for j in range(-1, 2): 
            for k in range(-1, 2): 
                idx_offsets = [i, j, k]
                temp_indices = best_rand_indices.copy()
                valid_config = True
                
                for idx, key in enumerate(nudge_keys):
                    new_idx = temp_indices[key] + idx_offsets[idx]
                    if 0 <= new_idx < len(param_grid[key]):
                        temp_indices[key] = new_idx
                    else:
                        valid_config = False
                
                if not valid_config: continue

                grid_config = {key: param_grid[key][temp_indices[key]] for key in keys}

                print(f"Grid Training (Nudge {i,j,k}): {grid_config}")

                per_fold_accuracies = []
                for fold in range(foldTries[1]):
                    train_idx, val_idx = getFold(folds, original_folds, fold)
                    
                    loader_args = {'batch_size': grid_config['batch_size'], 'shuffle': True, 'num_workers': 8, 'pin_memory': True}
                    trainLoader = DataLoader(data.MFCCDataset(mfccs, labels, train_idx), **loader_args)
                    
                    loader_args['shuffle'] = False
                    validationLoader = DataLoader(data.MFCCDataset(mfccs, labels, val_idx), **loader_args)

                    acc = tryModel(model, device, trainLoader, validationLoader, mfccs[0].shape[1], grid_config, epochs[1])
                    per_fold_accuracies.append(acc)

                avg_accuracy = sum(per_fold_accuracies) / len(per_fold_accuracies)
                print("Average accuracy: ", avg_accuracy)
                
                res_entry = {"val_accuracy": avg_accuracy, "per_fold_accuracies": per_fold_accuracies, "actual_config": grid_config}
                res_entry.update(temp_indices)
                results.append(res_entry)

    best_final = max(results, key=lambda x: x["val_accuracy"])
    final_params = {k: param_grid[k][best_final[k]] for k in keys}
    
    print("\nFinal Best Configuration Found:")
    for k, v in final_params.items():
        print(f"{k} = {v}")

    return final_params, results