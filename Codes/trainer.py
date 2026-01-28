import data as data
from torch.utils.data import DataLoader
import torch
import json

def TrainModel(model_class, device, train_idx, val_idx, test_idx, mfccs, labels, max_epochs, params):
    config = params.copy()
    lr = config.pop('lr')
    batch_size = config.pop('batch_size')

    network = model_class(n_mfcc=mfccs[0].shape[1], num_classes=6, **config)
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    base_args = {'batch_size': batch_size, 'num_workers': 8, 'pin_memory': True}
    
    trainLoader = DataLoader(data.MFCCDataset(mfccs, labels, train_idx), shuffle=True, **base_args)
    valLoader = DataLoader(data.MFCCDataset(mfccs, labels, val_idx), shuffle=False, **base_args)
    testLoader = DataLoader(data.MFCCDataset(mfccs, labels, test_idx), shuffle=False, **base_args)
    
    statsTrainLoader = DataLoader(data.MFCCDataset(mfccs, labels, train_idx), shuffle=False, **base_args)

    train_l, train_a, val_l, val_a = network.trainFully(optimizer, trainLoader, valLoader, device, max_epochs, patience=7)

    trainStats = network.evaluateDetailed(statsTrainLoader, device)
    testStats = network.evaluateDetailed(testLoader, device)

    model_name = model_class.__name__
    torch.save(network.state_dict(), f"{model_name}_final_model.pth")
    
    with open(f"{model_name}_best_config.json", "w") as f:
        json.dump(params, f, indent=4) 
        
    print(f"Final Model and Config for {model_name} saved to disk.")
    
    return train_l, train_a, val_l, val_a, trainStats, testStats