import audio_handler as audio
import split_sets as split
import trainer as trainer
import tuner as tuner
import grapher as grapher
import torch
import BiLSTM as BiLSTM
import TRANSFORMER as TRANSFORMER
import RESNET as RESNET

MODELS = [BiLSTM.BiLSTM, RESNET.ResNet]
PARAMS = {
    BiLSTM.BiLSTM: {
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 0.001,
        "batch_size": 32
    },
    TRANSFORMER.Transformer: None,
    RESNET.ResNet: {
        "hidden_size": 256,
        "num_layers": 5,
        "kernel_size": 11,
        "dropout": 0.2,
        "lr": 0.0005,
        "batch_size": 16
    }
}
PARAM_GRID = {
    BiLSTM.BiLSTM: {
        "hidden_size": [32, 64, 128, 192, 256], 
        "num_layers": [1, 2, 3, 4],
        "dropout": [0.0, 0.1, 0.2, 0.4],

        "lr": [0.0003, 0.0005, 0.001],
        "batch_size": [32, 64]
    },
    TRANSFORMER.Transformer: {
        "hidden_size": [64, 128, 192, 256], 
        "num_layers": [2, 3, 4, 5, 6],
        "n_heads": [2, 4, 8, 16],
        
        "dropout": [0.1, 0.2, 0.3],
        "lr": [0.00005, 0.0001, 0.0003],
        "batch_size": [16, 32] 
    },
    RESNET.ResNet: {
        "hidden_size": [64, 128, 256, 384, 512],
        "num_layers": [3, 5, 7, 9, 11],
        "kernel_size": [3, 5, 7, 9, 11],
        
        "dropout": [0.2, 0.3, 0.4],
        "lr": [0.0005, 0.001],
        "batch_size": [16, 32] 
    }
}

ACTOR_AUDIO_NO = 82 * 3 # 1 original + 2 augmented
FOLD_NO = 10
TRAIN_EPOCHS = 50

def main():
    labels, mfccs = audio.get_file_info()
    folds, original_folds = split.split_folds(len(labels), FOLD_NO, ACTOR_AUDIO_NO)

    train_folds = folds[:FOLD_NO - 2]
    train_idx = [idx for f in train_folds for idx in f]
    
    val_idx = folds[FOLD_NO - 2]
    test_idx = original_folds[FOLD_NO - 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model in MODELS:
        if PARAMS[model] is None:
            PARAMS[model], results = tuner.tune_model(
                model, device, train_folds, train_folds,
                mfccs, labels, PARAM_GRID[model]
            )
            grapher.save_tuning(results, model.__name__, PARAM_GRID[model])

        t_losses, t_accs, v_losses, v_accs, trainStats, testStats = trainer.TrainModel(
            model, device, train_idx, val_idx, test_idx, mfccs, labels, TRAIN_EPOCHS, PARAMS[model]
        )

        grapher.plot_training_curves(model.__name__, t_losses, v_losses, t_accs, v_accs)
        grapher.plot_guess_breakdown((model.__name__ + "_train"), trainStats)
        grapher.plot_guess_breakdown((model.__name__ + "_test"), testStats)
        grapher.plot_success_rates((model.__name__ + "_train"), trainStats)
        grapher.plot_success_rates((model.__name__ + "_test"), testStats)
        

        
if __name__ == "__main__":
    main()