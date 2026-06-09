"""
Train and store a model using K-fold hyperparameter search
"""

import json
from pathlib import Path

import joblib
import torch

from src.models.gru_classifier import GRUClassifier
from src.training.gru_training import train_gru_model
from src.training.k_fold_hyperparameter_search import hyperparameter_search_gru
from src.utils import file_to_sequences, make_label_array, make_binary_labels, standardize_sequences

# Resolve makes current path absolute and parent goes back one step to ML-Proj
ROOT = Path(__file__).resolve().parent
MODEL_FILE_NAME = "gru_final.pth"
SCALER_FILE_NAME = "scaler.pkl"
PARAMS_FILE_NAME = "best_params.json"
train_file_path = ROOT / "data" / "ae.train"
save_dir = ROOT / "saved_models"

N_FOLDS = 5


# These speakers are treated as authenticated; the remaining 6 are strangers
AUTHENTICATED_SPEAKERS = [0, 1, 2]


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training GRU on GPU...")
    else:
        device = torch.device("cpu")
        print("No GPU detected or using non-cuda torch version, training on CPU...")

    
    # === LOAD AND PREPROCESS DATA ===
    # Load raw padded sequences - shape (270, 29, 12)
    train_seqs_raw = file_to_sequences(train_file_path, pad_sequences=True, max_seq_len=29)
 
    # Standardize - fit only on training data, save scaler for later use on test data
    train_seqs_scaled, scaler = standardize_sequences(train_seqs_raw)
 
    # Convert to tensor for GRU - shape (270, 29, 12)
    train_seqs = torch.from_numpy(train_seqs_scaled)
 
    # === LABELS ===
    # First create original 9-class labels (30 samples per speaker)
    label_counts = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    original_labels = make_label_array(label_counts)
 
    # Remap to binary
    train_labels = torch.from_numpy(make_binary_labels(original_labels, AUTHENTICATED_SPEAKERS))
 
    print(f"\nAuthenticated speakers: {[s + 1 for s in AUTHENTICATED_SPEAKERS]}")
    print(f"Authenticated samples: {train_labels.sum().item()}")
    print(f"Stranger samples: {(train_labels == 0).sum().item()}")

    # === HYPERPARAMETER SEARCH ===
    # Note that epoch patience will not be used for final model
    # it is only there to help find the correct epoch number using early stopping
    param_grid = {
        "hidden_size": [32, 64, 128],
        "bidirectional": [True],
        "dropout_prob": [0.0, 0.2, 0.4],
        "epochs": [3000],
        "learning_rate": [1e-2, 1e-3, 1e-4],
        "l2_weight": [0, 1e-4, 1e-3, 1e-2],
        "batch_size": [8, 16, 32, 64],
        "epoch_patience": [5], 
    }
 
    best_params = hyperparameter_search_gru(param_grid, train_seqs, train_labels, N_FOLDS, device)
    print("Best found hyperparameters:", best_params)
 
    # === FINAL TRAINING ===
    # Only initialze the model with the best_parameters
    input_size = train_seqs.size(2)
    n_labels = len(torch.unique(train_labels))
 
    gru_model = GRUClassifier(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        n_classes=n_labels,
        bidirectional=best_params["bidirectional"],
        dropout_prob=best_params["dropout_prob"],
        device=device,
    ).to(device)
 
    # Train the final model on all training sequences without validation
    # Note that it will only track the training loss
    best_model, _, _ = train_gru_model(
        gru_model,
        train_seqs,
        train_labels,
        val_seqs=None,
        val_labels=None,
        epochs=best_params["epochs"],
        learning_rate=best_params["learning_rate"],
        l2_weight=best_params["l2_weight"],
        batch_size=best_params["batch_size"],
        epoch_patience=best_params["epoch_patience"],
        use_writer=True,
        log_dir="runs/final_training",
    )
 
    # SAVE MODEL, SCALER, AND METADATA 
    # Save all three so they can be used for evaluation and deployment
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / MODEL_FILE_NAME
    torch.save(best_model, save_path)
    print(f"Model saved to {save_path}.")
 
    scaler_path = save_dir / SCALER_FILE_NAME
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}.")
    
    # save best hyperparameters as JSON for the API 
    params_path = save_dir / PARAMS_FILE_NAME
    # Convert numpy values to plain Python types for JSON
    serializable_params = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in best_params.items()
    }
    # derived info the API will need
    serializable_params["n_classes"] = int(n_labels)
    serializable_params["input_size"] = int(input_size)
    serializable_params["authenticated_speakers"] = [int(s) for s in AUTHENTICATED_SPEAKERS]
    with open(params_path, "w") as f:
        json.dump(serializable_params, f, indent=2)
    print(f"Best hyperparameters saved to {params_path}.")

