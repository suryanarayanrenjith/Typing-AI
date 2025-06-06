# train_model.py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def generate_typing_data(num_sequences: int, seq_length: int, mean: float, std: float) -> np.ndarray:
    """
    Generate simulated keystroke timing data.
    """
    return np.random.normal(loc=mean, scale=std, size=(num_sequences, seq_length))

def compute_derivative(seq: np.ndarray) -> np.ndarray:
    """
    Compute the first-order difference (derivative) of a 1D sequence.
    Pads the beginning with 0 to maintain the original length.
    """
    diff = np.diff(seq, n=1)
    diff = np.insert(diff, 0, 0)
    return diff

def load_predefined_data() -> np.ndarray:
    """
    Create a predefined dataset simulating three types of typists:
      - Fast typists (mean=0.15 sec, std=0.03 sec)
      - Average typists (mean=0.20 sec, std=0.05 sec)
      - Slow typists (mean=0.25 sec, std=0.04 sec)
      
    For each sample, compute the derivative and stack it with the raw intervals.
    
    Returns:
        np.ndarray of shape (total_samples, seq_length, 2)
    """
    fast = generate_typing_data(500, 20, 0.15, 0.03)
    average = generate_typing_data(500, 20, 0.20, 0.05)
    slow = generate_typing_data(500, 20, 0.25, 0.04)
    
    data = np.concatenate([fast, average, slow], axis=0)
    np.random.shuffle(data)
    
    # For each sample, compute derivative and stack with the raw intervals.
    new_data = []
    for sample in data:
        derivative = compute_derivative(sample)
        sample_stacked = np.stack([sample, derivative], axis=-1)  # shape: (seq_length, 2)
        new_data.append(sample_stacked)
    new_data = np.array(new_data)
    return new_data

def compute_global_scaler(data: np.ndarray):
    """
    Compute the global mean and standard deviation for each channel (over all samples and timesteps).
    """
    global_mean = np.mean(data, axis=(0, 1))
    global_std = np.std(data, axis=(0, 1)) + 1e-6  # add epsilon to avoid division by zero
    return global_mean, global_std

def normalize_data(data: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray) -> np.ndarray:
    """
    Normalize the data using the provided global mean and std.
    """
    return (data - global_mean) / global_std

def build_autoencoder(timesteps: int, features: int, latent_dim: int = 64) -> Model:
    """
    Build an LSTM autoencoder model with dropout layers.
    """
    input_seq = Input(shape=(timesteps, features))
    x = LSTM(latent_dim, activation='relu')(input_seq)
    x = Dropout(0.2)(x)
    x = RepeatVector(timesteps)(x)
    x = LSTM(latent_dim, activation='relu', return_sequences=True)(x)
    x = Dropout(0.2)(x)
    output_seq = TimeDistributed(Dense(features))(x)
    
    autoencoder = Model(input_seq, output_seq)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def main():
    # Load predefined simulated data
    data = load_predefined_data()  # shape: (num_samples, 20, 2)
    timesteps, features = data.shape[1], data.shape[2]
    
    # Compute global scaler parameters and normalize the data
    global_mean, global_std = compute_global_scaler(data)
    normalized_data = normalize_data(data, global_mean, global_std)
    
    # Build and train the autoencoder
    autoencoder = build_autoencoder(timesteps, features)
    autoencoder.summary()
    
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(normalized_data, normalized_data,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[es],
                    verbose=1)
    
    # Compute reconstruction error on the training data to determine threshold
    errors = []
    for sample in normalized_data:
        sample = sample.reshape((1, timesteps, features))
        reconstructed = autoencoder.predict(sample, verbose=0)
        mse = ((sample - reconstructed) ** 2).mean()
        errors.append(mse)
    errors = np.array(errors)
    threshold = errors.mean() + 3 * errors.std()
    print(f"\nComputed anomaly threshold: {threshold:.6f}")
    
    # Save the trained model, threshold, and scaler parameters
    autoencoder.save("model.h5")
    with open("threshold.txt", "w") as f:
        f.write(str(threshold))
    np.savez("scaler.npz", global_mean=global_mean, global_std=global_std)
    print("Training complete. Model saved to 'model.h5', threshold saved to 'threshold.txt', and scaler parameters saved to 'scaler.npz'.")

if __name__ == "__main__":
    main()
