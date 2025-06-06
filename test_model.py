# test_model.py
import time
import numpy as np
from pynput import keyboard
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def compute_derivative(seq: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of a 1D sequence and pad it to maintain the original length.
    """
    diff = np.diff(seq, n=1)
    diff = np.insert(diff, 0, 0)
    return diff

def collect_keystroke_intervals(prompt: str, required_intervals: int = 20) -> np.ndarray:
    """
    Display a prompt and collect keystroke intervals (in seconds) until Enter is pressed.
    Returns an array of the first `required_intervals` intervals, or None if insufficient data is captured.
    """
    print(prompt)
    intervals = []
    last_time = None

    def on_press(key):
        nonlocal last_time, intervals
        current_time = time.time()
        if last_time is not None:
            intervals.append(current_time - last_time)
        last_time = current_time
        if key == keyboard.Key.enter:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
    if len(intervals) < required_intervals:
        print(f"Only captured {len(intervals)} intervals; need at least {required_intervals}.")
        return None
    return np.array(intervals[:required_intervals])

def load_scaler():
    """
    Load the scaler parameters (global mean and std) from file.
    """
    npzfile = np.load("scaler.npz")
    return npzfile["global_mean"], npzfile["global_std"]

def normalize_data(data: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray) -> np.ndarray:
    """
    Normalize data using the provided global scaler parameters.
    """
    return (data - global_mean) / global_std

def main():
    # Load the pre-trained model, threshold, and scaler parameters.
    model = load_model("model.h5", custom_objects={'mse': MeanSquaredError()})
    with open("threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    global_mean, global_std = load_scaler()
    print(f"Loaded model. Using threshold: {threshold:.6f}\n")
    
    print("=== Typing Pattern Evaluation ===")
    print("When prompted, please type the following sentence exactly and press Enter:")
    print("'The quick brown fox jumps over the lazy dog'\n")
    
    input("Press Enter to start a typing session (Ctrl+C to exit)...")
    intervals = collect_keystroke_intervals("Please type the sentence now:", required_intervals=20)
    if intervals is None:
        print("Insufficient data captured. Exiting.")
        return
    
    # Compute the derivative for the captured intervals and stack to form a two-channel sample.
    derivative = compute_derivative(intervals)
    sample = np.stack([intervals, derivative], axis=-1)  # shape: (20, 2)
    
    # Normalize the sample using the global scaler parameters
    normalized_sample = normalize_data(sample, global_mean, global_std)
    normalized_sample = normalized_sample.reshape((1, normalized_sample.shape[0], normalized_sample.shape[1]))
    
    # Compute the reconstruction error for the sample
    reconstructed = model.predict(normalized_sample, verbose=0)
    mse = ((normalized_sample - reconstructed) ** 2).mean()
    
    # If the reconstruction error exceeds the threshold, flag as abnormal.
    if mse > threshold:
        print(f"\nAbnormal typing detected! Reconstruction error: {mse:.6f} (Threshold: {threshold:.6f})")
    else:
        print(f"\nTyping pattern is normal. Reconstruction error: {mse:.6f} (Threshold: {threshold:.6f})")

if __name__ == "__main__":
    main()
