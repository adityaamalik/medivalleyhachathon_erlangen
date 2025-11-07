"""
Deep Learning Pipeline for PPG Arrhythmia Detection
Approach A: Minimal preprocessing + 1D CNN

Signal Characteristics:
- Sampling rate: 100 Hz
- Segment duration: 10 seconds
- Samples per segment: 1,000 values
- Binary classification: Healthy (0) vs Arrhythmic (1)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


class PPGPreprocessor:
    """Minimal preprocessing for PPG signals"""
    
    def __init__(self, sampling_rate=100, lowcut=0.5, highcut=8.0):
        """
        Initialize preprocessor
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 100)
            lowcut: Lower cutoff frequency for bandpass filter (default: 0.5 Hz)
            highcut: Upper cutoff frequency for bandpass filter (default: 8.0 Hz)
        """
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        
    def bandpass_filter(self, ppg_signal):
        """
        Apply bandpass filter to remove noise outside physiological range
        
        Args:
            ppg_signal: 1D array of PPG signal
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, ppg_signal)
        
        return filtered
    
    def normalize(self, ppg_signal):
        """
        Z-score normalization
        
        Args:
            ppg_signal: 1D array of PPG signal
            
        Returns:
            Normalized signal
        """
        mean = np.mean(ppg_signal)
        std = np.std(ppg_signal)
        
        if std == 0:
            return ppg_signal - mean
        
        return (ppg_signal - mean) / std
    
    def preprocess(self, ppg_signal):
        """
        Complete preprocessing pipeline
        
        Args:
            ppg_signal: 1D array of PPG signal
            
        Returns:
            Preprocessed signal
        """
        # Apply bandpass filter
        filtered = self.bandpass_filter(ppg_signal)
        
        # Normalize
        normalized = self.normalize(filtered)
        
        return normalized
    
    def preprocess_batch(self, ppg_signals):
        """
        Preprocess a batch of signals
        
        Args:
            ppg_signals: 2D array of shape (n_samples, signal_length)
            
        Returns:
            Preprocessed signals
        """
        preprocessed = np.zeros_like(ppg_signals)
        
        for i in range(len(ppg_signals)):
            preprocessed[i] = self.preprocess(ppg_signals[i])
        
        return preprocessed


class DataAugmentation:
    """Data augmentation for PPG signals"""
    
    @staticmethod
    def add_noise(ppg_signal, noise_level=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, ppg_signal.shape)
        return ppg_signal + noise
    
    @staticmethod
    def time_warp(ppg_signal, sigma=0.2):
        """Apply time warping"""
        from scipy.interpolate import interp1d
        
        orig_steps = np.arange(len(ppg_signal))
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(3,))
        
        warp_steps = np.zeros(len(ppg_signal))
        warp_steps[0] = 0
        warp_steps[-1] = len(ppg_signal) - 1
        
        for i in range(1, len(ppg_signal) - 1):
            warp_steps[i] = warp_steps[i-1] + random_warps[i % len(random_warps)]
        
        warp_steps = (warp_steps / warp_steps[-1]) * (len(ppg_signal) - 1)
        
        f = interp1d(warp_steps, ppg_signal, kind='cubic', fill_value='extrapolate')
        warped = f(orig_steps)
        
        return warped
    
    @staticmethod
    def magnitude_scale(ppg_signal, sigma=0.1):
        """Scale magnitude"""
        scale_factor = np.random.normal(loc=1.0, scale=sigma)
        return ppg_signal * scale_factor
    
    @staticmethod
    def time_shift(ppg_signal, max_shift=50):
        """Shift signal in time"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(ppg_signal, shift)


def build_1d_cnn_model(input_shape=(1000, 1), num_classes=1):
    """
    Build 1D CNN model for PPG arrhythmia detection
    
    Args:
        input_shape: Shape of input (signal_length, channels)
        num_classes: Number of output classes (1 for binary classification)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Fourth convolutional block
        layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def create_callbacks(model_path='best_model.keras'):
    """
    Create training callbacks
    
    Args:
        model_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    callback_list = [
        # Save best model
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    return callback_list


def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Keras training history object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train')
    axes[1, 0].plot(history.history['val_auc'], label='Validation')
    axes[1, 0].set_title('Model AUC', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision', linestyle='--')
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linestyle='--')
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linestyle='-')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linestyle='-')
    axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=['Healthy', 'Arrhythmic']):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
