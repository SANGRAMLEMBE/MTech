#!/usr/bin/env python3
"""
14_advanced_regularization.py

Purpose: Demonstrate advanced regularization techniques for deep neural networks
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements various advanced regularization techniques to improve
generalization and training stability in deep networks.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def implement_dropconnect():
    """Implement DropConnect regularization"""

    print("🔗 IMPLEMENTING DROPCONNECT REGULARIZATION")
    print("=" * 60)

    class DropConnect(tf.keras.layers.Layer):
        """DropConnect layer implementation"""

        def __init__(self, units, drop_rate=0.5, activation=None, **kwargs):
            super(DropConnect, self).__init__(**kwargs)
            self.units = units
            self.drop_rate = drop_rate
            self.activation = tf.keras.activations.get(activation)

        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='kernel'
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
            super(DropConnect, self).build(input_shape)

        def call(self, inputs, training=None):
            if training:
                # Create dropout mask for connections
                mask_shape = tf.shape(self.kernel)
                random_tensor = tf.random.uniform(mask_shape)
                dropout_mask = tf.cast(random_tensor > self.drop_rate, tf.float32)

                # Scale to maintain expected value
                scale = 1.0 / (1.0 - self.drop_rate)
                masked_kernel = self.kernel * dropout_mask * scale
            else:
                masked_kernel = self.kernel

            # Compute output
            output = tf.matmul(inputs, masked_kernel) + self.bias

            if self.activation is not None:
                output = self.activation(output)

            return output

        def get_config(self):
            config = super(DropConnect, self).get_config()
            config.update({
                'units': self.units,
                'drop_rate': self.drop_rate,
                'activation': tf.keras.activations.serialize(self.activation)
            })
            return config

    print("✅ DropConnect implemented")
    print("Key features:")
    print("  • Drops connections (weights) instead of neurons")
    print("  • More fine-grained regularization than Dropout")
    print("  • Maintains activation scaling")

    return DropConnect

def implement_stochastic_depth():
    """Implement Stochastic Depth regularization"""

    print("\n🎲 IMPLEMENTING STOCHASTIC DEPTH")
    print("=" * 60)

    class StochasticDepth(tf.keras.layers.Layer):
        """Stochastic Depth layer for residual blocks"""

        def __init__(self, drop_rate=0.1, **kwargs):
            super(StochasticDepth, self).__init__(**kwargs)
            self.drop_rate = drop_rate

        def call(self, inputs, training=None):
            if not training:
                return inputs

            # inputs should be [main_path, shortcut_path]
            main_path, shortcut_path = inputs

            # Randomly drop the main path during training
            batch_size = tf.shape(main_path)[0]
            random_tensor = tf.random.uniform([batch_size, 1, 1])
            keep_prob = 1.0 - self.drop_rate

            # Create binary mask
            binary_mask = tf.cast(random_tensor < keep_prob, tf.float32)

            # Apply stochastic depth
            output = shortcut_path + binary_mask * main_path

            return output

        def get_config(self):
            config = super(StochasticDepth, self).get_config()
            config.update({'drop_rate': self.drop_rate})
            return config

    print("✅ Stochastic Depth implemented")
    print("Key features:")
    print("  • Randomly skips residual blocks during training")
    print("  • Reduces effective network depth randomly")
    print("  • Improves gradient flow in very deep networks")

    return StochasticDepth

def implement_spectral_normalization():
    """Implement Spectral Normalization"""

    print("\n📏 IMPLEMENTING SPECTRAL NORMALIZATION")
    print("=" * 60)

    class SpectralNormalization(tf.keras.layers.Wrapper):
        """Spectral Normalization wrapper for layers"""

        def __init__(self, layer, power_iterations=1, **kwargs):
            super(SpectralNormalization, self).__init__(layer, **kwargs)
            self.power_iterations = power_iterations

        def build(self, input_shape):
            super(SpectralNormalization, self).build(input_shape)

            # Get the kernel shape
            if hasattr(self.layer, 'kernel'):
                kernel_shape = self.layer.kernel.shape
                # Initialize u and v vectors for power iteration
                self.u = self.add_weight(
                    shape=(1, kernel_shape[-1]),
                    initializer='random_normal',
                    trainable=False,
                    name='u'
                )

        def call(self, inputs, training=None):
            if hasattr(self.layer, 'kernel'):
                # Reshape kernel to 2D
                kernel = self.layer.kernel
                kernel_shape = tf.shape(kernel)
                kernel_2d = tf.reshape(kernel, [-1, kernel_shape[-1]])

                # Power iteration method
                u = self.u
                for _ in range(self.power_iterations):
                    v = tf.nn.l2_normalize(tf.matmul(u, kernel_2d, transpose_b=True))
                    u = tf.nn.l2_normalize(tf.matmul(v, kernel_2d))

                # Update u
                if training:
                    self.u.assign(u)

                # Compute spectral norm
                sigma = tf.reduce_sum(tf.matmul(u, kernel_2d) * v)

                # Normalize kernel
                normalized_kernel = kernel / sigma
                self.layer.kernel.assign(normalized_kernel)

            return self.layer(inputs, training=training)

        def get_config(self):
            config = super(SpectralNormalization, self).get_config()
            config.update({'power_iterations': self.power_iterations})
            return config

    print("✅ Spectral Normalization implemented")
    print("Key features:")
    print("  • Constrains spectral norm of weight matrices")
    print("  • Improves training stability")
    print("  • Popular in GANs and other generative models")

    return SpectralNormalization

def implement_cutout_mixup():
    """Implement Cutout and Mixup data augmentation techniques"""

    print("\n✂️ IMPLEMENTING CUTOUT AND MIXUP")
    print("=" * 60)

    def cutout_augmentation(images, mask_size=4):
        """Apply cutout augmentation to images"""
        batch_size = tf.shape(images)[0]
        image_height = tf.shape(images)[1]
        image_width = tf.shape(images)[2]

        # Random center for cutout
        center_x = tf.random.uniform([batch_size], 0, image_width, dtype=tf.int32)
        center_y = tf.random.uniform([batch_size], 0, image_height, dtype=tf.int32)

        # Create masks
        masks = []
        for i in range(batch_size):
            # Create coordinate grids
            y_coords, x_coords = tf.meshgrid(
                tf.range(image_height), tf.range(image_width), indexing='ij'
            )

            # Create mask
            mask = tf.logical_or(
                tf.logical_or(
                    x_coords < center_x[i] - mask_size // 2,
                    x_coords > center_x[i] + mask_size // 2
                ),
                tf.logical_or(
                    y_coords < center_y[i] - mask_size // 2,
                    y_coords > center_y[i] + mask_size // 2
                )
            )
            masks.append(mask)

        # Stack masks and apply
        final_mask = tf.stack(masks)
        final_mask = tf.cast(final_mask, tf.float32)
        final_mask = tf.expand_dims(final_mask, -1)

        return images * final_mask

    def mixup_data(x, y, alpha=1.0):
        """Apply mixup augmentation"""
        batch_size = tf.shape(x)[0]

        # Sample lambda from Beta distribution
        if alpha > 0:
            lam = tf.random.gamma([batch_size], alpha, 1) / tf.random.gamma([batch_size], alpha, 1)
            lam = tf.clip_by_value(lam, 0, 1)
        else:
            lam = tf.ones([batch_size])

        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))

        # Expand lambda for broadcasting
        lam_x = tf.reshape(lam, [-1] + [1] * (len(x.shape) - 1))
        lam_y = tf.reshape(lam, [-1] + [1] * (len(y.shape) - 1))

        # Mix inputs and targets
        mixed_x = lam_x * x + (1 - lam_x) * tf.gather(x, indices)
        mixed_y = lam_y * y + (1 - lam_y) * tf.gather(y, indices)

        return mixed_x, mixed_y

    class CutoutLayer(tf.keras.layers.Layer):
        """Cutout as a Keras layer"""

        def __init__(self, mask_size=4, **kwargs):
            super(CutoutLayer, self).__init__(**kwargs)
            self.mask_size = mask_size

        def call(self, inputs, training=None):
            if training:
                return cutout_augmentation(inputs, self.mask_size)
            return inputs

        def get_config(self):
            config = super(CutoutLayer, self).get_config()
            config.update({'mask_size': self.mask_size})
            return config

    print("✅ Cutout and Mixup implemented")
    print("Key features:")
    print("  • Cutout: Randomly masks patches in images")
    print("  • Mixup: Linearly interpolates between training examples")
    print("  • Both improve generalization and robustness")

    return CutoutLayer, cutout_augmentation, mixup_data

def implement_shake_shake():
    """Implement Shake-Shake regularization"""

    print("\n🤝 IMPLEMENTING SHAKE-SHAKE REGULARIZATION")
    print("=" * 60)

    class ShakeShake(tf.keras.layers.Layer):
        """Shake-Shake regularization for residual networks"""

        def __init__(self, **kwargs):
            super(ShakeShake, self).__init__(**kwargs)

        def call(self, inputs, training=None):
            # inputs should be [branch1, branch2, shortcut]
            branch1, branch2, shortcut = inputs

            if training:
                # Generate random shake factors
                batch_size = tf.shape(branch1)[0]
                alpha = tf.random.uniform([batch_size, 1, 1], 0, 1)

                # Forward shake
                forward_shake = alpha * branch1 + (1 - alpha) * branch2

                # Backward shake (different alpha for gradients)
                beta = tf.random.uniform([batch_size, 1, 1], 0, 1)
                backward_shake = beta * branch1 + (1 - beta) * branch2

                # Use stop_gradient trick for different forward/backward behavior
                shake_output = forward_shake + tf.stop_gradient(backward_shake - forward_shake)
            else:
                # During inference, use equal weights
                shake_output = 0.5 * branch1 + 0.5 * branch2

            return shortcut + shake_output

        def get_config(self):
            return super(ShakeShake, self).get_config()

    print("✅ Shake-Shake implemented")
    print("Key features:")
    print("  • Randomly mixes two residual branches during training")
    print("  • Different mixing for forward and backward passes")
    print("  • Reduces overfitting in residual networks")

    return ShakeShake

def compare_regularization_techniques():
    """Compare different regularization techniques"""

    print("\n⚔️ COMPARING REGULARIZATION TECHNIQUES")
    print("=" * 60)

    # Generate synthetic dataset
    def create_complex_dataset(num_samples=2000, noise_level=0.1):
        """Create a complex dataset prone to overfitting"""
        X = np.random.randn(num_samples, 20)

        # Create complex non-linear pattern
        y = (
            np.sin(X[:, 0]) * np.cos(X[:, 1]) +
            np.tanh(X[:, 2] * X[:, 3]) +
            np.exp(-0.5 * X[:, 4]**2) +
            0.1 * np.sum(X[:, 5:15], axis=1)
        )

        # Add noise
        y += np.random.normal(0, noise_level, y.shape)

        # Convert to classification
        y = (y > np.median(y)).astype(int)

        return X.astype(np.float32), y.astype(np.float32)

    # Create dataset
    X_full, y_full = create_complex_dataset(3000)

    # Split data
    train_size = 1500
    val_size = 500
    test_size = 1000

    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    X_val = X_full[train_size:train_size+val_size]
    y_val = y_full[train_size:train_size+val_size]
    X_test = X_full[train_size+val_size:]
    y_test = y_full[train_size+val_size:]

    print(f"Dataset splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Get regularization techniques
    DropConnect = implement_dropconnect()
    CutoutLayer, _, mixup_data = implement_cutout_mixup()

    # Define models with different regularization
    models = {}

    # 1. Baseline (no regularization)
    models['Baseline'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 2. Standard dropout
    models['Dropout'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 3. DropConnect
    models['DropConnect'] = tf.keras.Sequential([
        DropConnect(128, drop_rate=0.3, activation='relu', input_shape=(20,)),
        DropConnect(128, drop_rate=0.3, activation='relu'),
        DropConnect(64, drop_rate=0.2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 4. L2 Regularization
    models['L2 Regularization'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(20,),
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 5. Batch Normalization
    models['BatchNorm'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(20,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Train and evaluate models
    training_results = {}

    print(f"\n{'Model':<18} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
    print("-" * 75)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Calculate overfitting (train - validation gap)
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        overfitting = final_train_acc - final_val_acc

        # Store results
        training_results[name] = {
            'history': history.history,
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'test_acc': test_acc,
            'overfitting': overfitting
        }

        print(f"{name:<18} {final_train_acc:<12.4f} {final_val_acc:<12.4f} {test_acc:<12.4f} {overfitting:<12.4f}")

    print("-" * 75)

    # Create comparison visualization
    create_regularization_comparison_plots(training_results)

    return training_results

def create_regularization_comparison_plots(training_results):
    """Create comprehensive comparison plots for regularization techniques"""

    print("\n📊 Creating regularization comparison visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training curves
    ax1 = axes[0, 0]
    for name, results in training_results.items():
        epochs = range(1, len(results['history']['accuracy']) + 1)
        ax1.plot(epochs, results['history']['accuracy'], '--', alpha=0.7, label=f'{name} (train)')
        ax1.plot(epochs, results['history']['val_accuracy'], '-', linewidth=2, label=f'{name} (val)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Curves Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overfitting comparison
    ax2 = axes[0, 1]
    methods = list(training_results.keys())
    overfitting_scores = [training_results[m]['overfitting'] for m in methods]

    bars = ax2.bar(range(len(methods)), overfitting_scores, alpha=0.7,
                   color=['red' if score > 0.1 else 'orange' if score > 0.05 else 'green'
                          for score in overfitting_scores])

    ax2.set_xlabel('Regularization Method')
    ax2.set_ylabel('Overfitting Score (Train - Val Acc)')
    ax2.set_title('Overfitting Comparison (Lower is Better)')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45)

    # Add value labels
    for bar, score in zip(bars, overfitting_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild overfitting')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Severe overfitting')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Test accuracy comparison
    ax3 = axes[1, 0]
    test_accs = [training_results[m]['test_acc'] for m in methods]

    bars = ax3.bar(range(len(methods)), test_accs, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Regularization Method')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Generalization Performance (Test Accuracy)')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45)

    # Add value labels
    for bar, acc in zip(bars, test_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax3.grid(True, alpha=0.3)

    # Plot 4: Regularization effectiveness
    ax4 = axes[1, 1]

    # Create scatter plot of overfitting vs test accuracy
    for i, method in enumerate(methods):
        overfitting = training_results[method]['overfitting']
        test_acc = training_results[method]['test_acc']
        ax4.scatter(overfitting, test_acc, s=100, alpha=0.7, label=method)
        ax4.annotate(method, (overfitting, test_acc), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Overfitting Score')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Regularization Effectiveness\n(Lower left is better)')
    ax4.grid(True, alpha=0.3)

    # Add ideal region
    ax4.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(y=max(test_accs) - 0.02, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Regularization comparison saved as 'regularization_comparison.png'")
    plt.show()

def analyze_regularization_effects():
    """Analyze the effects of different regularization techniques"""

    print("\n🔍 ANALYZING REGULARIZATION EFFECTS")
    print("=" * 60)

    regularization_analysis = {
        'Dropout': {
            'Mechanism': 'Randomly sets neurons to zero during training',
            'Effect': 'Prevents co-adaptation of neurons',
            'Best For': 'Standard feedforward and convolutional networks',
            'Hyperparameters': 'dropout_rate (0.2-0.5)',
            'Computational Cost': 'Minimal'
        },
        'DropConnect': {
            'Mechanism': 'Randomly sets connections (weights) to zero',
            'Effect': 'More fine-grained regularization than dropout',
            'Best For': 'Dense layers with many parameters',
            'Hyperparameters': 'drop_rate (0.1-0.5)',
            'Computational Cost': 'Low'
        },
        'Batch Normalization': {
            'Mechanism': 'Normalizes inputs to each layer',
            'Effect': 'Reduces internal covariate shift, acts as regularizer',
            'Best For': 'Deep networks, convolutional networks',
            'Hyperparameters': 'momentum (0.9-0.99), epsilon (1e-8)',
            'Computational Cost': 'Low'
        },
        'L2 Regularization': {
            'Mechanism': 'Adds penalty term proportional to weight squares',
            'Effect': 'Prevents large weights, smooth decision boundaries',
            'Best For': 'Linear models, preventing weight explosion',
            'Hyperparameters': 'lambda (1e-4 to 1e-2)',
            'Computational Cost': 'Minimal'
        },
        'Stochastic Depth': {
            'Mechanism': 'Randomly skips layers during training',
            'Effect': 'Reduces effective depth, improves gradient flow',
            'Best For': 'Very deep residual networks',
            'Hyperparameters': 'drop_rate (0.1-0.5)',
            'Computational Cost': 'Negative (faster training)'
        },
        'Mixup': {
            'Mechanism': 'Linearly interpolates training examples',
            'Effect': 'Smooths decision boundaries, robust to label noise',
            'Best For': 'Image classification, noisy labels',
            'Hyperparameters': 'alpha (0.2-1.0)',
            'Computational Cost': 'Minimal'
        },
        'Cutout': {
            'Mechanism': 'Randomly masks patches in input',
            'Effect': 'Forces model to focus on multiple features',
            'Best For': 'Image classification, computer vision',
            'Hyperparameters': 'patch_size (4-16 for images)',
            'Computational Cost': 'Low'
        }
    }

    print("📊 REGULARIZATION TECHNIQUES ANALYSIS:")
    print(f"{'Technique':<18} {'Mechanism':<35} {'Best For':<25}")
    print("-" * 80)

    for technique, info in regularization_analysis.items():
        print(f"{technique:<18} {info['Mechanism']:<35} {info['Best For']:<25}")

    print("\n🎯 SELECTION GUIDELINES:")

    guidelines = {
        'Network Type': {
            'Feedforward': 'Dropout, L2 regularization, Batch normalization',
            'Convolutional': 'Dropout, Batch normalization, Cutout',
            'Residual': 'Batch normalization, Stochastic depth, Shake-shake',
            'Recurrent': 'Dropout (on non-recurrent connections), L2 regularization'
        },
        'Problem Type': {
            'Image Classification': 'Batch normalization, Cutout, Mixup',
            'Natural Language': 'Dropout, Layer normalization, Label smoothing',
            'Tabular Data': 'Dropout, L2 regularization, Feature noise',
            'Small Datasets': 'Strong regularization (high dropout, L2)',
            'Large Datasets': 'Batch normalization, mild regularization'
        },
        'Training Issues': {
            'Overfitting': 'Increase dropout, add L2 regularization',
            'Slow Convergence': 'Batch normalization, reduce regularization',
            'Gradient Problems': 'Batch normalization, residual connections',
            'Label Noise': 'Mixup, label smoothing'
        }
    }

    for category, recommendations in guidelines.items():
        print(f"\n💡 {category.upper()}:")
        for problem, solution in recommendations.items():
            print(f"  • {problem}: {solution}")

def create_regularization_implementation_guide():
    """Create practical guide for implementing regularization"""

    print("\n📋 REGULARIZATION IMPLEMENTATION GUIDE")
    print("=" * 80)

    print("🎯 IMPLEMENTATION PRIORITIES:")
    print("1. 🥇 FIRST: Always start with Batch Normalization")
    print("2. 🥈 SECOND: Add appropriate dropout (0.2-0.5)")
    print("3. 🥉 THIRD: Consider L2 regularization if still overfitting")
    print("4. 🏅 ADVANCED: Experiment with domain-specific techniques")

    print("\n🔧 TENSORFLOW IMPLEMENTATION EXAMPLES:")

    implementation_examples = {
        'Standard Dropout': '''
# Standard dropout implementation
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
''',
        'Batch Normalization': '''
# Batch normalization (recommended placement)
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
''',
        'L2 Regularization': '''
# L2 regularization
model.add(tf.keras.layers.Dense(128, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.01)))
''',
        'Mixup Data Augmentation': '''
# Mixup implementation
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    indices = tf.random.shuffle(tf.range(tf.shape(x)[0]))

    mixed_x = lam * x + (1 - lam) * tf.gather(x, indices)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, indices)

    return mixed_x, mixed_y
'''
    }

    for technique, code in implementation_examples.items():
        print(f"\n🔧 {technique}:")
        print(code)

    print("\n⚙️ HYPERPARAMETER TUNING GUIDELINES:")

    tuning_guidelines = [
        "🎯 Dropout Rate: Start with 0.3, adjust based on overfitting",
        "📏 L2 Lambda: Start with 1e-3, increase if overfitting persists",
        "🔄 Batch Norm Momentum: Use default 0.99, rarely needs tuning",
        "🎲 Mixup Alpha: 0.2-1.0, higher values = more aggressive mixing",
        "✂️ Cutout Size: 10-20% of image size for vision tasks",
        "📊 Monitor: Validation accuracy gap (train - val) as overfitting metric"
    ]

    for guideline in tuning_guidelines:
        print(f"  • {guideline}")

    print("\n⚠️ COMMON MISTAKES:")
    common_mistakes = [
        "❌ Applying dropout to input layer (usually unnecessary)",
        "❌ Using too high dropout rates (>0.8) in hidden layers",
        "❌ Forgetting to disable dropout during inference",
        "❌ Over-regularizing small models or small datasets",
        "❌ Not using batch normalization in deep networks",
        "❌ Applying regularization when model is already underfitting"
    ]

    for mistake in common_mistakes:
        print(f"  • {mistake}")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🛡️ ADVANCED REGULARIZATION TECHNIQUES")
    print("=" * 80)
    print()

    try:
        # Implement various regularization techniques
        DropConnect = implement_dropconnect()
        StochasticDepth = implement_stochastic_depth()
        SpectralNormalization = implement_spectral_normalization()
        CutoutLayer, _, _ = implement_cutout_mixup()
        ShakeShake = implement_shake_shake()

        # Compare regularization techniques
        training_results = compare_regularization_techniques()

        # Analyze regularization effects
        analyze_regularization_effects()

        # Create implementation guide
        create_regularization_implementation_guide()

        print("\n" + "=" * 80)
        print("✅ ADVANCED REGULARIZATION ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - regularization_comparison.png")
        print("🎓 Learning outcome: Understanding advanced regularization for better generalization")
        print("💡 Next step: Learn about future directions and emerging techniques")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()