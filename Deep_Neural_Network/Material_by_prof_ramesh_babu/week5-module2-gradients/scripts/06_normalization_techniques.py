#!/usr/bin/env python3
"""
06_normalization_techniques.py

Purpose: Comprehensive analysis and comparison of normalization techniques
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script demonstrates different normalization techniques (Batch, Layer, Group, Instance)
and their impact on training stability and gradient flow.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def demonstrate_internal_covariate_shift():
    """Demonstrate the internal covariate shift problem"""

    print("🔄 INTERNAL COVARIATE SHIFT DEMONSTRATION")
    print("=" * 60)

    print("Creating a network without normalization to observe covariate shift...")

    # Create a simple network
    model_without_norm = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Generate training data
    X_train = tf.random.normal((1000, 10))
    y_train = tf.cast(tf.reduce_sum(X_train, axis=1) > 0, tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    print(f"Training data shape: {X_train.shape}")

    # Build the model first by calling it with sample data
    _ = model_without_norm(X_train[:1])

    # Track layer activations during training
    activation_tracker = {}

    # Create intermediate models to extract activations (skip if fails)
    layer_outputs = []
    try:
        for i in range(len(model_without_norm.layers)):
            intermediate_model = tf.keras.Model(
                inputs=model_without_norm.input,
                outputs=model_without_norm.layers[i].output
            )
            layer_outputs.append(intermediate_model)
    except Exception as e:
        print(f"⚠️ Warning: Could not create intermediate models - {e}")
        print("Continuing with main demonstration...")

    print("\nTracking activations across training epochs...")

    # Simple training loop to track activations
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    epochs_to_track = [0, 5, 10, 15, 20]

    for epoch in range(21):
        with tf.GradientTape() as tape:
            predictions = model_without_norm(X_train)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train, predictions))

        gradients = tape.gradient(loss, model_without_norm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_without_norm.trainable_variables))

        # Track activations at specific epochs
        if epoch in epochs_to_track and layer_outputs:
            epoch_activations = {}
            for i, layer_model in enumerate(layer_outputs):
                try:
                    activations = layer_model(X_train[:100])  # Use subset for efficiency
                    epoch_activations[f'Layer_{i+1}'] = {
                        'mean': tf.reduce_mean(activations).numpy(),
                        'std': tf.math.reduce_std(activations).numpy(),
                        'activations': activations.numpy()
                    }
                except Exception:
                    continue
            activation_tracker[f'Epoch_{epoch}'] = epoch_activations

            print(f"   Epoch {epoch}: Loss = {loss:.4f}")

    # Visualize covariate shift
    create_covariate_shift_visualization(activation_tracker, epochs_to_track)

    return activation_tracker

def create_covariate_shift_visualization(activation_tracker, epochs):
    """Visualize the internal covariate shift problem"""

    print("\n📊 Creating covariate shift visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot activation distributions for each layer at different epochs
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    for layer_idx in range(3):  # First 3 hidden layers
        row = layer_idx // 2
        col = layer_idx % 2
        ax = axes[row, col] if row < 2 else axes[1, col]

        layer_name = f'Layer_{layer_idx + 1}'

        for i, epoch in enumerate(epochs):
            epoch_key = f'Epoch_{epoch}'
            if epoch_key in activation_tracker and layer_name in activation_tracker[epoch_key]:
                activations = activation_tracker[epoch_key][layer_name]['activations'].flatten()
                ax.hist(activations, bins=30, alpha=0.6, color=colors[i],
                       label=f'Epoch {epoch}', density=True)

        ax.set_title(f'{layer_name} Activation Distributions')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot mean and std changes over epochs
    ax_stats = axes[1, 2]
    layers_to_plot = ['Layer_1', 'Layer_2', 'Layer_3']

    for layer in layers_to_plot:
        means = []
        stds = []
        for epoch in epochs:
            epoch_key = f'Epoch_{epoch}'
            if epoch_key in activation_tracker and layer in activation_tracker[epoch_key]:
                means.append(activation_tracker[epoch_key][layer]['mean'])
                stds.append(activation_tracker[epoch_key][layer]['std'])

        ax_stats.plot(epochs[:len(means)], means, 'o-', linewidth=2,
                     label=f'{layer} Mean', markersize=6)
        ax_stats.plot(epochs[:len(stds)], stds, 's--', linewidth=2,
                     label=f'{layer} Std', markersize=6)

    ax_stats.set_xlabel('Epoch')
    ax_stats.set_ylabel('Activation Statistics')
    ax_stats.set_title('Activation Statistics Over Training')
    ax_stats.legend(fontsize=8)
    ax_stats.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('internal_covariate_shift.png', dpi=300, bbox_inches='tight')
    print("✅ Covariate shift visualization saved as 'internal_covariate_shift.png'")
    plt.show()

def compare_normalization_techniques():
    """Compare different normalization techniques"""

    print("\n🔬 NORMALIZATION TECHNIQUES COMPARISON")
    print("=" * 60)

    # Create models with different normalization techniques
    models = {}

    # 1. No Normalization (baseline)
    models['No Normalization'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 2. Batch Normalization
    models['Batch Normalization'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(32,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 3. Layer Normalization
    models['Layer Normalization'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(32,)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 4. Group Normalization (simulated with Layer Normalization)
    models['Group Normalization'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(32,)),
        tf.keras.layers.LayerNormalization(),  # Simplified as Group Norm
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Generate dataset
    X_train = tf.random.normal((2000, 32))
    y_train = tf.cast(tf.reduce_sum(X_train[:, :16], axis=1) > tf.reduce_sum(X_train[:, 16:], axis=1), tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((400, 32))
    y_val = tf.cast(tf.reduce_sum(X_val[:, :16], axis=1) > tf.reduce_sum(X_val[:, 16:], axis=1), tf.float32)
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

    # Train and compare models
    training_results = {}

    for name, model in models.items():
        print(f"\n🔧 Training model with {name}...")

        try:
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
                epochs=30,
                batch_size=64,
                verbose=0
            )

            training_results[name] = {
                'history': history.history,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'convergence_epoch': np.argmin(history.history['val_loss']) + 1,
                'best_val_acc': np.max(history.history['val_accuracy'])
            }

            print(f"   Final train accuracy: {training_results[name]['final_train_acc']:.4f}")
            print(f"   Final validation accuracy: {training_results[name]['final_val_acc']:.4f}")
            print(f"   Best validation accuracy: {training_results[name]['best_val_acc']:.4f}")
            print(f"   Convergence epoch: {training_results[name]['convergence_epoch']}")

        except Exception as e:
            print(f"   ❌ Training failed for {name}: {e}")

    # Create comprehensive comparison visualization
    create_normalization_comparison_plots(training_results)

    return training_results

def create_normalization_comparison_plots(training_results):
    """Create comprehensive plots comparing normalization techniques"""

    print("\n📊 Creating normalization comparison visualization...")

    if not training_results:
        print("❌ No training results to visualize")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for name, data in training_results.items():
        ax1.plot(data['history']['loss'], linewidth=2, label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for name, data in training_results.items():
        ax2.plot(data['history']['val_loss'], linewidth=2, label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    ax3 = axes[0, 2]
    for name, data in training_results.items():
        ax3.plot(data['history']['accuracy'], linewidth=2, label=name)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Accuracy')
    ax3.set_title('Training Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    ax4 = axes[1, 0]
    for name, data in training_results.items():
        ax4.plot(data['history']['val_accuracy'], linewidth=2, label=name)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Validation Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Final Performance Comparison
    ax5 = axes[1, 1]
    methods = list(training_results.keys())
    final_val_accs = [training_results[m]['final_val_acc'] for m in methods]
    best_val_accs = [training_results[m]['best_val_acc'] for m in methods]

    x_pos = np.arange(len(methods))
    width = 0.35

    bars1 = ax5.bar(x_pos - width/2, final_val_accs, width, label='Final Val Accuracy', alpha=0.7)
    bars2 = ax5.bar(x_pos + width/2, best_val_accs, width, label='Best Val Accuracy', alpha=0.7)

    ax5.set_xlabel('Normalization Method')
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Final Performance Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([m.split()[0] for m in methods], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 6: Convergence Speed
    ax6 = axes[1, 2]
    convergence_epochs = [training_results[m]['convergence_epoch'] for m in methods]

    bars = ax6.bar(methods, convergence_epochs, alpha=0.7, color='skyblue')
    ax6.set_xlabel('Normalization Method')
    ax6.set_ylabel('Convergence Epoch')
    ax6.set_title('Convergence Speed (Lower is Better)')
    ax6.set_xticklabels([m.split()[0] for m in methods], rotation=45)
    ax6.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, convergence_epochs):
        ax6.text(bar.get_x() + bar.get_width()/2, value + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('normalization_techniques_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Normalization comparison saved as 'normalization_techniques_comparison.png'")
    plt.show()

def analyze_batch_normalization_mechanics():
    """Analyze the mechanics of batch normalization in detail"""

    print("\n🔍 BATCH NORMALIZATION MECHANICS ANALYSIS")
    print("=" * 60)

    # Create a simple layer to demonstrate BN mechanics
    batch_size = 100
    feature_dim = 10

    print(f"Analyzing batch normalization with batch_size={batch_size}, features={feature_dim}")

    # Generate sample batch
    X = tf.random.normal((batch_size, feature_dim), mean=2.0, stddev=3.0)
    print(f"Input statistics - Mean: {tf.reduce_mean(X):.4f}, Std: {tf.math.reduce_std(X):.4f}")

    # Manual batch normalization implementation
    def manual_batch_norm(x, epsilon=1e-8):
        """Manual implementation of batch normalization"""
        # Calculate batch statistics
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.reduce_mean(tf.square(x - batch_mean), axis=0)

        # Normalize
        x_normalized = (x - batch_mean) / tf.sqrt(batch_var + epsilon)

        return x_normalized, batch_mean, batch_var

    # Apply manual batch normalization
    X_normalized, batch_mean, batch_var = manual_batch_norm(X)

    print(f"After normalization - Mean: {tf.reduce_mean(X_normalized):.6f}, Std: {tf.math.reduce_std(X_normalized):.6f}")

    # Compare with TensorFlow's BatchNormalization layer
    bn_layer = tf.keras.layers.BatchNormalization()
    X_tf_normalized = bn_layer(X, training=True)

    print(f"TF BatchNorm - Mean: {tf.reduce_mean(X_tf_normalized):.6f}, Std: {tf.math.reduce_std(X_tf_normalized):.6f}")

    # Visualize the normalization effect
    plt.figure(figsize=(15, 5))

    # Original distribution
    plt.subplot(1, 3, 1)
    plt.hist(X.numpy().flatten(), bins=50, alpha=0.7, color='red', density=True)
    plt.title(f'Original Distribution\nMean: {tf.reduce_mean(X):.2f}, Std: {tf.math.reduce_std(X):.2f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Normalized distribution (manual)
    plt.subplot(1, 3, 2)
    plt.hist(X_normalized.numpy().flatten(), bins=50, alpha=0.7, color='green', density=True)
    plt.title(f'Manual Normalization\nMean: {tf.reduce_mean(X_normalized):.6f}, Std: {tf.math.reduce_std(X_normalized):.6f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Normalized distribution (TensorFlow)
    plt.subplot(1, 3, 3)
    plt.hist(X_tf_normalized.numpy().flatten(), bins=50, alpha=0.7, color='blue', density=True)
    plt.title(f'TensorFlow BatchNorm\nMean: {tf.reduce_mean(X_tf_normalized):.6f}, Std: {tf.math.reduce_std(X_tf_normalized):.6f}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('batch_normalization_mechanics.png', dpi=300, bbox_inches='tight')
    print("✅ Batch normalization mechanics saved as 'batch_normalization_mechanics.png'")
    plt.show()

def demonstrate_normalization_during_inference():
    """Demonstrate how normalization behaves during training vs inference"""

    print("\n🔄 NORMALIZATION: TRAINING vs INFERENCE")
    print("=" * 60)

    # Create a model with batch normalization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(10,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Generate training data
    X_train = tf.random.normal((1000, 10))
    y_train = tf.random.uniform((1000, 1))

    # Compile and train briefly
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=5, verbose=0)

    # Test data for comparison
    X_test = tf.random.normal((100, 10), mean=1.0, stddev=2.0)  # Different distribution

    print("Testing batch normalization behavior...")

    # Predictions during training mode
    predictions_training = model(X_test, training=True)
    print(f"Training mode predictions - Mean: {tf.reduce_mean(predictions_training):.4f}, Std: {tf.math.reduce_std(predictions_training):.4f}")

    # Predictions during inference mode
    predictions_inference = model(X_test, training=False)
    print(f"Inference mode predictions - Mean: {tf.reduce_mean(predictions_inference):.4f}, Std: {tf.math.reduce_std(predictions_inference):.4f}")

    # Get batch norm layers for detailed analysis
    bn_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]

    print(f"\nAnalyzing {len(bn_layers)} BatchNormalization layers...")

    for i, bn_layer in enumerate(bn_layers):
        print(f"\nBatchNorm Layer {i+1}:")
        print(f"  Moving mean: {tf.reduce_mean(bn_layer.moving_mean):.6f}")
        print(f"  Moving variance: {tf.reduce_mean(bn_layer.moving_variance):.6f}")
        print(f"  Gamma (scale): {tf.reduce_mean(bn_layer.gamma):.6f}")
        print(f"  Beta (shift): {tf.reduce_mean(bn_layer.beta):.6f}")

def create_normalization_summary_table():
    """Create a comprehensive summary of normalization techniques"""

    print("\n📋 NORMALIZATION TECHNIQUES SUMMARY")
    print("=" * 80)

    normalization_data = {
        'Batch Normalization': {
            'Normalize Over': 'Batch dimension',
            'Training/Inference': 'Different',
            'Memory': 'O(features)',
            'Best For': 'Large batches, CNNs',
            'Pros': 'Faster training, higher LR',
            'Cons': 'Batch size dependent'
        },
        'Layer Normalization': {
            'Normalize Over': 'Feature dimension',
            'Training/Inference': 'Same',
            'Memory': 'O(features)',
            'Best For': 'RNNs, small batches',
            'Pros': 'Batch size independent',
            'Cons': 'Slower than BatchNorm'
        },
        'Group Normalization': {
            'Normalize Over': 'Channel groups',
            'Training/Inference': 'Same',
            'Memory': 'O(features)',
            'Best For': 'Small batches, computer vision',
            'Pros': 'Stable across batch sizes',
            'Cons': 'Extra hyperparameter'
        },
        'Instance Normalization': {
            'Normalize Over': 'Spatial dimensions',
            'Training/Inference': 'Same',
            'Memory': 'O(channels)',
            'Best For': 'Style transfer, GANs',
            'Pros': 'Good for style tasks',
            'Cons': 'Limited applicability'
        }
    }

    # Print comparison table
    headers = ['Technique', 'Normalize Over', 'Train/Inference', 'Best For', 'Main Advantage']
    print(f"{headers[0]:<20} {headers[1]:<18} {headers[2]:<15} {headers[3]:<20} {headers[4]:<25}")
    print("-" * 100)

    for technique, info in normalization_data.items():
        print(f"{technique:<20} {info['Normalize Over']:<18} {info['Training/Inference']:<15} "
              f"{info['Best For']:<20} {info['Pros']:<25}")

    print("-" * 100)

    print("\n🎯 QUICK SELECTION GUIDE:")
    print("✅ GENERAL PURPOSE: Batch Normalization (most common choice)")
    print("🔧 SMALL BATCHES: Layer Normalization or Group Normalization")
    print("🎨 STYLE TASKS: Instance Normalization")
    print("⚠️ SEQUENTIAL DATA: Layer Normalization (not Batch Normalization)")

    print("\n💡 IMPLEMENTATION TIPS:")
    print("1. Place normalization BEFORE activation (common) or AFTER (alternative)")
    print("2. Use momentum=0.99 for batch normalization moving averages")
    print("3. Epsilon=1e-8 prevents division by zero")
    print("4. Initialize gamma=1, beta=0 for identity transform initially")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("📊 NORMALIZATION TECHNIQUES COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()

    try:
        # Demonstrate internal covariate shift
        activation_tracker = demonstrate_internal_covariate_shift()

        # Compare normalization techniques
        training_results = compare_normalization_techniques()

        # Analyze batch normalization mechanics
        analyze_batch_normalization_mechanics()

        # Demonstrate training vs inference behavior
        demonstrate_normalization_during_inference()

        # Create summary table
        create_normalization_summary_table()

        print("\n" + "=" * 80)
        print("✅ NORMALIZATION TECHNIQUES ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - internal_covariate_shift.png")
        print("   - normalization_techniques_comparison.png")
        print("   - batch_normalization_mechanics.png")
        print("🎓 Learning outcome: Understanding normalization impact on training stability")
        print("💡 Next step: Learn about residual connections and skip connections")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()