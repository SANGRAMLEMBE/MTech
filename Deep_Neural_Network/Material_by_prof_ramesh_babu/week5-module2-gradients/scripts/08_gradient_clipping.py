#!/usr/bin/env python3
"""
08_gradient_clipping.py

Purpose: Comprehensive demonstration of gradient clipping techniques
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script demonstrates different gradient clipping methods to solve gradient explosion
and provides practical guidelines for implementation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_explosion_prone_network():
    """Create a network prone to gradient explosion"""

    print("🏗️ Creating gradient explosion-prone network...")

    # Network with poor initialization that causes gradient explosion
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='linear', input_shape=(20,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.5)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print("✅ Created explosion-prone network with poor initialization")
    return model

def demonstrate_gradient_explosion():
    """Demonstrate gradient explosion without clipping"""

    print("\n💥 DEMONSTRATING GRADIENT EXPLOSION")
    print("=" * 60)

    # Create explosion-prone model
    model = create_explosion_prone_network()

    # Generate training data
    X_train = tf.random.normal((1000, 20))
    y_train = tf.cast(tf.reduce_sum(X_train[:, :10], axis=1) > 0, tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    print(f"Training data shape: {X_train.shape}")

    # Track gradients during training without clipping
    gradient_history = []
    loss_history = []
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)  # High LR to trigger explosion

    print("\nTraining without gradient clipping...")
    print(f"{'Epoch':<8} {'Loss':<12} {'Max Gradient':<15} {'Status':<20}")
    print("-" * 55)

    for epoch in range(20):
        with tf.GradientTape() as tape:
            predictions = model(X_train)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train, predictions))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Calculate gradient statistics
        grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
        max_grad = max(grad_norms) if grad_norms else 0.0

        gradient_history.append(max_grad)
        loss_history.append(loss.numpy())

        # Determine status
        if max_grad > 100:
            status = "🚨 EXPLODING!"
        elif max_grad > 10:
            status = "⚠️ WARNING"
        elif np.isnan(max_grad) or np.isinf(max_grad):
            status = "💀 NaN/Inf"
            break
        else:
            status = "✅ Normal"

        print(f"{epoch+1:<8} {loss.numpy():<12.6f} {max_grad:<15.2f} {status:<20}")

        # Apply gradients (without clipping)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Stop if gradients become NaN
        if np.isnan(max_grad) or np.isinf(max_grad):
            print("   Training stopped due to NaN/Inf gradients!")
            break

    return gradient_history, loss_history

def implement_gradient_clipping_methods():
    """Implement and compare different gradient clipping methods"""

    print("\n✂️ GRADIENT CLIPPING METHODS COMPARISON")
    print("=" * 60)

    # Define different clipping methods
    clipping_methods = {
        'No Clipping': {'type': None, 'value': None},
        'Global Norm (5.0)': {'type': 'global_norm', 'value': 5.0},
        'Global Norm (2.0)': {'type': 'global_norm', 'value': 2.0},
        'Global Norm (1.0)': {'type': 'global_norm', 'value': 1.0},
        'By Value (±2.0)': {'type': 'by_value', 'value': 2.0},
        'By Value (±1.0)': {'type': 'by_value', 'value': 1.0},
    }

    # Generate training data
    X_train = tf.random.normal((1500, 20))
    y_train = tf.cast(tf.reduce_sum(X_train[:, :10], axis=1) > 0, tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((300, 20))
    y_val = tf.cast(tf.reduce_sum(X_val[:, :10], axis=1) > 0, tf.float32)
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Training data: {X_train.shape}, Validation: {X_val.shape}")

    # Train models with different clipping methods
    training_results = {}

    for method_name, config in clipping_methods.items():
        print(f"\n🔧 Training with {method_name}...")

        # Create fresh model for each method
        model = create_explosion_prone_network()

        # Create optimizer with or without clipping
        if config['type'] == 'global_norm':
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, clipnorm=config['value'])
        elif config['type'] == 'by_value':
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, clipvalue=config['value'])
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'max_gradient': []
        }

        # Training loop
        epochs = 30
        successful_epochs = 0

        for epoch in range(epochs):
            # Training step
            with tf.GradientTape() as tape:
                train_predictions = model(X_train, training=True)
                train_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train, train_predictions))

            gradients = tape.gradient(train_loss, model.trainable_variables)

            # Track gradient norms before clipping
            grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
            max_grad = max(grad_norms) if grad_norms else 0.0

            # Check for NaN/Inf
            if np.isnan(max_grad) or np.isinf(max_grad) or np.isnan(train_loss.numpy()):
                print(f"   Stopped at epoch {epoch+1} due to NaN/Inf")
                break

            # Apply gradients (clipping is handled by optimizer)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Validation step
            val_predictions = model(X_val, training=False)
            val_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_val, val_predictions))

            # Calculate accuracies
            train_acc = tf.reduce_mean(tf.cast(tf.round(train_predictions) == y_train, tf.float32))
            val_acc = tf.reduce_mean(tf.cast(tf.round(val_predictions) == y_val, tf.float32))

            # Store history
            history['loss'].append(train_loss.numpy())
            history['val_loss'].append(val_loss.numpy())
            history['accuracy'].append(train_acc.numpy())
            history['val_accuracy'].append(val_acc.numpy())
            history['max_gradient'].append(max_grad)

            successful_epochs += 1

        # Store results
        training_results[method_name] = {
            'history': history,
            'successful_epochs': successful_epochs,
            'final_val_acc': history['val_accuracy'][-1] if history['val_accuracy'] else 0.0,
            'training_stable': successful_epochs == epochs,
            'avg_gradient': np.mean(history['max_gradient']) if history['max_gradient'] else 0.0
        }

        print(f"   Completed {successful_epochs}/{epochs} epochs")
        if history['val_accuracy']:
            print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

    # Create comprehensive comparison
    create_clipping_comparison_plots(training_results)

    return training_results

def create_clipping_comparison_plots(training_results):
    """Create comprehensive plots comparing different clipping methods"""

    print("\n📊 Creating gradient clipping comparison visualization...")

    if not training_results:
        print("❌ No training results to visualize")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_results)))

    for i, (name, data) in enumerate(training_results.items()):
        if data['history']['loss']:
            epochs = range(1, len(data['history']['loss']) + 1)
            line_style = '-' if data['training_stable'] else '--'
            ax1.plot(epochs, data['history']['loss'], line_style,
                    color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss with Different Clipping Methods')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]

    for i, (name, data) in enumerate(training_results.items()):
        if data['history']['val_accuracy']:
            epochs = range(1, len(data['history']['val_accuracy']) + 1)
            line_style = '-' if data['training_stable'] else '--'
            ax2.plot(epochs, data['history']['val_accuracy'], line_style,
                    color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy with Different Clipping')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient Magnitudes
    ax3 = axes[0, 2]

    for i, (name, data) in enumerate(training_results.items()):
        if data['history']['max_gradient']:
            epochs = range(1, len(data['history']['max_gradient']) + 1)
            line_style = '-' if data['training_stable'] else '--'
            ax3.semilogy(epochs, data['history']['max_gradient'], line_style,
                        color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax3.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='Explosion threshold')
    ax3.axhline(y=5, color='orange', linestyle=':', alpha=0.7, label='Warning threshold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Max Gradient Magnitude (log scale)')
    ax3.set_title('Gradient Magnitude Control')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training Stability
    ax4 = axes[1, 0]
    methods = list(training_results.keys())
    success_rates = [training_results[m]['successful_epochs'] / 30 * 100 for m in methods]

    bars = ax4.bar(range(len(methods)), success_rates, alpha=0.7,
                  color=['green' if rate == 100 else 'orange' if rate > 50 else 'red'
                         for rate in success_rates])
    ax4.set_xlabel('Clipping Method')
    ax4.set_ylabel('Training Completion Rate (%)')
    ax4.set_title('Training Stability Comparison')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels([m.split()[0] for m in methods], rotation=45)
    ax4.set_ylim(0, 105)

    # Add value labels
    for bar, value in zip(bars, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, value + 2,
                f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')

    ax4.grid(True, alpha=0.3)

    # Plot 5: Final Performance
    ax5 = axes[1, 1]
    final_accs = [training_results[m]['final_val_acc'] for m in methods]

    bars = ax5.bar(range(len(methods)), final_accs, alpha=0.7, color='skyblue')
    ax5.set_xlabel('Clipping Method')
    ax5.set_ylabel('Final Validation Accuracy')
    ax5.set_title('Final Performance Comparison')
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels([m.split()[0] for m in methods], rotation=45)

    # Add value labels
    for bar, value in zip(bars, final_accs):
        ax5.text(bar.get_x() + bar.get_width()/2, value + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    ax5.grid(True, alpha=0.3)

    # Plot 6: Average Gradient Control
    ax6 = axes[1, 2]
    avg_grads = [training_results[m]['avg_gradient'] for m in methods]

    bars = ax6.bar(range(len(methods)), avg_grads, alpha=0.7, color='lightcoral')
    ax6.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Target range')
    ax6.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Explosion threshold')
    ax6.set_xlabel('Clipping Method')
    ax6.set_ylabel('Average Max Gradient')
    ax6.set_title('Gradient Control Effectiveness')
    ax6.set_xticks(range(len(methods)))
    ax6.set_xticklabels([m.split()[0] for m in methods], rotation=45)
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_clipping_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Gradient clipping comparison saved as 'gradient_clipping_comparison.png'")
    plt.show()

def demonstrate_manual_clipping():
    """Demonstrate manual implementation of gradient clipping"""

    print("\n🔧 MANUAL GRADIENT CLIPPING IMPLEMENTATION")
    print("=" * 60)

    def clip_by_global_norm(gradients, clip_norm):
        """Manual implementation of global norm clipping"""
        # Calculate global norm
        global_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]))

        # Calculate clipping ratio
        clip_ratio = clip_norm / (global_norm + 1e-8)
        clip_ratio = tf.minimum(clip_ratio, 1.0)

        # Apply clipping
        clipped_gradients = [g * clip_ratio if g is not None else g for g in gradients]

        return clipped_gradients, global_norm

    def clip_by_value(gradients, clip_value):
        """Manual implementation of value clipping"""
        clipped_gradients = [
            tf.clip_by_value(g, -clip_value, clip_value) if g is not None else g
            for g in gradients
        ]
        return clipped_gradients

    # Create model for demonstration
    model = create_explosion_prone_network()

    # Generate sample data
    X_sample = tf.random.normal((100, 20))
    y_sample = tf.random.uniform((100, 1))

    print("Testing manual clipping implementations...")

    with tf.GradientTape() as tape:
        predictions = model(X_sample)
        loss = tf.reduce_mean(tf.square(predictions - y_sample))

    gradients = tape.gradient(loss, model.trainable_variables)

    # Original gradients
    original_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
    original_global_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in gradients if g is not None]))

    print(f"\nOriginal gradients:")
    print(f"  Global norm: {original_global_norm:.4f}")
    print(f"  Max layer norm: {max(original_norms):.4f}")
    print(f"  Min layer norm: {min(original_norms):.4f}")

    # Test global norm clipping
    clipped_gradients_global, clipped_global_norm = clip_by_global_norm(gradients, clip_norm=5.0)
    clipped_norms_global = [tf.norm(g).numpy() for g in clipped_gradients_global if g is not None]
    new_global_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in clipped_gradients_global if g is not None]))

    print(f"\nAfter global norm clipping (clip_norm=5.0):")
    print(f"  Global norm: {new_global_norm:.4f}")
    print(f"  Max layer norm: {max(clipped_norms_global):.4f}")
    print(f"  Clipping applied: {'Yes' if original_global_norm > 5.0 else 'No'}")

    # Test value clipping
    clipped_gradients_value = clip_by_value(gradients, clip_value=2.0)
    clipped_norms_value = [tf.norm(g).numpy() for g in clipped_gradients_value if g is not None]

    print(f"\nAfter value clipping (clip_value=2.0):")
    print(f"  Max layer norm: {max(clipped_norms_value):.4f}")
    print(f"  Values clipped to range: [-2.0, 2.0]")

    # Visualize clipping effects
    plt.figure(figsize=(15, 5))

    # Original gradients
    plt.subplot(1, 3, 1)
    plt.bar(range(len(original_norms)), original_norms, alpha=0.7, color='red')
    plt.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, label='Global norm threshold')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Original Gradients')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Global norm clipped
    plt.subplot(1, 3, 2)
    plt.bar(range(len(clipped_norms_global)), clipped_norms_global, alpha=0.7, color='green')
    plt.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, label='Global norm threshold')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Global Norm Clipped')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Value clipped
    plt.subplot(1, 3, 3)
    plt.bar(range(len(clipped_norms_value)), clipped_norms_value, alpha=0.7, color='blue')
    plt.axhline(y=2.0, color='black', linestyle='--', alpha=0.7, label='Value threshold')
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Value Clipped')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('manual_clipping_demonstration.png', dpi=300, bbox_inches='tight')
    print("\n✅ Manual clipping demonstration saved as 'manual_clipping_demonstration.png'")
    plt.show()

def analyze_clipping_hyperparameters():
    """Analyze the effect of different clipping hyperparameters"""

    print("\n🎛️ CLIPPING HYPERPARAMETER ANALYSIS")
    print("=" * 60)

    # Test different clipping values
    global_norm_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    clip_value_values = [0.5, 1.0, 2.0, 5.0]

    hyperparameter_results = {}

    # Generate training data
    X_train = tf.random.normal((1000, 20))
    y_train = tf.cast(tf.reduce_sum(X_train[:, :10], axis=1) > 0, tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    print("Testing different hyperparameter values...")

    # Test global norm clipping values
    print("\n📊 Global Norm Clipping Values:")
    print(f"{'Clip Norm':<12} {'Final Loss':<12} {'Training Stable':<15} {'Final Accuracy':<15}")
    print("-" * 60)

    for clip_norm in global_norm_values:
        model = create_explosion_prone_network()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, clipnorm=clip_norm)

        # Short training
        losses = []
        stable = True
        final_acc = 0.0

        for epoch in range(20):
            with tf.GradientTape() as tape:
                predictions = model(X_train)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train, predictions))

            if np.isnan(loss.numpy()) or np.isinf(loss.numpy()):
                stable = False
                break

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            losses.append(loss.numpy())

        if stable and losses:
            final_acc = tf.reduce_mean(tf.cast(tf.round(model(X_train)) == y_train, tf.float32)).numpy()

        hyperparameter_results[f'GlobalNorm_{clip_norm}'] = {
            'final_loss': losses[-1] if losses else float('inf'),
            'stable': stable,
            'final_accuracy': final_acc
        }

        stable_text = "✅ Yes" if stable else "❌ No"
        final_loss = losses[-1] if losses else float('inf')

        print(f"{clip_norm:<12} {final_loss:<12.6f} {stable_text:<15} {final_acc:<15.4f}")

    return hyperparameter_results

def create_clipping_guidelines():
    """Create comprehensive guidelines for gradient clipping"""

    print("\n📋 GRADIENT CLIPPING IMPLEMENTATION GUIDELINES")
    print("=" * 70)

    print("🎯 WHEN TO USE GRADIENT CLIPPING:")
    print("✅ RNNs and LSTMs (highly prone to exploding gradients)")
    print("✅ Very deep networks without residual connections")
    print("✅ Networks with poor weight initialization")
    print("✅ When using high learning rates")
    print("✅ Training with small datasets (higher gradient variance)")

    print("\n🔧 CHOOSING CLIPPING METHOD:")

    methods = {
        'Global Norm Clipping': {
            'When': 'General purpose, most common',
            'Pros': 'Preserves gradient direction',
            'Cons': 'May not clip individual large gradients',
            'Recommended': 'clipnorm=1.0 to 5.0',
            'TensorFlow': 'optimizer = Adam(clipnorm=1.0)'
        },
        'By Value Clipping': {
            'When': 'When individual gradients are extreme',
            'Pros': 'Clips individual gradient values',
            'Cons': 'May change gradient direction',
            'Recommended': 'clipvalue=0.5 to 2.0',
            'TensorFlow': 'optimizer = Adam(clipvalue=1.0)'
        },
        'Adaptive Clipping': {
            'When': 'Dynamic adjustment needed',
            'Pros': 'Automatically adjusts threshold',
            'Cons': 'More complex implementation',
            'Recommended': 'Based on gradient statistics',
            'TensorFlow': 'Custom implementation required'
        }
    }

    print(f"{'Method':<20} {'Best For':<25} {'Typical Values':<20}")
    print("-" * 65)

    for method_name, info in methods.items():
        print(f"{method_name:<20} {info['When']:<25} {info['Recommended']:<20}")

    print("\n💡 HYPERPARAMETER SELECTION GUIDE:")
    print("🎯 START WITH: clipnorm=1.0 (global norm clipping)")
    print("🔍 IF UNSTABLE: Reduce to clipnorm=0.5")
    print("🚀 IF TOO CONSERVATIVE: Increase to clipnorm=5.0")
    print("⚖️ MONITOR: Gradient norms during training")
    print("📊 TRACK: Training loss and validation accuracy")

    print("\n⚠️ COMMON MISTAKES TO AVOID:")
    print("❌ Using both clipnorm and clipvalue simultaneously")
    print("❌ Setting clip values too low (slower convergence)")
    print("❌ Setting clip values too high (no effect)")
    print("❌ Not monitoring gradient statistics")
    print("❌ Applying clipping to networks that don't need it")

    print("\n🔍 MONITORING AND DEBUGGING:")
    monitoring_code = '''
# Example monitoring code
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Monitor gradient statistics
    grad_norms = [tf.norm(g) for g in gradients if g is not None]
    max_grad = tf.reduce_max(grad_norms)
    mean_grad = tf.reduce_mean(grad_norms)

    # Log statistics
    tf.summary.scalar('max_gradient', max_grad)
    tf.summary.scalar('mean_gradient', mean_grad)

    # Apply clipped gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, max_grad
'''

    print("📝 MONITORING TEMPLATE:")
    print(monitoring_code)

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("✂️ GRADIENT CLIPPING TECHNIQUES")
    print("=" * 70)
    print()

    try:
        # Demonstrate gradient explosion
        gradient_history, loss_history = demonstrate_gradient_explosion()

        # Compare clipping methods
        training_results = implement_gradient_clipping_methods()

        # Demonstrate manual clipping
        demonstrate_manual_clipping()

        # Analyze hyperparameters
        hyperparameter_results = analyze_clipping_hyperparameters()

        # Create implementation guidelines
        create_clipping_guidelines()

        print("\n" + "=" * 70)
        print("✅ GRADIENT CLIPPING ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - gradient_clipping_comparison.png")
        print("   - manual_clipping_demonstration.png")
        print("🎓 Learning outcome: Understanding and implementing gradient clipping")
        print("💡 Next step: Learn about advanced optimization algorithms")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()