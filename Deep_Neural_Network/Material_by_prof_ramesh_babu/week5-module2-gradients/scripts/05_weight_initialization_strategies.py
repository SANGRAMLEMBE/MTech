#!/usr/bin/env python3
"""
05_weight_initialization_strategies.py

Purpose: Comprehensive analysis and comparison of weight initialization strategies
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script demonstrates different weight initialization methods and their impact
on gradient flow and network training stability.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def analyze_initialization_methods():
    """Analyze different weight initialization methods"""

    print("🎯 WEIGHT INITIALIZATION METHODS ANALYSIS")
    print("=" * 60)

    # Define initialization methods
    initializers = {
        'Zero': tf.keras.initializers.Zeros(),
        'Random Normal (std=1.0)': tf.keras.initializers.RandomNormal(stddev=1.0),
        'Random Normal (std=0.1)': tf.keras.initializers.RandomNormal(stddev=0.1),
        'Random Uniform': tf.keras.initializers.RandomUniform(-0.1, 0.1),
        'Xavier/Glorot Uniform': tf.keras.initializers.GlorotUniform(),
        'Xavier/Glorot Normal': tf.keras.initializers.GlorotNormal(),
        'He Uniform': tf.keras.initializers.HeUniform(),
        'He Normal': tf.keras.initializers.HeNormal(),
        'LeCun Uniform': tf.keras.initializers.LecunUniform(),
        'LeCun Normal': tf.keras.initializers.LecunNormal()
    }

    # Test each initializer
    results = {}
    input_shape = (100, 64)

    print("\n📊 INITIALIZATION ANALYSIS:")
    print(f"{'Method':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 75)

    for name, initializer in initializers.items():
        try:
            # Create weight matrix
            weights = initializer(shape=input_shape)

            # Calculate statistics
            mean_val = tf.reduce_mean(weights).numpy()
            std_val = tf.math.reduce_std(weights).numpy()
            min_val = tf.reduce_min(weights).numpy()
            max_val = tf.reduce_max(weights).numpy()

            results[name] = {
                'weights': weights.numpy(),
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'initializer': initializer
            }

            print(f"{name:<25} {mean_val:<12.6f} {std_val:<12.6f} {min_val:<12.6f} {max_val:<12.6f}")

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    print("-" * 75)
    return results

def create_initialization_visualization(results):
    """Create visualization of weight initialization distributions"""

    print("\n📊 Creating weight initialization visualization...")

    # Select key initializers for visualization
    key_methods = [
        'Zero', 'Random Normal (std=1.0)', 'Random Normal (std=0.1)',
        'Xavier/Glorot Normal', 'He Normal', 'LeCun Normal'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, len(key_methods)))

    for i, method in enumerate(key_methods):
        if method in results:
            weights = results[method]['weights'].flatten()

            ax = axes[i]
            ax.hist(weights, bins=50, alpha=0.7, color=colors[i], density=True)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')

            ax.set_title(f'{method}\n(μ={results[method]["mean"]:.3f}, σ={results[method]["std"]:.3f})')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig('weight_initialization_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Weight distributions saved as 'weight_initialization_distributions.png'")
    plt.show()

def compare_initialization_impact_on_gradients():
    """Compare how different initializations affect gradient flow"""

    print("\n🔍 INITIALIZATION IMPACT ON GRADIENT FLOW")
    print("=" * 60)

    # Test key initialization methods
    test_methods = {
        'Poor (std=2.0)': tf.keras.initializers.RandomNormal(stddev=2.0),
        'Xavier/Glorot': tf.keras.initializers.GlorotNormal(),
        'He': tf.keras.initializers.HeNormal(),
        'Too Small (std=0.01)': tf.keras.initializers.RandomNormal(stddev=0.01)
    }

    # Generate test data
    X = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1))

    gradient_results = {}

    print(f"\n{'Method':<20} {'Max Grad':<12} {'Min Grad':<12} {'Vanished':<10} {'Exploded':<10} {'Status':<15}")
    print("-" * 85)

    for name, initializer in test_methods.items():
        try:
            # Create network with specific initialization
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,),
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Calculate gradients
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = tf.reduce_mean(tf.square(predictions - y))

            gradients = tape.gradient(loss, model.trainable_variables)
            grad_norms = [tf.norm(g).numpy() for i, g in enumerate(gradients) if i % 2 == 0]

            if grad_norms:
                max_grad = np.max(grad_norms)
                min_grad = np.min(grad_norms)
                vanished_count = sum(1 for g in grad_norms if g < 1e-6)
                exploded_count = sum(1 for g in grad_norms if g > 10.0)

                # Determine status
                if vanished_count > 0:
                    status = "🚨 Vanishing"
                elif exploded_count > 0:
                    status = "💥 Exploding"
                elif min_grad < 1e-4:
                    status = "⚠️ Weak"
                else:
                    status = "✅ Healthy"

                gradient_results[name] = {
                    'max_gradient': max_grad,
                    'min_gradient': min_grad,
                    'vanished_layers': vanished_count,
                    'exploded_layers': exploded_count,
                    'gradient_norms': grad_norms,
                    'status': status
                }

                print(f"{name:<20} {max_grad:<12.6f} {min_grad:<12.6f} {vanished_count:<10} {exploded_count:<10} {status:<15}")

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    print("-" * 85)

    # Create gradient comparison visualization
    if gradient_results:
        plt.figure(figsize=(15, 10))

        # Plot 1: Gradient magnitudes by layer
        plt.subplot(2, 2, 1)
        for name, data in gradient_results.items():
            layers = range(1, len(data['gradient_norms']) + 1)
            plt.semilogy(layers, data['gradient_norms'], 'o-', linewidth=2, label=name, markersize=6)

        plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Exploding threshold')
        plt.xlabel('Layer Number')
        plt.ylabel('Gradient Magnitude (log scale)')
        plt.title('Gradient Flow by Initialization Method')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Max gradient comparison
        plt.subplot(2, 2, 2)
        methods = list(gradient_results.keys())
        max_grads = [gradient_results[m]['max_gradient'] for m in methods]

        bars = plt.bar(range(len(methods)), max_grads, alpha=0.7)
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Exploding threshold')
        plt.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Ideal range')
        plt.xlabel('Initialization Method')
        plt.ylabel('Maximum Gradient')
        plt.title('Maximum Gradient by Method')
        plt.xticks(range(len(methods)), [m.split()[0] for m in methods], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Min gradient comparison
        plt.subplot(2, 2, 3)
        min_grads = [gradient_results[m]['min_gradient'] for m in methods]

        bars = plt.bar(range(len(methods)), min_grads, alpha=0.7, color='green')
        plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
        plt.axhline(y=1e-4, color='orange', linestyle='--', alpha=0.7, label='Weak threshold')
        plt.xlabel('Initialization Method')
        plt.ylabel('Minimum Gradient')
        plt.title('Minimum Gradient by Method')
        plt.xticks(range(len(methods)), [m.split()[0] for m in methods], rotation=45)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Problem layer counts
        plt.subplot(2, 2, 4)
        vanished_counts = [gradient_results[m]['vanished_layers'] for m in methods]
        exploded_counts = [gradient_results[m]['exploded_layers'] for m in methods]

        x_pos = np.arange(len(methods))
        width = 0.35

        plt.bar(x_pos - width/2, vanished_counts, width, label='Vanished Layers', color='red', alpha=0.7)
        plt.bar(x_pos + width/2, exploded_counts, width, label='Exploded Layers', color='purple', alpha=0.7)

        plt.xlabel('Initialization Method')
        plt.ylabel('Number of Problematic Layers')
        plt.title('Problem Layers by Method')
        plt.xticks(x_pos, [m.split()[0] for m in methods], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('initialization_gradient_impact.png', dpi=300, bbox_inches='tight')
        print("✅ Gradient impact analysis saved as 'initialization_gradient_impact.png'")
        plt.show()

    return gradient_results

def demonstrate_theoretical_basis():
    """Demonstrate theoretical basis for different initialization methods"""

    print("\n🧮 THEORETICAL BASIS FOR INITIALIZATION METHODS")
    print("=" * 70)

    # Xavier/Glorot initialization theory
    print("1. 📚 XAVIER/GLOROT INITIALIZATION:")
    print("   Theory: Maintain variance of activations and gradients")
    print("   Formula: Var(W) = 1/n_in (uniform) or 2/n_in (normal)")
    print("   Best for: Sigmoid, Tanh activations")
    print()

    # Demonstrate Xavier calculation
    layer_sizes = [784, 256, 128, 64, 10]
    print("   Example calculation for different layer sizes:")
    print(f"   {'Layer':<15} {'n_in':<8} {'n_out':<8} {'Xavier Std':<15} {'Xavier Range':<20}")
    print("   " + "-" * 70)

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        xavier_std = np.sqrt(2.0 / (n_in + n_out))
        xavier_range = np.sqrt(6.0 / (n_in + n_out))

        print(f"   {i+1:<15} {n_in:<8} {n_out:<8} {xavier_std:<15.6f} ±{xavier_range:<15.6f}")

    print()

    # He initialization theory
    print("2. 🎯 HE INITIALIZATION:")
    print("   Theory: Account for ReLU activation (kills half the neurons)")
    print("   Formula: Var(W) = 2/n_in")
    print("   Best for: ReLU family activations")
    print()

    print("   Example calculation for different layer sizes:")
    print(f"   {'Layer':<15} {'n_in':<8} {'He Std':<15} {'He Range':<20}")
    print("   " + "-" * 50)

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        he_std = np.sqrt(2.0 / n_in)
        he_range = np.sqrt(6.0 / n_in)

        print(f"   {i+1:<15} {n_in:<8} {he_std:<15.6f} ±{he_range:<15.6f}")

    print()

    # LeCun initialization theory
    print("3. 🔬 LECUN INITIALIZATION:")
    print("   Theory: Maintain variance of inputs")
    print("   Formula: Var(W) = 1/n_in")
    print("   Best for: SELU activation (self-normalizing networks)")

def compare_training_performance():
    """Compare actual training performance with different initializations"""

    print("\n🏃‍♂️ TRAINING PERFORMANCE COMPARISON")
    print("=" * 60)

    # Define initialization methods to test
    init_methods = {
        'Poor (std=2.0)': tf.keras.initializers.RandomNormal(stddev=2.0),
        'Xavier': tf.keras.initializers.GlorotNormal(),
        'He': tf.keras.initializers.HeNormal(),
        'Poor (std=0.01)': tf.keras.initializers.RandomNormal(stddev=0.01)
    }

    # Create dataset
    X_train = tf.random.normal((1000, 32))
    # Create non-linear target
    y_train = tf.cast(tf.reduce_sum(X_train[:, :16], axis=1) > tf.reduce_sum(X_train[:, 16:], axis=1), tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((200, 32))
    y_val = tf.cast(tf.reduce_sum(X_val[:, :16], axis=1) > tf.reduce_sum(X_val[:, 16:], axis=1), tf.float32)
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

    training_results = {}

    for name, initializer in init_methods.items():
        print(f"\n🔧 Training with {name} initialization...")

        try:
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(32,),
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(32, activation='relu',
                                     kernel_initializer=initializer),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

            # Train model
            history = model.fit(X_train, y_train,
                              validation_data=(X_val, y_val),
                              epochs=30,
                              batch_size=32,
                              verbose=0)

            training_results[name] = {
                'history': history.history,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }

            print(f"   Final train accuracy: {training_results[name]['final_accuracy']:.4f}")
            print(f"   Final val accuracy: {training_results[name]['final_val_accuracy']:.4f}")

        except Exception as e:
            print(f"   ❌ Training failed: {e}")

    # Create training comparison plots
    if training_results:
        plt.figure(figsize=(16, 12))

        # Loss curves
        plt.subplot(2, 3, 1)
        for name, data in training_results.items():
            plt.plot(data['history']['loss'], label=f'{name} (train)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        for name, data in training_results.items():
            plt.plot(data['history']['val_loss'], label=f'{name} (val)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy curves
        plt.subplot(2, 3, 3)
        for name, data in training_results.items():
            plt.plot(data['history']['accuracy'], label=f'{name} (train)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        for name, data in training_results.items():
            plt.plot(data['history']['val_accuracy'], label=f'{name} (val)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Final performance comparison
        plt.subplot(2, 3, 5)
        methods = list(training_results.keys())
        final_accs = [training_results[m]['final_accuracy'] for m in methods]
        final_val_accs = [training_results[m]['final_val_accuracy'] for m in methods]

        x_pos = np.arange(len(methods))
        width = 0.35

        plt.bar(x_pos - width/2, final_accs, width, label='Train Accuracy', alpha=0.7)
        plt.bar(x_pos + width/2, final_val_accs, width, label='Val Accuracy', alpha=0.7)
        plt.xlabel('Initialization Method')
        plt.ylabel('Final Accuracy')
        plt.title('Final Performance Comparison')
        plt.xticks(x_pos, [m.split()[0] for m in methods], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning stability (loss variance)
        plt.subplot(2, 3, 6)
        loss_vars = [np.var(training_results[m]['history']['val_loss']) for m in methods]

        bars = plt.bar(methods, loss_vars, alpha=0.7)
        plt.xlabel('Initialization Method')
        plt.ylabel('Validation Loss Variance')
        plt.title('Training Stability (Lower is Better)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, loss_vars):
            plt.text(bar.get_x() + bar.get_width()/2, value + np.max(loss_vars)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('initialization_training_performance.png', dpi=300, bbox_inches='tight')
        print("\n✅ Training performance comparison saved as 'initialization_training_performance.png'")
        plt.show()

def create_initialization_decision_guide():
    """Create a comprehensive decision guide for weight initialization"""

    print("\n🎯 WEIGHT INITIALIZATION DECISION GUIDE")
    print("=" * 70)

    decision_tree = {
        'Activation Function': {
            'Sigmoid/Tanh': {
                'Recommendation': 'Xavier/Glorot',
                'Reason': 'Maintains activation variance',
                'Formula': 'Normal(0, sqrt(2/(n_in + n_out)))'
            },
            'ReLU/Leaky ReLU/ELU': {
                'Recommendation': 'He',
                'Reason': 'Accounts for ReLU killing half neurons',
                'Formula': 'Normal(0, sqrt(2/n_in))'
            },
            'SELU': {
                'Recommendation': 'LeCun',
                'Reason': 'Maintains self-normalization',
                'Formula': 'Normal(0, sqrt(1/n_in))'
            },
            'Swish/GELU': {
                'Recommendation': 'He or Xavier',
                'Reason': 'Similar to ReLU behavior',
                'Formula': 'Normal(0, sqrt(2/n_in)) or Normal(0, sqrt(2/(n_in + n_out)))'
            }
        }
    }

    print("📋 DECISION MATRIX:")
    print("-" * 70)
    print(f"{'Activation':<15} {'Best Init':<15} {'Reason':<30} {'TensorFlow Code':<20}")
    print("-" * 80)

    for activation, info in decision_tree['Activation Function'].items():
        tf_code = {
            'Xavier/Glorot': 'GlorotNormal()',
            'He': 'HeNormal()',
            'LeCun': 'LecunNormal()',
            'He or Xavier': 'HeNormal()'
        }[info['Recommendation']]

        print(f"{activation:<15} {info['Recommendation']:<15} {info['Reason']:<30} {tf_code:<20}")

    print("-" * 80)

    print("\n💡 GENERAL GUIDELINES:")
    print("1. ✅ DEFAULT CHOICE: He initialization for ReLU networks")
    print("2. 🔧 SPECIAL CASES: Xavier for sigmoid/tanh networks")
    print("3. ⚠️ AVOID: Zero initialization (breaks symmetry)")
    print("4. ⚠️ AVOID: Too large initialization (gradient explosion)")
    print("5. ⚠️ AVOID: Too small initialization (vanishing gradients)")

    print("\n🔬 TROUBLESHOOTING:")
    print("- If gradients vanish: Try larger initialization (He instead of Xavier)")
    print("- If gradients explode: Try smaller initialization (Xavier instead of He)")
    print("- If training unstable: Try Xavier initialization with lower learning rate")
    print("- If dead ReLU neurons: Try leaky ReLU with He initialization")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🎯 WEIGHT INITIALIZATION STRATEGIES")
    print("=" * 70)
    print()

    try:
        # Analyze initialization methods
        init_results = analyze_initialization_methods()

        # Create visualization of weight distributions
        create_initialization_visualization(init_results)

        # Compare initialization impact on gradients
        gradient_results = compare_initialization_impact_on_gradients()

        # Demonstrate theoretical basis
        demonstrate_theoretical_basis()

        # Compare training performance
        compare_training_performance()

        # Create decision guide
        create_initialization_decision_guide()

        print("\n" + "=" * 70)
        print("✅ WEIGHT INITIALIZATION ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - weight_initialization_distributions.png")
        print("   - initialization_gradient_impact.png")
        print("   - initialization_training_performance.png")
        print("🎓 Learning outcome: Understanding weight initialization impact on training")
        print("💡 Next step: Learn about normalization techniques")

        print("\n📚 BOOK REFERENCES:")
        print("=" * 70)
        print("1. 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, Aaron Courville")
        print("   - Chapter 8: Optimization for Training Deep Models")
        print("   - Chapter 6: Deep Feedforward Networks")
        print()
        print("2. 'Neural Networks and Deep Learning' by Charu C. Aggarwal")
        print("   - Chapter 4: Teaching Deep Networks to Learn")
        print("   - Chapter 2: Machine Learning with Shallow Neural Networks")
        print()
        print("3. 'Deep Learning with Python' by François Chollet")
        print("   - Chapter 4: Getting started with neural networks")
        print("   - Chapter 5: Deep learning for computer vision")
        print()
        print("4. 'Deep Neural Network Architectures'")
        print("   - Chapter 3: Network Initialization and Training")
        print("   - Chapter 5: Optimization Techniques")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()