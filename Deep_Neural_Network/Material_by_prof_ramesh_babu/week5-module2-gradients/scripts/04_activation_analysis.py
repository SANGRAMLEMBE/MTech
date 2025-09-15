#!/usr/bin/env python3
"""
04_activation_analysis.py

Purpose: Comprehensive analysis and comparison of activation functions
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script provides detailed analysis of different activation functions,
their derivatives, gradient properties, and impact on deep networks.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def analyze_activation_functions():
    """Comprehensive analysis of activation functions and their properties"""

    print("🧪 ACTIVATION FUNCTION ANALYSIS")
    print("=" * 50)

    # Define range for analysis
    x = np.linspace(-5, 5, 1000)

    # Define activation functions
    activations = {
        'Sigmoid': {
            'func': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'derivative': lambda x: (1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 - (1 / (1 + np.exp(-np.clip(x, -500, 500))))),
            'color': 'red',
            'tf_name': 'sigmoid'
        },
        'Tanh': {
            'func': lambda x: np.tanh(x),
            'derivative': lambda x: 1 - np.tanh(x)**2,
            'color': 'blue',
            'tf_name': 'tanh'
        },
        'ReLU': {
            'func': lambda x: np.maximum(0, x),
            'derivative': lambda x: (x > 0).astype(float),
            'color': 'green',
            'tf_name': 'relu'
        },
        'Leaky ReLU': {
            'func': lambda x: np.where(x > 0, x, 0.01 * x),
            'derivative': lambda x: np.where(x > 0, 1, 0.01),
            'color': 'orange',
            'tf_name': 'leaky_relu'
        },
        'ELU': {
            'func': lambda x: np.where(x > 0, x, np.exp(x) - 1),
            'derivative': lambda x: np.where(x > 0, 1, np.exp(x)),
            'color': 'purple',
            'tf_name': 'elu'
        },
        'Swish': {
            'func': lambda x: x * (1 / (1 + np.exp(-np.clip(x, -500, 500)))),
            'derivative': lambda x: (1 / (1 + np.exp(-np.clip(x, -500, 500)))) + x * (1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 - (1 / (1 + np.exp(-np.clip(x, -500, 500))))),
            'color': 'brown',
            'tf_name': 'swish'
        }
    }

    # Calculate function values and derivatives
    results = {}
    for name, config in activations.items():
        try:
            func_values = config['func'](x)
            deriv_values = config['derivative'](x)

            results[name] = {
                'function': func_values,
                'derivative': deriv_values,
                'max_derivative': np.max(deriv_values),
                'min_derivative': np.min(deriv_values),
                'zero_centered': np.mean(func_values) < 0.1,
                'bounded': np.max(func_values) < 100,
                'color': config['color'],
                'tf_name': config['tf_name']
            }

            print(f"\n📊 {name} Analysis:")
            print(f"   Max derivative: {results[name]['max_derivative']:.4f}")
            print(f"   Min derivative: {results[name]['min_derivative']:.4f}")
            print(f"   Zero-centered output: {results[name]['zero_centered']}")
            print(f"   Bounded output: {results[name]['bounded']}")

        except Exception as e:
            print(f"❌ Error analyzing {name}: {e}")

    return x, results

def create_activation_visualization(x, results):
    """Create comprehensive visualization of activation functions"""

    print("\n📊 Creating activation function visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Activation functions
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(x, data['function'], color=data['color'], linewidth=2, label=name)

    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('Output f(x)')
    ax1.set_title('Activation Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 2: Derivatives
    ax2 = axes[0, 1]
    for name, data in results.items():
        ax2.plot(x, data['derivative'], color=data['color'], linewidth=2, label=name)

    ax2.set_xlabel('Input (x)')
    ax2.set_ylabel('Derivative f\'(x)')
    ax2.set_title('Activation Function Derivatives')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Plot 3: Max derivative comparison
    ax3 = axes[0, 2]
    names = list(results.keys())
    max_derivs = [results[name]['max_derivative'] for name in names]
    colors = [results[name]['color'] for name in names]

    bars = ax3.bar(range(len(names)), max_derivs, color=colors, alpha=0.7)
    ax3.set_xlabel('Activation Function')
    ax3.set_ylabel('Maximum Derivative')
    ax3.set_title('Maximum Derivative Comparison')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, max_derivs):
        ax3.text(bar.get_x() + bar.get_width()/2, value + 0.05,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Sigmoid problem visualization
    ax4 = axes[1, 0]
    sigmoid_func = results['Sigmoid']['function']
    sigmoid_deriv = results['Sigmoid']['derivative']

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(x, sigmoid_func, 'r-', linewidth=3, label='Sigmoid')
    line2 = ax4_twin.plot(x, sigmoid_deriv, 'r--', linewidth=2, label='Sigmoid Derivative')

    ax4.axhline(y=0.5, color='k', linestyle=':', alpha=0.7)
    ax4_twin.axhline(y=0.25, color='k', linestyle=':', alpha=0.7, label='Max derivative = 0.25')

    ax4.set_xlabel('Input (x)')
    ax4.set_ylabel('Sigmoid(x)', color='r')
    ax4_twin.set_ylabel('Sigmoid\'(x)', color='r')
    ax4.set_title('Sigmoid: The Vanishing Gradient Problem')

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: ReLU advantages
    ax5 = axes[1, 1]
    relu_func = results['ReLU']['function']
    relu_deriv = results['ReLU']['derivative']
    leaky_relu_func = results['Leaky ReLU']['function']
    leaky_relu_deriv = results['Leaky ReLU']['derivative']

    ax5_twin = ax5.twinx()
    ax5.plot(x, relu_func, 'g-', linewidth=3, label='ReLU')
    ax5.plot(x, leaky_relu_func, 'orange', linewidth=2, label='Leaky ReLU')
    ax5_twin.plot(x, relu_deriv, 'g--', linewidth=2, alpha=0.7, label='ReLU Derivative')
    ax5_twin.plot(x, leaky_relu_deriv, 'orange', linestyle='--', linewidth=2, alpha=0.7, label='Leaky ReLU Derivative')

    ax5.set_xlabel('Input (x)')
    ax5.set_ylabel('Function Output')
    ax5_twin.set_ylabel('Derivative')
    ax5.set_title('ReLU Family: Solving Vanishing Gradients')

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Modern activations
    ax6 = axes[1, 2]
    elu_func = results['ELU']['function']
    swish_func = results['Swish']['function']

    ax6.plot(x, elu_func, color='purple', linewidth=2, label='ELU')
    ax6.plot(x, swish_func, color='brown', linewidth=2, label='Swish')
    ax6.plot(x, results['ReLU']['function'], 'g--', alpha=0.7, label='ReLU (reference)')

    ax6.set_xlabel('Input (x)')
    ax6.set_ylabel('Output f(x)')
    ax6.set_title('Modern Activation Functions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('activation_functions_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'activation_functions_analysis.png'")
    plt.show()

def compare_networks_with_different_activations():
    """Compare gradient health across networks with different activations"""

    print("\n⚔️ NETWORK COMPARISON: DIFFERENT ACTIVATIONS")
    print("=" * 60)

    # Create networks with different activations
    activations_to_test = ['sigmoid', 'tanh', 'relu', 'elu']
    networks = {}

    for activation in activations_to_test:
        networks[activation.upper()] = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activation, input_shape=(10,)),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    # Generate test data
    X = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1))

    # Analyze gradients for each network
    gradient_analysis = {}

    print(f"\n{'Activation':<12} {'Max Grad':<12} {'Min Grad':<12} {'Vanished':<10} {'Status':<20}")
    print("-" * 75)

    for name, model in networks.items():
        try:
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = tf.reduce_mean(tf.square(predictions - y))

            gradients = tape.gradient(loss, model.trainable_variables)
            grad_norms = [tf.norm(g).numpy() for i, g in enumerate(gradients) if i % 2 == 0]

            max_grad = np.max(grad_norms)
            min_grad = np.min(grad_norms)
            vanished_count = sum(1 for g in grad_norms if g < 1e-6)

            # Determine status
            if vanished_count > 0:
                status = "🚨 Vanishing"
            elif max_grad > 10:
                status = "💥 Exploding"
            elif min_grad < 1e-4:
                status = "⚠️ Weak"
            else:
                status = "✅ Healthy"

            gradient_analysis[name] = {
                'max_gradient': max_grad,
                'min_gradient': min_grad,
                'vanished_layers': vanished_count,
                'gradient_norms': grad_norms,
                'status': status
            }

            print(f"{name:<12} {max_grad:<12.6f} {min_grad:<12.6f} {vanished_count:<10} {status:<20}")

        except Exception as e:
            print(f"❌ Error analyzing {name}: {e}")

    print("-" * 75)

    # Provide recommendations
    print("\n💡 ACTIVATION FUNCTION RECOMMENDATIONS:")
    print("1. 🚨 AVOID: Sigmoid and Tanh for deep networks (vanishing gradients)")
    print("2. ✅ PREFER: ReLU for most cases (simple, effective)")
    print("3. 🎯 CONSIDER: ELU for better negative region handling")
    print("4. 🔬 EXPERIMENT: Swish for potentially better performance")

    return gradient_analysis

def demonstrate_activation_impact_on_training():
    """Demonstrate how activation choice affects training dynamics"""

    print("\n🏃‍♂️ TRAINING DYNAMICS COMPARISON")
    print("=" * 50)

    # Create simple models with different activations
    activations = ['sigmoid', 'relu']
    training_histories = {}

    # Generate more complex dataset
    X_train = tf.random.normal((1000, 20))
    y_train = tf.cast(tf.reduce_sum(X_train[:, :10], axis=1) > 0, tf.float32)
    y_train = tf.reshape(y_train, (-1, 1))

    print("Training networks with different activations...")
    print(f"Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    for activation in activations:
        print(f"\n🔧 Training {activation.upper()} network...")

        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activation, input_shape=(20,)),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(16, activation=activation),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train
        try:
            history = model.fit(X_train, y_train, epochs=20, verbose=0, validation_split=0.2)

            training_histories[activation] = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }

            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]

            print(f"   Final loss: {final_loss:.4f}")
            print(f"   Final accuracy: {final_accuracy:.4f}")

        except Exception as e:
            print(f"   ❌ Training failed: {e}")

    # Plot training comparison
    if len(training_histories) > 1:
        plt.figure(figsize=(15, 5))

        # Loss comparison
        plt.subplot(1, 3, 1)
        for activation, history in training_histories.items():
            plt.plot(history['loss'], label=f'{activation.upper()} (train)')
            plt.plot(history['val_loss'], '--', label=f'{activation.upper()} (val)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accuracy comparison
        plt.subplot(1, 3, 2)
        for activation, history in training_histories.items():
            plt.plot(history['accuracy'], label=f'{activation.upper()} (train)')
            plt.plot(history['val_accuracy'], '--', label=f'{activation.upper()} (val)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Final performance
        plt.subplot(1, 3, 3)
        final_losses = [training_histories[act]['loss'][-1] for act in activations]
        final_accs = [training_histories[act]['accuracy'][-1] for act in activations]

        x_pos = np.arange(len(activations))
        width = 0.35

        plt.bar(x_pos - width/2, final_losses, width, label='Loss', alpha=0.7)
        plt.bar(x_pos + width/2, final_accs, width, label='Accuracy', alpha=0.7)
        plt.xlabel('Activation')
        plt.ylabel('Value')
        plt.title('Final Performance')
        plt.xticks(x_pos, [act.upper() for act in activations])
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('activation_training_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✅ Training comparison saved as 'activation_training_comparison.png'")
        plt.show()

def create_activation_summary_table():
    """Create a comprehensive summary table of activation functions"""

    print("\n📋 ACTIVATION FUNCTION SUMMARY TABLE")
    print("=" * 80)

    summary_data = {
        'Sigmoid': {
            'Range': '(0, 1)',
            'Zero-centered': 'No',
            'Max Derivative': '0.25',
            'Gradient Problem': 'Vanishing',
            'Use Case': 'Output layer (binary)',
            'Pros': 'Smooth, interpretable',
            'Cons': 'Vanishing gradients'
        },
        'Tanh': {
            'Range': '(-1, 1)',
            'Zero-centered': 'Yes',
            'Max Derivative': '1.0',
            'Gradient Problem': 'Vanishing',
            'Use Case': 'Better than sigmoid',
            'Pros': 'Zero-centered',
            'Cons': 'Still vanishing gradients'
        },
        'ReLU': {
            'Range': '[0, ∞)',
            'Zero-centered': 'No',
            'Max Derivative': '1.0',
            'Gradient Problem': 'None',
            'Use Case': 'Hidden layers',
            'Pros': 'Simple, no vanishing',
            'Cons': 'Dead neurons'
        },
        'Leaky ReLU': {
            'Range': '(-∞, ∞)',
            'Zero-centered': 'No',
            'Max Derivative': '1.0',
            'Gradient Problem': 'None',
            'Use Case': 'Alternative to ReLU',
            'Pros': 'No dead neurons',
            'Cons': 'Extra hyperparameter'
        },
        'ELU': {
            'Range': '(-1, ∞)',
            'Zero-centered': '~Yes',
            'Max Derivative': '1.0',
            'Gradient Problem': 'None',
            'Use Case': 'Better than ReLU',
            'Pros': 'Smooth, zero-centered',
            'Cons': 'Expensive computation'
        },
        'Swish': {
            'Range': '(-∞, ∞)',
            'Zero-centered': 'No',
            'Max Derivative': '~1.25',
            'Gradient Problem': 'None',
            'Use Case': 'State-of-the-art',
            'Pros': 'Self-gated, smooth',
            'Cons': 'More complex'
        }
    }

    # Print table header
    headers = ['Function', 'Range', 'Zero-centered', 'Max Deriv', 'Gradient Issue', 'Recommendation']
    print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<13} {headers[3]:<10} {headers[4]:<15} {headers[5]:<20}")
    print("-" * 85)

    # Print table rows
    for func_name, data in summary_data.items():
        print(f"{func_name:<12} {data['Range']:<12} {data['Zero-centered']:<13} "
              f"{data['Max Derivative']:<10} {data['Gradient Problem']:<15} {data['Use Case']:<20}")

    print("-" * 85)
    print("\n🎯 QUICK DECISION GUIDE:")
    print("✅ START WITH: ReLU (simple and effective)")
    print("🔧 IF DEAD NEURONS: Try Leaky ReLU or ELU")
    print("🚀 FOR BEST PERFORMANCE: Experiment with Swish")
    print("❌ AVOID: Sigmoid/Tanh in hidden layers")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🔄 ACTIVATION FUNCTION COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print()

    try:
        # Analyze activation functions
        x, results = analyze_activation_functions()

        # Create visualizations
        create_activation_visualization(x, results)

        # Compare networks with different activations
        gradient_analysis = compare_networks_with_different_activations()

        # Demonstrate training impact
        demonstrate_activation_impact_on_training()

        # Create summary table
        create_activation_summary_table()

        print("\n" + "=" * 70)
        print("✅ ACTIVATION FUNCTION ANALYSIS COMPLETE!")
        print("📊 Key files created: activation_functions_analysis.png, activation_training_comparison.png")
        print("🎓 Learning outcome: Understanding activation function properties and selection")
        print("💡 Next step: Learn about weight initialization strategies")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()