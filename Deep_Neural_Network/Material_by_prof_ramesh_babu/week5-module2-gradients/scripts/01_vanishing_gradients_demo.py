#!/usr/bin/env python3
"""
01_vanishing_gradients_demo.py

Purpose: Demonstrate the vanishing gradient problem in deep sigmoid networks
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script shows how gradients vanish exponentially in deep networks with sigmoid activations,
making it difficult for early layers to learn effectively.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def demonstrate_vanishing_gradients():
    """Show actual gradient magnitudes in a deep sigmoid network"""

    print("🧪 VANISHING GRADIENTS DEMONSTRATION")
    print("=" * 50)
    print("Creating Deep Sigmoid Network...")
    print("Network Architecture: 10 → 64 → 64 → 64 → 64 → 64 → 1")
    print("Activation: Sigmoid (the gradient killer!)")
    print()

    # Create a deep sigmoid network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(10,), name='layer_1'),
        tf.keras.layers.Dense(64, activation='sigmoid', name='layer_2'),
        tf.keras.layers.Dense(64, activation='sigmoid', name='layer_3'),
        tf.keras.layers.Dense(64, activation='sigmoid', name='layer_4'),
        tf.keras.layers.Dense(64, activation='sigmoid', name='layer_5'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    # Sample data
    print("📊 Generating sample data...")
    X = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1))
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print()

    # Calculate gradients
    print("🔍 Computing gradients...")
    try:
        with tf.GradientTape() as tape:
            predictions = model(X)
            loss = tf.reduce_mean(tf.square(predictions - y))
            print(f"Loss: {loss:.6f}")

        gradients = tape.gradient(loss, model.trainable_variables)
        print()

        # Analyze gradient magnitudes
        print("📈 GRADIENT ANALYSIS RESULTS:")
        print("=" * 60)
        print(f"{'Layer':<15} {'Gradient Norm':<20} {'Status':<25}")
        print("-" * 60)

        gradient_norms = []

        for i, grad in enumerate(gradients):
            if i % 2 == 0:  # Only weights (skip biases)
                layer_num = i // 2 + 1
                grad_norm = tf.norm(grad).numpy()
                gradient_norms.append(grad_norm)

                # Determine status
                if grad_norm < 1e-6:
                    status = "🚨 VANISHED! (< 1e-6)"
                elif grad_norm < 1e-4:
                    status = "⚠️ Very small (< 1e-4)"
                elif grad_norm < 1e-2:
                    status = "🟡 Small (< 1e-2)"
                else:
                    status = "✅ Reasonable"

                print(f"Layer {layer_num:<8} {grad_norm:<20.8f} {status:<25}")

        print("-" * 60)

        # Summary statistics
        vanished_count = sum(1 for g in gradient_norms if g < 1e-6)
        weak_count = sum(1 for g in gradient_norms if g < 1e-4)

        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total layers: {len(gradient_norms)}")
        print(f"Vanished layers (< 1e-6): {vanished_count}")
        print(f"Weak layers (< 1e-4): {weak_count}")
        print(f"Gradient range: {min(gradient_norms):.2e} to {max(gradient_norms):.2e}")

        if vanished_count > 0:
            print(f"\n🚨 CRITICAL: {vanished_count} layers have vanished gradients!")
            print("   These layers will not learn effectively.")

        return gradient_norms

    except Exception as e:
        print(f"❌ Error computing gradients: {e}")
        return None

def create_visualization(gradient_norms):
    """Create visualization of gradient magnitudes"""

    if gradient_norms is None:
        print("❌ Cannot create visualization - no gradient data")
        return

    print("\n📊 Creating gradient visualization...")

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    layers = list(range(1, len(gradient_norms) + 1))

    # Plot 1: Gradient magnitudes (linear scale)
    ax1 = axes[0, 0]
    colors = ['red' if g < 1e-6 else 'orange' if g < 1e-4 else 'green' for g in gradient_norms]
    bars = ax1.bar(layers, gradient_norms, color=colors, alpha=0.7)
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.set_title('Gradient Magnitudes by Layer\n(Linear Scale)')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, gradient_norms):
        if value > ax1.get_ylim()[1] * 0.01:  # Only show if visible
            ax1.text(bar.get_x() + bar.get_width()/2, value, f'{value:.2e}',
                    ha='center', va='bottom', rotation=45, fontsize=8)

    # Plot 2: Gradient magnitudes (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(layers, gradient_norms, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold (1e-6)')
    ax2.axhline(y=1e-4, color='orange', linestyle='--', alpha=0.7, label='Weak threshold (1e-4)')
    ax2.set_xlabel('Layer Number')
    ax2.set_ylabel('Gradient Magnitude (log scale)')
    ax2.set_title('Gradient Magnitudes by Layer\n(Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Theoretical sigmoid derivative decay
    ax3 = axes[1, 0]
    theoretical_decay = [0.25**i for i in layers]
    ax3.semilogy(layers, theoretical_decay, 'b^-', linewidth=2, markersize=8, label='Theoretical (0.25^layer)')
    ax3.semilogy(layers, gradient_norms, 'ro-', linewidth=2, markersize=6, label='Actual gradients')
    ax3.set_xlabel('Layer Depth')
    ax3.set_ylabel('Gradient Magnitude (log scale)')
    ax3.set_title('Theoretical vs Actual\nGradient Decay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sigmoid function and derivative
    ax4 = axes[1, 1]
    x = np.linspace(-6, 6, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_deriv = sigmoid * (1 - sigmoid)

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(x, sigmoid, 'b-', linewidth=2, label='Sigmoid')
    line2 = ax4_twin.plot(x, sigmoid_deriv, 'r-', linewidth=2, label='Sigmoid Derivative')
    ax4_twin.axhline(y=0.25, color='k', linestyle='--', alpha=0.7, label='Max derivative = 0.25')

    ax4.set_xlabel('x')
    ax4.set_ylabel('Sigmoid(x)', color='b')
    ax4_twin.set_ylabel('Sigmoid\'(x)', color='r')
    ax4.set_title('Sigmoid Function & Derivative')

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vanishing_gradients_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'vanishing_gradients_analysis.png'")
    plt.show()

def mathematical_analysis():
    """Provide mathematical analysis of the vanishing gradient problem"""

    print("\n🧮 MATHEMATICAL ANALYSIS:")
    print("=" * 50)

    # Sigmoid derivative maximum
    print("Sigmoid Function Analysis:")
    print("σ(x) = 1/(1 + e^(-x))")
    print("σ'(x) = σ(x)(1 - σ(x))")
    print()

    x = np.linspace(-10, 10, 1000)
    sigmoid_deriv = (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
    max_deriv = np.max(sigmoid_deriv)

    print(f"Maximum sigmoid derivative: {max_deriv:.6f}")
    print(f"This occurs at x = 0")
    print()

    print("Gradient reduction through layers (theoretical):")
    print("-" * 50)
    for layer in range(1, 11):
        reduction = max_deriv ** layer
        percentage = reduction * 100
        print(f"Layer {layer:2d}: Gradient = {reduction:.2e} ({percentage:.6f}% of original)")

    print()
    print("💡 KEY INSIGHTS:")
    print("1. Each sigmoid layer can reduce gradient by up to 75% (multiply by 0.25)")
    print("2. After 5 layers: gradient is ~0.1% of original")
    print("3. After 10 layers: gradient is ~0.0001% of original")
    print("4. Early layers receive almost no learning signal!")

    # Real-world analogy
    print("\n🏢 REAL-WORLD ANALOGY:")
    print("=" * 50)
    print("Imagine a corporate message traveling through hierarchy:")
    print("• CEO (Output): Makes important decision")
    print("• Each level loses 75% of message clarity")
    print("• Front-line workers (Input): Get garbled instructions")
    print("• Result: Poor execution despite good intentions")

def compare_with_relu():
    """Compare sigmoid network with ReLU network"""

    print("\n⚔️ COMPARISON: SIGMOID vs ReLU")
    print("=" * 50)

    # Create ReLU network for comparison
    relu_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Same data as sigmoid network
    X = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1))

    try:
        with tf.GradientTape() as tape:
            predictions = relu_model(X)
            loss = tf.reduce_mean(tf.square(predictions - y))

        gradients = tape.gradient(loss, relu_model.trainable_variables)
        relu_norms = [tf.norm(grad).numpy() for i, grad in enumerate(gradients) if i % 2 == 0]

        print("ReLU Network Gradient Analysis:")
        for i, norm in enumerate(relu_norms, 1):
            status = "✅ Healthy" if norm > 1e-4 else "⚠️ Weak"
            print(f"Layer {i}: {norm:.6f} {status}")

        print(f"\nComparison Summary:")
        print(f"ReLU - Vanished layers: {sum(1 for g in relu_norms if g < 1e-6)}")
        print(f"ReLU - Weak layers: {sum(1 for g in relu_norms if g < 1e-4)}")
        print("🎯 CONCLUSION: ReLU preserves gradients much better than sigmoid!")

    except Exception as e:
        print(f"❌ Error in ReLU comparison: {e}")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🕳️ VANISHING GRADIENTS DEMONSTRATION")
    print("=" * 60)
    print()

    try:
        # Run main demonstration
        gradient_norms = demonstrate_vanishing_gradients()

        # Create visualization
        create_visualization(gradient_norms)

        # Mathematical analysis
        mathematical_analysis()

        # Compare with ReLU
        compare_with_relu()

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETE!")
        print("📊 Key files created: vanishing_gradients_analysis.png")
        print("🎓 Learning outcome: Understanding why deep sigmoid networks fail to train")
        print("💡 Next step: Learn about solutions (ReLU, proper initialization, etc.)")

        print("\n📚 BOOK REFERENCES:")
        print("=" * 60)
        print("1. 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, Aaron Courville")
        print("   - Chapter 8: Optimization for Training Deep Models")
        print("   - Chapter 10: Sequence Modeling: Recurrent and Recursive Nets")
        print()
        print("2. 'Deep Learning with Python' by François Chollet")
        print("   - Chapter 4: Getting started with neural networks")
        print("   - Chapter 6: Deep learning for text and sequences")
        print()
        print("3. 'Neural Networks and Deep Learning' by Charu C. Aggarwal")
        print("   - Chapter 4: Teaching Deep Networks to Learn")
        print("   - Chapter 7: Recurrent Neural Networks")
        print()
        print("4. 'Deep Learning with Applications Using Python'")
        print("   - Chapter 3: Deep Neural Networks")
        print("   - Chapter 4: Improving Deep Networks")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()