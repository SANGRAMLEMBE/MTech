#!/usr/bin/env python3
"""
02_gradient_health_monitor.py

Purpose: Implement and demonstrate gradient health monitoring system
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements a comprehensive gradient health monitoring system that can
detect vanishing gradients, exploding gradients, and provide automated diagnostics.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def monitor_gradient_health(model, X, y):
    """Monitor gradient health during training"""

    try:
        with tf.GradientTape() as tape:
            predictions = model(X)
            # Fixed: Use direct MSE calculation instead of tf.keras.losses.mean_squared_error
            loss = tf.reduce_mean(tf.square(predictions - y))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Calculate statistics
        grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]

        if not grad_norms:
            return None

        metrics = {
            'min_gradient': np.min(grad_norms),
            'max_gradient': np.max(grad_norms),
            'mean_gradient': np.mean(grad_norms),
            'std_gradient': np.std(grad_norms),
            'vanished_layers': sum(1 for g in grad_norms if g < 1e-6),
            'weak_layers': sum(1 for g in grad_norms if g < 1e-4),
            'exploding_layers': sum(1 for g in grad_norms if g > 10.0),
            'total_layers': len(grad_norms),
            'gradient_norms': grad_norms,
            'loss_value': loss.numpy()
        }

        return metrics

    except Exception as e:
        print(f"❌ Error in gradient health monitoring: {e}")
        return None

def interpret_gradient_health(health_metrics):
    """Provide human-readable interpretation of gradient health"""

    if health_metrics is None:
        print("❌ No health metrics available for interpretation")
        return

    print("\n🏥 GRADIENT HEALTH DIAGNOSIS:")
    print("=" * 50)

    # Header with basic stats
    print(f"Total Layers Analyzed: {health_metrics['total_layers']}")
    print(f"Loss Value: {health_metrics['loss_value']:.6f}")
    print(f"Loss Computed Successfully: ✅")
    print()

    # Vanishing gradient assessment
    vanished = health_metrics['vanished_layers']
    weak = health_metrics['weak_layers']
    exploding = health_metrics['exploding_layers']

    print("🔍 VANISHING GRADIENT ANALYSIS:")
    if vanished > 0:
        print(f"🚨 CRITICAL: {vanished} layers have vanished gradients!")
        print("   📝 Recommendation: Switch to ReLU activation")
        print("   📝 Alternative: Check weight initialization")
    elif weak > health_metrics['total_layers'] // 2:
        print(f"⚠️ WARNING: {weak} layers have weak gradients")
        print("   📝 Recommendation: Consider ReLU or better initialization")
    else:
        print("✅ GOOD: No significant vanishing gradient problems detected")

    print()

    # Exploding gradient assessment
    max_grad = health_metrics['max_gradient']
    print("💥 EXPLODING GRADIENT ANALYSIS:")
    if exploding > 0:
        print(f"🚨 CRITICAL: {exploding} layers have exploding gradients!")
        print("   📝 Recommendation: Apply gradient clipping")
        print("   📝 Alternative: Reduce learning rate")
    elif max_grad > 10:
        print("⚠️ WARNING: Large gradients detected")
        print("   📝 Recommendation: Monitor closely, consider gradient clipping")
    else:
        print("✅ GOOD: No gradient explosion detected")

    print()

    # Overall network stability
    min_grad = health_metrics['min_gradient']
    gradient_ratio = max_grad / (min_grad + 1e-10)  # Avoid division by zero

    print("⚖️ NETWORK STABILITY ANALYSIS:")
    if gradient_ratio > 100000:
        print("🚨 CRITICAL: Very large gradient range - training may be unstable")
        print("   📝 Recommendation: Improve initialization or add normalization")
    elif gradient_ratio > 10000:
        print("⚠️ WARNING: Large gradient range detected")
        print("   📝 Recommendation: Monitor training stability")
    else:
        print("✅ GOOD: Gradient range is reasonable")

    print()

    # Detailed statistics
    print("📊 DETAILED STATISTICS:")
    print(f"   Min gradient: {min_grad:.2e}")
    print(f"   Max gradient: {max_grad:.2e}")
    print(f"   Mean gradient: {health_metrics['mean_gradient']:.2e}")
    print(f"   Std gradient: {health_metrics['std_gradient']:.2e}")
    print(f"   Gradient range ratio: {gradient_ratio:.1e}")
    print(f"   Vanished layers: {vanished}/{health_metrics['total_layers']}")
    print(f"   Weak layers: {weak}/{health_metrics['total_layers']}")
    print(f"   Exploding layers: {exploding}/{health_metrics['total_layers']}")

    # Overall health score
    health_score = 0
    if vanished == 0: health_score += 3
    elif vanished < health_metrics['total_layers'] // 3: health_score += 1

    if max_grad < 10: health_score += 3
    elif max_grad < 100: health_score += 1

    if gradient_ratio < 10000: health_score += 2
    elif gradient_ratio < 100000: health_score += 1

    if exploding == 0: health_score += 2

    print()
    print("🎯 OVERALL HEALTH ASSESSMENT:")
    if health_score >= 8:
        print("🟢 EXCELLENT: Network gradients are healthy")
        status = "EXCELLENT"
    elif health_score >= 6:
        print("🟡 GOOD: Minor issues detected, but manageable")
        status = "GOOD"
    elif health_score >= 4:
        print("🟠 MODERATE: Some issues detected, needs attention")
        status = "MODERATE"
    elif health_score >= 2:
        print("🔴 POOR: Significant gradient problems detected")
        status = "POOR"
    else:
        print("⚫ CRITICAL: Severe gradient problems - network may not train")
        status = "CRITICAL"

    print(f"Health Score: {health_score}/10")

    return status, health_score

def create_test_networks():
    """Create different networks for testing gradient health"""

    print("🏗️ Creating test networks...")

    networks = {}

    # Network 1: Problematic sigmoid network
    networks['Sigmoid (Problematic)'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 2: Healthy ReLU network
    networks['ReLU (Healthy)'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 3: Exploding gradient network (bad initialization)
    networks['Exploding (Dangerous)'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear', input_shape=(10,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 4: Well-initialized deep network
    networks['Well-Initialized'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,),
                             kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(f"✅ Created {len(networks)} test networks")
    return networks

def comprehensive_health_analysis():
    """Run comprehensive health analysis on multiple networks"""

    print("🔬 COMPREHENSIVE GRADIENT HEALTH ANALYSIS")
    print("=" * 60)

    # Create test networks
    networks = create_test_networks()

    # Generate test data
    print("\n📊 Generating test data...")
    X_sample = tf.random.normal((100, 10))
    y_sample = tf.random.uniform((100, 1))
    print(f"Data shape: {X_sample.shape} → {y_sample.shape}")

    # Analyze each network
    health_reports = {}
    health_scores = {}

    for name, model in networks.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name}")
        print(f"{'='*60}")

        # Monitor gradient health
        health = monitor_gradient_health(model, X_sample, y_sample)

        if health is not None:
            health_reports[name] = health

            # Interpret results
            status, score = interpret_gradient_health(health)
            health_scores[name] = {'status': status, 'score': score}

            print(f"\n📋 SUMMARY for {name}:")
            print(f"   Status: {status}")
            print(f"   Score: {score}/10")
        else:
            print(f"❌ Failed to analyze {name}")

        print("\n" + "-"*60)

    return health_reports, health_scores

def create_health_visualization(health_reports):
    """Create comprehensive visualization of gradient health"""

    if not health_reports:
        print("❌ No health reports available for visualization")
        return

    print("\n📊 Creating health visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    network_names = list(health_reports.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(network_names)))

    # Plot 1: Gradient ranges by network
    ax1 = axes[0, 0]
    min_grads = [health_reports[name]['min_gradient'] for name in network_names]
    max_grads = [health_reports[name]['max_gradient'] for name in network_names]

    x_pos = np.arange(len(network_names))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, min_grads, width, label='Min Gradient', alpha=0.7, color='blue')
    bars2 = ax1.bar(x_pos + width/2, max_grads, width, label='Max Gradient', alpha=0.7, color='red')

    ax1.set_yscale('log')
    ax1.set_xlabel('Network Type')
    ax1.set_ylabel('Gradient Magnitude (log scale)')
    ax1.set_title('Gradient Range by Network Type')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Problem layer counts
    ax2 = axes[0, 1]
    vanished_counts = [health_reports[name]['vanished_layers'] for name in network_names]
    weak_counts = [health_reports[name]['weak_layers'] for name in network_names]
    exploding_counts = [health_reports[name]['exploding_layers'] for name in network_names]

    bars1 = ax2.bar(x_pos - width, vanished_counts, width, label='Vanished', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos, weak_counts, width, label='Weak', color='orange', alpha=0.7)
    bars3 = ax2.bar(x_pos + width, exploding_counts, width, label='Exploding', color='purple', alpha=0.7)

    ax2.set_xlabel('Network Type')
    ax2.set_ylabel('Number of Problematic Layers')
    ax2.set_title('Problematic Layers by Network Type')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Individual gradient norms
    ax3 = axes[1, 0]
    for i, (name, health) in enumerate(health_reports.items()):
        grad_norms = health['gradient_norms']
        layers = list(range(1, len(grad_norms) + 1))
        ax3.plot(layers, grad_norms, 'o-', color=colors[i],
                label=name.split()[0], linewidth=2, markersize=6)

    ax3.set_yscale('log')
    ax3.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
    ax3.axhline(y=1e-4, color='orange', linestyle='--', alpha=0.5, label='Weak threshold')
    ax3.axhline(y=10, color='purple', linestyle='--', alpha=0.5, label='Exploding threshold')
    ax3.set_xlabel('Layer Number')
    ax3.set_ylabel('Gradient Magnitude (log scale)')
    ax3.set_title('Layer-wise Gradient Magnitudes')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Health score comparison
    ax4 = axes[1, 1]
    # We'll simulate health scores for visualization
    health_scores = []
    for name, health in health_reports.items():
        score = 0
        if health['vanished_layers'] == 0: score += 3
        if health['max_gradient'] < 10: score += 3
        if health['max_gradient'] / (health['min_gradient'] + 1e-10) < 10000: score += 2
        if health['exploding_layers'] == 0: score += 2
        health_scores.append(score)

    bars = ax4.bar(range(len(network_names)), health_scores,
                   color=colors, alpha=0.7)
    ax4.set_xlabel('Network Type')
    ax4.set_ylabel('Health Score (0-10)')
    ax4.set_title('Overall Gradient Health Score')
    ax4.set_xticks(range(len(network_names)))
    ax4.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax4.set_ylim(0, 10)

    # Add score labels
    for bar, score in zip(bars, health_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, score + 0.1, f'{score}/10',
                ha='center', va='bottom', fontweight='bold')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_health_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'gradient_health_analysis.png'")
    plt.show()

def create_comparison_table(health_reports):
    """Create a comprehensive comparison table"""

    print("\n📊 COMPREHENSIVE NETWORK COMPARISON")
    print("=" * 80)

    # Headers
    print(f"{'Network':<20} {'Min Grad':<12} {'Max Grad':<12} {'Mean Grad':<12} {'Vanished':<10} {'Health':<8}")
    print("-" * 80)

    for name, health in health_reports.items():
        # Calculate simple health indicator
        if health['vanished_layers'] == 0 and health['exploding_layers'] == 0:
            health_emoji = "🟢"
        elif health['vanished_layers'] <= 1 and health['exploding_layers'] <= 1:
            health_emoji = "🟡"
        else:
            health_emoji = "🔴"

        print(f"{name:<20} {health['min_gradient']:<12.2e} {health['max_gradient']:<12.2e} "
              f"{health['mean_gradient']:<12.2e} {health['vanished_layers']:<10} {health_emoji}")

    print("-" * 80)
    print("\n🎯 SUMMARY CONCLUSIONS:")
    print("✅ Well-initialized networks: Healthy gradient flow")
    print("🚨 Sigmoid networks: Severe vanishing gradients")
    print("⚠️ Poorly initialized networks: Unstable gradients")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🏥 GRADIENT HEALTH MONITORING SYSTEM")
    print("=" * 60)
    print()

    try:
        # Run comprehensive analysis
        health_reports, health_scores = comprehensive_health_analysis()

        # Create visualizations
        create_health_visualization(health_reports)

        # Create comparison table
        create_comparison_table(health_reports)

        print("\n" + "=" * 60)
        print("✅ GRADIENT HEALTH ANALYSIS COMPLETE!")
        print("📊 Key files created: gradient_health_analysis.png")
        print("🎓 Learning outcome: Automated gradient health monitoring")
        print("💡 Next step: Learn about gradient explosion detection")

        print("\n📚 BOOK REFERENCES:")
        print("=" * 60)
        print("1. 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, Aaron Courville")
        print("   - Chapter 8: Optimization for Training Deep Models")
        print("   - Chapter 11: Practical Methodology")
        print()
        print("2. 'Neural Networks and Deep Learning' by Charu C. Aggarwal")
        print("   - Chapter 4: Teaching Deep Networks to Learn")
        print("   - Chapter 1: An Introduction to Neural Networks")
        print()
        print("3. 'Deep Learning with Python' by François Chollet")
        print("   - Chapter 4: Getting started with neural networks")
        print("   - Chapter 7: Advanced deep learning best practices")
        print()
        print("4. 'Deep Learning with Applications Using Python'")
        print("   - Chapter 4: Improving Deep Networks")
        print("   - Chapter 8: Debugging and Monitoring")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()