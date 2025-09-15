#!/usr/bin/env python3
"""
03_gradient_explosion_detector.py

Purpose: Implement and demonstrate gradient explosion detection system
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements a comprehensive gradient explosion detection system that can
identify exploding gradients, provide automated diagnostics, and suggest remedies.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GradientExplosionDetector:
    """Advanced gradient explosion detection and monitoring system"""

    def __init__(self, explosion_threshold=10.0, warning_threshold=5.0):
        self.explosion_threshold = explosion_threshold
        self.warning_threshold = warning_threshold
        self.history = []

    def detect_explosion(self, model, X, y):
        """Detect gradient explosion in a neural network"""

        try:
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = tf.reduce_mean(tf.square(predictions - y))

            gradients = tape.gradient(loss, model.trainable_variables)

            # Calculate gradient statistics
            grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]

            if not grad_norms:
                return None

            max_grad = np.max(grad_norms)
            mean_grad = np.mean(grad_norms)
            std_grad = np.std(grad_norms)

            # Detect explosion
            exploding_layers = sum(1 for g in grad_norms if g > self.explosion_threshold)
            warning_layers = sum(1 for g in grad_norms if g > self.warning_threshold)

            explosion_info = {
                'max_gradient': max_grad,
                'mean_gradient': mean_grad,
                'std_gradient': std_grad,
                'gradient_norms': grad_norms,
                'exploding_layers': exploding_layers,
                'warning_layers': warning_layers,
                'total_layers': len(grad_norms),
                'loss_value': loss.numpy(),
                'is_exploding': max_grad > self.explosion_threshold,
                'has_warning': max_grad > self.warning_threshold,
                'explosion_severity': self._calculate_severity(max_grad)
            }

            # Store in history
            self.history.append(explosion_info)

            return explosion_info

        except Exception as e:
            print(f"❌ Error in explosion detection: {e}")
            return None

    def _calculate_severity(self, max_grad):
        """Calculate explosion severity level"""
        if max_grad > 1000:
            return "CRITICAL"
        elif max_grad > 100:
            return "SEVERE"
        elif max_grad > 50:
            return "HIGH"
        elif max_grad > self.explosion_threshold:
            return "MODERATE"
        elif max_grad > self.warning_threshold:
            return "LOW"
        else:
            return "NONE"

    def diagnose_explosion(self, explosion_info):
        """Provide detailed diagnosis of gradient explosion"""

        if explosion_info is None:
            print("❌ No explosion data available for diagnosis")
            return

        print("\n💥 GRADIENT EXPLOSION DIAGNOSIS:")
        print("=" * 60)

        # Basic metrics
        print(f"Loss Value: {explosion_info['loss_value']:.6f}")
        print(f"Max Gradient: {explosion_info['max_gradient']:.6f}")
        print(f"Mean Gradient: {explosion_info['mean_gradient']:.6f}")
        print(f"Gradient Std: {explosion_info['std_gradient']:.6f}")
        print()

        # Severity assessment
        severity = explosion_info['explosion_severity']
        print(f"🚨 EXPLOSION SEVERITY: {severity}")

        if severity == "CRITICAL":
            print("   🔥 DANGER: Training will likely fail completely!")
            print("   📝 IMMEDIATE ACTION: Apply aggressive gradient clipping (max_norm=0.5)")
            print("   📝 ALSO TRY: Reduce learning rate by 10x or more")
        elif severity == "SEVERE":
            print("   ⚠️ WARNING: Training instability likely")
            print("   📝 RECOMMENDATION: Apply gradient clipping (max_norm=1.0)")
            print("   📝 ALTERNATIVE: Reduce learning rate by 5x")
        elif severity == "HIGH":
            print("   🟡 CAUTION: Monitor training closely")
            print("   📝 RECOMMENDATION: Apply gradient clipping (max_norm=2.0)")
            print("   📝 ALTERNATIVE: Reduce learning rate by 2x")
        elif severity == "MODERATE":
            print("   🟠 MODERATE: Some instability possible")
            print("   📝 RECOMMENDATION: Consider gradient clipping (max_norm=5.0)")
        elif severity == "LOW":
            print("   🟡 LOW: Minor warning signs")
            print("   📝 RECOMMENDATION: Monitor gradient trends")
        else:
            print("   ✅ HEALTHY: No explosion detected")

        print()

        # Layer-wise analysis
        print("📊 LAYER-WISE ANALYSIS:")
        print(f"   Total layers: {explosion_info['total_layers']}")
        print(f"   Exploding layers (>{self.explosion_threshold}): {explosion_info['exploding_layers']}")
        print(f"   Warning layers (>{self.warning_threshold}): {explosion_info['warning_layers']}")

        if explosion_info['exploding_layers'] > 0:
            print(f"\n🚨 {explosion_info['exploding_layers']} layers have exploding gradients!")
            print("   These layers will destabilize training.")

        # Specific recommendations
        self._provide_recommendations(explosion_info)

    def _provide_recommendations(self, explosion_info):
        """Provide specific recommendations based on explosion analysis"""

        print("\n💡 SPECIFIC RECOMMENDATIONS:")
        print("-" * 40)

        max_grad = explosion_info['max_gradient']

        # Gradient clipping recommendation
        if max_grad > 100:
            clip_value = 0.5
        elif max_grad > 50:
            clip_value = 1.0
        elif max_grad > 20:
            clip_value = 2.0
        elif max_grad > 10:
            clip_value = 5.0
        else:
            clip_value = None

        if clip_value:
            print(f"1. 🎯 GRADIENT CLIPPING:")
            print(f"   tf.clip_by_global_norm(gradients, {clip_value})")
            print(f"   or optimizer.clipnorm = {clip_value}")

        # Learning rate adjustment
        if max_grad > 50:
            lr_factor = 0.01
        elif max_grad > 20:
            lr_factor = 0.1
        elif max_grad > 10:
            lr_factor = 0.2
        else:
            lr_factor = 0.5

        if max_grad > self.explosion_threshold:
            print(f"\n2. 🎯 LEARNING RATE ADJUSTMENT:")
            print(f"   Reduce learning rate by factor of {1/lr_factor:.1f}")
            print(f"   New LR = current_lr * {lr_factor}")

        # Weight initialization
        print(f"\n3. 🎯 WEIGHT INITIALIZATION:")
        print("   Consider Xavier/Glorot or He initialization")
        print("   tf.keras.initializers.glorot_uniform()")
        print("   tf.keras.initializers.he_uniform()")

        # Batch normalization
        print(f"\n4. 🎯 NORMALIZATION:")
        print("   Add BatchNormalization layers")
        print("   tf.keras.layers.BatchNormalization()")

def create_explosion_prone_networks():
    """Create networks that are prone to gradient explosion"""

    print("🏗️ Creating explosion-prone networks...")

    networks = {}

    # Network 1: Poor initialization with large weights
    networks['Large Weights'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear', input_shape=(10,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3.0)),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3.0)),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3.0)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 2: Very deep linear network
    networks['Deep Linear'] = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='linear', input_shape=(10,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.5)),
        tf.keras.layers.Dense(128, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.5)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 3: Healthy network for comparison
    networks['Healthy (Xavier)'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,),
                             kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Network 4: Network with batch normalization
    networks['BatchNorm Protected'] = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear', input_shape=(10,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=2.0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(f"✅ Created {len(networks)} test networks")
    return networks

def comprehensive_explosion_analysis():
    """Run comprehensive explosion analysis on multiple networks"""

    print("🔬 COMPREHENSIVE GRADIENT EXPLOSION ANALYSIS")
    print("=" * 70)

    # Create detector
    detector = GradientExplosionDetector(explosion_threshold=10.0, warning_threshold=5.0)

    # Create test networks
    networks = create_explosion_prone_networks()

    # Generate test data
    print("\n📊 Generating test data...")
    X_sample = tf.random.normal((100, 10))
    y_sample = tf.random.uniform((100, 1))
    print(f"Data shape: {X_sample.shape} → {y_sample.shape}")

    # Analyze each network
    explosion_reports = {}

    for name, model in networks.items():
        print(f"\n{'='*70}")
        print(f"ANALYZING: {name}")
        print(f"{'='*70}")

        # Detect explosion
        explosion_info = detector.detect_explosion(model, X_sample, y_sample)

        if explosion_info is not None:
            explosion_reports[name] = explosion_info

            # Provide diagnosis
            detector.diagnose_explosion(explosion_info)

            print(f"\n📋 SUMMARY for {name}:")
            print(f"   Max Gradient: {explosion_info['max_gradient']:.2f}")
            print(f"   Severity: {explosion_info['explosion_severity']}")
            print(f"   Exploding Layers: {explosion_info['exploding_layers']}/{explosion_info['total_layers']}")
        else:
            print(f"❌ Failed to analyze {name}")

        print("\n" + "-"*70)

    return explosion_reports, detector

def create_explosion_visualization(explosion_reports):
    """Create comprehensive visualization of explosion analysis"""

    if not explosion_reports:
        print("❌ No explosion reports available for visualization")
        return

    print("\n📊 Creating explosion visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    network_names = list(explosion_reports.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(network_names)))

    # Plot 1: Max gradient comparison
    ax1 = axes[0, 0]
    max_grads = [explosion_reports[name]['max_gradient'] for name in network_names]

    bars = ax1.bar(range(len(network_names)), max_grads, color=colors, alpha=0.7)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Explosion threshold')
    ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
    ax1.set_yscale('log')
    ax1.set_xlabel('Network Type')
    ax1.set_ylabel('Max Gradient (log scale)')
    ax1.set_title('Maximum Gradient by Network')
    ax1.set_xticks(range(len(network_names)))
    ax1.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, max_grads):
        ax1.text(bar.get_x() + bar.get_width()/2, value, f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold')

    # Plot 2: Exploding layer counts
    ax2 = axes[0, 1]
    exploding_counts = [explosion_reports[name]['exploding_layers'] for name in network_names]
    warning_counts = [explosion_reports[name]['warning_layers'] for name in network_names]

    x_pos = np.arange(len(network_names))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, exploding_counts, width,
                    label='Exploding (>10)', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, warning_counts, width,
                    label='Warning (>5)', color='orange', alpha=0.7)

    ax2.set_xlabel('Network Type')
    ax2.set_ylabel('Number of Problematic Layers')
    ax2.set_title('Problematic Layers by Network')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Layer-wise gradient magnitudes
    ax3 = axes[1, 0]
    for i, (name, report) in enumerate(explosion_reports.items()):
        grad_norms = report['gradient_norms']
        layers = list(range(1, len(grad_norms) + 1))
        ax3.plot(layers, grad_norms, 'o-', color=colors[i],
                label=name.split()[0], linewidth=2, markersize=6)

    ax3.set_yscale('log')
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Explosion threshold')
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax3.set_xlabel('Layer Number')
    ax3.set_ylabel('Gradient Magnitude (log scale)')
    ax3.set_title('Layer-wise Gradient Magnitudes')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Severity distribution
    ax4 = axes[1, 1]
    severities = [explosion_reports[name]['explosion_severity'] for name in network_names]
    severity_colors = {'NONE': 'green', 'LOW': 'yellow', 'MODERATE': 'orange',
                      'HIGH': 'red', 'SEVERE': 'darkred', 'CRITICAL': 'black'}
    bar_colors = [severity_colors.get(sev, 'gray') for sev in severities]

    bars = ax4.bar(range(len(network_names)), [1]*len(network_names),
                   color=bar_colors, alpha=0.7)
    ax4.set_xlabel('Network Type')
    ax4.set_ylabel('Explosion Severity')
    ax4.set_title('Explosion Severity by Network')
    ax4.set_xticks(range(len(network_names)))
    ax4.set_xticklabels([name.split()[0] for name in network_names], rotation=45)
    ax4.set_yticks([])

    # Add severity labels
    for i, (bar, severity) in enumerate(zip(bars, severities)):
        ax4.text(bar.get_x() + bar.get_width()/2, 0.5, severity,
                ha='center', va='center', fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig('gradient_explosion_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'gradient_explosion_analysis.png'")
    plt.show()

def demonstrate_gradient_clipping():
    """Demonstrate how gradient clipping solves explosion problem"""

    print("\n🎯 GRADIENT CLIPPING DEMONSTRATION")
    print("=" * 50)

    # Create explosion-prone network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear', input_shape=(10,),
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3.0)),
        tf.keras.layers.Dense(64, activation='linear',
                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=3.0)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Generate data
    X = tf.random.normal((100, 10))
    y = tf.random.uniform((100, 1))

    print("Testing gradient clipping effectiveness...")

    # Test different clipping values
    clip_values = [None, 10.0, 5.0, 2.0, 1.0, 0.5]
    results = {}

    for clip_value in clip_values:
        try:
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = tf.reduce_mean(tf.square(predictions - y))

            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply clipping if specified
            if clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_value)

            # Calculate statistics
            grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
            max_grad = np.max(grad_norms)

            results[clip_value] = {
                'max_gradient': max_grad,
                'mean_gradient': np.mean(grad_norms),
                'clipped': clip_value is not None
            }

            status = "CLIPPED" if clip_value is not None else "UNCLIPPED"
            print(f"Clip value: {str(clip_value):>6} | Max grad: {max_grad:>8.2f} | Status: {status}")

        except Exception as e:
            print(f"Error with clip_value {clip_value}: {e}")

    print("\n💡 CLIPPING EFFECTIVENESS:")
    if None in results and results[None]['max_gradient'] > 10:
        print("✅ Gradient clipping successfully controlled explosion!")
        for clip_val in [1.0, 0.5]:
            if clip_val in results:
                reduction = (results[None]['max_gradient'] - results[clip_val]['max_gradient']) / results[None]['max_gradient']
                print(f"   Clip={clip_val}: Reduced gradient by {reduction*100:.1f}%")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("💥 GRADIENT EXPLOSION DETECTION SYSTEM")
    print("=" * 70)
    print()

    try:
        # Run comprehensive analysis
        explosion_reports, detector = comprehensive_explosion_analysis()

        # Create visualizations
        create_explosion_visualization(explosion_reports)

        # Demonstrate gradient clipping
        demonstrate_gradient_clipping()

        print("\n" + "=" * 70)
        print("✅ GRADIENT EXPLOSION ANALYSIS COMPLETE!")
        print("📊 Key files created: gradient_explosion_analysis.png")
        print("🎓 Learning outcome: Automated gradient explosion detection")
        print("💡 Next step: Learn about gradient clipping implementation")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()