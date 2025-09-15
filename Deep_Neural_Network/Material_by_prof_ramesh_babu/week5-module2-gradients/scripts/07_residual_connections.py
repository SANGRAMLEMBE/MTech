#!/usr/bin/env python3
"""
07_residual_connections.py

Purpose: Demonstrate residual connections and skip connections for deep networks
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script shows how residual connections solve the degradation problem in very deep networks
and enable training of networks with hundreds of layers.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_residual_block(inputs, filters, kernel_size=3, strides=1, activation='relu'):
    """Create a basic residual block"""

    # Main path
    x = tf.keras.layers.Dense(filters, activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    x = tf.keras.layers.Dense(filters, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Shortcut connection
    if inputs.shape[-1] != filters:
        # Project shortcut to match dimensions
        shortcut = tf.keras.layers.Dense(filters, activation=None)(inputs)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    # Add shortcut
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)

    return x

def demonstrate_degradation_problem():
    """Demonstrate the degradation problem in very deep networks"""

    print("🔍 DEMONSTRATING THE DEGRADATION PROBLEM")
    print("=" * 60)

    # Create networks of different depths without residual connections
    depths = [5, 10, 20, 30, 40]
    plain_networks = {}

    print("Creating plain networks of different depths...")

    for depth in depths:
        print(f"  Creating {depth}-layer network...")

        layers = [tf.keras.layers.Dense(64, activation='relu', input_shape=(32,))]

        # Add hidden layers
        for i in range(depth - 2):
            layers.append(tf.keras.layers.Dense(64, activation='relu'))

        # Output layer
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

        plain_networks[f'{depth}_layers'] = tf.keras.Sequential(layers)

    # Generate complex dataset
    X_train = tf.random.normal((2000, 32))
    # Create complex non-linear target
    y_train = tf.cast(
        (tf.reduce_sum(X_train[:, :8], axis=1) * tf.reduce_sum(X_train[:, 8:16], axis=1)) >
        (tf.reduce_sum(X_train[:, 16:24], axis=1) * tf.reduce_sum(X_train[:, 24:], axis=1)),
        tf.float32
    )
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((400, 32))
    y_val = tf.cast(
        (tf.reduce_sum(X_val[:, :8], axis=1) * tf.reduce_sum(X_val[:, 8:16], axis=1)) >
        (tf.reduce_sum(X_val[:, 16:24], axis=1) * tf.reduce_sum(X_val[:, 24:], axis=1)),
        tf.float32
    )
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"\nTraining dataset: {X_train.shape}, Validation: {X_val.shape}")

    # Train networks and observe degradation
    degradation_results = {}

    print("\nTraining networks to observe degradation...")
    print(f"{'Depth':<8} {'Final Train Acc':<15} {'Final Val Acc':<15} {'Best Val Acc':<15}")
    print("-" * 60)

    for depth in depths:
        model_name = f'{depth}_layers'
        model = plain_networks[model_name]

        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=64,
                verbose=0
            )

            degradation_results[depth] = {
                'history': history.history,
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'best_val_acc': max(history.history['val_accuracy'])
            }

            print(f"{depth:<8} {degradation_results[depth]['final_train_acc']:<15.4f} "
                  f"{degradation_results[depth]['final_val_acc']:<15.4f} "
                  f"{degradation_results[depth]['best_val_acc']:<15.4f}")

        except Exception as e:
            print(f"Failed to train {depth}-layer network: {e}")

    # Visualize degradation problem
    create_degradation_visualization(degradation_results, depths)

    return degradation_results

def create_degradation_visualization(degradation_results, depths):
    """Visualize the degradation problem"""

    print("\n📊 Creating degradation problem visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training curves for different depths
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))

    for i, depth in enumerate(depths):
        if depth in degradation_results:
            history = degradation_results[depth]['history']
            ax1.plot(history['accuracy'], color=colors[i], linewidth=2,
                    label=f'{depth} layers', alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.set_title('Training Accuracy vs Network Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation curves
    ax2 = axes[0, 1]

    for i, depth in enumerate(depths):
        if depth in degradation_results:
            history = degradation_results[depth]['history']
            ax2.plot(history['val_accuracy'], color=colors[i], linewidth=2,
                    label=f'{depth} layers', alpha=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy vs Network Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best performance vs depth
    ax3 = axes[1, 0]
    best_accs = [degradation_results[d]['best_val_acc'] for d in depths if d in degradation_results]
    available_depths = [d for d in depths if d in degradation_results]

    ax3.plot(available_depths, best_accs, 'ro-', linewidth=3, markersize=8)
    ax3.set_xlabel('Network Depth (layers)')
    ax3.set_ylabel('Best Validation Accuracy')
    ax3.set_title('Degradation Problem:\nDeeper ≠ Better Performance')
    ax3.grid(True, alpha=0.3)

    # Annotate the degradation
    max_acc_idx = np.argmax(best_accs)
    ax3.annotate(f'Peak at {available_depths[max_acc_idx]} layers\n({best_accs[max_acc_idx]:.3f})',
                xy=(available_depths[max_acc_idx], best_accs[max_acc_idx]),
                xytext=(available_depths[max_acc_idx] + 5, best_accs[max_acc_idx] + 0.01),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Plot 4: Training difficulty visualization
    ax4 = axes[1, 1]
    final_train_accs = [degradation_results[d]['final_train_acc'] for d in depths if d in degradation_results]
    final_val_accs = [degradation_results[d]['final_val_acc'] for d in depths if d in degradation_results]

    x_pos = np.arange(len(available_depths))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, final_train_accs, width,
                    label='Final Train Accuracy', alpha=0.7, color='skyblue')
    bars2 = ax4.bar(x_pos + width/2, final_val_accs, width,
                    label='Final Val Accuracy', alpha=0.7, color='lightcoral')

    ax4.set_xlabel('Network Depth')
    ax4.set_ylabel('Final Accuracy')
    ax4.set_title('Training Difficulty Increases with Depth')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{d}L' for d in available_depths])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('degradation_problem_demonstration.png', dpi=300, bbox_inches='tight')
    print("✅ Degradation problem visualization saved as 'degradation_problem_demonstration.png'")
    plt.show()

def create_resnet_architectures():
    """Create ResNet-style architectures with residual connections"""

    print("\n🏗️ CREATING RESNET ARCHITECTURES")
    print("=" * 60)

    def create_plain_network(depth, input_dim=32):
        """Create plain network without residual connections"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,))
        ])

        for i in range(depth - 2):
            model.add(tf.keras.layers.Dense(64, activation='relu'))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def create_resnet_network(depth, input_dim=32):
        """Create ResNet-style network with residual connections"""
        inputs = tf.keras.layers.Input(shape=(input_dim,))

        # Initial dense layer
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)

        # Add residual blocks
        num_blocks = (depth - 2) // 2  # Each block has 2 dense layers

        for i in range(num_blocks):
            x = create_residual_block(x, filters=64)

        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs)

    # Create networks for comparison
    depths = [20, 40, 60]
    networks = {}

    print("Creating network architectures...")

    for depth in depths:
        print(f"  Creating {depth}-layer networks...")
        networks[f'Plain_{depth}L'] = create_plain_network(depth)
        networks[f'ResNet_{depth}L'] = create_resnet_network(depth)

    return networks

def compare_plain_vs_resnet():
    """Compare plain networks vs ResNet architectures"""

    print("\n⚔️ PLAIN NETWORKS vs RESNET COMPARISON")
    print("=" * 60)

    # Create networks
    networks = create_resnet_architectures()

    # Generate dataset
    X_train = tf.random.normal((3000, 32))
    # Complex target function
    y_train = tf.cast(
        tf.reduce_sum(tf.nn.relu(X_train[:, :16] - 0.5) * tf.nn.relu(X_train[:, 16:] + 0.5), axis=1) > 10,
        tf.float32
    )
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((600, 32))
    y_val = tf.cast(
        tf.reduce_sum(tf.nn.relu(X_val[:, :16] - 0.5) * tf.nn.relu(X_val[:, 16:] + 0.5), axis=1) > 10,
        tf.float32
    )
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Dataset: {X_train.shape[0]} train, {X_val.shape[0]} validation samples")

    # Train networks
    comparison_results = {}

    print("\nTraining networks for comparison...")
    print(f"{'Network':<15} {'Trainable Params':<18} {'Best Val Acc':<15} {'Final Val Acc':<15}")
    print("-" * 70)

    for name, model in networks.items():
        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                verbose=0
            )

            comparison_results[name] = {
                'history': history.history,
                'trainable_params': model.count_params(),
                'best_val_acc': max(history.history['val_accuracy']),
                'final_val_acc': history.history['val_accuracy'][-1],
                'convergence_epoch': np.argmax(history.history['val_accuracy']) + 1
            }

            print(f"{name:<15} {comparison_results[name]['trainable_params']:<18} "
                  f"{comparison_results[name]['best_val_acc']:<15.4f} "
                  f"{comparison_results[name]['final_val_acc']:<15.4f}")

        except Exception as e:
            print(f"Failed to train {name}: {e}")

    # Create comparison visualization
    create_resnet_comparison_plots(comparison_results)

    return comparison_results

def create_resnet_comparison_plots(comparison_results):
    """Create comprehensive ResNet vs Plain network comparison plots"""

    print("\n📊 Creating ResNet comparison visualization...")

    if not comparison_results:
        print("❌ No comparison results to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Separate plain and ResNet results
    plain_results = {k: v for k, v in comparison_results.items() if 'Plain' in k}
    resnet_results = {k: v for k, v in comparison_results.items() if 'ResNet' in k}

    # Plot 1: Training curves comparison
    ax1 = axes[0, 0]

    # Plot plain networks
    for name, data in plain_results.items():
        depth = name.split('_')[1]
        ax1.plot(data['history']['val_accuracy'], '--', linewidth=2,
                label=f'Plain {depth}', alpha=0.7)

    # Plot ResNet networks
    for name, data in resnet_results.items():
        depth = name.split('_')[1]
        ax1.plot(data['history']['val_accuracy'], '-', linewidth=2,
                label=f'ResNet {depth}', alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Training Curves: Plain vs ResNet')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best performance vs depth
    ax2 = axes[0, 1]

    # Extract depths and best accuracies
    plain_depths = [int(k.split('_')[1][:-1]) for k in plain_results.keys()]
    plain_best_accs = [plain_results[k]['best_val_acc'] for k in plain_results.keys()]

    resnet_depths = [int(k.split('_')[1][:-1]) for k in resnet_results.keys()]
    resnet_best_accs = [resnet_results[k]['best_val_acc'] for k in resnet_results.keys()]

    ax2.plot(plain_depths, plain_best_accs, 's--', linewidth=3, markersize=8,
            label='Plain Networks', color='red', alpha=0.7)
    ax2.plot(resnet_depths, resnet_best_accs, 'o-', linewidth=3, markersize=8,
            label='ResNet', color='green', alpha=0.8)

    ax2.set_xlabel('Network Depth (layers)')
    ax2.set_ylabel('Best Validation Accuracy')
    ax2.set_title('ResNet Solves Degradation Problem')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Convergence speed
    ax3 = axes[1, 0]

    plain_conv = [plain_results[k]['convergence_epoch'] for k in plain_results.keys()]
    resnet_conv = [resnet_results[k]['convergence_epoch'] for k in resnet_results.keys()]

    x_pos = np.arange(len(plain_depths))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, plain_conv, width, label='Plain Networks',
                    alpha=0.7, color='red')
    bars2 = ax3.bar(x_pos + width/2, resnet_conv, width, label='ResNet',
                    alpha=0.7, color='green')

    ax3.set_xlabel('Network Configuration')
    ax3.set_ylabel('Convergence Epoch')
    ax3.set_title('Convergence Speed Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{d}L' for d in plain_depths])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Performance improvement
    ax4 = axes[1, 1]

    improvements = []
    depths_for_improvement = []

    for i, depth in enumerate(plain_depths):
        if i < len(resnet_best_accs) and i < len(plain_best_accs):
            improvement = (resnet_best_accs[i] - plain_best_accs[i]) / plain_best_accs[i] * 100
            improvements.append(improvement)
            depths_for_improvement.append(depth)

    bars = ax4.bar(depths_for_improvement, improvements, alpha=0.7,
                  color=['green' if imp > 0 else 'red' for imp in improvements])
    ax4.set_xlabel('Network Depth')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('ResNet Performance Improvement over Plain')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, improvements):
        ax4.text(bar.get_x() + bar.get_width()/2, value + (1 if value > 0 else -1),
                f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top',
                fontweight='bold')

    plt.tight_layout()
    plt.savefig('resnet_vs_plain_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ ResNet comparison visualization saved as 'resnet_vs_plain_comparison.png'")
    plt.show()

def analyze_residual_block_variants():
    """Analyze different variants of residual blocks"""

    print("\n🔬 RESIDUAL BLOCK VARIANTS ANALYSIS")
    print("=" * 60)

    def create_basic_block(inputs, filters):
        """Basic residual block: Conv-BN-ReLU-Conv-BN-Add-ReLU"""
        x = tf.keras.layers.Dense(filters, activation=None)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(filters, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Match dimensions if necessary
        if inputs.shape[-1] != filters:
            shortcut = tf.keras.layers.Dense(filters, activation=None)(inputs)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = inputs

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    def create_preactivation_block(inputs, filters):
        """Pre-activation block: BN-ReLU-Conv-BN-ReLU-Conv-Add"""
        # Match dimensions first if necessary
        if inputs.shape[-1] != filters:
            shortcut = tf.keras.layers.Dense(filters, activation=None)(inputs)
        else:
            shortcut = inputs

        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(filters, activation=None)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(filters, activation=None)(x)

        x = tf.keras.layers.Add()([x, shortcut])
        return x

    def create_bottleneck_block(inputs, filters):
        """Bottleneck block: 1x1-reduce, 3x3, 1x1-expand"""
        bottleneck_filters = filters // 4

        x = tf.keras.layers.Dense(bottleneck_filters, activation=None)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(bottleneck_filters, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(filters, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Match dimensions if necessary
        if inputs.shape[-1] != filters:
            shortcut = tf.keras.layers.Dense(filters, activation=None)(inputs)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = inputs

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    block_variants = {
        'Basic Block': create_basic_block,
        'Pre-activation': create_preactivation_block,
        'Bottleneck': create_bottleneck_block
    }

    print("Testing different residual block variants...")

    # Create networks with different block types
    variant_networks = {}

    for variant_name, block_fn in block_variants.items():
        inputs = tf.keras.layers.Input(shape=(32,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)

        # Add multiple residual blocks
        for i in range(6):
            x = block_fn(x, 64)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        variant_networks[variant_name] = tf.keras.Model(inputs, outputs)

    print(f"Created {len(variant_networks)} network variants")

    # Print parameter counts
    print("\n📊 MODEL COMPLEXITY COMPARISON:")
    print(f"{'Variant':<20} {'Parameters':<15} {'Relative Size':<15}")
    print("-" * 50)

    param_counts = {}
    for name, model in variant_networks.items():
        param_count = model.count_params()
        param_counts[name] = param_count

    min_params = min(param_counts.values())

    for name, param_count in param_counts.items():
        relative_size = param_count / min_params
        print(f"{name:<20} {param_count:<15,} {relative_size:<15.2f}x")

    return variant_networks

def demonstrate_gradient_flow_improvement():
    """Demonstrate how residual connections improve gradient flow"""

    print("\n🌊 GRADIENT FLOW IMPROVEMENT DEMONSTRATION")
    print("=" * 60)

    # Create networks for gradient flow comparison
    def create_deep_plain_network(depth=30):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,))
        ])
        for i in range(depth - 2):
            model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def create_deep_resnet(depth=30):
        inputs = tf.keras.layers.Input(shape=(20,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)

        num_blocks = (depth - 2) // 2
        for i in range(num_blocks):
            x = create_residual_block(x, 64)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    # Create models
    plain_model = create_deep_plain_network(30)
    resnet_model = create_deep_resnet(30)

    # Generate data
    X_sample = tf.random.normal((100, 20))
    y_sample = tf.random.uniform((100, 1))

    print("Analyzing gradient flow in 30-layer networks...")

    # Analyze gradients
    models = {'Plain (30L)': plain_model, 'ResNet (30L)': resnet_model}
    gradient_analysis = {}

    for name, model in models.items():
        try:
            with tf.GradientTape() as tape:
                predictions = model(X_sample)
                loss = tf.reduce_mean(tf.square(predictions - y_sample))

            gradients = tape.gradient(loss, model.trainable_variables)

            # Calculate gradient norms for weight matrices only
            weight_grad_norms = []
            layer_names = []

            for i, grad in enumerate(gradients):
                if grad is not None and len(grad.shape) == 2:  # Weight matrices
                    weight_grad_norms.append(tf.norm(grad).numpy())
                    layer_names.append(f'Layer {len(weight_grad_norms)}')

            gradient_analysis[name] = {
                'gradient_norms': weight_grad_norms,
                'layer_names': layer_names,
                'max_gradient': max(weight_grad_norms) if weight_grad_norms else 0,
                'min_gradient': min(weight_grad_norms) if weight_grad_norms else 0,
                'mean_gradient': np.mean(weight_grad_norms) if weight_grad_norms else 0,
                'vanished_layers': sum(1 for g in weight_grad_norms if g < 1e-6)
            }

            print(f"\n{name}:")
            print(f"  Max gradient: {gradient_analysis[name]['max_gradient']:.6f}")
            print(f"  Min gradient: {gradient_analysis[name]['min_gradient']:.6f}")
            print(f"  Mean gradient: {gradient_analysis[name]['mean_gradient']:.6f}")
            print(f"  Vanished layers: {gradient_analysis[name]['vanished_layers']}")

        except Exception as e:
            print(f"Error analyzing {name}: {e}")

    # Visualize gradient flow
    if len(gradient_analysis) >= 2:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        for name, data in gradient_analysis.items():
            layers = range(1, len(data['gradient_norms']) + 1)
            plt.semilogy(layers, data['gradient_norms'], 'o-', linewidth=2,
                        label=name, markersize=4)

        plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Vanishing threshold')
        plt.xlabel('Layer Depth')
        plt.ylabel('Gradient Magnitude (log scale)')
        plt.title('Gradient Flow: Plain vs ResNet')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        names = list(gradient_analysis.keys())
        max_grads = [gradient_analysis[name]['max_gradient'] for name in names]
        min_grads = [gradient_analysis[name]['min_gradient'] for name in names]

        x_pos = np.arange(len(names))
        width = 0.35

        plt.bar(x_pos - width/2, max_grads, width, label='Max Gradient', alpha=0.7)
        plt.bar(x_pos + width/2, min_grads, width, label='Min Gradient', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Network Type')
        plt.ylabel('Gradient Magnitude (log scale)')
        plt.title('Gradient Range Comparison')
        plt.xticks(x_pos, names)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        mean_grads = [gradient_analysis[name]['mean_gradient'] for name in names]
        vanished_counts = [gradient_analysis[name]['vanished_layers'] for name in names]

        plt.bar(names, mean_grads, alpha=0.7, color='green')
        plt.xlabel('Network Type')
        plt.ylabel('Mean Gradient Magnitude')
        plt.title('Average Gradient Strength')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.bar(names, vanished_counts, alpha=0.7, color='red')
        plt.xlabel('Network Type')
        plt.ylabel('Number of Vanished Layers')
        plt.title('Gradient Vanishing Problem')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gradient_flow_improvement.png', dpi=300, bbox_inches='tight')
        print("\n✅ Gradient flow analysis saved as 'gradient_flow_improvement.png'")
        plt.show()

def create_residual_connection_guide():
    """Create a comprehensive guide for implementing residual connections"""

    print("\n📋 RESIDUAL CONNECTIONS IMPLEMENTATION GUIDE")
    print("=" * 70)

    print("🎯 KEY PRINCIPLES:")
    print("1. ✅ IDENTITY MAPPING: f(x) = x + F(x), where F(x) is the residual function")
    print("2. ✅ DIMENSION MATCHING: Use 1x1 convolutions or dense layers to match dimensions")
    print("3. ✅ PROPER PLACEMENT: Add skip connections every 2-3 layers")
    print("4. ✅ BATCH NORMALIZATION: Use BN before or after residual blocks")

    print("\n💡 DESIGN PATTERNS:")

    patterns = {
        'Basic Block': {
            'Structure': 'Dense-BN-ReLU-Dense-BN-[Add]-ReLU',
            'Use Case': 'General purpose, medium depth',
            'Pros': 'Simple, effective',
            'Cons': 'Not parameter efficient'
        },
        'Pre-activation': {
            'Structure': 'BN-ReLU-Dense-BN-ReLU-Dense-[Add]',
            'Use Case': 'Very deep networks (>100 layers)',
            'Pros': 'Better gradient flow',
            'Cons': 'Slightly more complex'
        },
        'Bottleneck': {
            'Structure': '1x1-reduce → 3x3 → 1x1-expand',
            'Use Case': 'Parameter efficiency, very deep networks',
            'Pros': 'Fewer parameters',
            'Cons': 'More complex'
        }
    }

    print(f"{'Pattern':<15} {'Structure':<40} {'Best For':<25}")
    print("-" * 80)

    for pattern_name, info in patterns.items():
        print(f"{pattern_name:<15} {info['Structure']:<40} {info['Use Case']:<25}")

    print("\n🔧 IMPLEMENTATION CHECKLIST:")
    print("□ Ensure input and output dimensions match for skip connections")
    print("□ Use proper weight initialization (He initialization recommended)")
    print("□ Add batch normalization for training stability")
    print("□ Start with simple basic blocks, then experiment with variants")
    print("□ Monitor gradient flow in very deep networks")
    print("□ Consider pre-activation blocks for networks >100 layers")

    print("\n⚠️ COMMON PITFALLS:")
    print("❌ Dimension mismatch between skip connection and main path")
    print("❌ Adding too many skip connections (every layer)")
    print("❌ Forgetting batch normalization")
    print("❌ Using residual connections in shallow networks (unnecessary)")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🔗 RESIDUAL CONNECTIONS AND SKIP CONNECTIONS")
    print("=" * 70)
    print()

    try:
        # Demonstrate degradation problem
        degradation_results = demonstrate_degradation_problem()

        # Compare plain vs ResNet
        comparison_results = compare_plain_vs_resnet()

        # Analyze residual block variants
        variant_networks = analyze_residual_block_variants()

        # Demonstrate gradient flow improvement
        demonstrate_gradient_flow_improvement()

        # Create implementation guide
        create_residual_connection_guide()

        print("\n" + "=" * 70)
        print("✅ RESIDUAL CONNECTIONS ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - degradation_problem_demonstration.png")
        print("   - resnet_vs_plain_comparison.png")
        print("   - gradient_flow_improvement.png")
        print("🎓 Learning outcome: Understanding how residual connections enable very deep networks")
        print("💡 Next step: Learn about advanced optimization algorithms")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()