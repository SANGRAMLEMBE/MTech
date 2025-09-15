#!/usr/bin/env python3
"""
09_optimization_algorithms.py

Purpose: Comprehensive comparison of optimization algorithms
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script demonstrates different optimization algorithms (SGD, Momentum, Adam, RMSprop, etc.)
and their performance characteristics on various optimization landscapes.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_optimization_landscapes():
    """Create different optimization landscapes for testing optimizers"""

    print("🗻 CREATING OPTIMIZATION LANDSCAPES")
    print("=" * 60)

    def quadratic_bowl(x, y):
        """Simple quadratic bowl - well-conditioned"""
        return x**2 + y**2

    def elongated_bowl(x, y):
        """Elongated bowl - ill-conditioned"""
        return 100 * x**2 + y**2

    def rosenbrock(x, y, a=1, b=100):
        """Rosenbrock function - curved valley"""
        return (a - x)**2 + b * (y - x**2)**2

    def beale(x, y):
        """Beale function - multiple local minima"""
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

    def saddle_point(x, y):
        """Saddle point function"""
        return x**2 - y**2

    landscapes = {
        'Quadratic Bowl': {
            'function': quadratic_bowl,
            'bounds': (-3, 3),
            'optimum': (0, 0),
            'description': 'Well-conditioned quadratic'
        },
        'Elongated Bowl': {
            'function': elongated_bowl,
            'bounds': (-1, 1),
            'optimum': (0, 0),
            'description': 'Ill-conditioned (high condition number)'
        },
        'Rosenbrock': {
            'function': rosenbrock,
            'bounds': (-2, 2),
            'optimum': (1, 1),
            'description': 'Curved valley (banana function)'
        },
        'Beale': {
            'function': beale,
            'bounds': (-2, 2),
            'optimum': (3, 0.5),
            'description': 'Multiple local minima'
        },
        'Saddle Point': {
            'function': saddle_point,
            'bounds': (-2, 2),
            'optimum': (0, 0),
            'description': 'Saddle point (indefinite Hessian)'
        }
    }

    print(f"Created {len(landscapes)} optimization landscapes:")
    for name, info in landscapes.items():
        print(f"  • {name}: {info['description']}")

    return landscapes

def visualize_optimization_landscapes(landscapes):
    """Visualize the optimization landscapes"""

    print("\n📊 Creating landscape visualization...")

    fig = plt.figure(figsize=(20, 12))

    for i, (name, landscape) in enumerate(landscapes.items()):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')

        # Create grid
        bounds = landscape['bounds']
        x = np.linspace(bounds[0], bounds[1], 50)
        y = np.linspace(bounds[0], bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = landscape['function'](X, Y)

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7,
                              linewidth=0, antialiased=False)

        # Mark optimum
        opt_x, opt_y = landscape['optimum']
        if bounds[0] <= opt_x <= bounds[1] and bounds[0] <= opt_y <= bounds[1]:
            opt_z = landscape['function'](opt_x, opt_y)
            ax.scatter([opt_x], [opt_y], [opt_z], color='red', s=100, label='Optimum')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        ax.set_title(f'{name}\n{landscape["description"]}')

        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig('optimization_landscapes.png', dpi=300, bbox_inches='tight')
    print("✅ Optimization landscapes saved as 'optimization_landscapes.png'")
    plt.show()

def implement_optimizers():
    """Implement various optimization algorithms"""

    print("\n🚀 IMPLEMENTING OPTIMIZATION ALGORITHMS")
    print("=" * 60)

    class CustomOptimizer:
        """Base class for custom optimizers"""

        def __init__(self, learning_rate=0.01):
            self.learning_rate = learning_rate

        def update(self, params, gradients):
            raise NotImplementedError

    class SGD(CustomOptimizer):
        """Stochastic Gradient Descent"""

        def __init__(self, learning_rate=0.01):
            super().__init__(learning_rate)

        def update(self, params, gradients):
            return [p - self.learning_rate * g for p, g in zip(params, gradients)]

    class MomentumSGD(CustomOptimizer):
        """SGD with Momentum"""

        def __init__(self, learning_rate=0.01, momentum=0.9):
            super().__init__(learning_rate)
            self.momentum = momentum
            self.velocities = None

        def update(self, params, gradients):
            if self.velocities is None:
                self.velocities = [np.zeros_like(p) for p in params]

            new_velocities = [self.momentum * v + self.learning_rate * g
                            for v, g in zip(self.velocities, gradients)]
            self.velocities = new_velocities

            return [p - v for p, v in zip(params, new_velocities)]

    class AdaGrad(CustomOptimizer):
        """Adaptive Gradient Algorithm"""

        def __init__(self, learning_rate=0.01, epsilon=1e-8):
            super().__init__(learning_rate)
            self.epsilon = epsilon
            self.sum_squared_gradients = None

        def update(self, params, gradients):
            if self.sum_squared_gradients is None:
                self.sum_squared_gradients = [np.zeros_like(p) for p in params]

            # Update sum of squared gradients
            self.sum_squared_gradients = [ssg + g**2
                                        for ssg, g in zip(self.sum_squared_gradients, gradients)]

            # Update parameters
            return [p - (self.learning_rate / (np.sqrt(ssg) + self.epsilon)) * g
                   for p, ssg, g in zip(params, self.sum_squared_gradients, gradients)]

    class RMSprop(CustomOptimizer):
        """Root Mean Square Propagation"""

        def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
            super().__init__(learning_rate)
            self.decay_rate = decay_rate
            self.epsilon = epsilon
            self.squared_gradients = None

        def update(self, params, gradients):
            if self.squared_gradients is None:
                self.squared_gradients = [np.zeros_like(p) for p in params]

            # Update squared gradients with exponential moving average
            self.squared_gradients = [self.decay_rate * sg + (1 - self.decay_rate) * g**2
                                    for sg, g in zip(self.squared_gradients, gradients)]

            # Update parameters
            return [p - (self.learning_rate / (np.sqrt(sg) + self.epsilon)) * g
                   for p, sg, g in zip(params, self.squared_gradients, gradients)]

    class Adam(CustomOptimizer):
        """Adaptive Moment Estimation"""

        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            super().__init__(learning_rate)
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.moment1 = None  # First moment (momentum)
            self.moment2 = None  # Second moment (RMSprop)
            self.t = 0           # Time step

        def update(self, params, gradients):
            self.t += 1

            if self.moment1 is None:
                self.moment1 = [np.zeros_like(p) for p in params]
                self.moment2 = [np.zeros_like(p) for p in params]

            # Update biased first moment estimate
            self.moment1 = [self.beta1 * m1 + (1 - self.beta1) * g
                           for m1, g in zip(self.moment1, gradients)]

            # Update biased second moment estimate
            self.moment2 = [self.beta2 * m2 + (1 - self.beta2) * g**2
                           for m2, g in zip(self.moment2, gradients)]

            # Compute bias-corrected moments
            m1_corrected = [m1 / (1 - self.beta1**self.t) for m1 in self.moment1]
            m2_corrected = [m2 / (1 - self.beta2**self.t) for m2 in self.moment2]

            # Update parameters
            return [p - (self.learning_rate * m1) / (np.sqrt(m2) + self.epsilon)
                   for p, m1, m2 in zip(params, m1_corrected, m2_corrected)]

    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'Momentum': MomentumSGD(learning_rate=0.01, momentum=0.9),
        'AdaGrad': AdaGrad(learning_rate=0.1),
        'RMSprop': RMSprop(learning_rate=0.01),
        'Adam': Adam(learning_rate=0.01)
    }

    print(f"Implemented {len(optimizers)} optimization algorithms:")
    for name in optimizers.keys():
        print(f"  • {name}")

    return optimizers

def optimize_on_landscape(optimizer, landscape_func, start_point, bounds, max_iterations=100):
    """Run optimization on a specific landscape"""

    def compute_gradient(func, x, y, h=1e-6):
        """Compute numerical gradient"""
        grad_x = (func(x + h, y) - func(x - h, y)) / (2 * h)
        grad_y = (func(x, y + h) - func(x, y - h)) / (2 * h)
        return [grad_x, grad_y]

    # Initialize
    params = list(start_point)
    trajectory = [params.copy()]
    losses = [landscape_func(*params)]

    for iteration in range(max_iterations):
        # Compute gradient
        gradients = compute_gradient(landscape_func, *params)

        # Update parameters
        new_params = optimizer.update(params, gradients)

        # Clip to bounds
        new_params = [np.clip(p, bounds[0], bounds[1]) for p in new_params]

        # Store trajectory
        params = new_params
        trajectory.append(params.copy())
        losses.append(landscape_func(*params))

        # Early stopping if converged
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-8:
            break

    return np.array(trajectory), np.array(losses)

def compare_optimizers_on_landscapes():
    """Compare all optimizers on all landscapes"""

    print("\n⚔️ COMPARING OPTIMIZERS ON DIFFERENT LANDSCAPES")
    print("=" * 60)

    landscapes = create_optimization_landscapes()
    optimizers = implement_optimizers()

    comparison_results = {}

    for landscape_name, landscape in landscapes.items():
        print(f"\n🗻 Testing on {landscape_name}...")

        landscape_results = {}
        start_point = (landscape['bounds'][0] * 0.8, landscape['bounds'][1] * 0.8)

        print(f"   Start point: {start_point}")
        print(f"   Target optimum: {landscape['optimum']}")

        for optimizer_name, optimizer in optimizers.items():
            try:
                # Create fresh optimizer instance to reset state
                if optimizer_name == 'SGD':
                    fresh_optimizer = SGD(learning_rate=0.01)
                elif optimizer_name == 'Momentum':
                    fresh_optimizer = MomentumSGD(learning_rate=0.01, momentum=0.9)
                elif optimizer_name == 'AdaGrad':
                    fresh_optimizer = AdaGrad(learning_rate=0.1)
                elif optimizer_name == 'RMSprop':
                    fresh_optimizer = RMSprop(learning_rate=0.01)
                elif optimizer_name == 'Adam':
                    fresh_optimizer = Adam(learning_rate=0.01)

                trajectory, losses = optimize_on_landscape(
                    fresh_optimizer,
                    landscape['function'],
                    start_point,
                    landscape['bounds']
                )

                # Calculate final distance to optimum
                final_point = trajectory[-1]
                optimum = landscape['optimum']
                distance_to_optimum = np.sqrt((final_point[0] - optimum[0])**2 +
                                            (final_point[1] - optimum[1])**2)

                landscape_results[optimizer_name] = {
                    'trajectory': trajectory,
                    'losses': losses,
                    'final_loss': losses[-1],
                    'iterations': len(losses) - 1,
                    'distance_to_optimum': distance_to_optimum,
                    'converged': len(losses) < 100
                }

                print(f"     {optimizer_name}: Final loss={losses[-1]:.6f}, "
                      f"Distance={distance_to_optimum:.4f}, Iterations={len(losses)-1}")

            except Exception as e:
                print(f"     {optimizer_name}: Failed - {e}")

        comparison_results[landscape_name] = landscape_results

    # Create comprehensive visualization
    create_optimizer_comparison_plots(comparison_results, landscapes)

    return comparison_results

def create_optimizer_comparison_plots(comparison_results, landscapes):
    """Create comprehensive plots comparing optimizers"""

    print("\n📊 Creating optimizer comparison visualization...")

    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))

    # Plot 1: Trajectories on each landscape
    for i, (landscape_name, landscape) in enumerate(landscapes.items()):
        ax = fig.add_subplot(5, 3, i*3 + 1)

        # Plot contours
        bounds = landscape['bounds']
        x = np.linspace(bounds[0], bounds[1], 50)
        y = np.linspace(bounds[0], bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = landscape['function'](X, Y)

        # Use log scale for better visualization
        contours = ax.contour(X, Y, np.log(Z + 1), levels=20, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=8)

        # Plot trajectories
        if landscape_name in comparison_results:
            colors = plt.cm.Set1(np.linspace(0, 1, len(comparison_results[landscape_name])))

            for j, (opt_name, result) in enumerate(comparison_results[landscape_name].items()):
                trajectory = result['trajectory']
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-',
                       color=colors[j], linewidth=2, markersize=3,
                       label=opt_name, alpha=0.8)

        # Mark optimum
        opt_x, opt_y = landscape['optimum']
        if bounds[0] <= opt_x <= bounds[1] and bounds[0] <= opt_y <= bounds[1]:
            ax.scatter([opt_x], [opt_y], color='red', s=100, marker='*',
                      label='Optimum', zorder=10)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{landscape_name}\nOptimization Trajectories')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss curves
        ax_loss = fig.add_subplot(5, 3, i*3 + 2)

        if landscape_name in comparison_results:
            for j, (opt_name, result) in enumerate(comparison_results[landscape_name].items()):
                losses = result['losses']
                ax_loss.semilogy(range(len(losses)), losses,
                               color=colors[j], linewidth=2, label=opt_name)

        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss (log scale)')
        ax_loss.set_title(f'{landscape_name}\nConvergence Curves')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        # Plot 3: Performance metrics
        ax_perf = fig.add_subplot(5, 3, i*3 + 3)

        if landscape_name in comparison_results:
            optimizers = list(comparison_results[landscape_name].keys())
            final_losses = [comparison_results[landscape_name][opt]['final_loss'] for opt in optimizers]
            distances = [comparison_results[landscape_name][opt]['distance_to_optimum'] for opt in optimizers]

            x_pos = np.arange(len(optimizers))
            width = 0.35

            # Normalize for comparison
            norm_losses = np.array(final_losses) / max(final_losses) if max(final_losses) > 0 else final_losses
            norm_distances = np.array(distances) / max(distances) if max(distances) > 0 else distances

            bars1 = ax_perf.bar(x_pos - width/2, norm_losses, width,
                              label='Normalized Final Loss', alpha=0.7)
            bars2 = ax_perf.bar(x_pos + width/2, norm_distances, width,
                              label='Normalized Distance', alpha=0.7)

            ax_perf.set_xlabel('Optimizer')
            ax_perf.set_ylabel('Normalized Performance (Lower is Better)')
            ax_perf.set_title(f'{landscape_name}\nPerformance Comparison')
            ax_perf.set_xticks(x_pos)
            ax_perf.set_xticklabels(optimizers, rotation=45)
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive optimizer comparison saved as 'optimizer_comprehensive_comparison.png'")
    plt.show()

def compare_tensorflow_optimizers():
    """Compare TensorFlow's built-in optimizers on a neural network task"""

    print("\n🧠 COMPARING TENSORFLOW OPTIMIZERS ON NEURAL NETWORK")
    print("=" * 60)

    # Create dataset
    X_train = tf.random.normal((2000, 20))
    # Complex non-linear target
    y_train = tf.cast(
        tf.reduce_sum(tf.nn.tanh(X_train[:, :10]) * tf.nn.relu(X_train[:, 10:]), axis=1) > 0,
        tf.float32
    )
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((400, 20))
    y_val = tf.cast(
        tf.reduce_sum(tf.nn.tanh(X_val[:, :10]) * tf.nn.relu(X_val[:, 10:]), axis=1) > 0,
        tf.float32
    )
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Dataset: {X_train.shape[0]} training, {X_val.shape[0]} validation samples")

    # Define optimizers
    tf_optimizers = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
        'SGD + Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'AdaGrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
        'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
        'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        'Nadam': tf.keras.optimizers.Nadam(learning_rate=0.001)
    }

    print(f"Testing {len(tf_optimizers)} TensorFlow optimizers...")

    tf_results = {}

    for opt_name, optimizer in tf_optimizers.items():
        print(f"\n🔧 Training with {opt_name}...")

        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=64,
                verbose=0
            )

            tf_results[opt_name] = {
                'history': history.history,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_train_acc': history.history['accuracy'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'best_val_acc': max(history.history['val_accuracy']),
                'convergence_epoch': np.argmin(history.history['val_loss']) + 1
            }

            print(f"   Final validation accuracy: {tf_results[opt_name]['final_val_acc']:.4f}")
            print(f"   Best validation accuracy: {tf_results[opt_name]['best_val_acc']:.4f}")
            print(f"   Convergence epoch: {tf_results[opt_name]['convergence_epoch']}")

        except Exception as e:
            print(f"   Failed: {e}")

    # Create TensorFlow optimizer comparison plots
    create_tensorflow_optimizer_plots(tf_results)

    return tf_results

def create_tensorflow_optimizer_plots(tf_results):
    """Create plots comparing TensorFlow optimizers"""

    print("\n📊 Creating TensorFlow optimizer comparison visualization...")

    if not tf_results:
        print("❌ No TensorFlow results to visualize")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for name, data in tf_results.items():
        ax1.plot(data['history']['loss'], linewidth=2, label=name, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for name, data in tf_results.items():
        ax2.plot(data['history']['val_loss'], linewidth=2, label=name, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax3 = axes[0, 2]
    for name, data in tf_results.items():
        ax3.plot(data['history']['val_accuracy'], linewidth=2, label=name, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Validation Accuracy Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Performance
    ax4 = axes[1, 0]
    optimizers = list(tf_results.keys())
    final_accs = [tf_results[opt]['final_val_acc'] for opt in optimizers]
    best_accs = [tf_results[opt]['best_val_acc'] for opt in optimizers]

    x_pos = np.arange(len(optimizers))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, final_accs, width, label='Final Val Acc', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, best_accs, width, label='Best Val Acc', alpha=0.7)

    ax4.set_xlabel('Optimizer')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([opt.split()[0] for opt in optimizers], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Convergence Speed
    ax5 = axes[1, 1]
    convergence_epochs = [tf_results[opt]['convergence_epoch'] for opt in optimizers]

    bars = ax5.bar(optimizers, convergence_epochs, alpha=0.7, color='skyblue')
    ax5.set_xlabel('Optimizer')
    ax5.set_ylabel('Convergence Epoch')
    ax5.set_title('Convergence Speed (Lower is Better)')
    ax5.set_xticklabels([opt.split()[0] for opt in optimizers], rotation=45)
    ax5.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, convergence_epochs):
        ax5.text(bar.get_x() + bar.get_width()/2, value + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    # Plot 6: Training Stability
    ax6 = axes[1, 2]
    val_loss_vars = [np.var(tf_results[opt]['history']['val_loss']) for opt in optimizers]

    bars = ax6.bar(optimizers, val_loss_vars, alpha=0.7, color='lightcoral')
    ax6.set_xlabel('Optimizer')
    ax6.set_ylabel('Validation Loss Variance')
    ax6.set_title('Training Stability (Lower is Better)')
    ax6.set_xticklabels([opt.split()[0] for opt in optimizers], rotation=45)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tensorflow_optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ TensorFlow optimizer comparison saved as 'tensorflow_optimizer_comparison.png'")
    plt.show()

def create_optimizer_selection_guide():
    """Create a comprehensive guide for selecting optimizers"""

    print("\n📋 OPTIMIZER SELECTION GUIDE")
    print("=" * 70)

    optimizer_guide = {
        'SGD': {
            'Best For': 'Simple problems, when you have time to tune',
            'Pros': 'Simple, interpretable, guaranteed convergence',
            'Cons': 'Slow convergence, requires learning rate tuning',
            'Hyperparameters': 'learning_rate (0.01-0.1)',
            'Use When': 'Small datasets, simple models'
        },
        'SGD + Momentum': {
            'Best For': 'Overcoming local minima, faster convergence',
            'Pros': 'Accelerated convergence, momentum helps escape local minima',
            'Cons': 'One more hyperparameter to tune',
            'Hyperparameters': 'learning_rate (0.01-0.1), momentum (0.9)',
            'Use When': 'Need faster convergence than plain SGD'
        },
        'AdaGrad': {
            'Best For': 'Sparse data, early in training',
            'Pros': 'Adaptive learning rates, good for sparse features',
            'Cons': 'Learning rate diminishes over time',
            'Hyperparameters': 'learning_rate (0.01-0.1)',
            'Use When': 'Text data, sparse features, short training'
        },
        'RMSprop': {
            'Best For': 'RNNs, non-convex optimization',
            'Pros': 'Solves AdaGrad\'s diminishing learning rate',
            'Cons': 'Can be unstable with large learning rates',
            'Hyperparameters': 'learning_rate (0.001), decay_rate (0.9)',
            'Use When': 'RNNs, when AdaGrad slows down too much'
        },
        'Adam': {
            'Best For': 'General purpose, default choice',
            'Pros': 'Combines momentum and RMSprop, adaptive, robust',
            'Cons': 'Can overshoot optimal solution, memory overhead',
            'Hyperparameters': 'learning_rate (0.001), beta1 (0.9), beta2 (0.999)',
            'Use When': 'Most deep learning tasks, starting point'
        },
        'AdamW': {
            'Best For': 'When you need weight decay regularization',
            'Pros': 'Better weight decay than Adam, improved generalization',
            'Cons': 'One more hyperparameter (weight decay)',
            'Hyperparameters': 'learning_rate (0.001), weight_decay (0.01)',
            'Use When': 'Large models, when regularization is important'
        },
        'Nadam': {
            'Best For': 'When you want Nesterov momentum with Adam',
            'Pros': 'Combines Adam with Nesterov momentum',
            'Cons': 'Slightly more complex, marginal improvements',
            'Hyperparameters': 'learning_rate (0.001)',
            'Use When': 'Fine-tuning Adam performance'
        }
    }

    print("🎯 OPTIMIZER COMPARISON TABLE:")
    print(f"{'Optimizer':<15} {'Best For':<25} {'Key Advantage':<30}")
    print("-" * 70)

    for opt_name, info in optimizer_guide.items():
        print(f"{opt_name:<15} {info['Best For']:<25} {info['Pros'].split(',')[0]:<30}")

    print("\n💡 SELECTION FLOWCHART:")
    print("1. 🚀 START HERE: Try Adam with default parameters (lr=0.001)")
    print("2. 📊 IF OVERFITTING: Switch to AdamW with weight_decay=0.01")
    print("3. 🐌 IF TRAINING TOO SLOW: Try higher learning rate or SGD with momentum")
    print("4. 📉 IF LOSS PLATEAUS: Reduce learning rate by 10x")
    print("5. 🎯 IF FINE-TUNING: Try Nadam or reduce learning rate for Adam")
    print("6. 💾 IF MEMORY CONSTRAINED: Use SGD or SGD+Momentum")
    print("7. 📊 IF SPARSE DATA: Try AdaGrad or RMSprop")

    print("\n⚙️ HYPERPARAMETER TUNING TIPS:")
    print("🔍 LEARNING RATE: Start with optimizer defaults, then search [1e-4, 1e-1]")
    print("📏 BATCH SIZE: Larger batches → higher learning rates often work")
    print("⏱️ SCHEDULING: Use learning rate decay for better final performance")
    print("🎛️ WARMUP: Consider learning rate warmup for large models")
    print("📊 MONITORING: Track both loss and gradient norms")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🚀 OPTIMIZATION ALGORITHMS COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()

    try:
        # Create and visualize optimization landscapes
        landscapes = create_optimization_landscapes()
        visualize_optimization_landscapes(landscapes)

        # Compare custom optimizers on mathematical functions
        comparison_results = compare_optimizers_on_landscapes()

        # Compare TensorFlow optimizers on neural network task
        tf_results = compare_tensorflow_optimizers()

        # Create selection guide
        create_optimizer_selection_guide()

        print("\n" + "=" * 80)
        print("✅ OPTIMIZATION ALGORITHMS ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - optimization_landscapes.png")
        print("   - optimizer_comprehensive_comparison.png")
        print("   - tensorflow_optimizer_comparison.png")
        print("🎓 Learning outcome: Understanding different optimization algorithms and their characteristics")
        print("💡 Next step: Learn about learning rate scheduling and advanced techniques")

        print("\n📚 BOOK REFERENCES:")
        print("=" * 80)
        print("1. 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, Aaron Courville")
        print("   - Chapter 8: Optimization for Training Deep Models")
        print("   - Chapter 4: Numerical Computation")
        print()
        print("2. 'Neural Networks and Deep Learning' by Charu C. Aggarwal")
        print("   - Chapter 4: Teaching Deep Networks to Learn")
        print("   - Chapter 1: An Introduction to Neural Networks")
        print()
        print("3. 'Deep Learning with Applications Using Python'")
        print("   - Chapter 4: Improving Deep Networks")
        print("   - Chapter 5: Optimization and Hyperparameter Tuning")
        print()
        print("4. 'Deep Neural Network Architectures'")
        print("   - Chapter 5: Optimization Techniques")
        print("   - Chapter 6: Advanced Training Methods")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()