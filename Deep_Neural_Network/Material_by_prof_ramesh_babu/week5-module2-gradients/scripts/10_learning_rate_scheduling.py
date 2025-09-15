#!/usr/bin/env python3
"""
10_learning_rate_scheduling.py

Purpose: Comprehensive demonstration of learning rate scheduling techniques
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script demonstrates different learning rate scheduling strategies and their impact
on training convergence, final performance, and optimization dynamics.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_scheduling_strategies():
    """Create different learning rate scheduling strategies"""

    print("📈 CREATING LEARNING RATE SCHEDULING STRATEGIES")
    print("=" * 60)

    def constant_schedule(initial_lr=0.01):
        """Constant learning rate (baseline)"""
        return initial_lr

    def step_decay_schedule(initial_lr=0.01, decay_factor=0.5, decay_steps=1000):
        """Step decay schedule"""
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[decay_steps, decay_steps*2],
            values=[initial_lr, initial_lr*decay_factor, initial_lr*decay_factor*decay_factor]
        )

    def exponential_decay_schedule(initial_lr=0.01, decay_rate=0.96, decay_steps=1000):
        """Exponential decay schedule"""
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

    def polynomial_decay_schedule(initial_lr=0.01, final_lr=1e-5, decay_steps=5000, power=1.0):
        """Polynomial decay schedule"""
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=final_lr,
            power=power
        )

    def cosine_decay_schedule(initial_lr=0.01, decay_steps=5000, alpha=0.0):
        """Cosine decay schedule"""
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=alpha
        )

    def cosine_restart_schedule(initial_lr=0.01, first_decay_steps=1000):
        """Cosine decay with restarts"""
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=first_decay_steps,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.0
        )

    def piecewise_schedule(boundaries, values):
        """Piecewise constant schedule"""
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values
        )

    def warmup_cosine_schedule(initial_lr=0.01, warmup_steps=1000, total_steps=10000):
        """Custom warmup + cosine decay schedule"""
        class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, warmup_steps, total_steps):
                self.initial_lr = initial_lr
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps

            def __call__(self, step):
                # Warmup phase
                warmup_lr = (self.initial_lr * tf.cast(step, tf.float32) /
                           tf.cast(self.warmup_steps, tf.float32))

                # Cosine decay phase
                cosine_lr = 0.5 * self.initial_lr * (
                    1 + tf.cos(math.pi * tf.cast(step - self.warmup_steps, tf.float32) /
                              tf.cast(self.total_steps - self.warmup_steps, tf.float32))
                )

                return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

        return WarmupCosineSchedule(initial_lr, warmup_steps, total_steps)

    # Create all schedules
    schedules = {
        'Constant': constant_schedule(0.01),
        'Step Decay': step_decay_schedule(0.01, 0.5, 2000),
        'Exponential Decay': exponential_decay_schedule(0.01, 0.96, 1000),
        'Polynomial Decay': polynomial_decay_schedule(0.01, 1e-5, 8000, 1.0),
        'Cosine Decay': cosine_decay_schedule(0.01, 8000, 0.0),
        'Cosine Restarts': cosine_restart_schedule(0.01, 1500),
        'Piecewise': piecewise_schedule([2000, 5000, 7000], [0.01, 0.005, 0.001, 0.0005]),
        'Warmup + Cosine': warmup_cosine_schedule(0.01, 1000, 8000)
    }

    print(f"Created {len(schedules)} learning rate schedules:")
    for name in schedules.keys():
        print(f"  • {name}")

    return schedules

def visualize_learning_rate_schedules(schedules, total_steps=10000):
    """Visualize different learning rate schedules"""

    print("\n📊 Creating learning rate schedules visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    steps = np.arange(0, total_steps, 10)

    for i, (name, schedule) in enumerate(schedules.items()):
        ax = axes[i]

        # Calculate learning rates
        try:
            lr_values = [float(schedule(step)) for step in steps]
        except:
            # For some schedules that might have issues with numpy arrays
            lr_values = []
            for step in steps:
                try:
                    lr_val = float(schedule(tf.constant(step, dtype=tf.int64)))
                    lr_values.append(lr_val)
                except:
                    lr_values.append(0.0)

        ax.plot(steps, lr_values, linewidth=2, color='blue', alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{name} Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Add annotations for key characteristics
        if 'Restart' in name:
            ax.text(0.5, 0.8, 'Periodic Restarts', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        elif 'Warmup' in name:
            ax.text(0.02, 0.8, 'Warmup Phase', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        elif 'Constant' in name:
            ax.text(0.5, 0.5, 'No Decay', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.savefig('learning_rate_schedules.png', dpi=300, bbox_inches='tight')
    print("✅ Learning rate schedules visualization saved as 'learning_rate_schedules.png'")
    plt.show()

def compare_schedules_on_training():
    """Compare different learning rate schedules on actual training"""

    print("\n🏃‍♂️ COMPARING SCHEDULES ON NEURAL NETWORK TRAINING")
    print("=" * 60)

    # Create dataset
    X_train = tf.random.normal((3000, 32))
    # Complex non-linear classification task
    y_train = tf.cast(
        (tf.reduce_sum(tf.nn.tanh(X_train[:, :16]), axis=1) *
         tf.reduce_sum(tf.nn.sigmoid(X_train[:, 16:]), axis=1)) > 0,
        tf.float32
    )
    y_train = tf.reshape(y_train, (-1, 1))

    X_val = tf.random.normal((600, 32))
    y_val = tf.cast(
        (tf.reduce_sum(tf.nn.tanh(X_val[:, :16]), axis=1) *
         tf.reduce_sum(tf.nn.sigmoid(X_val[:, 16:]), axis=1)) > 0,
        tf.float32
    )
    y_val = tf.reshape(y_val, (-1, 1))

    print(f"Dataset: {X_train.shape[0]} training, {X_val.shape[0]} validation samples")

    # Create schedules for comparison
    schedules = create_scheduling_strategies()

    # Track training results
    training_results = {}

    print("\nTraining with different learning rate schedules...")
    print(f"{'Schedule':<20} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Convergence Epoch':<15}")
    print("-" * 70)

    for schedule_name, schedule in schedules.items():
        print(f"\n🔧 Training with {schedule_name} schedule...")

        try:
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(32,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Create optimizer with schedule
            optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

            # Compile model
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Custom callback to track learning rate
            class LearningRateTracker(tf.keras.callbacks.Callback):
                def __init__(self):
                    self.learning_rates = []

                def on_epoch_end(self, epoch, logs=None):
                    lr = float(self.model.optimizer.learning_rate.numpy())
                    self.learning_rates.append(lr)

            lr_tracker = LearningRateTracker()

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                verbose=0,
                callbacks=[lr_tracker]
            )

            # Store results
            training_results[schedule_name] = {
                'history': history.history,
                'learning_rates': lr_tracker.learning_rates,
                'best_val_acc': max(history.history['val_accuracy']),
                'final_val_acc': history.history['val_accuracy'][-1],
                'convergence_epoch': np.argmax(history.history['val_accuracy']) + 1,
                'final_loss': history.history['val_loss'][-1]
            }

            print(f"   Best validation accuracy: {training_results[schedule_name]['best_val_acc']:.4f}")
            print(f"   Final validation accuracy: {training_results[schedule_name]['final_val_acc']:.4f}")
            print(f"   Convergence epoch: {training_results[schedule_name]['convergence_epoch']}")

            # Print summary row
            print(f"{schedule_name:<20} {training_results[schedule_name]['best_val_acc']:<15.4f} "
                  f"{training_results[schedule_name]['final_val_acc']:<15.4f} "
                  f"{training_results[schedule_name]['convergence_epoch']:<15}")

        except Exception as e:
            print(f"   ❌ Training failed: {e}")

    print("-" * 70)

    # Create comprehensive comparison plots
    create_schedule_training_plots(training_results)

    return training_results

def create_schedule_training_plots(training_results):
    """Create comprehensive plots comparing schedule performance"""

    print("\n📊 Creating schedule training comparison visualization...")

    if not training_results:
        print("❌ No training results to visualize")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for name, data in training_results.items():
        ax1.plot(data['history']['loss'], linewidth=2, label=name, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss with Different Schedules')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]
    for name, data in training_results.items():
        ax2.plot(data['history']['val_accuracy'], linewidth=2, label=name, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy with Different Schedules')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Evolution
    ax3 = axes[0, 2]
    for name, data in training_results.items():
        if 'learning_rates' in data:
            epochs = range(1, len(data['learning_rates']) + 1)
            ax3.semilogy(epochs, data['learning_rates'], linewidth=2, label=name, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate (log scale)')
    ax3.set_title('Learning Rate Evolution During Training')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Best Performance Comparison
    ax4 = axes[1, 0]
    schedules = list(training_results.keys())
    best_accs = [training_results[s]['best_val_acc'] for s in schedules]

    bars = ax4.bar(range(len(schedules)), best_accs, alpha=0.7, color='skyblue')
    ax4.set_xlabel('Schedule')
    ax4.set_ylabel('Best Validation Accuracy')
    ax4.set_title('Best Performance Comparison')
    ax4.set_xticks(range(len(schedules)))
    ax4.set_xticklabels([s.replace(' ', '\n') for s in schedules], rotation=0, fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, best_accs):
        ax4.text(bar.get_x() + bar.get_width()/2, value + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # Plot 5: Convergence Speed
    ax5 = axes[1, 1]
    convergence_epochs = [training_results[s]['convergence_epoch'] for s in schedules]

    bars = ax5.bar(range(len(schedules)), convergence_epochs, alpha=0.7, color='lightcoral')
    ax5.set_xlabel('Schedule')
    ax5.set_ylabel('Convergence Epoch')
    ax5.set_title('Convergence Speed (Lower is Better)')
    ax5.set_xticks(range(len(schedules)))
    ax5.set_xticklabels([s.replace(' ', '\n') for s in schedules], rotation=0, fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, convergence_epochs):
        ax5.text(bar.get_x() + bar.get_width()/2, value + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # Plot 6: Final vs Best Performance
    ax6 = axes[1, 2]
    final_accs = [training_results[s]['final_val_acc'] for s in schedules]

    x_pos = np.arange(len(schedules))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, best_accs, width, label='Best Val Acc', alpha=0.7)
    bars2 = ax6.bar(x_pos + width/2, final_accs, width, label='Final Val Acc', alpha=0.7)

    ax6.set_xlabel('Schedule')
    ax6.set_ylabel('Accuracy')
    ax6.set_title('Best vs Final Performance')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([s.replace(' ', '\n') for s in schedules], rotation=0, fontsize=8)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('schedule_training_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Schedule training comparison saved as 'schedule_training_comparison.png'")
    plt.show()

def demonstrate_adaptive_schedules():
    """Demonstrate adaptive learning rate schedules"""

    print("\n🎯 DEMONSTRATING ADAPTIVE LEARNING RATE SCHEDULES")
    print("=" * 60)

    # Create adaptive schedules
    class LossBasedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        """Adaptive schedule based on validation loss"""

        def __init__(self, initial_lr=0.01, patience=5, factor=0.5, min_lr=1e-6):
            self.initial_lr = initial_lr
            self.patience = patience
            self.factor = factor
            self.min_lr = min_lr
            self.current_lr = tf.Variable(initial_lr, trainable=False)
            self.best_loss = tf.Variable(float('inf'), trainable=False)
            self.wait = tf.Variable(0, trainable=False)

        def __call__(self, step):
            return self.current_lr

        def update_lr(self, current_loss):
            """Update learning rate based on current loss"""
            if current_loss < self.best_loss:
                self.best_loss.assign(current_loss)
                self.wait.assign(0)
            else:
                self.wait.assign_add(1)
                if self.wait >= self.patience:
                    new_lr = tf.maximum(self.current_lr * self.factor, self.min_lr)
                    self.current_lr.assign(new_lr)
                    self.wait.assign(0)
                    return True
            return False

    class GradientBasedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        """Adaptive schedule based on gradient norms"""

        def __init__(self, initial_lr=0.01, target_grad_norm=1.0, adaptation_rate=0.1):
            self.initial_lr = initial_lr
            self.target_grad_norm = target_grad_norm
            self.adaptation_rate = adaptation_rate
            self.current_lr = tf.Variable(initial_lr, trainable=False)

        def __call__(self, step):
            return self.current_lr

        def update_lr(self, grad_norm):
            """Update learning rate based on gradient norm"""
            if grad_norm > self.target_grad_norm:
                # Gradients too large, reduce learning rate
                new_lr = self.current_lr * (1 - self.adaptation_rate)
            elif grad_norm < self.target_grad_norm * 0.1:
                # Gradients too small, increase learning rate
                new_lr = self.current_lr * (1 + self.adaptation_rate)
            else:
                new_lr = self.current_lr

            self.current_lr.assign(tf.maximum(new_lr, 1e-6))

    # Demonstrate ReduceLROnPlateau callback
    def train_with_reduce_lr_on_plateau():
        """Train with ReduceLROnPlateau callback"""

        # Create dataset
        X_train = tf.random.normal((2000, 20))
        y_train = tf.cast(tf.reduce_sum(X_train[:, :10], axis=1) > 0, tf.float32)
        y_train = tf.reshape(y_train, (-1, 1))

        X_val = tf.random.normal((400, 20))
        y_val = tf.cast(tf.reduce_sum(X_val[:, :10], axis=1) > 0, tf.float32)
        y_val = tf.reshape(y_val, (-1, 1))

        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Create callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        # Custom callback to track learning rate
        class LRTracker(tf.keras.callbacks.Callback):
            def __init__(self):
                self.learning_rates = []

            def on_epoch_end(self, epoch, logs=None):
                lr = float(self.model.optimizer.learning_rate.numpy())
                self.learning_rates.append(lr)

        lr_tracker = LRTracker()

        print("Training with ReduceLROnPlateau callback...")

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=32,
            verbose=0,
            callbacks=[reduce_lr, lr_tracker]
        )

        return history.history, lr_tracker.learning_rates

    # Run demonstration
    history, learning_rates = train_with_reduce_lr_on_plateau()

    # Visualize adaptive schedule
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training with ReduceLROnPlateau')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_accuracy'], linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.semilogy(learning_rates, linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('Adaptive Learning Rate')
    plt.grid(True, alpha=0.3)

    # Mark learning rate reductions
    for i in range(1, len(learning_rates)):
        if learning_rates[i] < learning_rates[i-1] * 0.9:  # Significant reduction
            plt.axvline(x=i, color='red', linestyle='--', alpha=0.7)
            plt.text(i, learning_rates[i], f'  Reduced at epoch {i}',
                    rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('adaptive_learning_rate_demo.png', dpi=300, bbox_inches='tight')
    print("\n✅ Adaptive learning rate demonstration saved as 'adaptive_learning_rate_demo.png'")
    plt.show()

def analyze_schedule_characteristics():
    """Analyze characteristics of different scheduling strategies"""

    print("\n🔍 ANALYZING SCHEDULE CHARACTERISTICS")
    print("=" * 60)

    characteristics = {
        'Constant': {
            'Convergence Speed': 'Slow',
            'Final Performance': 'Good',
            'Stability': 'High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'High',
            'Best For': 'Simple tasks, when LR is well-tuned'
        },
        'Step Decay': {
            'Convergence Speed': 'Medium',
            'Final Performance': 'Very Good',
            'Stability': 'High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Medium',
            'Best For': 'Classification tasks, when you know when to reduce'
        },
        'Exponential Decay': {
            'Convergence Speed': 'Fast initially',
            'Final Performance': 'Good',
            'Stability': 'Medium',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Medium',
            'Best For': 'Tasks requiring fast initial progress'
        },
        'Polynomial Decay': {
            'Convergence Speed': 'Medium',
            'Final Performance': 'Very Good',
            'Stability': 'High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Low',
            'Best For': 'Long training runs, smooth decay needed'
        },
        'Cosine Decay': {
            'Convergence Speed': 'Medium',
            'Final Performance': 'Excellent',
            'Stability': 'High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Low',
            'Best For': 'Modern deep learning, transformer training'
        },
        'Cosine Restarts': {
            'Convergence Speed': 'Fast',
            'Final Performance': 'Excellent',
            'Stability': 'Medium',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Medium',
            'Best For': 'Avoiding local minima, ensemble methods'
        },
        'Warmup + Cosine': {
            'Convergence Speed': 'Excellent',
            'Final Performance': 'Excellent',
            'Stability': 'Very High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Low',
            'Best For': 'Large models, transformer architectures'
        },
        'Adaptive (ReduceLR)': {
            'Convergence Speed': 'Medium',
            'Final Performance': 'Good',
            'Stability': 'Very High',
            'Computational Cost': 'Low',
            'Hyperparameter Sensitivity': 'Very Low',
            'Best For': 'Unknown tasks, automatic tuning needed'
        }
    }

    print("📊 SCHEDULE CHARACTERISTICS COMPARISON:")
    print(f"{'Schedule':<20} {'Convergence':<15} {'Final Perf':<15} {'Stability':<12} {'Best For':<25}")
    print("-" * 90)

    for schedule, chars in characteristics.items():
        print(f"{schedule:<20} {chars['Convergence Speed']:<15} {chars['Final Performance']:<15} "
              f"{chars['Stability']:<12} {chars['Best For']:<25}")

    print("\n🎯 SELECTION CRITERIA:")

    selection_guide = {
        'Fast Convergence Needed': 'Cosine Restarts, Warmup + Cosine',
        'Best Final Performance': 'Cosine Decay, Warmup + Cosine',
        'High Stability Required': 'Polynomial Decay, Adaptive',
        'Low Hyperparameter Tuning': 'Adaptive, Cosine Decay',
        'Simple Implementation': 'Step Decay, Exponential Decay',
        'Large Models/Transformers': 'Warmup + Cosine',
        'Unknown Task': 'Adaptive (ReduceLROnPlateau)',
        'Research/Experimentation': 'Cosine Restarts'
    }

    for criterion, recommendation in selection_guide.items():
        print(f"• {criterion:<25}: {recommendation}")

def create_scheduling_implementation_guide():
    """Create practical implementation guide for learning rate scheduling"""

    print("\n📋 LEARNING RATE SCHEDULING IMPLEMENTATION GUIDE")
    print("=" * 70)

    print("🚀 QUICK START RECOMMENDATIONS:")
    print("1. 🎯 DEFAULT CHOICE: Cosine decay with warmup")
    print("2. 🔧 FOR FINE-TUNING: ReduceLROnPlateau")
    print("3. 🏃 FOR RESEARCH: Cosine decay with restarts")
    print("4. 📚 FOR BEGINNERS: Step decay")

    print("\n💡 IMPLEMENTATION EXAMPLES:")

    examples = {
        'TensorFlow Cosine Decay': '''
# Cosine decay schedule
initial_lr = 0.001
decay_steps = 10000

schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    alpha=0.0
)

optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
''',
        'Warmup + Cosine Decay': '''
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        warmup_lr = (self.initial_lr * tf.cast(step, tf.float32) /
                    tf.cast(self.warmup_steps, tf.float32))

        cosine_lr = 0.5 * self.initial_lr * (
            1 + tf.cos(math.pi * tf.cast(step - self.warmup_steps, tf.float32) /
                      tf.cast(self.total_steps - self.warmup_steps, tf.float32))
        )

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

schedule = WarmupCosineSchedule(0.001, 1000, 10000)
''',
        'ReduceLROnPlateau Callback': '''
# Adaptive learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          callbacks=[reduce_lr])
'''
    }

    for name, code in examples.items():
        print(f"\n🔧 {name}:")
        print(code)

    print("\n⚙️ HYPERPARAMETER TUNING GUIDELINES:")
    print("📈 INITIAL LEARNING RATE:")
    print("  • Start with optimizer defaults (Adam: 0.001, SGD: 0.01)")
    print("  • Use learning rate finder for optimal initial value")
    print("  • Scale with batch size: larger batch → higher learning rate")

    print("\n⏱️ DECAY TIMING:")
    print("  • Total steps = epochs × steps_per_epoch")
    print("  • Warmup steps = 1000-5000 for large models")
    print("  • Decay should reach minimum before training ends")

    print("\n🎛️ SPECIFIC PARAMETERS:")
    print("  • Step decay: factor=0.1-0.5, step_size=30-100 epochs")
    print("  • Exponential decay: decay_rate=0.9-0.99")
    print("  • Cosine decay: alpha=0.0-0.1 (final LR ratio)")
    print("  • ReduceLROnPlateau: patience=5-15, factor=0.1-0.5")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("📈 LEARNING RATE SCHEDULING COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()

    try:
        # Create and visualize different schedules
        schedules = create_scheduling_strategies()
        visualize_learning_rate_schedules(schedules)

        # Compare schedules on actual training
        training_results = compare_schedules_on_training()

        # Demonstrate adaptive schedules
        demonstrate_adaptive_schedules()

        # Analyze schedule characteristics
        analyze_schedule_characteristics()

        # Create implementation guide
        create_scheduling_implementation_guide()

        print("\n" + "=" * 80)
        print("✅ LEARNING RATE SCHEDULING ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - learning_rate_schedules.png")
        print("   - schedule_training_comparison.png")
        print("   - adaptive_learning_rate_demo.png")
        print("🎓 Learning outcome: Understanding and implementing learning rate scheduling")
        print("💡 Next step: Learn about advanced regularization techniques")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()