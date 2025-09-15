#!/usr/bin/env python3
"""
12_meta_learning_maml.py

Purpose: Demonstrate Meta-Learning concepts with MAML (Model-Agnostic Meta-Learning)
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements simplified MAML to demonstrate the concept of learning to learn
and fast adaptation to new tasks with few examples.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_meta_learning_tasks(num_tasks=100, num_classes=2, num_samples_per_class=10):
    """Generate synthetic tasks for meta-learning"""

    print(f"🎯 GENERATING META-LEARNING TASKS")
    print("=" * 60)

    tasks = []

    for task_id in range(num_tasks):
        # Generate task-specific parameters
        task_center = np.random.uniform(-2, 2, 2)
        task_rotation = np.random.uniform(0, 2*np.pi)
        task_scale = np.random.uniform(0.5, 2.0)

        # Create rotation matrix
        cos_angle = np.cos(task_rotation)
        sin_angle = np.sin(task_rotation)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])

        X_task = []
        y_task = []

        for class_id in range(num_classes):
            # Generate samples for this class
            if class_id == 0:
                # Class 0: samples around one side
                base_samples = np.random.normal([-1, 0], 0.3, (num_samples_per_class, 2))
            else:
                # Class 1: samples around other side
                base_samples = np.random.normal([1, 0], 0.3, (num_samples_per_class, 2))

            # Apply task-specific transformations
            transformed_samples = []
            for sample in base_samples:
                # Scale, rotate, and translate
                scaled_sample = sample * task_scale
                rotated_sample = rotation_matrix @ scaled_sample
                final_sample = rotated_sample + task_center
                transformed_samples.append(final_sample)

            X_task.extend(transformed_samples)
            y_task.extend([class_id] * num_samples_per_class)

        # Convert to arrays and shuffle
        X_task = np.array(X_task)
        y_task = np.array(y_task)

        # Shuffle samples
        indices = np.random.permutation(len(X_task))
        X_task = X_task[indices]
        y_task = y_task[indices]

        tasks.append({
            'X': X_task,
            'y': y_task,
            'task_id': task_id,
            'center': task_center,
            'rotation': task_rotation,
            'scale': task_scale
        })

    print(f"Generated {len(tasks)} tasks")
    print(f"Each task has {len(tasks[0]['X'])} samples ({num_classes} classes)")

    return tasks

def visualize_meta_learning_tasks(tasks, num_tasks_to_show=6):
    """Visualize sample meta-learning tasks"""

    print(f"\n📊 Visualizing {num_tasks_to_show} sample tasks...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['red', 'blue', 'green', 'orange']

    for i in range(min(num_tasks_to_show, len(tasks))):
        ax = axes[i]
        task = tasks[i]

        X = task['X']
        y = task['y']

        # Plot samples by class
        for class_id in np.unique(y):
            class_mask = y == class_id
            ax.scatter(X[class_mask, 0], X[class_mask, 1],
                      c=colors[class_id], alpha=0.7, s=30,
                      label=f'Class {class_id}')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Task {task["task_id"]} (Scale: {task["scale"]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig('meta_learning_tasks.png', dpi=300, bbox_inches='tight')
    print("✅ Meta-learning tasks visualization saved as 'meta_learning_tasks.png'")
    plt.show()

def create_base_model():
    """Create a simple neural network for meta-learning"""

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def compute_loss(model, X, y):
    """Compute loss for a batch"""
    predictions = model(X, training=True)
    y_reshaped = tf.reshape(y, (-1, 1))
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_reshaped, predictions))
    return loss

def compute_gradients(model, X, y):
    """Compute gradients for a batch"""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, X, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients

def apply_gradients(model, gradients, learning_rate):
    """Apply gradients to model parameters"""
    for i, (param, grad) in enumerate(zip(model.trainable_variables, gradients)):
        if grad is not None:
            param.assign_sub(learning_rate * grad)

def copy_model(source_model):
    """Create a copy of a model with the same weights"""
    target_model = create_base_model()
    target_model.build(input_shape=(None, 2))

    # Copy weights
    for target_param, source_param in zip(target_model.trainable_variables,
                                        source_model.trainable_variables):
        target_param.assign(source_param)

    return target_model

def maml_inner_loop(model, support_X, support_y, inner_lr=0.01, inner_steps=1):
    """Perform inner loop adaptation for MAML"""

    # Create a copy of the model for adaptation
    adapted_model = copy_model(model)

    # Perform gradient descent steps
    for step in range(inner_steps):
        loss, gradients = compute_gradients(adapted_model, support_X, support_y)
        apply_gradients(adapted_model, gradients, inner_lr)

    return adapted_model

def maml_outer_loop(model, tasks_batch, inner_lr=0.01, outer_lr=0.001,
                   inner_steps=1, support_size=5):
    """Perform outer loop update for MAML"""

    # Accumulate meta-gradients
    meta_gradients = [tf.zeros_like(param) for param in model.trainable_variables]
    total_loss = 0.0

    for task in tasks_batch:
        # Split task into support and query sets
        X, y = task['X'], task['y']

        # Random split
        indices = np.random.permutation(len(X))
        support_indices = indices[:support_size]
        query_indices = indices[support_size:support_size*2]  # Same size as support

        support_X = tf.constant(X[support_indices], dtype=tf.float32)
        support_y = tf.constant(y[support_indices], dtype=tf.float32)
        query_X = tf.constant(X[query_indices], dtype=tf.float32)
        query_y = tf.constant(y[query_indices], dtype=tf.float32)

        # Inner loop: adapt to support set
        adapted_model = maml_inner_loop(model, support_X, support_y, inner_lr, inner_steps)

        # Compute loss on query set
        with tf.GradientTape() as tape:
            # Watch the original model parameters
            for param in model.trainable_variables:
                tape.watch(param)

            query_loss = compute_loss(adapted_model, query_X, query_y)
            total_loss += query_loss

        # Compute gradients w.r.t. original model parameters
        task_gradients = tape.gradient(query_loss, model.trainable_variables)

        # Accumulate meta-gradients
        for i, grad in enumerate(task_gradients):
            if grad is not None:
                meta_gradients[i] += grad

    # Average meta-gradients
    batch_size = len(tasks_batch)
    meta_gradients = [grad / batch_size for grad in meta_gradients]

    # Apply meta-gradients
    apply_gradients(model, meta_gradients, outer_lr)

    return total_loss / batch_size

def train_maml(model, train_tasks, num_epochs=100, batch_size=4,
               inner_lr=0.01, outer_lr=0.001, inner_steps=1, support_size=5):
    """Train model using MAML algorithm"""

    print(f"\n🧠 TRAINING WITH MAML")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  • Inner learning rate: {inner_lr}")
    print(f"  • Outer learning rate: {outer_lr}")
    print(f"  • Inner steps: {inner_steps}")
    print(f"  • Support set size: {support_size}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Epochs: {num_epochs}")

    training_history = {'epoch': [], 'meta_loss': []}

    for epoch in range(num_epochs):
        # Sample batch of tasks
        batch_tasks = np.random.choice(train_tasks, size=batch_size, replace=False)

        # Perform MAML update
        meta_loss = maml_outer_loop(
            model, batch_tasks, inner_lr, outer_lr, inner_steps, support_size
        )

        # Log progress
        training_history['epoch'].append(epoch)
        training_history['meta_loss'].append(meta_loss.numpy())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Meta-loss = {meta_loss:.4f}")

    return training_history

def evaluate_few_shot_performance(model, test_tasks, support_sizes=[1, 3, 5, 10],
                                inner_lr=0.01, inner_steps=5):
    """Evaluate few-shot learning performance"""

    print(f"\n📊 EVALUATING FEW-SHOT PERFORMANCE")
    print("=" * 60)

    results = {}

    for support_size in support_sizes:
        print(f"\nTesting with {support_size}-shot learning...")

        accuracies = []

        for task in test_tasks[:20]:  # Test on subset for speed
            X, y = task['X'], task['y']

            # Split into support and query
            indices = np.random.permutation(len(X))
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]

            support_X = tf.constant(X[support_indices], dtype=tf.float32)
            support_y = tf.constant(y[support_indices], dtype=tf.float32)
            query_X = tf.constant(X[query_indices], dtype=tf.float32)
            query_y = y[query_indices]

            # Fast adaptation
            adapted_model = maml_inner_loop(model, support_X, support_y,
                                          inner_lr, inner_steps)

            # Evaluate on query set
            query_predictions = adapted_model(query_X, training=False)
            query_pred_binary = (query_predictions.numpy() > 0.5).astype(int).flatten()

            accuracy = accuracy_score(query_y, query_pred_binary)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        results[support_size] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'accuracies': accuracies
        }

        print(f"  {support_size}-shot accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")

    return results

def compare_with_baseline(train_tasks, test_tasks):
    """Compare MAML with baseline approach (training from scratch)"""

    print(f"\n⚔️ COMPARING MAML WITH BASELINE")
    print("=" * 60)

    # Train MAML model
    print("Training MAML model...")
    maml_model = create_base_model()
    maml_model.build(input_shape=(None, 2))

    maml_history = train_maml(
        maml_model, train_tasks, num_epochs=50, batch_size=4,
        inner_lr=0.01, outer_lr=0.001, inner_steps=1, support_size=5
    )

    # Evaluate MAML
    print("\nEvaluating MAML performance...")
    maml_results = evaluate_few_shot_performance(
        maml_model, test_tasks, support_sizes=[1, 3, 5, 10],
        inner_lr=0.01, inner_steps=5
    )

    # Baseline: Train from scratch for each task
    print("\nEvaluating baseline (training from scratch)...")
    baseline_results = {}

    support_sizes = [1, 3, 5, 10]

    for support_size in support_sizes:
        print(f"\nBaseline with {support_size} samples...")

        accuracies = []

        for task in test_tasks[:10]:  # Fewer tasks for baseline (slower)
            X, y = task['X'], task['y']

            # Split into support and query
            indices = np.random.permutation(len(X))
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]

            support_X = X[support_indices]
            support_y = y[support_indices]
            query_X = X[query_indices]
            query_y = y[query_indices]

            # Train new model from scratch
            baseline_model = create_base_model()
            baseline_model.compile(optimizer='adam', loss='binary_crossentropy')

            # Train on support set
            try:
                baseline_model.fit(
                    support_X, support_y,
                    epochs=100,
                    batch_size=min(8, len(support_X)),
                    verbose=0
                )

                # Evaluate on query set
                query_predictions = baseline_model(query_X, training=False)
                query_pred_binary = (query_predictions.numpy() > 0.5).astype(int).flatten()

                accuracy = accuracy_score(query_y, query_pred_binary)
                accuracies.append(accuracy)

            except:
                # If training fails, use random performance
                accuracies.append(0.5)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        baseline_results[support_size] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }

        print(f"  Baseline {support_size}-shot accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")

    # Create comparison visualization
    create_comparison_plots(maml_history, maml_results, baseline_results)

    return maml_results, baseline_results

def create_comparison_plots(maml_history, maml_results, baseline_results):
    """Create comprehensive comparison plots"""

    print(f"\n📊 Creating comparison visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: MAML Training History
    ax1 = axes[0, 0]
    ax1.plot(maml_history['epoch'], maml_history['meta_loss'],
             'b-', linewidth=2, label='Meta-loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Meta-loss')
    ax1.set_title('MAML Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Few-shot Performance Comparison
    ax2 = axes[0, 1]
    support_sizes = list(maml_results.keys())
    maml_means = [maml_results[k]['mean_accuracy'] for k in support_sizes]
    maml_stds = [maml_results[k]['std_accuracy'] for k in support_sizes]

    baseline_means = [baseline_results[k]['mean_accuracy'] for k in support_sizes]
    baseline_stds = [baseline_results.get(k, {'std_accuracy': 0})['std_accuracy'] for k in support_sizes]

    x_pos = np.arange(len(support_sizes))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, maml_means, width, yerr=maml_stds,
                    label='MAML', alpha=0.7, capsize=5)
    bars2 = ax2.bar(x_pos + width/2, baseline_means, width, yerr=baseline_stds,
                    label='Baseline', alpha=0.7, capsize=5)

    ax2.set_xlabel('Number of Support Samples')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Few-shot Learning Performance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{k}-shot' for k in support_sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bars, means in [(bars1, maml_means), (bars2, baseline_means)]:
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Accuracy Distribution
    ax3 = axes[1, 0]

    # Show distribution for 5-shot learning
    shot_size = 5
    if shot_size in maml_results:
        maml_accs = maml_results[shot_size]['accuracies']
        ax3.hist(maml_accs, bins=15, alpha=0.7, label='MAML', density=True)
        ax3.axvline(np.mean(maml_accs), color='blue', linestyle='--',
                   label=f'MAML Mean: {np.mean(maml_accs):.3f}')

    ax3.axvline(0.5, color='red', linestyle=':', label='Random Baseline')
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{shot_size}-shot Learning Accuracy Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning Curves
    ax4 = axes[1, 1]

    # Simulate learning curves for different shot sizes
    shot_sizes_subset = [1, 5, 10]
    for shot_size in shot_sizes_subset:
        if shot_size in maml_results:
            mean_acc = maml_results[shot_size]['mean_accuracy']
            # Simulate improvement with more adaptation steps
            steps = np.arange(1, 11)
            # Exponential approach to final accuracy
            learning_curve = mean_acc * (1 - np.exp(-steps/3))
            ax4.plot(steps, learning_curve, 'o-', linewidth=2,
                    label=f'{shot_size}-shot MAML', markersize=4)

    ax4.set_xlabel('Adaptation Steps')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Simulated Adaptation Learning Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('maml_comparison_results.png', dpi=300, bbox_inches='tight')
    print("✅ MAML comparison results saved as 'maml_comparison_results.png'")
    plt.show()

def demonstrate_fast_adaptation(model, test_task):
    """Demonstrate fast adaptation on a single task"""

    print(f"\n⚡ DEMONSTRATING FAST ADAPTATION")
    print("=" * 60)

    X, y = test_task['X'], test_task['y']

    # Use only 3 samples for adaptation
    support_size = 3
    indices = np.random.permutation(len(X))
    support_indices = indices[:support_size]
    query_indices = indices[support_size:]

    support_X = tf.constant(X[support_indices], dtype=tf.float32)
    support_y = tf.constant(y[support_indices], dtype=tf.float32)
    query_X = tf.constant(X[query_indices], dtype=tf.float32)
    query_y = y[query_indices]

    print(f"Task ID: {test_task['task_id']}")
    print(f"Support set: {support_size} samples")
    print(f"Query set: {len(query_y)} samples")

    # Track adaptation progress
    adaptation_steps = 10
    adaptation_history = {'step': [], 'support_loss': [], 'query_accuracy': []}

    # Start with original model
    adapted_model = copy_model(model)

    print(f"\n{'Step':<6} {'Support Loss':<15} {'Query Accuracy':<15}")
    print("-" * 40)

    for step in range(adaptation_steps):
        # Evaluate current performance
        support_loss = compute_loss(adapted_model, support_X, support_y)
        query_predictions = adapted_model(query_X, training=False)
        query_pred_binary = (query_predictions.numpy() > 0.5).astype(int).flatten()
        query_accuracy = accuracy_score(query_y, query_pred_binary)

        # Log
        adaptation_history['step'].append(step)
        adaptation_history['support_loss'].append(support_loss.numpy())
        adaptation_history['query_accuracy'].append(query_accuracy)

        print(f"{step:<6} {support_loss.numpy():<15.4f} {query_accuracy:<15.3f}")

        # Adapt model
        if step < adaptation_steps - 1:  # Don't update after last evaluation
            _, gradients = compute_gradients(adapted_model, support_X, support_y)
            apply_gradients(adapted_model, gradients, 0.01)

    # Visualize adaptation
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(adaptation_history['step'], adaptation_history['support_loss'],
             'bo-', linewidth=2, markersize=4)
    plt.xlabel('Adaptation Step')
    plt.ylabel('Support Set Loss')
    plt.title('Support Loss During Adaptation')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(adaptation_history['step'], adaptation_history['query_accuracy'],
             'go-', linewidth=2, markersize=4)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    plt.xlabel('Adaptation Step')
    plt.ylabel('Query Set Accuracy')
    plt.title('Query Accuracy During Adaptation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    # Visualize the task and predictions
    colors = ['red', 'blue']
    for class_id in [0, 1]:
        class_mask = query_y == class_id
        pred_mask = query_pred_binary == class_id

        # Correct predictions
        correct_mask = class_mask & pred_mask
        plt.scatter(query_X[correct_mask, 0], query_X[correct_mask, 1],
                   c=colors[class_id], marker='o', s=30, alpha=0.7,
                   label=f'Class {class_id} (Correct)')

        # Incorrect predictions
        incorrect_mask = class_mask & ~pred_mask
        if np.any(incorrect_mask):
            plt.scatter(query_X[incorrect_mask, 0], query_X[incorrect_mask, 1],
                       c=colors[class_id], marker='x', s=30, alpha=0.7)

    # Show support samples
    for class_id in [0, 1]:
        support_class_mask = support_y.numpy() == class_id
        plt.scatter(support_X.numpy()[support_class_mask, 0],
                   support_X.numpy()[support_class_mask, 1],
                   c=colors[class_id], marker='s', s=100, edgecolors='black',
                   label=f'Support {class_id}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Final Predictions (Accuracy: {query_accuracy:.3f})')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fast_adaptation_demo.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Fast adaptation demonstration saved as 'fast_adaptation_demo.png'")
    plt.show()

def create_maml_implementation_guide():
    """Create practical guide for implementing MAML"""

    print(f"\n📋 MAML IMPLEMENTATION GUIDE")
    print("=" * 70)

    print("🎯 WHEN TO USE MAML:")
    print("✅ Few-shot learning scenarios")
    print("✅ Fast adaptation to new tasks")
    print("✅ Limited data for new domains")
    print("✅ Personalization applications")
    print("✅ Robotics and control tasks")

    print("\n🔧 KEY COMPONENTS:")

    components = {
        'Inner Loop': {
            'Purpose': 'Task-specific adaptation',
            'Parameters': 'Inner learning rate, adaptation steps',
            'Typical Values': 'α = 0.01, K = 1-5 steps'
        },
        'Outer Loop': {
            'Purpose': 'Meta-parameter optimization',
            'Parameters': 'Outer learning rate, batch size',
            'Typical Values': 'β = 0.001, batch = 4-32 tasks'
        },
        'Task Distribution': {
            'Purpose': 'Training task variety',
            'Parameters': 'Support/query split, task complexity',
            'Typical Values': 'Support = 1-10, Query = 5-15'
        }
    }

    print(f"{'Component':<15} {'Purpose':<25} {'Typical Values':<25}")
    print("-" * 70)

    for component, info in components.items():
        print(f"{component:<15} {info['Purpose']:<25} {info['Typical Values']:<25}")

    print("\n💡 IMPLEMENTATION TIPS:")
    print("1. 🎯 START SIMPLE: Begin with simple task distributions")
    print("2. 📏 BALANCE SETS: Keep support/query sets balanced")
    print("3. ⚖️ TUNE CAREFULLY: Inner/outer learning rates are critical")
    print("4. 🔄 TASK VARIETY: Ensure diverse training tasks")
    print("5. 📊 VALIDATE: Test on completely unseen task types")

    print("\n⚠️ COMMON CHALLENGES:")
    print("❌ Overfitting to training task distribution")
    print("❌ Instability with large inner learning rates")
    print("❌ Expensive computation (second-order gradients)")
    print("❌ Memory requirements for large models")
    print("❌ Difficulty with very different task distributions")

    print("\n🚀 EXTENSIONS AND VARIANTS:")
    print("• First-Order MAML (FOMAML): Cheaper approximation")
    print("• Reptile: Simpler algorithm, similar performance")
    print("• Meta-SGD: Learn inner learning rates")
    print("• ANIL: Almost No Inner Loop, for representation learning")
    print("• MAML++: Various improvements and tricks")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🤖 META-LEARNING WITH MAML")
    print("=" * 80)
    print()

    try:
        # Generate meta-learning tasks
        print("Generating meta-learning task distribution...")
        all_tasks = generate_meta_learning_tasks(num_tasks=80, num_classes=2, num_samples_per_class=15)

        # Split into train/test
        train_tasks = all_tasks[:60]
        test_tasks = all_tasks[60:]

        print(f"Split: {len(train_tasks)} training tasks, {len(test_tasks)} test tasks")

        # Visualize sample tasks
        visualize_meta_learning_tasks(all_tasks[:6])

        # Compare MAML with baseline
        maml_results, baseline_results = compare_with_baseline(train_tasks, test_tasks)

        # Demonstrate fast adaptation on a single task
        demonstrate_fast_adaptation(create_base_model(), test_tasks[0])

        # Create implementation guide
        create_maml_implementation_guide()

        print("\n" + "=" * 80)
        print("✅ META-LEARNING MAML ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - meta_learning_tasks.png")
        print("   - maml_comparison_results.png")
        print("   - fast_adaptation_demo.png")
        print("🎓 Learning outcome: Understanding meta-learning and few-shot adaptation")
        print("💡 Next step: Learn about attention mechanisms")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()