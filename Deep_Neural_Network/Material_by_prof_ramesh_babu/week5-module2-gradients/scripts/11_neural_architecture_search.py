#!/usr/bin/env python3
"""
11_neural_architecture_search.py

Purpose: Demonstrate Neural Architecture Search (NAS) concepts and techniques
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements simplified NAS approaches to automatically discover
optimal neural network architectures for given tasks.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def define_search_space():
    """Define the neural architecture search space"""

    print("🔍 DEFINING NEURAL ARCHITECTURE SEARCH SPACE")
    print("=" * 60)

    search_space = {
        'layers': {
            'min_layers': 2,
            'max_layers': 6,
            'layer_types': ['dense', 'dense_bn', 'dense_dropout', 'dense_bn_dropout']
        },
        'units': {
            'choices': [16, 32, 64, 128, 256]
        },
        'activations': {
            'choices': ['relu', 'tanh', 'elu', 'swish']
        },
        'dropout_rates': {
            'choices': [0.0, 0.1, 0.2, 0.3, 0.5]
        },
        'optimizers': {
            'choices': ['adam', 'sgd', 'rmsprop']
        },
        'learning_rates': {
            'choices': [0.001, 0.01, 0.1]
        }
    }

    print("Search space components:")
    for component, config in search_space.items():
        print(f"  • {component}: {config}")

    # Calculate total search space size
    total_architectures = (
        (search_space['layers']['max_layers'] - search_space['layers']['min_layers'] + 1) *
        len(search_space['units']['choices']) ** search_space['layers']['max_layers'] *
        len(search_space['activations']['choices']) ** search_space['layers']['max_layers'] *
        len(search_space['dropout_rates']['choices']) ** search_space['layers']['max_layers'] *
        len(search_space['optimizers']['choices']) *
        len(search_space['learning_rates']['choices'])
    )

    print(f"\nTotal possible architectures: {total_architectures:,}")
    print("(This demonstrates the need for efficient search strategies!)")

    return search_space

def generate_random_architecture(search_space):
    """Generate a random architecture from the search space"""

    num_layers = random.randint(search_space['layers']['min_layers'],
                               search_space['layers']['max_layers'])

    architecture = {
        'num_layers': num_layers,
        'layer_configs': [],
        'optimizer': random.choice(search_space['optimizers']['choices']),
        'learning_rate': random.choice(search_space['learning_rates']['choices'])
    }

    for i in range(num_layers):
        layer_config = {
            'type': random.choice(search_space['layers']['layer_types']),
            'units': random.choice(search_space['units']['choices']),
            'activation': random.choice(search_space['activations']['choices']),
            'dropout_rate': random.choice(search_space['dropout_rates']['choices'])
        }
        architecture['layer_configs'].append(layer_config)

    return architecture

def build_model_from_architecture(architecture, input_shape=(20,)):
    """Build a TensorFlow model from architecture specification"""

    try:
        model = tf.keras.Sequential()

        # Add input layer
        first_layer_config = architecture['layer_configs'][0]
        model.add(tf.keras.layers.Dense(
            first_layer_config['units'],
            activation=first_layer_config['activation'],
            input_shape=input_shape
        ))

        # Add batch normalization if specified
        if 'bn' in first_layer_config['type']:
            model.add(tf.keras.layers.BatchNormalization())

        # Add dropout if specified
        if 'dropout' in first_layer_config['type'] and first_layer_config['dropout_rate'] > 0:
            model.add(tf.keras.layers.Dropout(first_layer_config['dropout_rate']))

        # Add remaining layers
        for layer_config in architecture['layer_configs'][1:]:
            model.add(tf.keras.layers.Dense(
                layer_config['units'],
                activation=layer_config['activation']
            ))

            # Add batch normalization if specified
            if 'bn' in layer_config['type']:
                model.add(tf.keras.layers.BatchNormalization())

            # Add dropout if specified
            if 'dropout' in layer_config['type'] and layer_config['dropout_rate'] > 0:
                model.add(tf.keras.layers.Dropout(layer_config['dropout_rate']))

        # Add output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        return model

    except Exception as e:
        print(f"Error building model: {e}")
        return None

def evaluate_architecture(architecture, X_train, y_train, X_val, y_val, epochs=20):
    """Evaluate a single architecture's performance"""

    try:
        # Build model
        model = build_model_from_architecture(architecture)
        if model is None:
            return {'validation_accuracy': 0.0, 'error': 'Model build failed'}

        # Compile model
        optimizer_name = architecture['optimizer']
        learning_rate = architecture['learning_rate']

        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = 'adam'

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # Return performance metrics
        best_val_acc = max(history.history['val_accuracy'])
        final_val_acc = history.history['val_accuracy'][-1]
        convergence_epoch = np.argmax(history.history['val_accuracy']) + 1

        return {
            'validation_accuracy': best_val_acc,
            'final_accuracy': final_val_acc,
            'convergence_epoch': convergence_epoch,
            'training_history': history.history,
            'model_params': model.count_params()
        }

    except Exception as e:
        return {'validation_accuracy': 0.0, 'error': str(e)}

def random_search_nas(search_space, X_train, y_train, X_val, y_val, num_trials=20):
    """Implement random search NAS"""

    print(f"\n🎲 RANDOM SEARCH NAS ({num_trials} trials)")
    print("=" * 60)

    results = []
    best_architecture = None
    best_performance = 0.0

    print(f"{'Trial':<8} {'Layers':<8} {'Params':<10} {'Val Acc':<10} {'Status':<15}")
    print("-" * 60)

    for trial in range(num_trials):
        # Generate random architecture
        architecture = generate_random_architecture(search_space)

        # Evaluate architecture
        performance = evaluate_architecture(architecture, X_train, y_train, X_val, y_val, epochs=15)

        # Store results
        result = {
            'trial': trial + 1,
            'architecture': architecture,
            'performance': performance
        }
        results.append(result)

        # Update best architecture
        val_acc = performance.get('validation_accuracy', 0.0)
        if val_acc > best_performance:
            best_performance = val_acc
            best_architecture = architecture

        # Print progress
        status = "✅ Success" if 'error' not in performance else "❌ Failed"
        params = performance.get('model_params', 0)

        print(f"{trial+1:<8} {architecture['num_layers']:<8} {params:<10} {val_acc:<10.4f} {status:<15}")

    print("-" * 60)
    print(f"Best validation accuracy: {best_performance:.4f}")

    return results, best_architecture

def evolutionary_search_nas(search_space, X_train, y_train, X_val, y_val,
                           population_size=10, generations=5):
    """Implement evolutionary search NAS"""

    print(f"\n🧬 EVOLUTIONARY SEARCH NAS")
    print(f"Population size: {population_size}, Generations: {generations}")
    print("=" * 60)

    def mutate_architecture(architecture, mutation_rate=0.3):
        """Mutate an architecture"""
        mutated = architecture.copy()
        mutated['layer_configs'] = [layer.copy() for layer in architecture['layer_configs']]

        if random.random() < mutation_rate:
            # Mutate number of layers
            if random.random() < 0.5 and mutated['num_layers'] > search_space['layers']['min_layers']:
                mutated['num_layers'] -= 1
                mutated['layer_configs'].pop()
            elif mutated['num_layers'] < search_space['layers']['max_layers']:
                mutated['num_layers'] += 1
                new_layer = {
                    'type': random.choice(search_space['layers']['layer_types']),
                    'units': random.choice(search_space['units']['choices']),
                    'activation': random.choice(search_space['activations']['choices']),
                    'dropout_rate': random.choice(search_space['dropout_rates']['choices'])
                }
                mutated['layer_configs'].append(new_layer)

        # Mutate layer configurations
        for layer_config in mutated['layer_configs']:
            if random.random() < mutation_rate:
                layer_config['units'] = random.choice(search_space['units']['choices'])
            if random.random() < mutation_rate:
                layer_config['activation'] = random.choice(search_space['activations']['choices'])
            if random.random() < mutation_rate:
                layer_config['dropout_rate'] = random.choice(search_space['dropout_rates']['choices'])

        # Mutate optimizer settings
        if random.random() < mutation_rate:
            mutated['optimizer'] = random.choice(search_space['optimizers']['choices'])
        if random.random() < mutation_rate:
            mutated['learning_rate'] = random.choice(search_space['learning_rates']['choices'])

        return mutated

    def crossover_architectures(parent1, parent2):
        """Create offspring from two parent architectures"""
        child = parent1.copy()

        # Crossover layer configurations
        min_layers = min(len(parent1['layer_configs']), len(parent2['layer_configs']))
        crossover_point = random.randint(1, min_layers)

        child['layer_configs'] = (
            parent1['layer_configs'][:crossover_point] +
            parent2['layer_configs'][crossover_point:]
        )
        child['num_layers'] = len(child['layer_configs'])

        # Randomly choose optimizer settings
        child['optimizer'] = random.choice([parent1['optimizer'], parent2['optimizer']])
        child['learning_rate'] = random.choice([parent1['learning_rate'], parent2['learning_rate']])

        return child

    # Initialize population
    print("Initializing population...")
    population = []
    fitness_scores = []

    for i in range(population_size):
        architecture = generate_random_architecture(search_space)
        performance = evaluate_architecture(architecture, X_train, y_train, X_val, y_val, epochs=10)
        population.append(architecture)
        fitness_scores.append(performance.get('validation_accuracy', 0.0))

    evolution_history = []

    # Evolution loop
    for generation in range(generations):
        print(f"\n📊 Generation {generation + 1}/{generations}")

        # Selection (tournament selection)
        selected_parents = []
        for _ in range(population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected_parents.append(population[winner_idx])

        # Create new population
        new_population = []
        new_fitness_scores = []

        # Keep best individual (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx])
        new_fitness_scores.append(fitness_scores[best_idx])

        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)

            # Crossover
            if random.random() < 0.7:  # Crossover probability
                child = crossover_architectures(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])

            # Mutation
            child = mutate_architecture(child)

            # Evaluate child
            performance = evaluate_architecture(child, X_train, y_train, X_val, y_val, epochs=10)
            fitness = performance.get('validation_accuracy', 0.0)

            new_population.append(child)
            new_fitness_scores.append(fitness)

        # Update population
        population = new_population
        fitness_scores = new_fitness_scores

        # Track evolution
        generation_stats = {
            'generation': generation + 1,
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores)
        }
        evolution_history.append(generation_stats)

        print(f"   Best fitness: {generation_stats['best_fitness']:.4f}")
        print(f"   Average fitness: {generation_stats['avg_fitness']:.4f}")

    # Get best architecture
    best_idx = np.argmax(fitness_scores)
    best_architecture = population[best_idx]
    best_fitness = fitness_scores[best_idx]

    return evolution_history, best_architecture, best_fitness

def grid_search_nas(search_space, X_train, y_train, X_val, y_val, limited=True):
    """Implement limited grid search NAS"""

    print("\n🔲 GRID SEARCH NAS (Limited)")
    print("=" * 60)

    if limited:
        # Create a smaller search space for demonstration
        limited_space = {
            'num_layers': [2, 3, 4],
            'units': [32, 64, 128],
            'activations': ['relu', 'elu'],
            'dropout_rates': [0.0, 0.2],
            'optimizers': ['adam'],
            'learning_rates': [0.001, 0.01]
        }

        print("Using limited search space for demonstration:")
        for key, values in limited_space.items():
            print(f"  • {key}: {values}")

        # Generate all combinations
        combinations = list(product(
            limited_space['num_layers'],
            limited_space['units'],
            limited_space['activations'],
            limited_space['dropout_rates'],
            limited_space['optimizers'],
            limited_space['learning_rates']
        ))

        print(f"Total combinations to evaluate: {len(combinations)}")

        results = []
        best_architecture = None
        best_performance = 0.0

        print(f"\n{'Trial':<8} {'Config':<30} {'Val Acc':<10} {'Status':<15}")
        print("-" * 70)

        for i, (num_layers, units, activation, dropout, optimizer, lr) in enumerate(combinations[:20]):  # Limit to 20
            # Create architecture
            architecture = {
                'num_layers': num_layers,
                'layer_configs': [
                    {
                        'type': 'dense_dropout' if dropout > 0 else 'dense',
                        'units': units,
                        'activation': activation,
                        'dropout_rate': dropout
                    }
                ] * num_layers,
                'optimizer': optimizer,
                'learning_rate': lr
            }

            # Evaluate
            performance = evaluate_architecture(architecture, X_train, y_train, X_val, y_val, epochs=15)
            val_acc = performance.get('validation_accuracy', 0.0)

            # Store results
            results.append({
                'architecture': architecture,
                'performance': performance
            })

            # Update best
            if val_acc > best_performance:
                best_performance = val_acc
                best_architecture = architecture

            # Print progress
            config_str = f"L{num_layers}_U{units}_{activation[:3]}"
            status = "✅ Success" if 'error' not in performance else "❌ Failed"
            print(f"{i+1:<8} {config_str:<30} {val_acc:<10.4f} {status:<15}")

        print("-" * 70)
        print(f"Best validation accuracy: {best_performance:.4f}")

        return results, best_architecture

def visualize_nas_results(random_results, evolutionary_history, grid_results):
    """Visualize NAS results comparison"""

    print("\n📊 Creating NAS results visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Random Search Results
    ax1 = axes[0, 0]
    random_accs = [r['performance'].get('validation_accuracy', 0) for r in random_results]
    trials = range(1, len(random_accs) + 1)

    ax1.plot(trials, random_accs, 'bo-', linewidth=2, markersize=4, alpha=0.7)
    ax1.axhline(y=max(random_accs), color='red', linestyle='--', alpha=0.7, label=f'Best: {max(random_accs):.3f}')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Random Search NAS Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Evolutionary Search Progress
    ax2 = axes[0, 1]
    if evolutionary_history:
        generations = [h['generation'] for h in evolutionary_history]
        best_fitness = [h['best_fitness'] for h in evolutionary_history]
        avg_fitness = [h['avg_fitness'] for h in evolutionary_history]

        ax2.plot(generations, best_fitness, 'go-', linewidth=2, label='Best Fitness', markersize=6)
        ax2.plot(generations, avg_fitness, 'bo-', linewidth=2, label='Average Fitness', markersize=6, alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness (Validation Accuracy)')
        ax2.set_title('Evolutionary Search Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Method Comparison
    ax3 = axes[1, 0]
    method_names = ['Random Search', 'Evolutionary', 'Grid Search']
    best_scores = []

    # Random search best
    if random_results:
        best_scores.append(max(r['performance'].get('validation_accuracy', 0) for r in random_results))
    else:
        best_scores.append(0)

    # Evolutionary best
    if evolutionary_history:
        best_scores.append(max(h['best_fitness'] for h in evolutionary_history))
    else:
        best_scores.append(0)

    # Grid search best
    if grid_results and grid_results[0]:
        best_scores.append(max(r['performance'].get('validation_accuracy', 0) for r in grid_results[0]))
    else:
        best_scores.append(0)

    bars = ax3.bar(method_names, best_scores, alpha=0.7, color=['blue', 'green', 'orange'])
    ax3.set_ylabel('Best Validation Accuracy')
    ax3.set_title('NAS Methods Comparison')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, best_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, value + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Architecture Complexity vs Performance
    ax4 = axes[1, 1]
    if random_results:
        params = [r['performance'].get('model_params', 0) for r in random_results]
        accs = [r['performance'].get('validation_accuracy', 0) for r in random_results]

        scatter = ax4.scatter(params, accs, alpha=0.6, c=accs, cmap='viridis', s=50)
        ax4.set_xlabel('Model Parameters')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Model Complexity vs Performance')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Validation Accuracy')

    plt.tight_layout()
    plt.savefig('nas_comparison_results.png', dpi=300, bbox_inches='tight')
    print("✅ NAS results visualization saved as 'nas_comparison_results.png'")
    plt.show()

def analyze_architecture_patterns(results):
    """Analyze patterns in successful architectures"""

    print("\n🔍 ANALYZING SUCCESSFUL ARCHITECTURE PATTERNS")
    print("=" * 60)

    # Filter successful architectures (top 25%)
    performances = [r['performance'].get('validation_accuracy', 0) for r in results]
    threshold = np.percentile(performances, 75)
    successful_results = [r for r in results if r['performance'].get('validation_accuracy', 0) >= threshold]

    print(f"Analyzing {len(successful_results)} top-performing architectures (>={threshold:.3f} accuracy)")

    if not successful_results:
        print("No successful architectures found to analyze.")
        return

    # Analyze patterns
    patterns = {
        'num_layers': [],
        'units': [],
        'activations': [],
        'dropout_rates': [],
        'optimizers': [],
        'learning_rates': []
    }

    for result in successful_results:
        arch = result['architecture']
        patterns['num_layers'].append(arch['num_layers'])
        patterns['optimizers'].append(arch['optimizer'])
        patterns['learning_rates'].append(arch['learning_rate'])

        for layer in arch['layer_configs']:
            patterns['units'].append(layer['units'])
            patterns['activations'].append(layer['activation'])
            patterns['dropout_rates'].append(layer['dropout_rate'])

    # Statistical analysis
    print("\n📊 PATTERN ANALYSIS:")
    print(f"{'Component':<15} {'Most Common':<20} {'Frequency':<12} {'Mean/Mode':<15}")
    print("-" * 65)

    for component, values in patterns.items():
        if values:
            if component in ['num_layers', 'units', 'dropout_rates', 'learning_rates']:
                # Numerical data
                mean_val = np.mean(values)
                most_common = max(set(values), key=values.count)
                frequency = values.count(most_common) / len(values)
                print(f"{component:<15} {str(most_common):<20} {frequency:<12.2%} {mean_val:<15.3f}")
            else:
                # Categorical data
                most_common = max(set(values), key=values.count)
                frequency = values.count(most_common) / len(values)
                print(f"{component:<15} {most_common:<20} {frequency:<12.2%} {'-':<15}")

    # Recommendations
    print("\n💡 ARCHITECTURE DESIGN RECOMMENDATIONS:")

    # Number of layers
    optimal_layers = max(set(patterns['num_layers']), key=patterns['num_layers'].count)
    print(f"• Optimal depth: {optimal_layers} layers")

    # Units per layer
    optimal_units = max(set(patterns['units']), key=patterns['units'].count)
    print(f"• Preferred layer width: {optimal_units} units")

    # Activation function
    optimal_activation = max(set(patterns['activations']), key=patterns['activations'].count)
    print(f"• Best activation: {optimal_activation}")

    # Dropout
    dropout_values = [d for d in patterns['dropout_rates'] if d > 0]
    if dropout_values:
        avg_dropout = np.mean(dropout_values)
        print(f"• Recommended dropout: {avg_dropout:.2f} (when used)")
    else:
        print("• Dropout: Not beneficial for this task")

    # Optimizer
    optimal_optimizer = max(set(patterns['optimizers']), key=patterns['optimizers'].count)
    optimal_lr = max(set(patterns['learning_rates']), key=patterns['learning_rates'].count)
    print(f"• Best optimizer: {optimal_optimizer} with LR={optimal_lr}")

def create_nas_implementation_guide():
    """Create practical guide for implementing NAS"""

    print("\n📋 NEURAL ARCHITECTURE SEARCH IMPLEMENTATION GUIDE")
    print("=" * 80)

    print("🎯 WHEN TO USE NAS:")
    print("✅ New domains with unknown optimal architectures")
    print("✅ Resource-constrained environments (mobile, edge)")
    print("✅ When manual architecture design is too time-consuming")
    print("✅ For discovering novel architectural components")

    print("\n🔧 NAS STRATEGY SELECTION:")

    strategies = {
        'Random Search': {
            'Best For': 'Quick baseline, small search spaces',
            'Pros': 'Simple, parallelizable, good baseline',
            'Cons': 'Inefficient, no learning from previous trials',
            'When': 'Starting point, limited time/resources'
        },
        'Evolutionary': {
            'Best For': 'Medium search spaces, novel architectures',
            'Pros': 'Learns from population, handles complex spaces',
            'Cons': 'Slower convergence, population management',
            'When': 'Need exploration, discovering new patterns'
        },
        'Bayesian Optimization': {
            'Best For': 'Continuous search spaces, expensive evaluations',
            'Pros': 'Sample efficient, principled uncertainty',
            'Cons': 'Complex implementation, scalability issues',
            'When': 'Limited evaluation budget, continuous parameters'
        },
        'Gradient-based (DARTS)': {
            'Best For': 'Large search spaces, differentiable operations',
            'Pros': 'Very efficient, end-to-end differentiable',
            'Cons': 'Limited to differentiable operations, memory intensive',
            'When': 'Large search spaces, standard operations'
        }
    }

    print(f"{'Strategy':<20} {'Best For':<30} {'Key Advantage':<25}")
    print("-" * 80)

    for strategy, info in strategies.items():
        print(f"{strategy:<20} {info['Best For']:<30} {info['Pros'].split(',')[0]:<25}")

    print("\n💡 IMPLEMENTATION BEST PRACTICES:")
    print("1. 🎯 START SMALL: Begin with limited search space")
    print("2. 📊 MEASURE EFFICIENTLY: Use early stopping, weight sharing")
    print("3. 🔄 ITERATE: Gradually expand search space based on findings")
    print("4. ⚖️ BALANCE: Consider accuracy vs efficiency trade-offs")
    print("5. 📈 VALIDATE: Test final architectures thoroughly")

    print("\n⚠️ COMMON PITFALLS:")
    print("❌ Search space too large (intractable)")
    print("❌ Not validating on separate test set")
    print("❌ Ignoring computational constraints")
    print("❌ Over-optimizing for validation set")
    print("❌ Not considering deployment requirements")

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("🔍 NEURAL ARCHITECTURE SEARCH (NAS)")
    print("=" * 80)
    print()

    try:
        # Define search space
        search_space = define_search_space()

        # Generate dataset
        print("\n📊 Generating dataset for NAS evaluation...")
        X_train = tf.random.normal((2000, 20))
        y_train = tf.cast(
            (tf.reduce_sum(tf.nn.relu(X_train[:, :10] - 0.5), axis=1) *
             tf.reduce_sum(tf.nn.tanh(X_train[:, 10:]), axis=1)) > 0,
            tf.float32
        )
        y_train = tf.reshape(y_train, (-1, 1))

        X_val = tf.random.normal((400, 20))
        y_val = tf.cast(
            (tf.reduce_sum(tf.nn.relu(X_val[:, :10] - 0.5), axis=1) *
             tf.reduce_sum(tf.nn.tanh(X_val[:, 10:]), axis=1)) > 0,
            tf.float32
        )
        y_val = tf.reshape(y_val, (-1, 1))

        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Random Search NAS
        random_results, best_random_arch = random_search_nas(
            search_space, X_train, y_train, X_val, y_val, num_trials=15
        )

        # Evolutionary Search NAS
        evolutionary_history, best_evo_arch, best_evo_fitness = evolutionary_search_nas(
            search_space, X_train, y_train, X_val, y_val,
            population_size=8, generations=3
        )

        # Grid Search NAS (limited)
        grid_results, best_grid_arch = grid_search_nas(
            search_space, X_train, y_train, X_val, y_val, limited=True
        )

        # Visualize results
        visualize_nas_results(random_results, evolutionary_history, (grid_results, best_grid_arch))

        # Analyze patterns
        analyze_architecture_patterns(random_results)

        # Create implementation guide
        create_nas_implementation_guide()

        print("\n" + "=" * 80)
        print("✅ NEURAL ARCHITECTURE SEARCH ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - nas_comparison_results.png")
        print("🎓 Learning outcome: Understanding automated architecture design")
        print("💡 Next step: Learn about meta-learning approaches")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()