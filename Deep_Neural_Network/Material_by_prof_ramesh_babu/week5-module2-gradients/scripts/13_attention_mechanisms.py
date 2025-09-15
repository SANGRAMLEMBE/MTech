#!/usr/bin/env python3
"""
13_attention_mechanisms.py

Purpose: Demonstrate attention mechanisms and their applications
From: Week 5 Lecture Notes - Gradient Problems in Deep Neural Networks
Course: 21CSE558T - Deep Neural Network Architectures

This script implements various attention mechanisms to show how they solve
gradient flow problems and enable better information processing in deep networks.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def implement_basic_attention():
    """Implement basic attention mechanism from scratch"""

    print("🎯 IMPLEMENTING BASIC ATTENTION MECHANISM")
    print("=" * 60)

    class BasicAttention(tf.keras.layers.Layer):
        """Basic attention layer implementation"""

        def __init__(self, units, **kwargs):
            super(BasicAttention, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            # Attention weights
            self.W_query = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_query'
            )

            self.W_key = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_key'
            )

            self.W_value = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_value'
            )

            super(BasicAttention, self).build(input_shape)

        def call(self, inputs, return_attention=False):
            # inputs shape: (batch_size, sequence_length, features)

            # Compute queries, keys, and values
            queries = tf.matmul(inputs, self.W_query)  # (batch, seq_len, units)
            keys = tf.matmul(inputs, self.W_key)       # (batch, seq_len, units)
            values = tf.matmul(inputs, self.W_value)   # (batch, seq_len, units)

            # Compute attention scores
            scores = tf.matmul(queries, keys, transpose_b=True)  # (batch, seq_len, seq_len)
            scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))  # Scale

            # Apply softmax to get attention weights
            attention_weights = tf.nn.softmax(scores, axis=-1)

            # Apply attention to values
            attended_values = tf.matmul(attention_weights, values)

            if return_attention:
                return attended_values, attention_weights
            return attended_values

        def get_config(self):
            config = super(BasicAttention, self).get_config()
            config.update({'units': self.units})
            return config

    print("✅ Basic attention mechanism implemented")
    print("Key components:")
    print("  • Query, Key, Value transformations")
    print("  • Scaled dot-product attention")
    print("  • Softmax normalization")

    return BasicAttention

def implement_self_attention():
    """Implement self-attention mechanism"""

    print("\n🔄 IMPLEMENTING SELF-ATTENTION MECHANISM")
    print("=" * 60)

    class SelfAttention(tf.keras.layers.Layer):
        """Self-attention layer implementation"""

        def __init__(self, units, num_heads=1, **kwargs):
            super(SelfAttention, self).__init__(**kwargs)
            self.units = units
            self.num_heads = num_heads
            self.head_dim = units // num_heads

            assert units % num_heads == 0, "units must be divisible by num_heads"

        def build(self, input_shape):
            self.W_query = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_query'
            )

            self.W_key = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_key'
            )

            self.W_value = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_value'
            )

            self.W_output = self.add_weight(
                shape=(self.units, self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_output'
            )

            super(SelfAttention, self).build(input_shape)

        def call(self, inputs, return_attention=False):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            # Linear transformations
            queries = tf.matmul(inputs, self.W_query)
            keys = tf.matmul(inputs, self.W_key)
            values = tf.matmul(inputs, self.W_value)

            # Reshape for multi-head attention
            queries = tf.reshape(queries, (batch_size, seq_len, self.num_heads, self.head_dim))
            keys = tf.reshape(keys, (batch_size, seq_len, self.num_heads, self.head_dim))
            values = tf.reshape(values, (batch_size, seq_len, self.num_heads, self.head_dim))

            # Transpose for attention computation
            queries = tf.transpose(queries, [0, 2, 1, 3])  # (batch, heads, seq_len, head_dim)
            keys = tf.transpose(keys, [0, 2, 1, 3])
            values = tf.transpose(values, [0, 2, 1, 3])

            # Compute attention scores
            scores = tf.matmul(queries, keys, transpose_b=True)
            scores = scores / tf.sqrt(tf.cast(self.head_dim, tf.float32))

            # Apply softmax
            attention_weights = tf.nn.softmax(scores, axis=-1)

            # Apply attention to values
            attended = tf.matmul(attention_weights, values)

            # Transpose back and reshape
            attended = tf.transpose(attended, [0, 2, 1, 3])
            attended = tf.reshape(attended, (batch_size, seq_len, self.units))

            # Output projection
            output = tf.matmul(attended, self.W_output)

            if return_attention:
                return output, attention_weights
            return output

        def get_config(self):
            config = super(SelfAttention, self).get_config()
            config.update({
                'units': self.units,
                'num_heads': self.num_heads
            })
            return config

    print("✅ Self-attention mechanism implemented")
    print("Features:")
    print("  • Multi-head attention support")
    print("  • Self-attention (queries, keys, values from same input)")
    print("  • Output projection")

    return SelfAttention

def demonstrate_attention_on_sequences():
    """Demonstrate attention mechanisms on sequence data"""

    print("\n📊 DEMONSTRATING ATTENTION ON SEQUENCE DATA")
    print("=" * 60)

    # Create synthetic sequence data
    def create_sequence_data(num_samples=1000, seq_length=10, feature_dim=8):
        """Create synthetic sequence classification data"""

        X = []
        y = []

        for _ in range(num_samples):
            # Generate random sequence
            sequence = np.random.randn(seq_length, feature_dim)

            # Create pattern: if sum of first half > sum of second half, label = 1
            first_half_sum = np.sum(sequence[:seq_length//2])
            second_half_sum = np.sum(sequence[seq_length//2:])
            label = 1 if first_half_sum > second_half_sum else 0

            # Add noise to make it more challenging
            sequence += np.random.normal(0, 0.1, sequence.shape)

            X.append(sequence)
            y.append(label)

        return np.array(X), np.array(y)

    # Generate data
    seq_length, feature_dim = 12, 16
    X_train, y_train = create_sequence_data(2000, seq_length, feature_dim)
    X_val, y_val = create_sequence_data(400, seq_length, feature_dim)

    print(f"Generated sequence data:")
    print(f"  Training: {X_train.shape} → {y_train.shape}")
    print(f"  Validation: {X_val.shape} → {y_val.shape}")

    # Test different models
    models = {}

    # 1. Baseline: Simple RNN
    print("\n🔧 Creating baseline RNN model...")
    models['RNN'] = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(seq_length, feature_dim)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 2. RNN with attention
    print("🔧 Creating RNN + Attention model...")
    BasicAttention = implement_basic_attention()

    rnn_input = tf.keras.layers.Input(shape=(seq_length, feature_dim))
    rnn_output = tf.keras.layers.LSTM(64, return_sequences=True)(rnn_input)
    attention_output = BasicAttention(64)(rnn_output)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    dense1 = tf.keras.layers.Dense(32, activation='relu')(pooled)
    dropout = tf.keras.layers.Dropout(0.2)(dense1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

    models['RNN + Attention'] = tf.keras.Model(inputs=rnn_input, outputs=output)

    # 3. Self-attention only
    print("🔧 Creating Self-Attention model...")
    SelfAttention = implement_self_attention()

    sa_input = tf.keras.layers.Input(shape=(seq_length, feature_dim))
    sa_output = SelfAttention(64, num_heads=4)(sa_input)
    sa_pooled = tf.keras.layers.GlobalAveragePooling1D()(sa_output)
    sa_dense1 = tf.keras.layers.Dense(32, activation='relu')(sa_pooled)
    sa_dropout = tf.keras.layers.Dropout(0.2)(sa_dense1)
    sa_output_final = tf.keras.layers.Dense(1, activation='sigmoid')(sa_dropout)

    models['Self-Attention'] = tf.keras.Model(inputs=sa_input, outputs=sa_output_final)

    # Train and compare models
    training_results = {}

    print(f"\n{'Model':<20} {'Val Accuracy':<15} {'Epochs to Converge':<20}")
    print("-" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0
        )

        # Store results
        best_val_acc = max(history.history['val_accuracy'])
        convergence_epoch = np.argmax(history.history['val_accuracy']) + 1

        training_results[name] = {
            'history': history.history,
            'best_val_acc': best_val_acc,
            'convergence_epoch': convergence_epoch
        }

        print(f"{name:<20} {best_val_acc:<15.4f} {convergence_epoch:<20}")

    print("-" * 60)

    # Create comparison plots
    create_attention_comparison_plots(training_results)

    return training_results, models

def visualize_attention_weights():
    """Visualize attention weights to understand what the model focuses on"""

    print("\n👁️ VISUALIZING ATTENTION WEIGHTS")
    print("=" * 60)

    # Create a simple example with interpretable attention
    class AttentionVisualizer(tf.keras.layers.Layer):
        """Attention layer that returns attention weights for visualization"""

        def __init__(self, units, **kwargs):
            super(AttentionVisualizer, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W_attention = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='W_attention'
            )

            self.v_attention = self.add_weight(
                shape=(self.units, 1),
                initializer='glorot_uniform',
                trainable=True,
                name='v_attention'
            )

            super(AttentionVisualizer, self).build(input_shape)

        def call(self, inputs):
            # Compute attention scores
            energy = tf.nn.tanh(tf.matmul(inputs, self.W_attention))
            scores = tf.matmul(energy, self.v_attention)
            scores = tf.squeeze(scores, axis=-1)

            # Compute attention weights
            attention_weights = tf.nn.softmax(scores, axis=1)

            # Apply attention
            attended = tf.reduce_sum(
                inputs * tf.expand_dims(attention_weights, axis=-1),
                axis=1
            )

            return attended, attention_weights

    # Create simple sequence data with clear patterns
    def create_pattern_data(num_samples=100):
        """Create sequences with clear attention patterns"""
        X, y, patterns = [], [], []

        for i in range(num_samples):
            seq_len = 8
            sequence = np.random.randn(seq_len, 4) * 0.1

            # Create different patterns
            pattern_type = i % 3

            if pattern_type == 0:
                # Important information at the beginning
                sequence[0:2] += 2.0
                label = 1
                patterns.append("Beginning")
            elif pattern_type == 1:
                # Important information in the middle
                sequence[3:5] += 2.0
                label = 1
                patterns.append("Middle")
            else:
                # Important information at the end
                sequence[6:8] += 2.0
                label = 0
                patterns.append("End")

            X.append(sequence)
            y.append(label)

        return np.array(X), np.array(y), patterns

    # Generate pattern data
    X_pattern, y_pattern, pattern_labels = create_pattern_data(150)

    print(f"Created pattern data: {X_pattern.shape}")
    print("Patterns: Beginning (label 1), Middle (label 1), End (label 0)")

    # Build model with attention visualizer
    input_layer = tf.keras.layers.Input(shape=(8, 4))
    attended, attention_weights = AttentionVisualizer(16)(input_layer)
    dense = tf.keras.layers.Dense(16, activation='relu')(attended)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    attention_model = tf.keras.Model(inputs=input_layer, outputs=[output, attention_weights])

    # Compile and train
    attention_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining attention visualizer model...")
    history = attention_model.fit(
        X_pattern, y_pattern,
        epochs=50,
        batch_size=16,
        verbose=0
    )

    # Get attention weights for visualization
    _, attention_weights = attention_model.predict(X_pattern[:30])

    # Create attention heatmap
    create_attention_heatmap(X_pattern[:30], attention_weights, pattern_labels[:30])

    return attention_model, attention_weights

def create_attention_heatmap(X_sample, attention_weights, pattern_labels):
    """Create heatmap visualization of attention weights"""

    print("\n📊 Creating attention heatmap...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Group samples by pattern
    pattern_types = ['Beginning', 'Middle', 'End']

    for i, pattern in enumerate(pattern_types):
        ax = axes[i]

        # Filter samples for this pattern
        pattern_indices = [j for j, label in enumerate(pattern_labels) if label == pattern]
        pattern_weights = attention_weights[pattern_indices]

        # Create heatmap
        sns.heatmap(
            pattern_weights,
            cmap='Blues',
            cbar=True,
            ax=ax,
            xticklabels=[f'Pos {j+1}' for j in range(8)],
            yticklabels=False
        )

        ax.set_title(f'Attention Weights - {pattern} Pattern')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Sample')

        # Add pattern annotation
        if pattern == 'Beginning':
            ax.axvspan(0, 2, alpha=0.3, color='red', label='Important Region')
        elif pattern == 'Middle':
            ax.axvspan(3, 5, alpha=0.3, color='red', label='Important Region')
        else:  # End
            ax.axvspan(6, 8, alpha=0.3, color='red', label='Important Region')

        ax.legend()

    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Attention heatmap saved as 'attention_heatmap.png'")
    plt.show()

def create_attention_comparison_plots(training_results):
    """Create plots comparing different attention mechanisms"""

    print("\n📊 Creating attention mechanism comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training curves
    ax1 = axes[0, 0]
    for name, results in training_results.items():
        ax1.plot(results['history']['val_accuracy'], linewidth=2, label=name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    ax2 = axes[0, 1]
    for name, results in training_results.items():
        ax2.plot(results['history']['val_loss'], linewidth=2, label=name)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Loss Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best performance comparison
    ax3 = axes[1, 0]
    models = list(training_results.keys())
    accuracies = [training_results[model]['best_val_acc'] for model in models]

    bars = ax3.bar(models, accuracies, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax3.set_ylabel('Best Validation Accuracy')
    ax3.set_title('Best Performance Comparison')
    ax3.set_xticklabels(models, rotation=45)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence speed
    ax4 = axes[1, 1]
    convergence_epochs = [training_results[model]['convergence_epoch'] for model in models]

    bars = ax4.bar(models, convergence_epochs, alpha=0.7, color=['orange', 'green', 'purple'])
    ax4.set_ylabel('Epochs to Convergence')
    ax4.set_title('Convergence Speed Comparison')
    ax4.set_xticklabels(models, rotation=45)

    # Add value labels
    for bar, epoch in zip(bars, convergence_epochs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{epoch}', ha='center', va='bottom', fontweight='bold')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Attention comparison plots saved as 'attention_comparison.png'")
    plt.show()

def demonstrate_transformer_block():
    """Demonstrate a simplified transformer block"""

    print("\n🤖 DEMONSTRATING TRANSFORMER BLOCK")
    print("=" * 60)

    class TransformerBlock(tf.keras.layers.Layer):
        """Simplified transformer block implementation"""

        def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
            super(TransformerBlock, self).__init__(**kwargs)
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.dropout_rate = dropout_rate

        def build(self, input_shape):
            # Multi-head attention
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim
            )

            # Feed-forward network
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.ff_dim, activation='relu'),
                tf.keras.layers.Dense(self.embed_dim),
            ])

            # Layer normalization
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

            # Dropout
            self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
            self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

            super(TransformerBlock, self).build(input_shape)

        def call(self, inputs, training=None):
            # Multi-head attention with residual connection
            attention_output = self.attention(inputs, inputs)
            attention_output = self.dropout1(attention_output, training=training)
            out1 = self.layernorm1(inputs + attention_output)

            # Feed-forward with residual connection
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)

            return out2

        def get_config(self):
            config = super(TransformerBlock, self).get_config()
            config.update({
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'ff_dim': self.ff_dim,
                'dropout_rate': self.dropout_rate
            })
            return config

    # Build model with transformer block
    def create_transformer_model(seq_length, feature_dim):
        inputs = tf.keras.layers.Input(shape=(seq_length, feature_dim))

        # Add positional encoding (simplified)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=seq_length, output_dim=feature_dim
        )(positions)
        x = inputs + position_embedding

        # Transformer blocks
        x = TransformerBlock(embed_dim=feature_dim, num_heads=4, ff_dim=64)(x)
        x = TransformerBlock(embed_dim=feature_dim, num_heads=4, ff_dim=64)(x)

        # Global pooling and classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs, outputs)

    print("✅ Transformer block implemented with:")
    print("  • Multi-head self-attention")
    print("  • Feed-forward networks")
    print("  • Residual connections")
    print("  • Layer normalization")

    return TransformerBlock, create_transformer_model

def analyze_attention_benefits():
    """Analyze the benefits of attention mechanisms"""

    print("\n🔍 ANALYZING ATTENTION MECHANISM BENEFITS")
    print("=" * 60)

    benefits = {
        'Gradient Flow': {
            'Problem': 'Vanishing gradients in long sequences',
            'Solution': 'Direct connections via attention weights',
            'Impact': 'Better training of deep networks'
        },
        'Long Dependencies': {
            'Problem': 'RNNs struggle with long-range dependencies',
            'Solution': 'Attention can connect distant positions',
            'Impact': 'Capture relationships across entire sequence'
        },
        'Interpretability': {
            'Problem': 'Black box models difficult to understand',
            'Solution': 'Attention weights show what model focuses on',
            'Impact': 'Better model understanding and debugging'
        },
        'Parallelization': {
            'Problem': 'RNNs process sequences sequentially',
            'Solution': 'Attention allows parallel computation',
            'Impact': 'Faster training and inference'
        },
        'Selective Focus': {
            'Problem': 'Fixed representations lose information',
            'Solution': 'Dynamic attention adapts to input',
            'Impact': 'More efficient information processing'
        }
    }

    print("📊 ATTENTION MECHANISM BENEFITS:")
    print(f"{'Benefit':<18} {'Problem Solved':<35} {'Key Impact':<30}")
    print("-" * 85)

    for benefit, info in benefits.items():
        print(f"{benefit:<18} {info['Problem']:<35} {info['Impact']:<30}")

    print("\n💡 PRACTICAL APPLICATIONS:")
    applications = [
        "🔤 Natural Language Processing (BERT, GPT, T5)",
        "👁️ Computer Vision (Vision Transformers, DETR)",
        "🎵 Speech Processing (Listen, Attend and Spell)",
        "🧬 Protein Folding (AlphaFold attention mechanisms)",
        "🎯 Recommendation Systems (Attentive collaborative filtering)",
        "🚗 Autonomous Driving (Spatial attention for objects)",
        "📈 Time Series Forecasting (Temporal attention patterns)"
    ]

    for app in applications:
        print(f"  • {app}")

def create_attention_implementation_guide():
    """Create practical guide for implementing attention mechanisms"""

    print("\n📋 ATTENTION MECHANISMS IMPLEMENTATION GUIDE")
    print("=" * 80)

    print("🎯 CHOOSING THE RIGHT ATTENTION:")

    attention_types = {
        'Basic Attention': {
            'When': 'Simple sequence-to-sequence tasks',
            'Complexity': 'Low',
            'Memory': 'O(n²)',
            'Best For': 'Learning attention concepts'
        },
        'Self-Attention': {
            'When': 'Within-sequence dependencies',
            'Complexity': 'Medium',
            'Memory': 'O(n²)',
            'Best For': 'Document classification, time series'
        },
        'Multi-Head Attention': {
            'When': 'Complex patterns, multiple relationships',
            'Complexity': 'High',
            'Memory': 'O(n²)',
            'Best For': 'Language models, complex sequences'
        },
        'Sparse Attention': {
            'When': 'Very long sequences',
            'Complexity': 'High',
            'Memory': 'O(n√n)',
            'Best For': 'Long documents, large images'
        }
    }

    print(f"{'Type':<20} {'Best For':<30} {'Complexity':<12} {'Memory':<10}")
    print("-" * 75)

    for att_type, info in attention_types.items():
        print(f"{att_type:<20} {info['Best For']:<30} {info['Complexity']:<12} {info['Memory']:<10}")

    print("\n🔧 IMPLEMENTATION BEST PRACTICES:")
    print("1. 🎯 START SIMPLE: Begin with basic attention before multi-head")
    print("2. 📏 SCALE APPROPRIATELY: Use scaling factor √d_k for stability")
    print("3. 🎭 ADD MASKS: Use padding masks and causal masks when needed")
    print("4. 🔄 RESIDUAL CONNECTIONS: Always use with layer normalization")
    print("5. 📊 VISUALIZE: Plot attention weights to understand behavior")
    print("6. ⚖️ MEMORY AWARE: Consider O(n²) memory complexity for long sequences")

    print("\n⚠️ COMMON PITFALLS:")
    print("❌ Not scaling attention scores (leads to vanishing gradients)")
    print("❌ Forgetting position information (use positional encoding)")
    print("❌ Ignoring padding tokens (use attention masks)")
    print("❌ Too many heads (diminishing returns after 8-16 heads)")
    print("❌ Not using layer normalization with residuals")
    print("❌ Applying attention everywhere (not always beneficial)")

    print("\n💻 TensorFlow IMPLEMENTATION TIPS:")
    implementation_tips = '''
# Basic attention implementation
def scaled_dot_product_attention(q, k, v, mask=None):
    scores = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_scores = scores / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_scores += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Using TensorFlow's built-in MultiHeadAttention
attention_layer = tf.keras.layers.MultiHeadAttention(
    num_heads=8, key_dim=64, dropout=0.1
)
'''

    print(implementation_tips)

def main():
    """Main demonstration function"""

    print("🧠 DEEP NEURAL NETWORK ARCHITECTURES - WEEK 5")
    print("👁️ ATTENTION MECHANISMS")
    print("=" * 80)
    print()

    try:
        # Implement and demonstrate basic attention
        BasicAttention = implement_basic_attention()
        SelfAttention = implement_self_attention()

        # Demonstrate on sequence data
        training_results, models = demonstrate_attention_on_sequences()

        # Visualize attention weights
        attention_model, attention_weights = visualize_attention_weights()

        # Demonstrate transformer block
        TransformerBlock, create_transformer_model = demonstrate_transformer_block()

        # Analyze benefits
        analyze_attention_benefits()

        # Create implementation guide
        create_attention_implementation_guide()

        print("\n" + "=" * 80)
        print("✅ ATTENTION MECHANISMS ANALYSIS COMPLETE!")
        print("📊 Key files created:")
        print("   - attention_comparison.png")
        print("   - attention_heatmap.png")
        print("🎓 Learning outcome: Understanding attention mechanisms and their applications")
        print("💡 Next step: Learn about advanced techniques and future directions")

        print("\n📚 BOOK REFERENCES:")
        print("=" * 80)
        print("1. 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, Aaron Courville")
        print("   - Chapter 10: Sequence Modeling: Recurrent and Recursive Nets")
        print("   - Chapter 12: Applications")
        print()
        print("2. 'Deep Learning with Python' by François Chollet")
        print("   - Chapter 6: Deep learning for text and sequences")
        print("   - Chapter 10: Introduction to artificial neural networks")
        print()
        print("3. 'Neural Networks and Deep Learning' by Charu C. Aggarwal")
        print("   - Chapter 7: Recurrent Neural Networks")
        print("   - Chapter 8: Attention Mechanisms")
        print()
        print("4. 'Deep Learning with Applications Using Python'")
        print("   - Chapter 7: Sequential Models and NLP")
        print("   - Chapter 9: Transformers and Attention")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Suggestion: Check TensorFlow installation and version")

if __name__ == "__main__":
    main()