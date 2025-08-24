# Week 2 - Hour 2: XOR Implementation & TensorFlow Basics
## Tutorial Session T2: Working With Tensors (60 minutes)

---

## 🎯 **Learning Objectives**
By the end of this session, students will be able to:
- Implement XOR problem solution using TensorFlow/Keras
- Create and manipulate tensors for neural network inputs
- Build a simple MLP using TensorFlow's high-level API
- Understand the connection between theory and practical implementation

---

## 🔗 **1. Bridging Theory to Practice (5 minutes)**

### **What We Just Learned**
- ✅ **Biological neurons** → **Perceptrons** → **MLP architecture**
- ✅ **XOR problem** exposed perceptron limitations
- ✅ **Hidden layers** transform input space to solve complex problems
- 🎯 **Now**: Let's build it with real code!

### **Today's Coding Mission**
*"We're going to implement the XOR solution we just designed, but using TensorFlow - the same framework used by Google, Netflix, and thousands of AI companies!"*

### **Session Structure**
1. **TensorFlow Environment** (15 min): Setup and basic operations
2. **Tensor Fundamentals** (20 min): Creating and manipulating data
3. **XOR Neural Network** (20 min): Building and training our first MLP

---

## 💻 **2. TensorFlow Environment Setup (15 minutes)**

### **2.1 Google Colab Introduction (5 minutes)**

#### **Why Google Colab?**
- **Free GPU access**: Train neural networks faster
- **Pre-installed libraries**: TensorFlow, NumPy, Matplotlib ready to use
- **Collaborative**: Share notebooks easily
- **No installation needed**: Works in any web browser

#### **Opening Your First Notebook**
```python
# Let's start with a simple check
print("Welcome to Deep Learning with TensorFlow!")
print("🚀 Ready to build neural networks!")
```

### **2.2 TensorFlow Basics (10 minutes)**

#### **Import and Version Check**
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

#### **First TensorFlow Operations**
```python
# Creating your first tensors
a = tf.constant(5)
b = tf.constant(3)
result = a + b

print(f"5 + 3 = {result}")
print(f"Type: {type(result)}")
```

#### **Interactive Exercise**
*"Everyone try this: Create two tensors with your favorite numbers and multiply them!"*

```python
# Student exercise template
my_number1 = tf.constant(???)  # Your favorite number
my_number2 = tf.constant(???)  # Another favorite number
my_result = my_number1 * my_number2
print(f"My calculation: {my_result}")
```

---

## 🔢 **3. Tensor Fundamentals (20 minutes)**

### **3.1 Understanding Tensors (8 minutes)**

#### **What Are Tensors?**
*"Tensors are just multi-dimensional arrays - the building blocks of neural networks!"*

#### **Tensor Dimensions**
```python
# 0D tensor (scalar)
scalar = tf.constant(42)
print(f"Scalar: {scalar}, Shape: {scalar.shape}")

# 1D tensor (vector)
vector = tf.constant([1, 2, 3, 4])
print(f"Vector: {vector}, Shape: {vector.shape}")

# 2D tensor (matrix)
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])
print(f"Matrix:\n{matrix}\nShape: {matrix.shape}")

# 3D tensor (like a color image: height × width × channels)
tensor_3d = tf.constant([[[1, 2, 3], [4, 5, 6]], 
                         [[7, 8, 9], [10, 11, 12]]])
print(f"3D Tensor shape: {tensor_3d.shape}")
```

#### **Real-World Tensor Examples**
- **Image**: `[height, width, channels]` → `[224, 224, 3]` for RGB image
- **Text**: `[sentence_length, vocabulary_size]` → `[50, 10000]`
- **Batch of images**: `[batch_size, height, width, channels]` → `[32, 224, 224, 3]`

### **3.2 Tensor Operations for Neural Networks (12 minutes)**

#### **Basic Operations**
```python
# Element-wise operations
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.constant([[2, 3], [4, 5]], dtype=tf.float32)

print(f"Addition:\n{x + y}")
print(f"Multiplication:\n{x * y}")
print(f"Matrix multiplication:\n{tf.matmul(x, y)}")
```

#### **Neural Network Operations**
```python
# Simulating a simple perceptron calculation
inputs = tf.constant([0.5, 0.8, 0.3], dtype=tf.float32)
weights = tf.constant([0.4, 0.6, 0.2], dtype=tf.float32)
bias = tf.constant(0.1, dtype=tf.float32)

# Weighted sum (dot product)
weighted_sum = tf.reduce_sum(inputs * weights) + bias
print(f"Weighted sum: {weighted_sum}")

# Apply activation function (sigmoid)
output = tf.nn.sigmoid(weighted_sum)
print(f"After sigmoid: {output}")
```

#### **Preparing XOR Data**
```python
# XOR dataset as tensors
XOR_inputs = tf.constant([[0, 0],
                          [0, 1], 
                          [1, 0],
                          [1, 1]], dtype=tf.float32)

XOR_outputs = tf.constant([[0],
                           [1],
                           [1], 
                           [0]], dtype=tf.float32)

print("XOR Inputs:")
print(XOR_inputs)
print("\nXOR Expected Outputs:")
print(XOR_outputs)
```

---

## 🧠 **4. Building XOR Neural Network (20 minutes)**

### **4.1 Creating the MLP Model (8 minutes)**

#### **Using Keras Sequential API**
```python
# Create our XOR-solving MLP
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Display model architecture
model.summary()
```

#### **Understanding the Architecture**
```python
# Let's understand what we just built
print("Model layers:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i+1}: {layer.name}")
    print(f"  - Units: {layer.units}")
    print(f"  - Activation: {layer.activation.__name__}")
    print(f"  - Input shape: {layer.input_shape}")
    print()
```

### **4.2 Model Compilation and Training (8 minutes)**

#### **Compiling the Model**
```python
# Configure the learning process
model.compile(
    optimizer='adam',           # Learning algorithm
    loss='binary_crossentropy', # Error measurement
    metrics=['accuracy']        # What to track during training
)

print("Model compiled successfully!")
```

#### **Training the Network**
```python
# Train the model on XOR data
print("Training the XOR network...")
history = model.fit(
    XOR_inputs, 
    XOR_outputs,
    epochs=1000,        # Number of training iterations
    verbose=0           # Reduce output for classroom
)

print("Training completed!")
```

#### **Testing Our Trained Model**
```python
# Make predictions
predictions = model.predict(XOR_inputs)

print("XOR Results:")
print("Input | Expected | Predicted | Rounded")
print("------|----------|-----------|--------")
for i in range(len(XOR_inputs)):
    input_vals = XOR_inputs[i].numpy()
    expected = XOR_outputs[i].numpy()[0]
    predicted = predictions[i][0]
    rounded = round(predicted)
    
    print(f" {input_vals} |    {expected}     |   {predicted:.3f}   |   {rounded}")
```

### **4.3 Visualizing the Results (4 minutes)**

#### **Training Progress**
```python
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
```

#### **Understanding What Happened**
*"Look at the loss curve - it should decrease over time, meaning our network is learning! The accuracy should approach 100%, meaning it's solving XOR correctly."*

---

## 🎯 **5. Wrap-up & Next Steps (5 minutes)**

### **What We Accomplished Today**
✅ **Theory**: Understood biological neurons → perceptrons → MLPs  
✅ **Problem**: Analyzed XOR and linear separability challenges  
✅ **Solution**: Built MLP architecture to solve XOR  
✅ **Practice**: Implemented everything in TensorFlow!

### **Key Programming Concepts Learned**
- **Tensors**: Multi-dimensional arrays for neural network data
- **Sequential Model**: Building layers step by step
- **Dense Layers**: Fully connected neural network layers
- **Training Loop**: How networks learn from data

### **The Big Picture Connection**
*"You just built the same type of network that powers modern AI! The principles scale from XOR to image recognition, language translation, and autonomous vehicles."*

### **Next Week Preview**
- **Backpropagation**: How networks actually learn (the math behind training)
- **Gradient Descent**: The optimization algorithm that updates weights
- **More Complex Problems**: Beyond simple logic gates

### **Homework Assignment**
- **T2 Completion**: Modify the XOR network to solve other logic gates (AND, OR, NOT)
- **Exploration**: Experiment with different numbers of hidden neurons
- **Reflection**: Write a short summary of today's learning journey

---

## 📝 **Instructor Notes**

### **Teaching Tips:**
- **Live Coding**: Type code step-by-step with students following along
- **Error Handling**: Expect and address common TensorFlow errors
- **Interactive Elements**: Have students modify parameters and observe results
- **Check Understanding**: "What happens if we change the number of hidden neurons?"

### **Common Issues & Solutions:**
**Issue**: "Model not converging"  
**Solution**: Increase epochs, check data types, verify network architecture

**Issue**: "Import errors"  
**Solution**: Verify Colab environment, restart runtime if needed

**Issue**: "Predictions not improving"  
**Solution**: Check learning rate, ensure proper data normalization

### **Equipment Needed:**
- Stable internet for Google Colab
- Projector for live coding demonstration
- Backup notebooks in case of technical issues

### **Interactive Elements:**
- Students predict tensor shapes before revealing answers
- Modify network parameters and observe training changes
- Compare different activation functions

### **Timing Breakdown:**
- TensorFlow setup: 15 minutes
- Tensor fundamentals: 20 minutes
- XOR implementation: 20 minutes
- Wrap-up: 5 minutes
- **Total: 60 minutes**

---

## 🔄 **Connection to Hour 1**
*"We spent Hour 1 understanding the theory of why MLPs work. Now we've seen that theory come to life in code - solving the exact XOR problem that stumped early AI researchers!"*

## 🚀 **Student Success Indicators**
- Successfully running all code cells
- Understanding tensor shape concepts
- Seeing XOR network achieve >95% accuracy
- Asking questions about modifying the architecture
- Connecting today's code to Hour 1's theory

---

## 💡 **Extension Activities (Time Permitting)**

### **Quick Experiments**
```python
# Try different numbers of hidden neurons
model_small = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Can it still solve XOR with only 1 hidden neuron?
```

### **Advanced Challenge**
```python
# Build a network for 3-input XOR
# XOR(A,B,C) = (A XOR B) XOR C
three_input_XOR = tf.constant([[0,0,0], [0,0,1], [0,1,0], [0,1,1],
                               [1,0,0], [1,0,1], [1,1,0], [1,1,1]], 
                              dtype=tf.float32)
```