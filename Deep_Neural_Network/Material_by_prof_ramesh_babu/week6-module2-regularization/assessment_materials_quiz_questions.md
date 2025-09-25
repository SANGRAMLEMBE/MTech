# Week 6 Day 3: Assessment Materials & Quiz Questions
## Overfitting, Underfitting & Classical Regularization

**Course:** 21CSE558T - Deep Neural Network Architectures
**Date:** September 15, 2025
**Instructor:** Prof. Ramesh Babu
**Assessment Type:** Formative & Unit Test 1 Preparation

---

## 📋 Assessment Overview

This assessment package contains multiple question types aligned with Bloom's Taxonomy and Course Outcomes (CO-1, CO-2). Questions incorporate our analogy-based learning approach for enhanced understanding.

**Assessment Components:**
- **Immediate Feedback Quiz** (During lecture)
- **Unit Test 1 Practice Questions** (Sep 19 preparation)
- **Conceptual Understanding Checks**
- **Practical Implementation Tasks**
- **Analogy-Based Explanations**

---

# 🎯 IMMEDIATE FEEDBACK QUIZ (During Lecture)

## Poll Questions (For Interactive Response)

### **Poll 1: Opening Warm-Up**
*"Which chef would you hire for your restaurant?"*
- A) Chef who knows only basic recipes (Simple but consistent)
- B) Chef who adapts principles to new situations (Balanced approach)
- C) Chef who memorizes every customer's exact preference (Perfect but inflexible)

**Correct Answer:** B
**Learning Objective:** Introduce bias-variance concept through analogy

### **Poll 2: Bias-Variance Understanding**
*"An archer consistently shoots to the left of the bullseye with arrows clustered together. This represents:"*
- A) High Bias, Low Variance
- B) Low Bias, High Variance
- C) High Bias, High Variance
- D) Low Bias, Low Variance

**Correct Answer:** A
**Explanation:** Consistent error (bias) but consistent placement (low variance)

### **Poll 3: Overfitting Recognition**
*"A student gets 100% on practice tests but fails the real exam. In ML, this is:"*
- A) Good generalization
- B) Overfitting
- C) Underfitting
- D) Perfect learning

**Correct Answer:** B
**Explanation:** Perfect training performance, poor test performance = overfitting

### **Poll 4: Regularization Choice**
*"Marie Kondo's decluttering philosophy 'keep only what sparks joy' best represents:"*
- A) No regularization
- B) L1 regularization
- C) L2 regularization
- D) Early stopping

**Correct Answer:** B
**Explanation:** L1 eliminates features (sets weights to zero) like decluttering

### **Poll 5: L2 Regularization**
*"An 'Equal Opportunity Employer' management style is most similar to:"*
- A) L1 regularization - eliminates some employees
- B) L2 regularization - gives everyone a balanced role
- C) No regularization - lets star employees dominate
- D) Dropout - randomly removes employees

**Correct Answer:** B
**Explanation:** L2 balances weights across all features

---

# 📝 UNIT TEST 1 PRACTICE QUESTIONS

## Section A: Multiple Choice Questions (1 mark each)

### **Question A1:**
The mathematical expression for total prediction error in the bias-variance decomposition is:
- A) Total Error = Bias + Variance
- B) Total Error = Bias² + Variance + Noise
- C) Total Error = Bias² + Variance + Irreducible Error
- D) Total Error = |Bias| + |Variance|

**Answer:** C

### **Question A2:**
In L1 regularization, the penalty term added to the loss function is:
- A) λ∑wᵢ
- B) λ∑wᵢ²
- C) λ∑|wᵢ|
- D) λ∑√wᵢ

**Answer:** C

### **Question A3:**
Which regularization technique is most likely to create sparse models (many weights = 0)?
- A) L1 regularization
- B) L2 regularization
- C) No regularization
- D) Both L1 and L2 equally

**Answer:** A

### **Question A4:**
The geometric constraint region for L2 regularization in 2D is:
- A) Diamond shaped
- B) Square shaped
- C) Circular shaped
- D) Triangular shaped

**Answer:** C

### **Question A5:**
A model shows training accuracy of 98% and validation accuracy of 72%. This indicates:
- A) Healthy learning
- B) Underfitting
- C) Overfitting
- D) Perfect generalization

**Answer:** C

## Section B: Short Answer Questions (2 marks each)

### **Question B1:**
Calculate the L1 penalty for weight vector w = [1.5, -2.0, 0.8, -0.3] with λ = 0.1.

**Solution:**
L1 penalty = λ∑|wᵢ| = 0.1 × (|1.5| + |-2.0| + |0.8| + |-0.3|)
= 0.1 × (1.5 + 2.0 + 0.8 + 0.3) = 0.1 × 4.6 = 0.46

**Marking Scheme:**
- Correct formula (1 mark)
- Correct calculation (1 mark)

### **Question B2:**
Explain why L1 regularization leads to feature selection using the "Marie Kondo" analogy.

**Model Answer:**
Marie Kondo's philosophy is "keep only items that spark joy." L1 regularization works similarly by keeping only features that contribute significantly to prediction accuracy. The L1 penalty (λ∑|wᵢ|) forces weights toward zero, and due to its absolute value nature and diamond-shaped constraint, it pushes less important weights to exactly zero, effectively "decluttering" the model by eliminating irrelevant features.

**Marking Scheme:**
- Analogy connection (1 mark)
- Mathematical explanation (1 mark)

### **Question B3:**
A learning curve shows training loss decreasing continuously while validation loss initially decreases then increases after epoch 20. Diagnose the problem and suggest two solutions.

**Model Answer:**
**Diagnosis:** Overfitting starting around epoch 20. The model begins memorizing training data rather than learning generalizable patterns.

**Solutions:**
1. Early stopping - halt training around epoch 20 when validation loss starts increasing
2. Add regularization (L1 or L2) - penalize model complexity to prevent memorization

**Marking Scheme:**
- Correct diagnosis (1 mark)
- Two valid solutions (1 mark)

## Section C: Long Answer Questions (5 marks each)

### **Question C1:**
*"A restaurant chain wants to predict customer satisfaction using 50 different factors (food quality, service speed, price, music volume, etc.). Some factors are highly correlated (e.g., food temperature and cooking time), and some might be irrelevant (e.g., server's hair color)."*

**Part A:** Which regularization technique would you recommend? Justify using appropriate analogies. (3 marks)

**Part B:** How would you determine the optimal λ value? Describe your validation strategy. (2 marks)

**Model Answer:**

**Part A:** I would recommend **L1 regularization (LASSO)** for this scenario.

**Justification using analogies:**
- **Marie Kondo approach:** L1 regularization will automatically "declutter" the model by eliminating irrelevant factors (like server's hair color) that don't contribute to customer satisfaction, setting their weights to exactly zero.
- **Budget allocation:** Like allocating a limited budget only to departments that generate profit, L1 forces the model to invest only in features that actually predict satisfaction.
- **Feature selection benefit:** With 50 factors, many likely irrelevant, L1's natural feature selection will create a simpler, more interpretable model that restaurant managers can understand and act upon.

**Part B:** Optimal λ determination strategy:
1. **K-fold cross-validation** (k=5 or 10) with different λ values [0.001, 0.01, 0.1, 1.0]
2. **Monitor training vs validation performance** - select λ that minimizes validation error while maintaining reasonable training performance
3. **Grid search** around the best performing λ for fine-tuning
4. **Final evaluation** on hold-out test set to confirm generalization

**Marking Scheme:**
- Correct technique choice (1 mark)
- Appropriate analogies (2 marks)
- Validation strategy (2 marks)

### **Question C2:**
Compare and contrast L1 and L2 regularization across five different dimensions. Use mathematical formulations and real-world analogies in your explanation.

**Model Answer:**

| Dimension | L1 Regularization | L2 Regularization |
|-----------|-------------------|-------------------|
| **Mathematical Form** | Loss + λ∑\|wᵢ\| | Loss + λ∑wᵢ² |
| **Geometric Constraint** | Diamond: \|w₁\| + \|w₂\| ≤ λ | Circle: w₁² + w₂² ≤ λ |
| **Effect on Weights** | Forces some weights to exactly zero | Shrinks all weights toward zero |
| **Analogy** | Marie Kondo - eliminates clutter | Equal Opportunity - fair distribution |
| **Use Case** | Feature selection, interpretability | Handling multicollinearity, stability |

**Detailed Explanation:**
1. **Sparsity:** L1 creates sparse models (many zero weights) due to diamond constraint's sharp corners at axes. L2 shrinks weights proportionally but rarely eliminates them completely.

2. **Interpretability:** L1 produces highly interpretable models with fewer active features. L2 maintains all features, making interpretation more complex.

3. **Stability:** L1 can be unstable with correlated features (may randomly select one). L2 is more stable, distributing weights among correlated features.

4. **Computational:** L1 is non-differentiable at zero, requiring special optimization. L2 is smooth and easier to optimize.

5. **Selection:** Choose L1 for feature selection in high-dimensional data. Choose L2 for correlated features and when you want all features to contribute.

**Marking Scheme:**
- Mathematical formulations (1 mark)
- Geometric explanations (1 mark)
- Analogies (1 mark)
- Practical differences (1 mark)
- Use case guidance (1 mark)

---

# 🧩 CONCEPTUAL UNDERSTANDING CHECKS

## Analogy Completion Tasks

### **Task 1: Complete the Analogy**
*"Overfitting is to _____________ as understanding is to _____________"*
- A) memorizing; adapting
- B) learning; cramming
- C) practicing; testing
- D) studying; forgetting

**Answer:** A
**Explanation:** Overfitting memorizes training data while good learning adapts to new situations

### **Task 2: Analogy Application**
*"If L1 regularization is like Marie Kondo's 'keep only what sparks joy' philosophy, then L2 regularization is most like:"*
- A) A hoarder who keeps everything
- B) An equal opportunity employer who gives everyone a chance
- C) A minimalist who owns nothing
- D) A perfectionist who controls everything

**Answer:** B

### **Task 3: Extended Analogy**
*"In the restaurant chef analogy, what would 'increasing λ (lambda) in regularization' be equivalent to?"*
- A) Hiring more chefs
- B) Adding more recipes to memorize
- C) Enforcing stricter cooking rules/constraints
- D) Serving more customers

**Answer:** C
**Explanation:** Higher λ means stricter regularization constraints, like stricter cooking rules

## Scenario-Based Questions

### **Scenario 1: Medical AI System**
*"You're developing an AI system to diagnose rare diseases. The system must be highly interpretable for doctors and work reliably across different hospitals. You have 200 potential symptoms, but only 15-20 are typically relevant for each disease."*

**Questions:**
1. Which regularization would you choose and why?
2. How does interpretability requirement affect your choice?
3. What λ range would you start testing with?

**Model Answers:**
1. **L1 regularization** - automatic feature selection identifies the 15-20 relevant symptoms
2. **High interpretability need** supports L1 choice - doctors can see exactly which symptoms the model considers important
3. **Start with λ = [0.001, 0.01, 0.1]** - medical data often needs moderate regularization

### **Scenario 2: Financial Trading**
*"A trading algorithm uses 100 technical indicators, many of which are highly correlated (e.g., different types of moving averages). The model needs to be stable and robust to market changes."*

**Questions:**
1. Which regularization technique would be most appropriate?
2. Why might L1 be problematic in this scenario?
3. How would you validate your approach?

**Model Answers:**
1. **L2 regularization** - handles correlated features well, provides stability
2. **L1 problems:** With correlated indicators, L1 might randomly select one moving average and eliminate others, losing valuable information
3. **Time-series validation:** Use walk-forward analysis, not random splits, due to temporal nature of financial data

---

# 🔬 PRACTICAL IMPLEMENTATION ASSESSMENT

## Code Analysis Questions

### **Question P1: Debug the Regularization**
```python
# Student's code with issues
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l1(1.0)),
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
```

**Task:** Identify and fix three potential issues with this regularization setup.

**Issues & Solutions:**
1. **λ = 1.0 for L1 is too large** - likely to cause severe underfitting. Start with 0.01-0.1
2. **Missing metrics=['accuracy']** in compile - important for monitoring performance
3. **No early stopping** - might overfit despite regularization, should add callback
4. **Mixed L1/L2 without justification** - should choose based on problem requirements

### **Question P2: Complete the Implementation**
```python
def build_regularized_model(reg_type, lambda_val):
    """
    Build a model with specified regularization

    Args:
        reg_type: 'l1', 'l2', or 'l1_l2'
        lambda_val: regularization strength
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=_______),  # Fill this
        tf.keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=_______),   # Fill this
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

**Solution:**
```python
if reg_type == 'l1':
    regularizer = tf.keras.regularizers.l1(lambda_val)
elif reg_type == 'l2':
    regularizer = tf.keras.regularizers.l2(lambda_val)
elif reg_type == 'l1_l2':
    regularizer = tf.keras.regularizers.l1_l2(l1=lambda_val, l2=lambda_val)
else:
    regularizer = None
```

---

# 📊 FORMATIVE ASSESSMENT RUBRIC

## Evaluation Criteria

### **Level 4 - Excellent (90-100%)**
- **Conceptual Understanding:** Explains regularization using analogies fluently, connects mathematical formulations to real-world applications
- **Mathematical Skills:** Calculates penalties accurately, derives constraints, understands geometric interpretations
- **Implementation:** Writes clean, correct TensorFlow code with appropriate hyperparameters
- **Analysis:** Diagnoses overfitting/underfitting correctly, chooses appropriate regularization with clear justification
- **Communication:** Uses analogies effectively to explain complex concepts to others

### **Level 3 - Proficient (80-89%)**
- **Conceptual Understanding:** Good grasp of regularization concepts, makes some analogy connections
- **Mathematical Skills:** Mostly accurate calculations with minor errors
- **Implementation:** Correct basic implementation with some hyperparameter guidance
- **Analysis:** Identifies most overfitting cases, reasonable regularization choices
- **Communication:** Can explain concepts with some analogy support

### **Level 2 - Developing (70-79%)**
- **Conceptual Understanding:** Basic understanding of regularization purpose, limited analogy use
- **Mathematical Skills:** Calculation errors but shows understanding of process
- **Implementation:** Basic code structure correct, needs guidance on parameters
- **Analysis:** Recognizes obvious overfitting, may choose suboptimal regularization
- **Communication:** Explains concepts in technical terms but struggles with analogies

### **Level 1 - Beginning (60-69%)**
- **Conceptual Understanding:** Confused about regularization purpose and effects
- **Mathematical Skills:** Significant calculation errors, formula confusion
- **Implementation:** Code issues, incorrect regularization application
- **Analysis:** Cannot reliably identify overfitting or choose appropriate methods
- **Communication:** Cannot explain concepts clearly or use analogies effectively

---

# 🎯 QUICK ASSESSMENT TOOLS

## 5-Minute Exit Ticket

**Complete these sentences:**

1. "Overfitting is like a student who _____________ because _____________"
2. "L1 regularization is like Marie Kondo because it _____________"
3. "The main difference between L1 and L2 regularization is _____________"
4. "If I see training accuracy 95% and validation accuracy 70%, I should _____________"
5. "One thing I'm still confused about is _____________"

## Red Light / Green Light Check

**Show green (understood) or red (confused) for each concept:**

- ✅/❌ Bias-variance tradeoff
- ✅/❌ Overfitting detection from learning curves
- ✅/❌ L1 regularization creates sparsity
- ✅/❌ L2 regularization handles correlated features
- ✅/❌ How to choose λ (lambda) values
- ✅/❌ TensorFlow implementation of regularization

## Peer Teaching Assessment

**Partner with someone and explain these topics using analogies:**
1. Person A explains overfitting using the "student cramming" analogy
2. Person B explains L1 regularization using the "Marie Kondo" analogy
3. Switch and Person A explains L2 using "equal opportunity employer"
4. Person B explains bias-variance using "restaurant chef" analogy

**Assessment criteria:** Can your partner understand without ML background?

---

# 📚 HOMEWORK ASSESSMENT

## Tutorial T6 Implementation Checklist

**Required implementations with assessment criteria:**

### **Task H1: Basic Regularization Implementation (40%)**
- [ ] Correctly implements L1 regularization in TensorFlow
- [ ] Correctly implements L2 regularization in TensorFlow
- [ ] Uses appropriate λ values (0.001-0.1 range)
- [ ] Compares performance with unregularized baseline

### **Task H2: Hyperparameter Tuning (30%)**
- [ ] Tests at least 5 different λ values
- [ ] Uses proper validation methodology (train/val/test split)
- [ ] Plots performance vs λ curves
- [ ] Identifies optimal λ for both L1 and L2

### **Task H3: Analysis and Interpretation (20%)**
- [ ] Analyzes weight sparsity patterns for L1
- [ ] Compares weight distributions between L1 and L2
- [ ] Correctly interprets overfitting/underfitting patterns
- [ ] Provides justified recommendations for real-world use

### **Task H4: Communication (10%)**
- [ ] Uses class analogies in explanations
- [ ] Code is well-commented and readable
- [ ] Explains results in plain English
- [ ] Connects findings to practical applications

---

# 🔄 CONTINUOUS ASSESSMENT PLAN

## Week 6 Assessment Timeline

### **Day 3 (Today):**
- Immediate feedback polls during lecture
- Exit ticket for understanding check
- Peer teaching exercises

### **Day 4:**
- Quick quiz on L1/L2 concepts (5 minutes)
- Practical implementation check
- Tutorial T6 progress review

### **Week 7:**
- Application of regularization in image processing tasks
- Integration with CNN architectures

### **Unit Test 1 (Sep 19):**
- 20% of questions from regularization topics
- Mix of calculation, conceptual, and application questions
- Include analogy-based explanations

---

# 🎓 LEARNING OUTCOMES ASSESSMENT

## Course Outcome Alignment

### **CO-1: Create simple deep neural networks**
**Assessment Evidence:**
- Students implement regularized networks in TensorFlow ✅
- Students choose appropriate regularization for given scenarios ✅
- Students debug and fix regularization implementation issues ✅

### **CO-2: Build neural networks with multiple layers with appropriate activations**
**Assessment Evidence:**
- Students apply regularization across multiple layers ✅
- Students understand how regularization affects layer weights ✅
- Students optimize multi-layer networks using regularization ✅

### **PO-1: Engineering Knowledge (Level 3)**
**Assessment Evidence:**
- Mathematical calculations of L1/L2 penalties ✅
- Understanding of bias-variance tradeoff mathematics ✅
- Application of optimization principles to regularization ✅

### **PO-2: Problem Analysis (Level 2)**
**Assessment Evidence:**
- Diagnosis of overfitting from learning curves ✅
- Selection of appropriate regularization for different scenarios ✅
- Analysis of regularization effects on model performance ✅

---

## 📋 INSTRUCTOR ASSESSMENT CHECKLIST

### **Pre-Class Preparation:**
- [ ] Review student background questionnaires for regularization knowledge
- [ ] Prepare interactive polling system (Kahoot, Mentimeter, etc.)
- [ ] Set up demonstration code and data
- [ ] Print assessment rubrics and checklists

### **During Class Monitoring:**
- [ ] Use polls to check understanding throughout lecture
- [ ] Monitor hands-on exercise completion rates
- [ ] Identify students struggling with implementation
- [ ] Note common misconceptions for addressing

### **Post-Class Assessment:**
- [ ] Review exit ticket responses for understanding gaps
- [ ] Analyze poll results to identify challenging concepts
- [ ] Plan remediation for next session based on feedback
- [ ] Update Tutorial T6 expectations based on class performance

### **Follow-up Actions:**
- [ ] Provide individual feedback on homework submissions
- [ ] Create additional practice materials for struggling students
- [ ] Prepare enhanced explanations for difficult concepts
- [ ] Plan Unit Test 1 questions based on demonstrated understanding

---

**🎯 Assessment Philosophy:** *"Assess to improve learning, not just to measure it."*

*© 2025 Prof. Ramesh Babu | SRM University | Deep Neural Network Architectures*
*"Evidence-Based Teaching Through Continuous Assessment"*