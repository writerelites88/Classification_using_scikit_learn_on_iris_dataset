from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
print('Class lebles: ',np.unique(y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc=StandardScaler()
sc.fit(X_train, y_train)
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Create a perceptron object and train it

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train, y_train)

# Test the model

y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (y_test!= y_pred).sum())
print('Accuracy: %.2f' % ppn.score(X_test, y_test))
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Visualize the decision boundaries

# First, train a new perceptron using only 2 features for visualization
X_viz = X[:, [0, 1]]  # Using only sepal length and sepal width
X_train_viz, X_test_viz, y_train_viz, y_test_viz = train_test_split(
    X_viz, y, test_size=0.3, random_state=0, stratify=y
)

# Scale the visualization features
sc_viz = StandardScaler()
X_train_viz = sc_viz.fit_transform(X_train_viz)
X_test_viz = sc_viz.transform(X_test_viz)

# Train a separate perceptron for visualization
ppn_viz = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn_viz.fit(X_train_viz, y_train_viz)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Plot contours
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                   y=X[y == cl, 1],
                   alpha=0.8, 
                   c=[cmap(idx)],
                   marker=markers[idx], 
                   label=f'Class {cl}')
    
    # Highlight test samples if provided
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                   c='none', 
                   alpha=1.0,
                   linewidth=1, 
                   marker='o',
                   s=55, 
                   label='Test set')

# Create visualizations
plt.figure(figsize=(15, 5))

# Plot 1: Decision boundary for first two features
plt.subplot(121)
X_combined_viz = np.vstack((X_train_viz, X_test_viz))
y_combined = np.hstack((y_train_viz, y_test_viz))
plot_decision_regions(X_combined_viz, y_combined, classifier=ppn_viz, 
                     test_idx=range(len(y_train_viz), len(y_combined)))
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Sepal width [standardized]')
plt.title('Decision Boundary (Sepal length vs Sepal width)')
plt.legend(loc='upper left')

# Plot 2: Feature importance/weights for all features
plt.subplot(122)
plt.bar(range(X.shape[1]), abs(ppn.coef_[0]), alpha=0.8)
plt.xticks(range(X.shape[1]), iris.feature_names)
plt.xlabel('Features')
plt.ylabel('Absolute weight magnitude')
plt.title('Feature Importance (All Features)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print accuracies for comparison
print("\nAccuracy with all 4 features:", accuracy_score(y_test, y_pred))
print("Accuracy with 2 features:", accuracy_score(y_test_viz, ppn_viz.predict(X_test_viz)))
