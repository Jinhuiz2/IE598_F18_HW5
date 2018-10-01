import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score





#Setup datasets and some data analysis
print(60 * '=')
print('Part 1: Exploratory Data Analysis')
print(60 * '=')
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()



# Splitting the data into 70% training and 30% test subsets.
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y,
                                                    random_state=0)


print('Dataset excerpt:\n\n', df_wine.head())


#############################################################################

sns.set(style='whitegrid', context='notebook')
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

sns.pairplot(df_wine[cols], size=2.5)
# plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()


#heatmap
cm = np.corrcoef(df_wine.values.T)
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(18,18)) 
hm = sns.heatmap(cm,
            cbar=True,
            annot=True,
            square=True,
           fmt='0.09f',
            annot_kws={'size': 10},
           yticklabels=df_wine.columns,
            xticklabels=df_wine.columns, ax=ax)
plt.tight_layout()
plt.show()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values




# Standardizing the data.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#define a function to draw decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)



#Logistic and SVM classifier - baseline
print(60 * '=')
print('Part 2: Logistic regression classifier v. SVM classifier - baseline')
print(60 * '=')

svm = SVC(kernel='linear', C=1.0, random_state=1)
# Training logistic regression classifier baseline
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Training SVM classifier baseline
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm = svm.fit(X_train, y_train)

y_svm_train_pred = svm.predict(X_train)
y_svm_test_pred = svm.predict(X_test)
print('Accuracy of baseline LR train set:')
print( metrics.accuracy_score(y_train, y_lr_train_pred) )
print('Accuracy of baseline LR test set:')
print( metrics.accuracy_score(y_test, y_lr_test_pred) )
print('Accuracy of baseline SVM train set:')
print( metrics.accuracy_score(y_train, y_svm_train_pred) )
print('Accuracy of baseline SVM test set:')
print( metrics.accuracy_score(y_test, y_svm_test_pred) )





#Logistic and SVM classifier on PCA transformed datasets
print(60 * '=')
print('Part 3: Perform a PCA on both datasets')
print(60 * '=')



# Eigendecomposition of the covariance matrix.
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)


# ## Total and explained variance
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

# Select the 2 largest eigenvalues and their eigenvectors to form tramformation metrix W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('\n')
print('Matrix W:\n', w)


#Projecting samples onto the new feature space
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()                
        
# Training logistic regression classifier using the first 2 principal components.
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
y_pca_lr_train_pred = lr.predict(X_train_pca)
y_pca_lr_test_pred = lr.predict(X_test_pca)

# Training SVM classifier using the first 2 principal components.
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm = svm.fit(X_train_pca, y_train)
y_pca_svm_train_pred = svm.predict(X_train_pca)
y_pca_svm_test_pred = svm.predict(X_test_pca)

#PCA LR train plot and accuracy scores
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('PCA LR train')
plt.show()
print('Accuracy of PCA LR train set:')
print( metrics.accuracy_score(y_train, y_pca_lr_train_pred) )



#PCA LR test plot and accuracy scores
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('PCA LR test')
plt.show()
print('Accuracy of PCA LR test set:')
print( metrics.accuracy_score(y_test, y_pca_lr_test_pred) )

#PCA SVM train plot and accuracy scores
plot_decision_regions(X_train_pca, y_train, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('PCA SVM train')
plt.show()
print('Accuracy of PCA SVM train set:')
print( metrics.accuracy_score(y_train, y_pca_svm_train_pred) )

#PCA SVM test plot and accuracy scores
plot_decision_regions(X_test_pca, y_test, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('PCA SVM test')
plt.show()
print('Accuracy of PCA SVM test set:')
print( metrics.accuracy_score(y_test, y_pca_svm_test_pred) )




#Logistic and SVM classifier on LDA transformed datasets
print(60 * '=')
print('Part 4: Perform a LDA on both datasets')
print(60 * '=')

# Compute the mean vectors for each class:
np.set_printoptions(precision=4)
mean_vecs = []
print('Mean vector of each 3 class')
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))


# Compute the within-class scatter matrix:
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices
print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# Better: covariance matrix since classes are not equally distributed:
print('Class label distribution: %s' 
      % np.bincount(y_train)[1:])
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                     S_W.shape[1]))


# Compute the between-class scatter matrix:
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
print('')



# Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# Sort eigenvectors in descending order of the eigenvalues:
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\nEigenvalues in descending order:')
for eigen_val in eigen_pairs:
    print(eigen_val[0])



tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()



#get the d*k tranformation matrix W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)



#Projecting samples onto the new feature space
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()






# Training logistic regression classifier using the first 2 principal components.
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
y_lda_lr_train_pred = lr.predict(X_train_lda)
y_lda_lr_test_pred = lr.predict(X_test_lda)
# Training SVM classifier using the first 2 principal components.
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm = svm.fit(X_train_lda, y_train)
y_lda_svm_train_pred = svm.predict(X_train_lda)
y_lda_svm_test_pred = svm.predict(X_test_lda)


#LDA LR train plot and accuracy scores
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('LDA LR train')
plt.show()
print('Accuracy of LDA LR train set:')
print( metrics.accuracy_score(y_train, y_lda_lr_train_pred) )

#LDA LR test plot and accuracy scores
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('LDA LR test')
plt.show()
print('Accuracy of LDA LR test set:')
print( metrics.accuracy_score(y_test, y_lda_lr_test_pred) )

#LDA SVM train plot and accuracy scores
plot_decision_regions(X_train_lda, y_train, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('LDA SVM train')
plt.show()
print('Accuracy of LDA SVM train set:')
print( metrics.accuracy_score(y_train, y_lda_svm_train_pred) )

#LDA SVM test plot and accuracy scores
plot_decision_regions(X_test_lda, y_test, classifier=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('LDA SVM test')
plt.show()
print('Accuracy of LDA SVM test set:')
print( metrics.accuracy_score(y_test, y_lda_svm_test_pred) )





#Logistic and SVM classifier on kPCA transformed datasets
print(60 * '=')
print('Part 5: Perform a kPCA on both datasets')
print(60 * '=')
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kpca = kpca.fit_transform(X_train_std)
X_test_kpca = kpca.transform(X_test_std)

svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
svm.fit(X_train_kpca, y_train)

kpca_train_pred = svm.predict(X_train_kpca)
kpca_test_pred = svm.predict(X_test_kpca)
kpca_train_score = accuracy_score(y_train, kpca_train_pred)
kpca_test_score = accuracy_score(y_test, kpca_test_pred)
print('KPCA train/test accuracies %f/%f'
      % (kpca_train_score, kpca_test_score))

gammas = [0.01, 0.05, 0.25, 0.30, 0.45, 0.6, 0.75, 0.9, 1., 1.15, 1.30, 1.45, 1.6, 1.75, 2]
kpca_lr_train_scores = []
kpca_lr_test_scores = []
kpca_svm_train_scores = []
kpca_svm_test_scores = []

for i in range(0,15):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gammas[i])
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    kpca_lr_train_pred = svm.predict(X_train_kpca)
    kpca_lr_test_pred = svm.predict(X_test_kpca) 
    lr = LogisticRegression()
    lr.fit(X_train_kpca, y_train)

    print("Gamma: ", gammas[i])
    kpca_lr_train_scores.append(accuracy_score(y_train, kpca_lr_train_pred))
    kpca_lr_test_scores.append(accuracy_score(y_test, kpca_lr_test_pred))
    print('kPCA LR train accuracies %f' % (kpca_lr_train_scores[i]))
    print('kPCA LR test accuracies %f'% (kpca_lr_test_scores[i]))
    
    svm = SVC(kernel = 'linear', C= 1.0, random_state= 42)
    svm.fit(X_train_kpca, y_train)
    kpca_svm_train_pred = svm.predict(X_train_kpca)
    kpca_svm_test_pred = svm.predict(X_test_kpca)
    
    kpca_svm_train_scores.append(accuracy_score(y_train, kpca_svm_train_pred))
    kpca_svm_test_scores.append(accuracy_score(y_test, kpca_svm_test_pred))
    print('kPCA SVM train accuracies %f' % (kpca_svm_train_scores[i]))
    print('kPCA SVM test accuracies %f'% (kpca_svm_test_scores[i]))

plt.plot(gammas, kpca_lr_train_scores, label='kPCA LR training')
plt.plot(gammas, kpca_lr_test_scores, label='kPCA LR test')
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.legend()
plt.title("kPCA_LR accuracy scores")
plt.show()

plt.plot(gammas, kpca_svm_train_scores, label='kPCA SVM training')
plt.plot(gammas, kpca_svm_test_scores, label='kPCA SVM test')
plt.xlabel('gamma')
plt.ylabel('accuracy score')
plt.legend()
plt.title("kPCA_SVM accuracy scores")
plt.show()
    
print("My name is Jinhui Zhang")
print("My NetID is: jinhuiz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






