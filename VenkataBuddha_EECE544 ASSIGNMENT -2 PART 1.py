#!/usr/bin/env python
# coding: utf-8

# # HW2-PART1
# ### Venkata Buddha Work

# In[49]:


# Widget to manipulate plots in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'widget')

from sys import float_info  # Threshold smallest positive floating value

import matplotlib.pyplot as plt # For general plotting

from math import ceil, floor 

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


# ## Utility Functions

# In[50]:


def generate_data_from_gmm(N, pdf_params, L=None, fig_ax=None): #where L = A list of the labels for each class distribution
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['m'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N)

    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes

    if L.all() == None: # Default, no classes specified
        L = np.array(range(1, len(pdf_params['priors'])+1))

    for distribution in np.array(range(0, L.size)):
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[distribution] <= u) & (u <= thresholds[distribution+1]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)  
        labels[indices] = L[distribution]
        X[indices, :] =  multivariate_normal.rvs(pdf_params['m'][distribution], pdf_params['C'][distribution], Nl)
    
    return X, labels


# ## Evaluation Functions

# In[51]:


# Generate ROC curve samples
def estimate_roc(discriminant_score, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)

    # Use gamma values that will account for every possible classification split
    # The epsilon is just to account for the two extremes of the ROC curve (TPR=FPR=0 and TPR=FPR=1)
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]
    
    # Retrieve indices where FPs occur
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    # Compute FP rates (FPR) as a fraction of total samples in the negative class
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    # Retrieve indices where TPs occur
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    # Compute TP rates (TPR) as a fraction of total samples in the positive class
    p11 = [len(inds) / N_labels[1] for inds in ind11]
    
    # ROC has FPR on the x-axis and TPR on the y-axis, but return others as well for convenience
    roc = {}
    roc['p10'] = np.array(p10)
    roc['p11'] = np.array(p11)

    return roc, gammas

def get_binary_classification_metrics(predictions, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))

    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)
    class_metrics = {}
    
    # True Negative Probability Rate
    ind_00 = np.argwhere((predictions == 0) & (labels == 0))
    class_metrics['TNR'] = len(ind_00) / N_labels[0]
    # False Positive Probability Rate
    ind_10 = np.argwhere((predictions == 1) & (labels == 0))
    class_metrics['FPR'] = len(ind_10) / N_labels[0]
    # False Negative Probability Rate
    ind_01 = np.argwhere((predictions == 0) & (labels == 1))
    class_metrics['FNR'] = len(ind_01) / N_labels[1]
    # True Positive Probability Rate
    ind_11 = np.argwhere((predictions == 1) & (labels == 1))
    class_metrics['TPR'] = len(ind_11) / N_labels[1]

    return class_metrics


# ## Solution
# 
# #### Given
# 
# ![ASSIGN2-QUESTION%201.JPG](attachment:ASSIGN2-QUESTION%201.JPG)

# In[52]:


# Generate dataset from three different distributions/categories
gmm_pdf = {}

# Class priors, splitting the first class into two halves
gmm_pdf['priors'] = np.array([0.325, 0.325, 0.35])
# True priors
gmm_pdf['true priors'] = np.array([0.65, 0.35])

# Mean and covariance of data pdfs conditioned on labels
gmm_pdf['m'] = np.array([[3, 0],
                         [0, 3],
                         [2, 2]])  # Gaussian distributions means
gmm_pdf['C'] = np.array([[[2, 0],
                          [0, 1]],
                         [[1, 0],
                          [0, 2]],
                         [[1, 0],
                          [0, 1]]])  # Gaussian distributions covariance matrices


# In[53]:


# Initialize dictionary and figure
datasets = {}
fig = plt.figure(figsize=(10, 10))

L = np.array([0, 0, 1]) # A list of the labels for each class distribution
N_sets = np.array([20, 200, 2000, 10000]) # Size of each dataset to be genreated

# Generate the data for each specified set size:
for i in np.array(range(0, N_sets.size)):
    N = N_sets[i] # Get number of samples
    
    X, labels = generate_data_from_gmm(N, gmm_pdf, L) # Generate data based on size, distribution, and labels

    # Plot the original data and their true labels

    ax_raw = fig.add_subplot(2, 2, i+1)
    ax_raw.scatter(X[labels == 0, 0], X[labels == 0, 1], s=8, label="Class 0")
    ax_raw.scatter(X[labels == 1, 0], X[labels == 1, 1], s=8, label="Class 1")
    ax_raw.set_xlabel(r"$x_1$")
    ax_raw.set_ylabel(r"$x_2$")
    ax_raw.set_aspect('equal', 'box') # Set equal axes for 3D plots

    plt.title("Data for N = {} samples".format(N))
    plt.legend()
    plt.tight_layout()

    # Store into dictionary:
    datasets['D'+str(N)] = {'N': N, 'X': X, 'labels': labels}


# ## Part 1
# Determine the theoretically optimal classifier that achieves minimum probability of error using knowledge of the true pdf. Specify the classifier mathematically and implement
# it; then apply it to all samples in D10K valid. From the decision results and true labels for this validation
# set, estimate and plot the ROC curve of this min-Pr(error) classifier. Report the optimal threshold
# and probability error estimate of the theoretical min-Pr(error) classifier, indicating on the ROC
# curve with a special marker its location. Also report the empirical threshold and associated minimum probability of error estimate for this classifier based on counts of decision-truth label pairs
# on D10K valid.
# 
# The theoretically optimal classifier for the dataset will be MAP, because we have true knowledge of the pdf.

# In[54]:


valid_set = 'D10000'

# Compute class conditional likelihoods to express ratio test, where ratio is discriminant score


class_conditional_likelihoods_0 = np.array([multivariate_normal.pdf(datasets[valid_set]['X'], gmm_pdf['m'][0], gmm_pdf['C'][0])]) + np.array([multivariate_normal.pdf(datasets[valid_set]['X'], gmm_pdf['m'][1], gmm_pdf['C'][1])])
class_conditional_likelihoods_1 = np.array([multivariate_normal.pdf(datasets[valid_set]['X'], gmm_pdf['m'][2], gmm_pdf['C'][2])])

class_conditional_likelihoods = np.append(class_conditional_likelihoods_0, class_conditional_likelihoods_1, axis =0)

print(class_conditional_likelihoods)

# Class conditional log likelihoods equate to decision boundary log gamma in the 0-1 loss case
discriminant_score_erm = np.log(class_conditional_likelihoods[1]) - np.log(class_conditional_likelihoods[0])

# Construct the ROC for ERM by changing log(gamma)
roc_erm, gammas_empirical = estimate_roc(discriminant_score_erm, labels)

plt.ioff() # Interactive plotting off
fig_roc, ax_roc = plt.subplots(figsize=(10, 10));
plt.ion()

ax_roc.plot(roc_erm['p10'], roc_erm['p11'], label="Empirical ERM Classifier ROC Curve")
ax_roc.set_xlabel(r"Probability of False Alarm $P(D=1|L=0)$")
ax_roc.set_ylabel(r"Probability of True Positive $P(D=1|L=1)$")

plt.grid(True)
display(fig_roc)
fig_roc;


# In[55]:


N_per_l = np.array([sum(labels == l) for l in [0,1]])

# ROC returns FPR vs TPR, but prob error needs FNR so take 1-TPR

prob_error_empirical = np.array((roc_erm['p10'], 1 - roc_erm['p11'])).T.dot(N_per_l / datasets[valid_set]['N'])

# Min prob error for the empirically-selected gamma thresholds
min_prob_error_empirical = np.min(prob_error_empirical)
min_ind_empirical = np.argmin(prob_error_empirical)

# Compute theoretical gamma as log-ratio of true priors (0-1 loss) -> MAP classification rule
gamma_map = gmm_pdf['true priors'][0]/gmm_pdf['true priors'][1]
decisions_map = discriminant_score_erm >= np.log(gamma_map)

class_metrics_map = get_binary_classification_metrics(decisions_map, labels)
# To compute probability of error, we need FPR and FNR
min_prob_error_map = np.array((class_metrics_map['FPR'] * gmm_pdf['true priors'][0] + 
                               class_metrics_map['FNR'] * gmm_pdf['true priors'][1]))

# Plot theoretical and empirical
ax_roc.plot(roc_erm['p10'][min_ind_empirical], roc_erm['p11'][min_ind_empirical], 'gx', label="Empirical Min P(Error) ERM",
            markersize=14)
ax_roc.plot(class_metrics_map['FPR'], class_metrics_map['TPR'], 'ro', label="Theoretical Min P(Error) ERM", markersize=14)
plt.legend()


# ##### The Final results and ROC curve are:

# In[56]:


print("Min Empirical Pr(error) for ERM = {:.3f}".format(min_prob_error_empirical))
print("Min Empirical Gamma = {:.3f}".format(np.exp(gammas_empirical[min_ind_empirical])))

print("Min Theoretical Pr(error) for ERM = {:.3f}".format(min_prob_error_map))
print("Min Theoretical Gamma = {:.3f}".format(gamma_map))

display(fig_roc)

