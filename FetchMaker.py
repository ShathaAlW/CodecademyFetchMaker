import numpy as np
import pandas as pd
import scipy.stats as stat
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt


dogs = pd.read_csv('dog_data.csv')
sig_threshold = 0.05

# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

# the number of whippet rescued dogs
whippet_rescue = dogs.is_rescue[dogs.breed == 'whippet_rescue']
num_whippet_rescues = np.sum(whippet_rescue)
print('Number of whippet rescued dogs:')
print(num_whippet_rescues)

# the number of whippets in the data sample
num_whippets = np.sum(dogs.breed == 'whippet')
print('Number of whippets in dataset:')
print(num_whippets)


# to test, null: 8% of of whippets are rescues. alternative: more or less than 8% of whippets are rescues
pval = stat.binom_test(0, 100, p = 0.80)
result = ('There is a significant difference in whippets rescued rate and 8% expected dog rescue rate' if pval < sig_threshold else 'There is NO significant difference in whippets rescued rate and 8% expected dog rescue rate')
print(result, 'p-value:', pval)


#  extract the weights of whippets, terrirs and pitbulls from data
wt_whippets = dogs.weight[dogs.breed == 'whippet']
wt_terriers = dogs.weight[dogs.breed == 'terrier']
wt_pitbulls = dogs.weight[dogs.breed == 'pitbull']

# to test if there is a significant difference in the average weights of three dog breeds (whippets, terriers and pitbulls)
fstat, pval = stat.f_oneway(wt_whippets, wt_terriers, wt_pitbulls)
result = ('There is a significant weight average difference between at least one pair of dogs' if pval < sig_threshold else 'There is NO significant weight difference. The three dog breeds weigh the same amount on average')
print(result, 'p-value:', pval)


# to determine which pair(s) of dogs has/have the significat difference in average weight
tukey_result = pairwise_tukeyhsd(dogs_wtp.weight,dogs_wtp.breed, 0.05)
print(tukey_result)
# two pairs of dogs showed a signicant difference in average weight, pitbulls & terriers and terriers & whippets


# boxplot of weights of each breed type
ax = sns.boxplot(x= dogs.breed, y= dogs.weight)
ax.set_xticklabels(ax.get_xticklabels(), rotation= 90)
plt.title('Weight of rescued dogs by breed')
plt.show()

# to check the range of dog weights
avg_weights = []
for dog in dogs.breed:
  avg_wt = np.mean(dogs.weight[dogs.breed == dog])
  if not avg_wt in avg_weights:
    avg_weights.append(avg_wt)
print('Range of dog weights (lbs):', min(avg_weights) ,'-', max(avg_weights))


# to test the association between breed (poodle vs. shihtzu) and color
Xtab = pd.crosstab(dogs_ps.breed, dogs_ps.color)
print(Xtab)
chi2, pval, dof, expected = stat.chi2_contingency(Xtab)
result = ('There is a significant association between dog breed (poodle vs. shihtzu) and color' if pval < sig_threshold else 'There is NO association between breed (poodle vs. shihtzu) and color')
print(result, 'p-value:', pval)


# to test, if there is an association between dog breed and being hypoallergenic or not
Ytab = pd.crosstab(dogs.breed, dogs.is_hypoallergenic)
print(Ytab)
chi2, pval, dof, expected = stat.chi2_contingency(Ytab)
result = ('There is a significant association between dog breed and being hypoallergenic' if pval < sig_threshold else 'There is NO association between dog breed and being hypoallergenic')
print(result, 'p-value:', pval)

# barplot of dog breeds and being hypoallergenic or not
plt.clf()
ax2 = sns.barplot(dogs.breed, dogs.is_hypoallergenic)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation= 90)
ax2.set(title='Dog breed vs. Hypoallergenic', xlabel= 'breed', ylabel= 'hypoallergy')
plt.show()
