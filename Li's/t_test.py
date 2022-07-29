
from scipy.stats import levene, ttest_ind

x = [1,2,4,7,2,4]
y = [3,2,5,7,4]

if levene(x,y)[1] > 0.05:
    res = ttest_ind(x,y)
else:
    res = ttest_ind(x,y,equal_var=False)
print(res)
print(res[1])

