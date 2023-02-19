```python
%%html 
<style type='text/css'>p {font-size: 18px; font-family: "Monaco"}</style>
```


<style type='text/css'>p {font-size: 18px; font-family: "Monaco"}</style>



# Estimating Time Saved by a 5% better Student Model.

&emsp; Knowledge tracers are typically fit to student data in the form of seqeuences of binary values indicating whether or not a student performed correctly at a particular practice opportunity. These datasets are typically fit using statistical models like Baysian Knowledge Tracing (a kind of hidden Markov Model), mixed effect logistic regression models like the Additive Factors Model (AFM), or with deep-learning (i.e. DKT). These models predict the probability that a student will get their next problem attempt correct. In this calculation we'll assume to know the ground-truth probabilities---an assumption which allows us to analytically calculate the expected Mean Square Error between a candidate model and the ground-truth one.

## Calculating Expected RMSE 

The Mean-Square Error (MSE) for these models is typically expressed as the Brier Score:
    
$BS = MSE = \frac{1}{N}\sum_{t=1}^{N}(\hat{p}_t-o_t)^2$
    
Which compares the individual continuous probability predicted by the model $\hat{p}_t$ with the real 0 or 1 observations $o_t$.

For binary random variable $X$ with ground-truth probability $P(X=1) = p$ and a model prediction $\hat{p}$ we can express the expected MSE as:

$E[MSE] = E[(\hat{p}-X)^2]$

$= E[\hat{p}^2] - E[2\hat{p}X] + E[(X)^2]$

$= \hat{p}^2 - 2\hat{p}E[X] + \frac{1}{N}\sum[1^2p + 0^2(1-p)]$

$= \hat{p}^2 - 2\hat{p}p + p$

$E[RMSE] = \sqrt{\hat{p}^2 - 2\hat{p}p + p}$

(alternative derivation: https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/met.21)

Let's define Python functions for the analytical MSE and RMSE, plus some helper functions to help translate between log-odds and probabilities:


```python
import numpy as np
# Log Likelihood
ll = lambda p : np.log(p/(1-p))
# Sigmoid: i.e. inverse log-likelihood
sig = lambda l : 1/(1+np.exp(-l))

# E[MSE] (i.e. Expected Brier Score) Expressed as E[(_p-X)**2] => _p**2 + p - 2*_p*p
E_mse = lambda p,_p : _p**2 + p - 2*_p*p
E_rmse = lambda p,_p : np.sqrt(E_mse(p,_p))
```

## The Calculating Expect Time Saved

### Find the old 5% worse threshold intercept 


```python
# A mastery threshold intercept of 95%
thresh = .95
old_p_at_thresh = .8785 #Here we're assuming an underestimate of mastery

# Ensure this choice the old threshold intercept has 5% worse RMSE than the new generating model.
assert np.abs(E_rmse(thresh, old_p_at_thresh)*.95-E_rmse(thresh,thresh))  < 1e-4
```

### Set Ground-Truth Parameters

Below we're assuming that the model is a logistic regression model like AFM. We'll set a ground-truth intercept `new_intr` and slope `new_slope`. Since we're doing a simple back-of-the-napkin sort of calculation these two parameters are considerably simplified compared to the typical parameterization where there is a per-student intercept paramater, and a per KC intercept and slope. With just two paramers we're only modeling an average set of parameters consistent with the properties of a typical dataset.


```python
# Ground-truth parameters (1) stu intercept .35 (2) avg 12 problems to mastery
intr_p = .35
avg_probs = 12

#Model ground-truth model parameters  
new_intr = ll(intr_p)
new_slope = (ll(thresh)-new_intr)/avg_probs
```

### Calculate the parameters of the old model running through this intercept


```python
def gen_old_model_params(old_p_at_thresh):
    '''Generates the intercept and slope for the 'old' model assuming a particular
       predicted probability of the old model after 'avg_probs' '''
    # The probability difference at the threshold intercept
    p_diff = thresh-old_p_at_thresh

    # Half the improvement from old to new was due to intercept change
    old_intr = .5 * ll(intr_p-p_diff) + \
               .5 * new_intr
    # Adjust the slope to account for the rest of the change
    old_slope = (ll(old_p_at_thresh)-old_intr)/avg_probs 
    
    # Ensure that these parameters reproduce 'old_p_at_thresh'
    assert np.isclose(sig(old_intr+avg_probs*old_slope), old_p_at_thresh)
    return old_intr, old_slope

old_intr, old_slope = gen_old_model_params(old_p_at_thresh)
print(f"intercept old/new: {old_intr:.3f}  {new_intr:.3f}")
print(f"slope old/new: \t    {old_slope:.3f}   {new_slope:.3f}")
```

    intercept old/new: -0.785  -0.619
    slope old/new: 	    0.230   0.297


### Then Interpolate the time saved by the new model 


```python
avg_probs_old = (ll(thresh)-old_intr)/old_slope
print(f"avg_extra_probs_old_model: {avg_probs_old-avg_probs:.2f}")

attempt_length = 15 #seconds
n_kcs = 500
total_time_saved = (avg_probs_old-avg_probs)*n_kcs*attempt_length
total_time = n_kcs*attempt_length*avg_probs
print(f"Total Time Saved: {total_time_saved/60:.2f} minutes or {total_time_saved/3600:.2f} hours. ({100*total_time_saved/total_time:.2f})% of {total_time/3600} total hours")
```

    avg_extra_probs_old_model: 4.19
    Total Time Saved: 524.36 minutes or 8.74 hours. (34.96)% of 25.0 total hours


&emsp; In total a the 5% better model gives us a rather large expected time saving of 35%. 

## What about whole model RMSE?

&emsp; We can do a similar calculation for the RMSE of the whole model (not just at the threshold point). However, the RMSE outside of the neighborhood of the mastery threshold has no bearing on the problem selection behavior of a knowledge-tracer. Nonetheless in this case if we tweak the parameters so that the RMSE difference of the whole model is 5% then, we get a similar RMSE difference in the neighborhood of the mastery threshold. It's hard to say if this is true in general or  a quirk of comparing two versions of the same model: one with ground-truth parameters and one with perturbed parameters. The more typical comparison would be to compare two models with different choices of independant variables, like Performance Factors Analysis (PFA) versus Additive Factors Model (AFM), or a different model variety entirely like Baysian Knowledge Tracing (BKT) or Deep Knowledge Tracing (DKT).  The reality is that we really ought to be in the habit of reporting model performance statistics around reasonable threshold choices (i.e. 85%, 90%, 95%).


```python
def E_model_rmse(intr, slope, opp_cutoff=15):
    ''' Given intr and slope generate the expected RMSE of the model compared to ground-truth'''
    p_ground_truth = lambda i : sig(new_intr+i*new_slope)
    p_model = lambda i : sig(intr+i*slope)
    mse_s = []
    for i in range(opp_cutoff+1):
        mse_s.append(np.mean([E_mse(p_ground_truth(i), p_model(i)) ]))
    return np.mean(mse_s)

old_p_at_thresh = .8797
old_intr, old_slope = gen_old_model_params(old_p_at_thresh)

assert np.abs(E_model_rmse(old_intr, old_slope)*.95-E_model_rmse(new_intr,new_slope)) < 1e-4

print(f"Actual RMSE difference at mastery threshold: {100*(1-E_rmse(thresh, thresh)/E_rmse(thresh, old_p_at_thresh)):.2f}%" )

print(f"old/new intr.: \t{old_intr:.3f}  {new_intr:.3f}")
print(f"old/new slope: \t{old_slope:.3f}  {new_slope:.3f}")

avg_probs_old = (ll(thresh)-old_intr)/old_slope
print(f"avg_extra_probs_old_model: {avg_probs_old-avg_probs:.2f}")

attempt_length = 15 #seconds
n_kcs = 500
total_time_saved = (avg_probs_old-avg_probs)*n_kcs*attempt_length
print(f"Total Time Saved: {total_time_saved/60:.2f} minutes or {total_time_saved/3600:.2f} hours. ({100*total_time_saved/total_time:.2f})% of {total_time/3600} total hours")
```

    Actual RMSE difference at mastery threshold: 4.83%
    old/new intr.: 	-0.782  -0.619
    old/new slope: 	0.231  0.297
    avg_extra_probs_old_model: 4.13
    Total Time Saved: 516.68 minutes or 8.61 hours. (34.45)% of 25.0 total hours



```python
from numpy.random import random as rand

# Sanity Check : Does analytic expected RMSE match an empirically generate one
N = 10000000
print(f'{np.sqrt(np.mean( np.square( (rand(N)<=.95).astype(np.float64) - .8785 ))):.5f}')
print(f'{E_rmse(thresh, .8785):.5f}')
    
```

    0.22959
    0.22937

