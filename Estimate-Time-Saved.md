```python
%%html 
<style type='text/css'>p {font-size: 18px; font-family: "Monaco"}</style>
```


<style type='text/css'>p {font-size: 18px; font-family: "Monaco"}</style>



&emsp; Knowledge tracers are predictors of individual students' mastery of fine-grained learning objectives called knowledge components. Knowledge tracers make this prediction on the basis of student's prior performance on practice or assessment problems, and serve the purpose determining when students should continue practicing problems of certain types or move on to new content. Knowledge tracers are typically fit to existing student data that has been reduced to seqeuences of binary values indicating whether particular practice opportunities resulted in a correct first attempt response. Knowledge tracers and can come in a variety of types including Baysian Knowledge Tracing (a kind of hidden Markov Model)[2], mixed effect logistic regression models like the Additive Factors Model (AFM) [1], and deep-learning based methods like DKT[4]. Although the original DKT work in fact makes a different sort of prediction---pure item-by-item prediction---which is distinct from the kind of prediction required of a knowledge tracer, which must have a characterization of item to KC mappings so that items associated with particular KCs can be omitted from future practice.

# Estimating Time Saved by a 5% better Student Model.

&emsp; In the following calculation we'll attempt to quantify the estimated time saved if we improved the fit of a knowledge tracer by 5% in terms of Root Mean Square Error. First we'll analytically calculate the difference in prediction probability an intial knowledge tracer and a 5% better one (which we'll assume to be the ground truth model). Then we'll use this prediction difference to interplote the extraneous practice time imposed on students by the 5% worse model (assuming it was underpredicting mastery).

## Calculating Expected RMSE 

The Mean-Square Error (MSE) of a knowledge tracer is typically expressed as the Brier Score:
    
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

### Calculate the old 5% worse threshold intercept 

To find the threshold intercept of the 5% worse model we can solve for $\hat{p}$ in our equation above. Set the left-hand side to 5% worse than the RMSE of the ground truth model (i.e. $\frac{RMSE^{*}}{.95}$) then complete the square to solve for $\hat{p}$.

$\frac{RMSE^{*}}{.95} = \sqrt{\hat{p}^2 - 2\hat{p}p + p}$

$(\frac{RMSE^{*}}{.95})^2 = \hat{p}^2 - 2\hat{p}p + p^2 - p^2 + p$  (Complete the square)

$(\frac{RMSE^{*}}{.95})^2 = (\hat{p}-p)^2 - p^2 + p $ (Factor the square)

$(\hat{p}-p)^2 = (\frac{RMSE^{*}}{.95})^2 + p^2 - p$

$\hat{p} = p \pm \sqrt{(\frac{RMSE^{*}}{.95})^2 + p(p-1)}$ 




```python
err = 0.05 # How much worse the old model is
thresh = .95 # A mastery threshold intercept of 95%

# Considering the choice of +/- above, we'll pick the negative (-) choice. This assumes an underestimate.
# In fact there isn't a %5 worse over-estimate for this choice of threshold since to achieve the same 
# 5% worse RMSE with an over-estimate we would need a model prediction probability of > 1.0.
p_for_worse = lambda p : p - np.sqrt((E_rmse(p,p)/(1-err))**2 + p*(p-1))

old_p_at_thresh = p_for_worse(thresh) 
print(f"Probability at threshold intercept of old {100*err}% worse model: \n{100*old_p_at_thresh:.2f}%")

# Sanity Check: the choice the old threshold intercept has 5% worse RMSE than the new generating model.
assert np.abs(E_rmse(thresh, old_p_at_thresh)*.95-E_rmse(thresh,thresh))  < 1e-4
```

    Probability at threshold intercept of old 5.0% worse model: 
    87.84%


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

    intercept old/new: -0.786  -0.619
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

    avg_extra_probs_old_model: 4.20
    Total Time Saved: 525.22 minutes or 8.75 hours. (35.01)% of 25.0 total hours


&emsp; In total a the 5% better model gives us a rather large expected time saving of 35%. 

## What about whole model RMSE?

&emsp; We can do a similar calculation for the RMSE of the whole model (not just at the threshold point). However, the RMSE outside of the neighborhood of the mastery threshold has no bearing on the problem selection behavior of a knowledge-tracer. Nonetheless in this case if we tweak the parameters so that the RMSE difference of the whole model is 5% then, we get a similar RMSE difference in the neighborhood of the mastery threshold. It's hard to say if this is true in general or  a quirk of comparing two versions of the same model: one with ground-truth parameters and one with perturbed parameters. The more typical comparison would be to compare two models with different choices of independant variables, like Performance Factors Analysis (PFA)[3] versus Additive Factors Model (AFM)[1], or a different model variety entirely like Baysian Knowledge Tracing (BKT)[2] or Deep Knowledge Tracing (DKT)[4].  The reality is that we really ought to be in the habit of reporting model performance statistics around reasonable threshold choices (i.e. 85%, 90%, 95%).


```python
def E_model_rmse(intr, slope, opp_cutoff=15):
    ''' Given intr and slope generate the expected RMSE of the model compared to ground-truth'''
    p_ground_truth = lambda i : sig(new_intr+i*new_slope)
    p_model = lambda i : sig(intr+i*slope)
    mse_s = []
    for i in range(opp_cutoff+1):
        mse_s.append(np.mean([E_mse(p_ground_truth(i), p_model(i)) ]))
    return np.mean(mse_s)

old_p_at_thresh = .8797 # No simple analytical solution in this case, just guess and check with assertion.
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


### An additional Sanity Check on our Math / Code


```python
from numpy.random import random as rand

# Sanity Check : Does analytic expected RMSE match an empirically generate one
N = 10000000
print(f'{np.sqrt(np.mean( np.square( (rand(N)<=.95).astype(np.float64) - .8785 ))):.5f}')
print(f'{E_rmse(thresh, .8785):.5f}')
    
```

    0.22941
    0.22937


## Considerations for extending this approach to other models like Performance Factors Analysis (PFA) or Baysian Knowledge Tracing (BKT) that have multiple features (instead of just number of opportunities)?

Our initial set of calculations were model agnostic up-to a point. We did not need to make any assumptions about the nature of the model to determine the mastery probability predicted by a 5% worse than ground-truth model that is underestimating at the ground-truth theshold point. Nonetheless, we did need to make model assumptions to interpolate beyond that point to determine how many extra problems that model would give each student. By choosing a model where number of opportunities was the only feature [1] we simplified this interplolation. In this section we discuss how in principle this approach can be extended to other kinds of models.

Performance Factors Analysis (PFA) [3] is another kind of logistic regression-based model, which counts number of correct and incorrect responses for opportunities independantly instead of combining them into a total opportunity count. In principle, our approach still works in this case. Although, yet more assumptions must be made to continue purely analytically. We would need to, for instance, make a choice of the relative contributions of positive and negative responses. For any choice of parameters there are multiple consistent positive and negative counts that produce a particular correctness probability. Nonetheless, we can interpolate from any of them abitrarily without loss of generality since the model is linear. To perfom this interpolation, we could for instance make a simplifying assumption that all of the remaining practice opportunities after the threshold point will be correct. 

Baysian Knowledge Tracing (BKT) [2] is yet another kind of knowledge tracing model (in fact it is among the first). Like PFA it utilizes both positive and negative responses. But, BKT is a hidden markov model, so it's mastery prediction is not a linear function of the number of positive and negative responses. If two BKT models updated with the same set of correct and incorrect responses, but in different orders, then the models may not make the same predictions. Luckily BKT's predications are characterized by a single latent variable $L_t$, which is the probability that the student has "mastered" a particular skill, concept, or fact at opportunity $t$. To set the point from which we interpolate from we can set $L_t$ such that the probability that the next item is answered correctly $P(C|L_t)$ is equal to the probability associated with the 5% worse than ground truth model at the threshold point, then simply update the model assuming the remaining responses are correct until it exceeds the mastery threshold (the approach take for instance by [5][6]), or alternatively analytically solve for the number of additional problems, which could in principle be done by expressing the BKT update as a markov process with a transition matrix.



## Works Cited

[1] Cen, H., Koedinger, K., Junker, B.: Learning factors analysis–a general method for cognitive model evaluation and improvement. In: International Conference on Intelligent Tutoring Systems. pp. 164–175. Springer (2006)

[2] Corbett, A.T., Anderson, J.R.: Knowledge tracing: Modeling the acquisition of procedural knowledge. User modeling and user-adapted interaction 4(4), 253–278 (1994)

[3] Pavlik Jr, P. I., Cen, H., & Koedinger, K. R. (2009). Performance Factors Analysis--A New Alternative to Knowledge Tracing. Online Submission.

[4] Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. Advances in neural information processing systems, 28.

[5] Yudelson, M., Ritter, S.: Small improvement for the model accuracy–big difference for the students. In: Industry Track Proceedings of 17th International Conference on Artificial Intelligence in Education (AIED 2015), Madrid, Spain (2015)

[6] Yudelson, M., Koedinger, K.: Estimating the benefits of student model improvements on a substantive scale. In: Educational Data Mining 2013 (2013)

