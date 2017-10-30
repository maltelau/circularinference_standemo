
functions {
    vector F(vector L, real w) {
        return log((w * exp(L) + 1 - w)./((1 - w) * exp(L) + w));
    }
}

data{
    int<lower=1> N;
    int<lower=0, upper=1> ChoiceRed[N];
    vector[N] lSelf;
    vector[N] lOthers;
    int<lower=1> N_params;
}

parameters{
    real a;
    real<lower=.5, upper=1> wSelf;
    real<lower=.5, upper=1> wOthers;
    real<lower=0> aOthers;
    real<lower=0> aSelf;   
}

transformed parameters {
    vector[N_params] fixed_effects;
    
    vector[N] mu;
    vector[N] I; 
    
    fixed_effects[1] = a;
    fixed_effects[2] = wSelf;
    fixed_effects[3] = wOthers;
    fixed_effects[4] = aSelf;
    fixed_effects[5] = aOthers;
    

    
    # reverberating "noise" parameter
    # gets counted multiple times
    I = F(aOthers * lOthers, wOthers) +
        F(aSelf * lSelf, wSelf);
    
    mu = a +
        F(lSelf + I, wSelf) +
        F(lOthers + I, wOthers);
    
}



model{
    // Grand means
    fixed_effects ~ normal(0, 1);
    
    ChoiceRed ~ bernoulli_logit(mu);
}


generated quantities{
    vector[N] log_lik;
    
    for (i in 1:N) {
        log_lik[i] = bernoulli_logit_lpmf( ChoiceRed[i] | mu[i] );
    }
}
