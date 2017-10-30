
functions {
    vector F(vector L, vector w) {
        return log((w .* exp(L) + 1 - w)./((1 - w) .* exp(L) + w));
    }
}

data{
    int<lower=1> N;
    int<lower=1> P;
    int<lower=0, upper=1> ChoiceRed[N];
    vector[N] lSelf;
    vector[N] lOthers;
    int<lower=1, upper=P> Participant[N];
    int<lower=1> N_params;
}

parameters{
    real a;
    vector[P] aP;
    real wSelf;
    vector[P] wSelfP;
    real wOthers;
    vector[P] wOthersP;
    # these bounds are not perfect, but I can't think of how to do them better
    real<lower=0> aOthers;
    vector<lower=-aOthers>[P] aOthersP;
    real<lower=0> aSelf;
    vector<lower=-aSelf>[P] aSelfP;
    
    vector<lower=0>[N_params] sigma_participant;
    cholesky_factor_corr[N_params] Lcorr;      
}

transformed parameters {
    vector[N_params] participant_effects[P];
    vector[N_params] fixed_effects;
    
    vector[N] mu;
    vector[N] I; 
    vector[N] intercept;
    vector<lower=0.5, upper=1>[N] ws;
    vector<lower=0.5, upper=1>[N] wo;
    vector<lower=0>[N] as;
    vector<lower=0>[N] ao;
    
    # some variable redefinitions to make it easier to debug
    for ( p in 1:P ) {
        participant_effects[p,1] = aP[p];
        participant_effects[p,2] = wSelfP[p];
        participant_effects[p,3] = wOthersP[p];
        participant_effects[p,4] = aSelfP[p];
        participant_effects[p,5] = aOthersP[p];
    }
    
    fixed_effects[1] = a;
    fixed_effects[2] = wSelf;
    fixed_effects[3] = wOthers;
    fixed_effects[4] = aSelf;
    fixed_effects[5] = aOthers;
    
    
    # overall bias against one choice or the other
    intercept = a + aP[Participant];
    
    # "weighting" parameter - how much does perceptual evidence count
    # both w parameters have to be on a scale of 0.5 to 1, that's why we use logit
    ws = inv_logit(wSelf + wSelfP[Participant])/2+.5;
    
    # weighting for prior information
    wo = inv_logit(wOthers + wOthersP[Participant])/2+.5;
    
    # how much does perceptual evidence reverberate
    # both a parameters have to be above 0
    as = aSelf   + aSelfP[Participant];
    
    # how much does prior information reverberate
    ao = aOthers + aOthersP[Participant];
    
    # reverberating "noise" parameter
    # gets counted multiple times
    I = F(ao .* lOthers, wo) +
        F(as .* lSelf, ws);
    
    mu = intercept +
        F(lSelf + I, ws) +
        F(lOthers + I, wo);
    
}



model{
    // Grand means
    fixed_effects ~ normal(0, 1);
    
    // Participant effects
    // lkj prior for the correlations of the multivariate distribution of varying effects
    Lcorr ~ lkj_corr_cholesky(5);
    sigma_participant ~ cauchy(0, 2);
    participant_effects ~ multi_normal_cholesky(rep_vector(0, N_params), 
                                                diag_pre_multiply(sigma_participant, Lcorr));
    
    ChoiceRed ~ bernoulli_logit(mu);
}


generated quantities{
    vector[N] log_lik;
    
    for (i in 1:N) {
        log_lik[i] = bernoulli_logit_lpmf( ChoiceRed[i] | mu[i] );
    }
}
