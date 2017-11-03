
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
    vector[N] Symptoms;
    int<lower=1, upper=P> Participant[N];
    int<lower=1> N_params;
}

parameters{
    real a;
    vector[P] aP;
    real aSymptoms;
    
    real<lower=.5,upper=1> wSelf;
    real<lower=(.5-wSelf)/20, upper=(1-wSelf)/20> wSelfSymptoms;
    vector<lower=.5-wSelf, upper=1-wSelf>[P] wSelfP;
    
    real<lower=.5,upper=1> wOthers;
    real<lower=(.5-wOthers)/20, upper=(1-wOthers)/20> wOthersSymptoms;
    vector<lower=.5-wOthers, upper=1-wOthers>[P] wOthersP;

    real<lower=0> aOthers;
    real<lower=-aOthers/20> aOthersSymptoms;
    vector<lower=-aOthers>[P] aOthersP;
    
    real<lower=0> aSelf;
    real<lower=-aSelf/20> aSelfSymptoms;
    vector<lower=-aSelf>[P] aSelfP;
    
    vector<lower=0>[N_params] sigma_participant;
    cholesky_factor_corr[N_params] Lcorr;      
}

transformed parameters {
    vector[N_params] participant_effects[P];
    vector[N_params] fixed_effects;
    vector[N_params] symptom_effects;
    
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
    fixed_effects[2] = wSelf-1;
    fixed_effects[3] = wOthers-1;
    fixed_effects[4] = aSelf-1;
    fixed_effects[5] = aOthers-1;
    
    symptom_effects[1] = aSymptoms;
    symptom_effects[2] = wSelfSymptoms;
    symptom_effects[3] = wOthersSymptoms;
    symptom_effects[4] = aSelfSymptoms;
    symptom_effects[5] = aOthersSymptoms;
    
    # overall bias against one choice or the other
    intercept = a + aP[Participant] + aSymptoms * Symptoms;
    
    # "weighting" parameter - how much does perceptual evidence count
    # both w parameters have to be on a scale of 0.5 to 1, that's why we use logit
    ws = wSelf + wSelfP[Participant] + wSelfSymptoms * Symptoms;
    
    # weighting for prior information
    wo = wOthers + wOthersP[Participant] + wOthersSymptoms * Symptoms;
    
    # how much does perceptual evidence reverberate
    # both a parameters have to be above 0
    as = aSelf   + aSelfP[Participant]   + aSelfSymptoms * Symptoms;
    
    # how much does prior information reverberate
    ao = aOthers + aOthersP[Participant] + aOthersSymptoms * Symptoms;
    
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
    fixed_effects ~ normal(0, .5);
    // Symptom effects
    symptom_effects ~ normal(0, .1);
    
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
