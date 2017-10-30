library(pacman)
p_load(tidyverse, rstan, parallel)

# run time for 1000 iterations: ~10 minutes on an i5-7200U 

# stan settings
N_ITER = 1000
CONTROL = list(max_treedepth = 15,
               adapt_delta = 0.99)

set.seed(1000)
d=read_csv('demo_data.csv')
standata = list(
        Participant = d$Participant,
        ChoiceRed = d$RedChoice,
        lSelf= d$lSelf,
        lOthers= d$lOthers,
        Symptoms=d$Symptoms,
        N = nrow(d),
        P = length(unique(d$Participant)),
        N_params = 5)

# run models
symptom_model = stan(
    file = "CircularInference_stamdemo_symptoms.stan",
    data = standata,
    chains = parallel::detectCores()-1,   
    cores = parallel::detectCores()-1,
    iter = N_ITER, 
    control = CONTROL)
# n_eff is about 10-50 of 1500


nosymptom_model = stan(
    file = "CircularInference_stamdemo_nosymptoms.stan",
    data = standata,
    chains = parallel::detectCores()-1,   
    cores = parallel::detectCores()-1,
    iter = N_ITER, 
    control = CONTROL)
# n_eff is about 50-500 of 1500

# diagnostics plots are clear: something's wrong
pairs(symptom_model, pars = "fixed_effects")
pairs(nosymptom_model, pars = "fixed_effects")

traceplot(symptom_model, pars = "fixed_effects")
traceplot(nosymptom_model, pars = "fixed_effects")

# traceplot(symptom_model, pars = "sigma_participant")
# traceplot(nosymptom_model, pars = "sigma_participant")

# traceplot(symptom_model, pars = "symptom_effects")


