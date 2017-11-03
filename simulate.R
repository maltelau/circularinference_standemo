# simulating the circular inference model

library(pacman)
p_load(tidyverse, rstan, parallel, bayesplot, tidybayes, cowplot, boot)
d=read_csv('csv/demo_data.csv') %>% mutate(i = 1:n())

# stan settings
N_ITER = 1600
N_CHAINS = parallel::detectCores()
CONTROL = list(max_treedepth = 15,
               adapt_delta = 0.99)

# simualtion settings
N = 1000
p = 5
symptom_prior = .01
symptoms = sample(0:20, p)

F <- function(L, w) {
    eL = exp(L)
    log((w * eL + 1 - w)/((1 - w) * eL + w))
}

params = data_frame(
    Participant = sample(1:p, N, replace = TRUE),
    Symptoms = symptoms[Participant],
    lOthers = sample(d$lOthers, N, replace = TRUE),
    lSelf = sample(d$lSelf, N, replace = TRUE),
    wSelf = .75,
    wOthers = .75,
    aSelf = 1,
    aOthers = 0,
    a = 0) %>%
    mutate(I = F((aOthers+Symptoms*symptom_prior*10)*lOthers,(wOthers+Symptoms*symptom_prior)) +
               F((aSelf+Symptoms*symptom_prior*10)*lSelf,(wSelf+Symptoms*symptom_prior)),
           mu = (a+Symptoms*symptom_prior) +
              F(lSelf + I, (wSelf+Symptoms*symptom_prior)) +
               F(lOthers + I, (wOthers+Symptoms*symptom_prior)),
           ChoiceRed = rbernoulli(N, inv.logit(mu)) %>% as.numeric(),
           i = 1:n())

simulated_standata = list(
    Participant = params$Participant,
    ChoiceRed = params$ChoiceRed,
    lSelf= params$lSelf,
    lOthers= params$lOthers,
    Symptoms=params$Symptoms,
    N = nrow(params),
    P = length(unique(params$Participant)),
    N_params = 5)

simulate_1 = stan(
    file = "stan/CircularInference_stamdemo_symptoms_bounds.stan",
    data = simulated_standata,
    chains = N_CHAINS,   
    cores = N_CHAINS,
    iter = N_ITER, 
    control = CONTROL)

# diagnostic plots
traceplot(simulate_1, pars = c("fixed_effects"))
pairs(simulate_1, pars = c("fixed_effects"))
pairs(simulate_1, pars = c("fixed_effects", "symptom_effects"))


## mcmc_parcoord
mc_simulate_1 <- As.mcmc.list(simulate_1)
np_simulate_1 = nuts_params(simulate_1)
mcmc_parcoord(mc_simulate_1, regex_pars = "participant_effects", np = np_simulate_1)

# predictive posteriors
true_values = data_frame(
    term = rep(c("intercept", "ao", "as", "wo", "ws"), each = 2),
    Symptoms = rep(c(0,20), times = 5)) %>%
    mutate(true_value = rep(c(0, 0, 1, .75, .75), each = 2) + 
               Symptoms * rep(c(0, symptom_prior*10, symptom_prior*10, symptom_prior, symptom_prior), each =  2))

param_names = c("a", "wSelf", "wOthers", "aSelf", "aOthers")
true_symptom = data_frame(
    term = param_names,
    true_value = c(0, symptom_prior, symptom_prior, symptom_prior*10, symptom_prior*10))


pred <- gather_samples(simulate_1, intercept[i], ws[i], wo[i], as[i], ao[i]) %>%
    left_join(params) 

ppcheck <- pred %>%
    filter(estimate < 3) %>%
    ggplot(aes(Symptoms, estimate, group = Participant)) +
    geom_line(aes(y = true_value, group = NA), data = true_values, linetype = "dashed") +
    geom_violin() +
    # geom_pointrange(stat = "summary", fun.data = mean_hdi, position = position_dodge(width = 1)) +
    facet_wrap(~ term, scales = "free") +
    labs(title = "Predictive posterior (with symptom effect)")

# save_plot("simulation/ppcheck.png", plot = ppcheck, base_width = 10)


effects <- gather_samples(simulate_1, symptom_effects[Participant]) %>%
    ungroup() %>%
    mutate(term = param_names[Participant]) %>%
    ggplot(aes(estimate)) +
    geom_vline(aes(xintercept = true_value, group = NA), data = true_symptom, linetype = "dashed") +
    geom_density() +
    facet_wrap(~ term, scales = "free") +
    labs(title = "Symptom effects")

# save_plot("simulation/effects.png", plot = effects, base_width = 10)




## comparing prior to posterior plots
true_fixed = data_frame(
    term = param_names,
    true_value = c(0, .75, .75, 0, 1))


posterior <- gather_samples(simulate_1, a, wSelf, wOthers, aSelf, aOthers) %>%
    mutate(distribution = "posterior") %>%
    ungroup() %>%
    select(term, distribution, estimate)

to_bound <- function(x, lower, upper) x[x > lower & x < upper]

prior = data_frame(term = param_names,
                   distribution = "prior",
                   estimate = lapply(1:5, function(i) rnorm(1e5, 0, .5))) %>%
    mutate(estimate = map2(estimate, c(0, 1, 1, 1, 1), ~.x + .y),
           estimate = purrr::pmap(list(estimate, 
                                       c(-100, .5, .5, 0, 0), 
                                       c(100, 1, 1, 100, 100)), 
                                  to_bound)) %>% 
    unnest()

both = rbind(prior, posterior)
ggplot(both, aes(estimate, colour = distribution)) +
    geom_density() +
    facet_wrap(~ term, scales = "free") +
    geom_vline(aes(xintercept = true_value), data = true_fixed, linetype = "dashed") +
    labs(colour = NULL, title = "Comparing prior to posterior")
