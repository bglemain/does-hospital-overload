


functions {
  // Custom distribution function icar_normal_lpdf for an ICAR random
  // variable phi (see Morris et al., 2019)
  real icar_normal_lpdf(
    vector phi, int N_dept, array[] int node1, array[] int node2
  ) {
    return -0.5 * dot_self(phi[node1] - phi[node2])
    + normal_lpdf(sum(phi) | 0, 0.001 * N_dept);
  }
}


data {
  
  int<lower=0> N_edges;
  array[N_edges] int<lower=1> node1;
  array[N_edges] int<lower=1> node2;
  
  int<lower=0> N_age;
  int<lower=0> N_dept;
  
  int<lower=0> N_pos_incid;
  array[N_pos_incid] int<lower=0> age_pos_incid;
  array[N_pos_incid] int<lower=0> dept_pos_incid;
  
  int<lower=0> N_sens_sapris;
  int<lower=0> y_sens_sapris;
  
  array[N_age, N_dept] int<lower=0> mat_N_age_dept;
  array[N_age, N_dept] int<lower=0> mat_y_age_dept;
  
  vector<lower=0, upper=1>[N_dept] prop_dept;
  vector<lower=0, upper=1>[N_age] prop_age;
  matrix<lower=0, upper=1>[N_dept, N_age] mat_dept_age;
  matrix<lower=0, upper=1>[N_age, N_dept] mat_age_dept;
  
  int<lower=0> N_grid_Y;
  vector<lower=0, upper=1>[N_grid_Y] grid_Y;
  
  vector<lower=0>[N_dept] pop_dept;
  array[N_dept] int<lower=0> hospit_dept;
  array[N_dept] int<lower=0> deces_dept;
  
  vector<lower=0, upper=1>[N_dept] age_pop;
  vector<lower=0, upper=1>[N_dept] beds;
  vector<lower=0, upper=1>[N_dept] diabetes;
  
  int<lower=0> N_draws;
  
}


parameters {
  
  vector[N_dept] phi_raw; // Corresponds to the parameter phi of the manuscript
  real<lower=0> sigma_phi;
  
  real<lower=0, upper=1> sens;
  real<lower=0, upper=1> spe;
  
  real mu_dept;
  vector[N_dept] beta_age_raw;
  real intercept_beta_age;
  real<lower=0> sigma_beta_age;
  real slope_beta_age;
  
  vector[N_dept] ihr_dept_raw;
  real intercept_ihr;
  real beta_ihr_age;
  real beta_ihr_beds;
  real beta_ihr_diabetes;
  real slope_ihr;
  real<lower=0> sigma_ihr_dept;
  
  vector[N_dept] ifr_dept_raw;
  real intercept_ifr;
  real beta_ifr_age;
  real beta_ifr_beds;
  real beta_ifr_diabetes;
  real slope_ifr;
  real<lower=0> sigma_ifr_dept;
  
}


transformed parameters {
  
  vector[N_dept] phi = sigma_phi * phi_raw;
  vector[N_dept] alpha_dept = mu_dept + phi;
  
  vector[N_dept] beta_age = beta_age_raw * sigma_beta_age +
  intercept_beta_age + slope_beta_age * alpha_dept;
  
  matrix<lower=0, upper=1>[N_age, N_dept] cuminc_age_dept;
  for (i in 1:N_age) {
    for (j in 1:N_dept) {
      cuminc_age_dept[i, j] = inv_logit(
        alpha_dept[j] + beta_age[j] * (i - 1)
      );
    }
  }
  
  matrix<lower=0, upper=1>[N_age, N_dept] seroprev_age_dept;
  for (i in 1:N_age) {
    for (j in 1:N_dept) {
      seroprev_age_dept[i, j] =
        sens * cuminc_age_dept[i, j] +
        (1 - spe) * (1 - cuminc_age_dept[i, j]);
    }
  }
  
  vector<lower=0, upper=1>[N_dept] seroprev_dept;
  for (i in 1:N_dept) {
    seroprev_dept[i] = mat_age_dept[, i]' * seroprev_age_dept[, i];
  }
  
  real seroprev_global_via_dept;
  seroprev_global_via_dept = seroprev_dept' * prop_dept;
  
  vector<lower=0, upper=1>[N_dept] cuminc_dept;
  for (i in 1:N_dept) {
    cuminc_dept[i] = mat_age_dept[, i]' * cuminc_age_dept[, i];
  }
  
  vector[N_dept] logit_ihr_dept;
  for (j in 1:N_dept) {
    logit_ihr_dept[j] =
      ihr_dept_raw[j] * sigma_ihr_dept +
      intercept_ihr +
      beta_ihr_age * age_pop[j] + beta_ihr_beds * beds[j] +
      beta_ihr_diabetes * diabetes[j] +
      slope_ihr * cuminc_dept[j];
  }
  
  vector[N_dept] logit_ifr_dept;
  for (j in 1:N_dept) {
    logit_ifr_dept[j] =
      ifr_dept_raw[j] * sigma_ifr_dept +
      intercept_ifr +
      beta_ifr_age * age_pop[j] + beta_ifr_beds * beds[j] +
      beta_ifr_diabetes * diabetes[j] +
      slope_ifr * cuminc_dept[j];
  }
  
  vector<lower=0, upper=1>[N_dept] ihr_dept = inv_logit(logit_ihr_dept) / 10;
  vector<lower=0, upper=1>[N_dept] ifr_dept = inv_logit(logit_ifr_dept) / 20;
  
}


model {
  
  phi_raw ~ icar_normal(N_dept, node1, node2);
  
  sens ~ beta(585, 56);
  spe ~ beta(953, 15);
  
  y_sens_sapris ~ binomial(N_sens_sapris, sens);
  
  seroprev_global_via_dept ~ beta(101, 1948);
  seroprev_global_via_dept ~ beta(1147, 17212);

  sigma_phi ~ exponential(1);
  sigma_ifr_dept ~ exponential(1);
  sigma_ihr_dept ~ exponential(1);
  beta_age_raw ~ normal(0, 1);
  
  for (k in 1:N_pos_incid) {
    1 ~ bernoulli(cuminc_age_dept[age_pos_incid[k] + 1, dept_pos_incid[k]]);
  }

  for (i in 1:N_age) {
    for (j in 1:N_dept) {
      mat_y_age_dept[i, j] ~ binomial(
        mat_N_age_dept[i, j], seroprev_age_dept[i, j]
      );
    }
  }

  ihr_dept_raw ~ normal(0, 1);
  ifr_dept_raw ~ normal(0, 1);
  
  for (j in 1:N_dept) {
    hospit_dept[j] ~ poisson(pop_dept[j] * cuminc_dept[j] * ihr_dept[j]);
    deces_dept[j] ~ poisson(pop_dept[j] * cuminc_dept[j] * ifr_dept[j]);
  }
  
}


generated quantities {
  
  // Association between incidence in persons under 60
  // and the proportion of persons above 60 among those infected
  
  vector<lower=0, upper=1>[N_dept] age_infected;
  for (j in 1:N_dept) {
    age_infected[j] =
      (cuminc_age_dept[2, j] * mat_age_dept[2, j]) /
      (cuminc_age_dept[2, j] * mat_age_dept[2, j] +
      cuminc_age_dept[1, j] * mat_age_dept[1, j]);
  }
  
  vector[N_grid_Y] g_cuminc_60_logit;
  for (i in 1:N_grid_Y) {
    g_cuminc_60_logit[i] =
      logit(grid_Y[i]) + intercept_beta_age +
      slope_beta_age * logit(grid_Y[i]);
  }

  vector<lower=0, upper=1>[N_grid_Y] g_cuminc_60;
  for (i in 1:N_grid_Y) {
    g_cuminc_60[i] = mean(inv_logit(normal_rng(
      rep_vector(g_cuminc_60_logit[i], N_draws),
      sigma_beta_age
    )));
  }
  
  vector<lower=0, upper=1>[N_grid_Y] g_age_infected;
  for (i in 1:N_grid_Y) {
    g_age_infected[i] =
      (g_cuminc_60[i] * prop_age[2]) /
      (g_cuminc_60[i] * prop_age[2] + grid_Y[i] * prop_age[1]);
  }

  real douze_moins_six_age_infected = g_age_infected[41] - g_age_infected[17];


  // Causal effects (incidence on IFR and IHR)
  
  vector<lower=0, upper=1>[N_grid_Y] g_ihr_backdoor;
  g_ihr_backdoor = rep_vector(0, N_grid_Y);
  
  vector<lower=0, upper=1>[N_grid_Y] g_ifr_backdoor;
  g_ifr_backdoor = rep_vector(0, N_grid_Y);
  
  for (i in 1:N_dept) {
    for (s in 1:N_grid_Y) {
      
      real g_ihr_logit = 
        intercept_ihr +
        beta_ihr_age * age_pop[i] + beta_ihr_beds * beds[i] +
        beta_ihr_diabetes * diabetes[i] +
        slope_ihr * grid_Y[s];
      
      real g_ifr_logit = 
        intercept_ifr +
        beta_ifr_age * age_pop[i] + beta_ifr_beds * beds[i] +
        beta_ifr_diabetes * diabetes[i] +
        slope_ifr * grid_Y[s];
      
      real g_ihr = mean(inv_logit(to_vector(normal_rng(
        rep_vector(g_ihr_logit, N_draws), sigma_ihr_dept
      )))) / 10;
      
      real g_ifr = mean(inv_logit(to_vector(normal_rng(
        rep_vector(g_ifr_logit, N_draws), sigma_ifr_dept
      )))) / 20;
      
      g_ihr_backdoor[s] +=
        g_ihr / N_dept;
        
      g_ifr_backdoor[s] += 
        g_ifr / N_dept;
        
    }
  }
  
  real nine_minus_three_backdoor_ifr = 
    g_ifr_backdoor[29] - g_ifr_backdoor[5];
    
  real nine_minus_three_backdoor_ihr =
    g_ihr_backdoor[29] - g_ihr_backdoor[5];
  
  
  // Global estimates for metropolitan France (incidence, IFR, IHR)
  
  real cuminc_global_via_dept;
  cuminc_global_via_dept = cuminc_dept' * prop_dept;
  
  vector<lower=0>[N_dept] n_infected_dept;
  for (j in 1:N_dept) {
    n_infected_dept[j] = cuminc_dept[j] * pop_dept[j];
  }
  
  real ifr_global;
  ifr_global = (ifr_dept' * n_infected_dept) / sum(n_infected_dept);
  
  real ihr_global;
  ihr_global = (ihr_dept' * n_infected_dept) / sum(n_infected_dept);
  
  
  // Posterior predictive samples
  
  array[N_age, N_dept] int<lower=0> y_rep_age_dept;
  for (i in 1:N_age) {
    for (j in 1:N_dept) {
      y_rep_age_dept[i ,j] = binomial_rng(
        mat_N_age_dept[i ,j], seroprev_age_dept[i ,j]
      );
    }
  }
  
  vector<lower=0>[N_dept] hospit_rep_dept;
  vector<lower=0>[N_dept] deces_rep_dept;
  
  for (j in 1:N_dept) {
    hospit_rep_dept[j] = poisson_rng(
      pop_dept[j] * cuminc_dept[j] * ihr_dept[j]
    );
    
    deces_rep_dept[j] = poisson_rng(
      pop_dept[j] * cuminc_dept[j] * ifr_dept[j]
    );
  }
  
}



