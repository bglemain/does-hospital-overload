


functions {
  // Custom distribution function icar_normal_lpdf for an ICAR random
  // variable phi (see Morris et al., 2019)
  real icar_normal_lpdf(
    vector phi, int J_dept, array[] int node1, array[] int node2
  ) {
    return -0.5 * dot_self(phi[node1] - phi[node2])
    + normal_lpdf(sum(phi) | 0, 0.001 * J_dept);
  }
}


data {
  
  int<lower=0> N_edges;
  array[N_edges] int<lower=1> node1;
  array[N_edges] int<lower=1> node2;
  
  int<lower=0> J_age;
  int<lower=0> J_dept;
  
  int<lower=0> N_pos_incid;
  array[N_pos_incid] int<lower=0> age_pos_incid;
  array[N_pos_incid] int<lower=0> dept_pos_incid;
  
  int<lower=0> N_sens_sapris;
  int<lower=0> y_sens_sapris;
  
  array[J_age, J_dept] int<lower=0> mat_N_age_dept;
  array[J_age, J_dept] int<lower=0> mat_y_age_dept;
  
  vector<lower=0, upper=1>[J_dept] prop_dept;
  vector<lower=0, upper=1>[J_age] prop_age;
  matrix<lower=0, upper=1>[J_dept, J_age] mat_dept_age;
  matrix<lower=0, upper=1>[J_age, J_dept] mat_age_dept;
  
  int<lower=0> N_grid_Y;
  vector<lower=0, upper=1>[N_grid_Y] grid_Y;
  
  vector<lower=0>[J_dept] pop_dept;
  array[J_dept] int<lower=0> hospit_dept;
  array[J_dept] int<lower=0> deces_dept;
  
  array[J_dept] int<lower=0, upper=1> binary_age;
  array[J_dept] int<lower=0, upper=1> binary_beds;
  array[J_dept] int<lower=0, upper=1> binary_diabetes;
  array[2,2,2] int<lower=0> strata_counts;
  
  int<lower=0> n_draws;
  
}


parameters {
  
  vector[J_dept] phi_raw; // Corresponds to the parameter phi of the manuscript
  real<lower=0> sigma_phi;
  
  real<lower=0, upper=1> sens;
  real<lower=0, upper=1> spe;
  
  real mu_dept;
  vector[J_dept] beta_age_raw;
  real intercept_beta_age;
  real<lower=0> sigma_beta_age;
  real slope_beta_age;
  
  vector[J_dept] ihr_dept_raw;
  real intercept_ihr;
  real beta_ihr_age;
  real beta_ihr_beds;
  real beta_ihr_diabetes;
  real slope_ihr;
  real<lower=0> sigma_ihr_dept;
  
  vector[J_dept] ifr_dept_raw;
  real intercept_ifr;
  real beta_ifr_age;
  real beta_ifr_beds;
  real beta_ifr_diabetes;
  real slope_ifr;
  real<lower=0> sigma_ifr_dept;
  
}


transformed parameters {
  
  vector[J_dept] phi = sigma_phi * phi_raw;
  vector[J_dept] alpha_dept = mu_dept + phi;
  
  vector[J_dept] beta_age = beta_age_raw * sigma_beta_age +
  intercept_beta_age + slope_beta_age * alpha_dept;
  
  matrix<lower=0, upper=1>[J_age, J_dept] cuminc_age_dept;
  for (i in 1:J_age) {
    for (j in 1:J_dept) {
      cuminc_age_dept[i, j] = inv_logit(
        alpha_dept[j] + beta_age[j] * (i - 1)
      );
    }
  }
  
  matrix<lower=0, upper=1>[J_age, J_dept] seroprev_age_dept;
  for (i in 1:J_age) {
    for (j in 1:J_dept) {
      seroprev_age_dept[i, j] =
        sens * cuminc_age_dept[i, j] +
        (1 - spe) * (1 - cuminc_age_dept[i, j]);
    }
  }
  
  vector<lower=0, upper=1>[J_dept] seroprev_dept;
  for (i in 1:J_dept) {
    seroprev_dept[i] = mat_age_dept[, i]' * seroprev_age_dept[, i];
  }
  
  real seroprev_global_via_dept;
  seroprev_global_via_dept = seroprev_dept' * prop_dept;
  
  vector<lower=0, upper=1>[J_dept] cuminc_dept;
  for (i in 1:J_dept) {
    cuminc_dept[i] = mat_age_dept[, i]' * cuminc_age_dept[, i];
  }
  
  vector[J_dept] logit_ihr_dept;
  for (j in 1:J_dept) {
    logit_ihr_dept[j] =
      ihr_dept_raw[j] * sigma_ihr_dept +
      intercept_ihr +
      beta_ihr_age * binary_age[j] + beta_ihr_beds * binary_beds[j] +
      beta_ihr_diabetes * binary_diabetes[j] +
      slope_ihr * cuminc_dept[j];
  }
  
  vector[J_dept] logit_ifr_dept;
  for (j in 1:J_dept) {
    logit_ifr_dept[j] =
      ifr_dept_raw[j] * sigma_ifr_dept +
      intercept_ifr +
      beta_ifr_age * binary_age[j] + beta_ifr_beds * binary_beds[j] +
      beta_ifr_diabetes * binary_diabetes[j] +
      slope_ifr * cuminc_dept[j];
  }
  
  vector<lower=0, upper=1>[J_dept] ihr_dept = inv_logit(logit_ihr_dept) / 10;
  vector<lower=0, upper=1>[J_dept] ifr_dept = inv_logit(logit_ifr_dept) / 20;
  
}


model {
  
  phi_raw ~ icar_normal(J_dept, node1, node2);
  
  sens ~ beta(585, 56);
  spe ~ beta(953, 15);
  
  y_sens_sapris ~ binomial(N_sens_sapris, sens);
  
  seroprev_global_via_dept ~ beta(101, 1948);
  seroprev_global_via_dept ~ beta(1147, 17212);

  sigma_phi ~ exponential(1);
  beta_age_raw ~ normal(0, 1);
  
  for (k in 1:N_pos_incid) {
    1 ~ bernoulli(cuminc_age_dept[age_pos_incid[k] + 1, dept_pos_incid[k]]);
  }

  for (i in 1:J_age) {
    for (j in 1:J_dept) {
      mat_y_age_dept[i, j] ~ binomial(
        mat_N_age_dept[i, j], seroprev_age_dept[i, j]
      );
    }
  }

  ihr_dept_raw ~ normal(0, 1);
  ifr_dept_raw ~ normal(0, 1);
  
  for (j in 1:J_dept) {
    hospit_dept[j] ~ poisson(pop_dept[j] * cuminc_dept[j] * ihr_dept[j]);
    deces_dept[j] ~ poisson(pop_dept[j] * cuminc_dept[j] * ifr_dept[j]);
  }
  
}


generated quantities {
  
  vector<lower=0, upper=1>[J_dept] prop_60;
  for (j in 1:J_dept) {
    prop_60[j] =
      (cuminc_age_dept[2, j] * mat_age_dept[2, j]) /
      (cuminc_age_dept[2, j] * mat_age_dept[2, j] +
      cuminc_age_dept[1, j] * mat_age_dept[1, j]);
  }
  
  vector[N_grid_Y] g_cuminc_60_logit;
  for (i in 1:N_grid_Y) {
    g_cuminc_60_logit[i] = logit(
      grid_Y[i]) + intercept_beta_age + slope_beta_age * logit(grid_Y[i]
    );
  }

  vector<lower=0, upper=1>[N_grid_Y] g_cuminc_60;
  for (i in 1:N_grid_Y) {
    g_cuminc_60[i] = mean(inv_logit(normal_rng(rep_vector(
      g_cuminc_60_logit[i], n_draws), sigma_beta_age))
    );
  }
  
  vector<lower=0, upper=1>[N_grid_Y] g_prop_60;
  for (i in 1:N_grid_Y) {
    g_prop_60[i] =
      (g_cuminc_60[i] * prop_age[2]) /
      (g_cuminc_60[i] * prop_age[2] + grid_Y[i] * prop_age[1]);
  }

  real douze_moins_six_prop_60 = g_prop_60[48] - g_prop_60[24];
  
  array[2, 2, 2] vector[N_grid_Y] g_ihr_logit;
  array[2, 2, 2] vector[N_grid_Y] g_ifr_logit;
  
  for (i in 1:2) {
    for (j in 1:2) {
      for (k in 1:2) {
        g_ihr_logit[i, j, k] = 
          intercept_ihr +
          beta_ihr_age * (i-1) + beta_ihr_beds * (j-1) +
          beta_ihr_diabetes * (k-1) +
          slope_ihr * grid_Y;
      
        g_ifr_logit[i, j, k] = 
          intercept_ifr +
          beta_ifr_age * (i-1) + beta_ifr_beds * (j-1) +
          beta_ifr_diabetes * (k-1) +
          slope_ifr * grid_Y;
      }      
    }
  }
  
  array[2, 2, 2] vector<lower=0, upper=1>[N_grid_Y] g_ihr;
  array[2, 2, 2] vector<lower=0, upper=1>[N_grid_Y] g_ifr;
  
  for (s in 1:N_grid_Y) {
    for (i in 1:2) {
      for (j in 1:2) {
        for (k in 1:2) {
        g_ihr[i, j, k][s] = mean(inv_logit(to_vector(normal_rng(
          rep_vector(g_ihr_logit[i, j, k][s], n_draws), sigma_ihr_dept
        ))))/10;
      
        g_ifr[i, j, k][s] = mean(inv_logit(to_vector(normal_rng(
          rep_vector(g_ifr_logit[i, j, k][s], n_draws), sigma_ifr_dept
        ))))/20;
        }      
      }
    }
  }
  
  vector<lower=0, upper=1>[N_grid_Y] g_ihr_backdoor;
  g_ihr_backdoor = rep_vector(0, N_grid_Y);
  for (i in 1:2) {
    for (j in 1:2) {
      for (k in 1:2) {
        g_ihr_backdoor +=
          g_ihr[i, j, k] * strata_counts[i, j, k] / J_dept;
      }
    }
  }
  
  vector<lower=0, upper=1>[N_grid_Y] g_ifr_backdoor;
  g_ifr_backdoor = rep_vector(0, N_grid_Y);
  for (i in 1:2) {
    for (j in 1:2) {
      for (k in 1:2) {
        g_ifr_backdoor += 
          g_ifr[i, j, k] * strata_counts[i, j, k] / J_dept;
      }
    }
  }
  
  real neuf_moins_trois_backdoor_ifr = 
    g_ifr_backdoor[36] - g_ifr_backdoor[12];
    
  real neuf_moins_trois_backdoor_ihr =
    g_ihr_backdoor[36] - g_ihr_backdoor[12];
  
  vector<lower=0, upper=1>[J_age] cuminc_age;
  for (i in 1:J_age) {
    cuminc_age[i] = cuminc_age_dept[i, ] * mat_dept_age[, i];
  }
  
  real cuminc_global_via_dept;
  cuminc_global_via_dept = cuminc_dept' * prop_dept;
  
  vector<lower=0>[J_dept] n_infected_dept;
  for (j in 1:J_dept) {
    n_infected_dept[j] = cuminc_dept[j] * pop_dept[j];
  }
  
  real ifr_global;
  ifr_global = (ifr_dept' * n_infected_dept) / sum(n_infected_dept);
  
  real ihr_global;
  ihr_global = (ihr_dept' * n_infected_dept) / sum(n_infected_dept);
  
  
  // Posterior predictive samples
  
  array[J_age, J_dept] int<lower=0> y_rep_age_dept;
  for (i in 1:J_age) {
    for (j in 1:J_dept) {
      y_rep_age_dept[i ,j] = binomial_rng(
        mat_N_age_dept[i ,j], seroprev_age_dept[i ,j]
      );
    }
  }
  
  vector<lower=0>[J_dept] hospit_rep_dept;
  vector<lower=0>[J_dept] deces_rep_dept;
  
  for (j in 1:J_dept) {
    hospit_rep_dept[j] = poisson_rng(
      pop_dept[j] * cuminc_dept[j] * ihr_dept[j]
    );
    
    deces_rep_dept[j] = poisson_rng(
      pop_dept[j] * cuminc_dept[j] * ifr_dept[j]
    );
  }
  
}



