require(data.table) 
require(tidyverse) # data mangling
require(ggplot2) # plotting
require(hexbin) # plotting
require(bayesplot) # Stan plotting
require(knitr)
require(kableExtra)
require(cmdstanr)
library(readr)
library(terra)
library(posterior)
library(dplyr)
library(tidyr)
library(lubridate)
out.dir <- "/Users/jyx/M4R"
in.dir  <- "/Users/jyx/M4R/"

model_text <- "
functions {
  real car_normal_mcar_lpdf(
    vector phi, 
    int N, 
    int N_edges,
    array[] int node1, 
    array[] int node2,
    matrix Sigma
  ) {
    // Split phi into two N-dimensional vectors
    vector[N] phi1 = phi[1:N];
    vector[N] phi2 = phi[(N+1):(2*N)];

    // Invert Sigma once
    matrix[2,2] Sigma_inv = inverse(Sigma);

    real accum = 0;
    // Loop over edges
    for (e in 1:N_edges) {
      // difference of the two fields across edge e
      vector[2] d;
      d[1] = phi1[node1[e]] - phi1[node2[e]];
      d[2] = phi2[node1[e]] - phi2[node2[e]];
      accum += d' * Sigma_inv * d;
    }

    // Negative half of the quadratic form
    return -0.5 * accum;
  }
}

data {
  // Node-level adjacency
  int<lower=0> N;               // number of nodes
  int<lower=0> N_edges;         // number of edges
  array[N_edges] int node1;     // adjacency 'from'
  array[N_edges] int node2;     // adjacency 'to'

  // Outcomes
  vector<lower=0>[N] y1;        // outcome 1
  vector<lower=0>[N] y2;        // outcome 2
}

parameters {
  // Intercepts & noise
  real beta0_1;
  real beta0_2;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;

  // Node-level effects for the 2 fields
  vector[2*N] phi_raw; // non-centered
  real<lower=0> sigma_phi;  

  // Cholesky factor for the 2×2 correlation of (phi1, phi2)
  cholesky_factor_corr[2] L_corr;
}

transformed parameters {
  // Scale the node-level effects
  vector[2*N] phi = phi_raw * sigma_phi;

  // Build the full 2×2 covariance (or correlation) matrix
  matrix[2,2] Sigma = multiply_lower_tri_self_transpose(L_corr);
}

model {
  // Split phi for readability
  vector[N] phi1 = phi[1:N];
  vector[N] phi2 = phi[(N+1):(2*N)];

  // Likelihood
  y1 ~ normal(beta0_1 + phi1, sigma_1 + 1e-6);
  y2 ~ normal(beta0_2 + phi2, sigma_2 + 1e-6);

  // Priors for intercepts and noise
  beta0_1 ~ normal(0, 1);
  beta0_2 ~ normal(0, 1);
  sigma_1 ~ normal(1, 0.5);
  sigma_2 ~ normal(1, 0.5);

  // Priors for the random effects
  phi_raw ~ normal(0, 1);
  sigma_phi ~ normal(0, 1);

  // Prior for the bivariate correlation
  // This is a weakly informative prior on L_corr
  L_corr ~ lkj_corr_cholesky(2);

  // Bivariate CAR penalty on (phi1, phi2)
  // using a single 2×2 Sigma
  target += car_normal_mcar_lpdf(phi | N, N_edges, node1, node2, Sigma);

  // Sum-to-zero constraints (for identifiability)
  sum(phi1) ~ normal(0, 0.001 * N);
  sum(phi2) ~ normal(0, 0.001 * N);
}

generated quantities {
  // Split phi for readability
  vector[N] phi1 = phi[1:N];
  vector[N] phi2 = phi[(N + 1):(2 * N)];

  // Posterior fitted means
  vector[N] mu1 = beta0_1 + phi1;
  vector[N] mu2 = beta0_2 + phi2;
  
  // Log-likelihoods
  vector[N] log_lik_y1;
  vector[N] log_lik_y2;

  for (i in 1:N) {
    log_lik_y1[i] = normal_lpdf(y1[i] | mu1[i], sigma_1 + 1e-6);
    log_lik_y2[i] = normal_lpdf(y2[i] | mu2[i], sigma_2 + 1e-6);
  }
}


"
# Write the Stan model to a file
model_filename <- cmdstanr::write_stan_file(
  gsub('\t',' ', model_text),  # Use the updated Stan model code
  dir = out.dir,
  basename = "model",
  force_overwrite = TRUE,
  hash_salt = ""
)

# Compile the Stan model
model_compiled_cmdstanr.rds <- cmdstanr::cmdstan_model(model_filename)
#-------------------------------------------------------------------
# 1. Load & preprocess the CSV data
#-------------------------------------------------------------------
df <- read_csv("/Users/jyx/M4R/data_combined/ANDEREW_1992_08_combined.csv")
# Add w10
df$w10 <- sqrt(df$u10^2 + df$v10^2)
df[c("u10", "v10")] <- NULL

# Use time index 225 from unfiltered data
original_times <- unique(df$valid_time)
selected_time <- original_times[200]

# Calculate thresholds
tp_quantile  <- quantile(df$tp,  0.90, na.rm = TRUE)
w10_quantile <- quantile(df$w10, 0.90, na.rm = TRUE)

# Filter
df <- df %>%
  filter(tp >= tp_quantile , w10 >= w10_quantile) %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%  # ensure coordinates are clean
  arrange(valid_time, longitude, latitude)

# Subset to selected time
df_first_time <- df[df$valid_time == selected_time, ]

#-------------------------------------------------------------------
# 2. Build node-level adjacency 
#-------------------------------------------------------------------
grid <- unique(df_first_time[, c("latitude", "longitude")])
grid <- grid[order(-grid$latitude, grid$longitude), ]
grid$id <- seq_len(nrow(grid))  # ID from 1..N

# Helper function: find 4-neighbors at +/-0.25 deg
find_neighbors <- function(id, grid) {
  current <- grid %>% filter(id == !!id)
  neighbors <- grid %>%
    filter(
      (latitude == current$latitude + 0.25 & longitude == current$longitude) | 
        (latitude == current$latitude - 0.25 & longitude == current$longitude) |
        (latitude == current$latitude & longitude == current$longitude - 0.25) |
        (latitude == current$latitude & longitude == current$longitude + 0.25)
    )
  return(neighbors$id)
}

neighbors_list <- lapply(grid$id, function(i) {
  nbrs <- find_neighbors(i, grid)
  if (length(nbrs) == 0) {
    return(NULL)
  } else {
    list(node = i, neighbors = nbrs)
  }
})

neighbors_list <- Filter(Negate(is.null), neighbors_list)

# Convert adjacency to a data frame of edges
adjacency_list <- do.call(rbind, lapply(neighbors_list, function(x) {
  data.frame(node1 = x$node, node2 = x$neighbors)
}))

# Make sure node1 < node2 to avoid duplicates
filtered_list <- adjacency_list %>%
  mutate(
    smaller_node = pmin(node1, node2),
    larger_node  = pmax(node1, node2)
  ) %>%
  dplyr::select(node1 = smaller_node, node2 = larger_node) %>%
  distinct()

#-------------------------------------------------------------------
# 3. Prepare Stan data
#-------------------------------------------------------------------
N <- nrow(grid)
N_edges <- nrow(filtered_list)
node1 <- filtered_list$node1
node2 <- filtered_list$node2

# Merge outcomes with grid so that row i corresponds to grid$id=i
df_first_time <- df_first_time %>%
  inner_join(grid, by = c("latitude", "longitude")) %>%
  arrange(id)

# Outcome vectors 
y1 <- df_first_time$tp
y2 <- df_first_time$w10
y1[is.na(y1) | y1 < 0] <- 0
y2[is.na(y2) | y2 < 0] <- 0

# Build final data list
stan_data <- list(
  N = N,
  N_edges = N_edges,
  node1 = node1,
  node2 = node2,
  y1 = y1,
  y2 = y2
)

#-------------------------------------------------------------------
# 4. Compile & Sample
#-------------------------------------------------------------------
# Compile
model_compiled_cmdstanr_cholesky <- cmdstanr::cmdstan_model(model_filename)


# Initialization function for the Cholesky-based MCAR
init_function <- function() {
  list(
    beta0_1   = rnorm(1, 0, 0.5),
    beta0_2   = rnorm(1, 0, 0.5),
    sigma_1   = runif(1, 0.5, 1.5),
    sigma_2   = runif(1, 0.5, 1.5),
    phi_raw   = rnorm(2*N, 0, 0.1),
    sigma_phi = runif(1, 0.5, 1.5),
    L_corr    = diag(2)  # identity as initial guess
  )
}

model_fit_cholesky <- model_compiled_cmdstanr_cholesky$sample(
  data = stan_data,
  seed = 5,
  chains = 2,
  parallel_chains = 2,
  iter_warmup = 1000,
  iter_sampling = 1500,
  refresh = 200,
  save_warmup = TRUE,
  adapt_delta = 0.99,
  max_treedepth = 15,
  init = init_function
)

# Save to RDS
model_fit_cholesky$save_object(
  file = file.path("/Users/jyx/M4R", "model_fit_cholesky.rds")
)

# Example trace plot
draws_array_cholesky <- model_fit_cholesky$draws(format = "array")
mcmc_trace(draws_array_cholesky, pars = c("beta0_1", "lp__"))

##############################################
# Compute DIC:
##############################################
log_lik_y1 <- model_fit_cholesky$draws("log_lik_y1", format = "matrix")
log_lik_y2 <- model_fit_cholesky$draws("log_lik_y2", format = "matrix")

log_lik <- log_lik_y1 + log_lik_y2
# Compute deviance for each draw
deviance_samples <- -2 * rowSums(log_lik)

# Posterior mean deviance
D_bar <- mean(deviance_samples)

# Pointwise posterior means (i.e. E[loglik_i])
log_lik_mean <- colMeans(log_lik)
D_hat <- -2 * sum(log_lik_mean)

# Effective number of parameters
p_D <- D_bar - D_hat

# DIC
DIC <- D_hat + 2 * p_D

cat("DIC:", DIC, "\n")
loo_result_constant <- loo(log_lik)
print(loo_result_constant)
