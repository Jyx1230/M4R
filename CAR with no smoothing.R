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
model_text <-"
//------------------------------------------------------------
// Stan model: Bivariate Spatio Model
//             with per-edge correlation on phi (no ICAR on θ)
//------------------------------------------------------------
functions {
  // A bivariate CAR prior using edge-based correlation (theta_corr).
  // phi is a 2*N vector, split into phi1 and phi2.
  // node1, node2 define the adjacency among N nodes (N_edges total edges).
  // theta_corr[e] is the correlation parameter for edge e in (-1, 1).
  real car_normal_edgecorr_lpdf(
    vector phi, 
    int N, 
    int N_edges,
    array[] int node1, 
    array[] int node2,
    vector theta_corr
  ) {
    // 1) Split phi into two N-dimensional fields
    vector[N] phi1 = phi[1:N];
    vector[N] phi2 = phi[(N+1):(2*N)];

    // 2a) Edge-based differences
    vector[N_edges] diff_phi1 = phi1[node1] - phi1[node2];
    vector[N_edges] diff_phi2 = phi2[node1] - phi2[node2];
    real edge_term = dot_self(diff_phi1) + dot_self(diff_phi2);

    // 2b) Cross-term
    real cross_term = -2 * dot_product(phi1[node1] .* phi2[node2], theta_corr);

    // 3) Node-based terms
    // For each edge e, add |theta_corr[e]| to the nodes it connects.
    vector[N] rho_sum_1 = rep_vector(0, N);
    vector[N] rho_sum_2 = rep_vector(0, N);
    rho_sum_1[node1] += abs(theta_corr);
    rho_sum_2[node2] += abs(theta_corr);

    real node_term_1 = dot_product(phi1 .* phi1, rho_sum_1);
    real node_term_2 = dot_product(phi2 .* phi2, rho_sum_2);
    real node_term   = node_term_1 + node_term_2;

    return -0.5 * (edge_term + cross_term + node_term);
  }
}

data {
  // Global spatio-temporal nodes
  int<lower=0> N;                // Total number of nodes
  int<lower=0> N_edges;          // Number of edges among nodes
  array[N_edges] int node1;      // For each edge, first node
  array[N_edges] int node2;      // For each edge, second node

  // Outcomes
  vector<lower=0>[N] y1;         // e.g., precipitation
  vector<lower=0>[N] y2;         // e.g., wind speed

  int<lower=1> M;    // Typically equals N_edges
}

parameters {
  // Regression intercepts & observation noise
  real beta0_1;
  real beta0_2;
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;

  // Non-centered spatial effects
  vector[2*N] phi_raw;
  real<lower=0> sigma_phi;

  // Edge-level correlation parameters (no longer ICAR)
  vector[M] theta_raw;
  real<lower=0> sigma_theta; // scale for the correlation
  // Removed tau_theta
}

transformed parameters {
  // Node-level fields
  vector[2*N] phi = phi_raw * sigma_phi;
  vector[N]   phi1 = phi[1:N];
  vector[N]   phi2 = phi[(N+1):(2*N)];

  // Per-edge correlation, each mapped to (-1,1)
  vector[M] theta = theta_raw * sigma_theta;
  vector[M] theta_corr = tanh(theta);
}

model {
  // Likelihood
  y1 ~ normal(beta0_1 + phi1, sigma_1 + 1e-6);
  y2 ~ normal(beta0_2 + phi2, sigma_2 + 1e-6);

  // Priors
  beta0_1 ~ normal(0, 1);
  beta0_2 ~ normal(0, 1);
  sigma_1 ~ normal(1, 0.5);
  sigma_2 ~ normal(1, 0.5);

  phi_raw  ~ normal(0, 1);
  sigma_phi ~ normal(0, 1);

  theta_raw ~ normal(0, 1);   // simpler prior
  sigma_theta ~ normal(0, 1); // simpler prior

  // Bivariate CAR prior on phi using edge-based correlation
  target += car_normal_edgecorr_lpdf(phi | N, N_edges, node1, node2, theta_corr);

  // Removed the line about icar_normal_lpdf(...)

  // Sum-to-zero constraints for identifiability
  sum(phi1) ~ normal(0, 0.001 * N);
  sum(phi2) ~ normal(0, 0.001 * N);
  sum(theta) ~ normal(0, 0.001 * M);
}

generated quantities {
  // Fitted means
  vector[N] mu1 = beta0_1 + phi1;
  vector[N] mu2 = beta0_2 + phi2;

  // Per-edge correlations in (-1, 1)
  vector[M] theta_corr_out = theta_corr;
  
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
  basename = "model_nosmooth",
  force_overwrite = TRUE,
  hash_salt = ""
)

# Compile the Stan model
model_compiled_cmdstanr <- cmdstanr::cmdstan_model(model_filename)
##############################################

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
# 2. Build adjacency 
#-------------------------------------------------------------------
# Create a grid of unique lat-lon for this single time point
grid <- unique(df_first_time[, c("latitude", "longitude")])
grid <- grid[order(-grid$latitude, grid$longitude), ]
grid$id <- seq_len(nrow(grid))  # ID from 1..N

# A helper function to find 4-neighbors (N, S, E, W) at +/-0.25 deg
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

# Build adjacency for each node
neighbors_list <- lapply(grid$id, function(i) {
  nbrs <- find_neighbors(i, grid)
  if (length(nbrs) == 0) {
    return(NULL)  # No neighbors => skip
  } else {
    return(list(node = i, neighbors = nbrs))
  }
})

# Filter out any nodes with no neighbors
neighbors_list <- Filter(Negate(is.null), neighbors_list)

# Convert to a data.frame of edges
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



# Suppose filtered_list has columns: node1, node2
edges_df <- filtered_list %>%
  mutate(edge_id = row_number())  # Label each edge uniquely


# Suppose edges_df has one row per edge with column "edge_id"
M <- nrow(edges_df)


#-------------------------------------------------------------------
# 4. Prepare Stan data
#-------------------------------------------------------------------
# Define N based on the number of unique lat-lon in the first time step
N <- nrow(grid)
node1 <- filtered_list$node1
node2 <- filtered_list$node2
N_edges <- nrow(filtered_list)

# Extract outcomes (tp, w10) for the first time step
#   - We need them in the same row order as 'grid'
#     so let's merge or match by lat-lon
df_first_time <- df_first_time %>%
  inner_join(grid, by = c("latitude", "longitude"))

# Sort so that row i corresponds to grid$id = i
df_first_time <- df_first_time[order(df_first_time$id), ]

# Now outcomes match each row i = 1..N
y1 <- df_first_time$tp
y2 <- df_first_time$w10

# Replace NAs with 0
y1[is.na(y1)] <- 0
y2[is.na(y2)] <- 0


y1[y1 < 0] <- 0
y2[y2 < 0] <- 0

# Build the final data list for Stan
data <- list(
  N = N, 
  N_edges = N_edges, 
  node1 = node1, 
  node2 = node2, 
  y1 = y1, 
  y2 = y2,
  
  M = M
)
##############################################
### Sampling

# Updated initialization function for the model with an ICAR prior on psi
init_function <- function() {
  list(
    beta0_1 = rnorm(1, 0, 0.5),
    beta0_2 = rnorm(1, 0, 0.5),
    sigma_1 = runif(1, 0.5, 1.5),
    sigma_2 = runif(1, 0.5, 1.5),
    phi_raw = rnorm(2 * N, 0, 0.1),
    sigma_phi = runif(1, 0.5, 1.5),
    theta_raw = rnorm(M, 0, 0.1),
    sigma_theta = runif(1, 0.5, 1.5)
  )
}

# Run the sampler
model_fit_nosmooth <- model_compiled_cmdstanr$sample(
  data = data,
  seed = 5,
  chains = 2, 
  parallel_chains = 2,
  iter_warmup = 1000,
  iter_sampling = 1500,
  refresh = 200, 
  save_warmup = TRUE,
  save_cmdstan_config = TRUE,
  adapt_delta = 0.99,  
  max_treedepth = 15,
  init = init_function
)

# Save output to RDS
model_fit_nosmooth$save_object(
  file = file.path(out.dir, "model_compiled_cmdstanr.rds")
)


# Diagnostic plots
draws_array_nosmooth <- model_fit_nosmooth$draws(format = "array")
mcmc_trace(draws_array_nosmooth, pars = c("beta0_1", "lp__"))



##############################################
# Compute DIC:
##############################################
log_lik_y1 <- model_fit_nosmooth$draws("log_lik_y1", format = "matrix")
log_lik_y2 <- model_fit_nosmooth$draws("log_lik_y2", format = "matrix")

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


# Compute LOO
loo_result_constant <- loo(log_lik)

# Print result
print(loo_result_constant)
