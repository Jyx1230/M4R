
out.dir <- "/Users/jyx/M4R"
in.dir  <- "/Users/jyx/M4R/"
model_text <-"
//------------------------------------------------------------
// Stan model: Bivariate Spatio‐Temporal Model with ICAR on ψ
//    – y₁ (precipitation) ∼ log-Student-t
//    – y₂ (wind speed) ∼ lognormal
//------------------------------------------------------------
functions {
  // Bivariate CAR prior with edge-specific correlation θ_corr
  real car_normal_edgecorr_lpdf(
    vector phi, 
    int N, 
    int N_edges, 
    array[] int node1, 
    array[] int node2, 
    vector theta_corr
  ) {
    vector[N] phi1 = phi[1:N];
    vector[N] phi2 = phi[(N+1):(2*N)];

    vector[N_edges] diff1 = phi1[node1] - phi1[node2];
    vector[N_edges] diff2 = phi2[node1] - phi2[node2];
    real edge_term = dot_self(diff1) + dot_self(diff2);

    real cross_term = -2 * dot_product(phi1[node1] .* phi2[node2], theta_corr);

    vector[N] rho_sum_1 = rep_vector(0, N);
    vector[N] rho_sum_2 = rep_vector(0, N);
    rho_sum_1[node1] += abs(theta_corr);
    rho_sum_2[node2] += abs(theta_corr);

    real node_term_1 = dot_product(phi1 .* phi1, rho_sum_1);
    real node_term_2 = dot_product(phi2 .* phi2, rho_sum_2);
    real node_term   = node_term_1 + node_term_2;

    return -0.5 * (edge_term + cross_term + node_term);
  }

  // ICAR prior on the edge graph for θ
  real icar_normal_lpdf(
    vector x, 
    int N_edges, 
    int E_edges, 
    array[] int e1, 
    array[] int e2, 
    real tau
  ) {
    vector[E_edges] diff;
    for (i in 1:E_edges)
      diff[i] = x[e1[i]] - x[e2[i]];
    return -0.5 * tau * dot_self(diff);
  }
}

data {
  // node-graph
  int<lower=1> N;
  int<lower=1> N_edges;
  array[N_edges] int node1;
  array[N_edges] int node2;

  // observations
  vector<lower=0>[N] y1;   // precipitation
  vector<lower=0>[N] y2;   // wind speed

  // edge-graph for θ
  int<lower=1> M;           // should equal N_edges
  int<lower=1> E_edges;
  array[E_edges] int e1;
  array[E_edges] int e2;
}

parameters {
  // intercepts
  real beta0_1;
  real beta0_2;

  // y1 ∼ log-Student-t
  real<lower=0> sigma1;
  real<lower=2> nu_tp;

  // y2 ∼ lognormal
  real<lower=0> sigma2;

  // spatial effects (non‐centered)
  vector[2 * N] phi_raw;
  real<lower=0> sigma_phi;

  // edge‐level latent for correlation
  vector[M] theta_raw;
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> tau_theta;
}

transformed parameters {
  // scale spatial effects
  vector[2 * N] phi = phi_raw * sigma_phi;
  vector[N] phi1 = phi[1:N];
  vector[N] phi2 = phi[(N+1):(2*N)];

  // scale & map θ to (–1,1)
  vector[M] theta = mu_theta + sigma_theta * theta_raw;
  vector[M] theta_corr = tanh(theta);
}

model {
  // 1) Likelihoods
  for (i in 1:N) {
    // y1: log(y1) ∼ Student-t
    target += student_t_lpdf(log(y1[i]) | nu_tp,
                             beta0_1 + phi1[i],
                             sigma1)
              - log(y1[i]);
    // y2: lognormal
    target += lognormal_lpdf(y2[i] | beta0_2 + phi2[i],
                             sigma2);
  }

  // 2) Priors – intercepts & dispersions
  beta0_1 ~ student_t(3, 0, 2.5);
  beta0_2 ~ student_t(3, 0, 2.5);
  sigma1  ~ student_t(3, 0, 2.5);
  nu_tp   ~ gamma(2, 0.1);
  sigma2  ~ student_t(3, 0, 2.5);

  // 3) Priors – spatial effects
  phi_raw   ~ normal(0, 1);
  sigma_phi ~ cauchy(0, 2.5);

  // 4) Priors – edge‐correlation θ
  theta_raw   ~ student_t(3, 0, 1);
  sigma_theta ~ cauchy(0, 2.5);
  mu_theta    ~ student_t(3, 0, 1);
  tau_theta   ~ gamma(0.1, 0.1);

  // 5) CAR / ICAR
  target += car_normal_edgecorr_lpdf(phi       |
                                     N, N_edges,
                                     node1, node2,
                                     theta_corr);
  target += icar_normal_lpdf(theta   |
                             M,
                             E_edges,
                             e1, e2,
                             tau_theta);

  // 6) Soft sum-to-zero constraints
  sum(phi1)   ~ normal(0, 0.001 * N);
  sum(phi2)   ~ normal(0, 0.001 * N);
  sum(theta)  ~ normal(0, 0.001 * M);
}

generated quantities {
  vector[N] mu1 = beta0_1 + phi1;
  vector[N] mu2 = beta0_2 + phi2;

  vector[N] log_lik_y1;
  vector[N] log_lik_y2;

  for (i in 1:N) {
    log_lik_y1[i] = student_t_lpdf(log(y1[i]) | nu_tp,
                                   mu1[i],
                                   sigma1)
                    - log(y1[i]);
    log_lik_y2[i] = lognormal_lpdf(y2[i] | mu2[i],
                                   sigma2);
  }
}


"
# Write the Stan model to a file
model_filename <- cmdstanr::write_stan_file(
  gsub('\t',' ', model_text),  # Use the updated Stan model code
  dir = out.dir,
  basename = "model_nodis",
  force_overwrite = TRUE,
  hash_salt = ""
)

# Compile the Stan model
model_compiled_cmdstanr_nodis <- cmdstanr::cmdstan_model(model_filename)
##############################################

#-------------------------------------------------------------------
# 1. Load & preprocess the CSV data
#-------------------------------------------------------------------
df <- read_csv("/Users/jyx/M4R/data_combined/CHARLEY_2004_08_combined.csv")
# Add w10
df$w10 <- sqrt(df$u10^2 + df$v10^2)
df[c("u10", "v10")] <- NULL

# Use time index 225 from unfiltered data
original_times <- unique(df$valid_time)
selected_time <- original_times[127]

# Calculate thresholds
tp_quantile  <- quantile(df$tp,  0.85, na.rm = TRUE)
w10_quantile <- quantile(df$w10, 0.85, na.rm = TRUE)

# Filter
df <- df %>%
  filter(tp >= tp_quantile , w10 >= w10_quantile) %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%  # ensure coordinates are clean
  arrange(valid_time, longitude, latitude)

# Subset to selected time
df_first_time <- df[df$valid_time == selected_time, ]

#-------------------------------------------------------------------
# 3. Build adjacency for just the first time point
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

# Create a mapping of nodes to their associated edges
node_to_edges <- edges_df %>%
  tidyr::gather(key = "end", value = "node", node1, node2) %>%
  group_by(node) %>%
  summarize(edge_ids = list(edge_id), .groups = "drop")

# Create a mapping of edges to their neighboring edges
edge_neighbors <- edges_df %>%
  rowwise() %>%
  mutate(
    # Get all edges connected to node1 and node2
    neighbors = list(unique(c(
      unlist(node_to_edges$edge_ids[node_to_edges$node == node1]),
      unlist(node_to_edges$edge_ids[node_to_edges$node == node2])
    ))),
    # Remove the current edge from its own neighbors
    neighbors = list(setdiff(neighbors, edge_id))
  ) %>%
  ungroup()

# Convert the list of neighbors into a tidy format
edge_adjacency <- edge_neighbors %>%
  dplyr::select(edge_id, neighbors) %>%
  tidyr::unnest_longer(neighbors) %>%
  filter(!is.na(neighbors)) %>%  # Remove any NA values
  distinct()  # Ensure no duplicate entries

# Ensure edge_id < neighbors and remove duplicates
edge_adjacency <- edge_adjacency %>%
  mutate(
    smaller = pmin(edge_id, neighbors),
    larger = pmax(edge_id, neighbors)
  ) %>%
  dplyr::select(edge_id = smaller, neighbors = larger) %>%
  distinct()  # Remove duplicate pairs

# Sort the adjacency list for clarity
edge_adjacency <- edge_adjacency %>%
  arrange(edge_id, neighbors)


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
  
  M = M,
  E_edges  = nrow(edge_adjacency),  # or whatever you call your edge-adjacency df
  e1       = edge_adjacency$edge_id,
  e2       = edge_adjacency$neighbors
)
##############################################
### Sampling

# Updated initialization function for the model with:
#   • Log–Student-t likelihood on tp (y₁)
#   • Gamma–log-link likelihood on ws (y₂)
#   • ICAR prior on spatial psi (φ)
init_function <- function() {
  list(
    # ─ Intercepts (Student-t(3,0,2.5) priors) ─
    beta0_1    = rnorm(1, 0, 2.0),
    beta0_2    = rnorm(1, 0, 2.0),
    
    # ─ y1: log-Student-t(ν_tp, μ1, σ1) ─
    sigma1     = runif(1, 0.1, 5.0),
    nu_tp      = runif(1, 2.1, 30.0),
    
    # ─ y2: Lognormal(μ2, σ2) ─
    sigma2     = runif(1, 0.1, 5.0),
    
    # ─ Spatial random effects (non-centered) ─
    #    phi_raw ~ Normal(0,1), sigma_phi ~ half-Cauchy(0,2.5)
    phi_raw    = rnorm(2 * N, 0, 1.0),
    sigma_phi  = runif(1, 0.1, 2.5),
    
    # ─ Edge‐CAR parameters for correlation θ ─
    theta_raw   = rnorm(M, 0, 1.0),
    mu_theta    = rnorm(1, 0, 1.0),
    sigma_theta = runif(1, 0.1, 2.5),
    tau_theta   = runif(1, 0.01, 10.0)
  )
}


# Run the sampler
model_fit_nodis <- model_compiled_cmdstanr_nodis$sample(
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
model_fit_nodis$save_object(
  file = file.path(out.dir, "model_compiled_cmdstanr_nodis.rds")
)


# Diagnostic plots
draws_array_nodis <- model_fit_nodis$draws(format = "array")
mcmc_trace(draws_array_nodis, pars = c("beta0_1", "lp__"))


summary_subset_nodis <- model_fit_nodis$summary(
  variables = c("beta0_1", "beta0_2", "sigma_1","tau_theta", "sigma_2","phi_raw")
)
print(summary_subset_nodis)


##############################################
# Compute DIC:
##############################################
log_lik_y1 <- model_fit_nodis$draws("log_lik_y1", format = "matrix")
log_lik_y2 <- model_fit_nodis$draws("log_lik_y2", format = "matrix")

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

