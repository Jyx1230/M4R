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
library(geosphere)


out.dir <- "/Users/jyx/M4R"
in.dir  <- "/Users/jyx/M4R/"

model_text <-"
//------------------------------------------------------------
// Stan model: Bivariate Spatio Model with ICAR on ψ
// (Modified so that y₁ uses Log-Student-t and y₂ uses Lognormal)
//------------------------------------------------------------
functions {
  // A bivariate CAR prior using edge-based correlation (theta_corr).
  // phi is a 2*N vector, split into phi1 and phi2.
  // node1, node2 define the adjacency among global spatio-temporal nodes (length = N_edges).
  // theta_corr[e] is the correlation parameter for edge e, in (-1, 1).
  real car_normal_edgecorr_lpdf(
    vector phi, 
    int N, int N_edges,
    array[] int node1, array[] int node2,
    vector theta_corr
  ) {
    vector[N] phi1 = phi[1:N];
    vector[N] phi2 = phi[(N+1):(2*N)];

    vector[N_edges] diff_phi1 = phi1[node1] - phi1[node2];
    vector[N_edges] diff_phi2 = phi2[node1] - phi2[node2];
    real edge_term = dot_self(diff_phi1) + dot_self(diff_phi2);

    real cross_term = -2 * dot_product(phi1[node1] .* phi2[node2], theta_corr);

    vector[N] rho_sum_1 = rep_vector(0, N);
    vector[N] rho_sum_2 = rep_vector(0, N);
    rho_sum_1[node1] += abs(theta_corr);
    rho_sum_2[node2] += abs(theta_corr);

    real node_term_1 = dot_product(phi1 .* phi1, rho_sum_1);
    real node_term_2 = dot_product(phi2 .* phi2, rho_sum_2);
    real node_term = node_term_1 + node_term_2;

    return -0.5 * (edge_term + cross_term + node_term);
  }

  // Standard ICAR prior for an edge-level parameter vector x,
  // using e1, e2 adjacency among edges.
  real icar_normal_lpdf(
    vector x, 
    int N_edges,          // number of edges in the global node graph
    int E_edges,          // number of adjacency relationships among edges
    array[] int e1, 
    array[] int e2, 
    real tau
  ) {
    vector[E_edges] diff;
    for (i in 1:E_edges) {
      diff[i] = x[e1[i]] - x[e2[i]];
    }
    return -0.5 * tau * dot_self(diff);
  }
}

data {
  // Global spatio-temporal nodes (each represents a unique (time, location) pair)
  int<lower=0> N;               // Total number of nodes
  int<lower=0> N_edges;         // Number of edges among nodes
  array[N_edges] int node1;     // For each edge, first node (in [1, N])
  array[N_edges] int node2;     // For each edge, second node

  // Outcomes (combined from all time steps)
  vector<lower=0>[N] y1;        // Total precipitation for each node
  vector<lower=0>[N] y2;        // Wind speed for each node

  // For edge-level ICAR prior (on the edge-level parameters)
  int<lower=1> M;               // Total number of edges 
  int<lower=0> E_edges;         // Number of adjacency relationships among edges
  array[E_edges] int e1;        // For each edge-adjacency pair, first edge index
  array[E_edges] int e2;        // For each edge-adjacency pair, second edge index
  
  int<lower=1> S;               // Number of time steps
  vector[S] land_fraction;      // Fraction of land for each time step
  array[M] int time_idx;        // Mapping from each observation to its time index
}

parameters {
  // Regression intercepts and observation‐noise parameters
  real beta0_1;                // Intercept for log-Student-t on y1
  real beta0_2;                // Intercept for lognormal on y2
  real<lower=0> sigma_1;       // Scale of Student-t on log(y1)
  real<lower=2> nu_1;          // Degrees of freedom for log-Student-t (y1)
  real<lower=0> sigma_2;       // Scale of lognormal for y2

  // Non-centered spatial effects (for both outcomes)
  vector[2 * N] phi_raw; 
  real<lower=0> sigma_phi;
  real<lower=0.0001> tau_phi;   // Precision parameter for ICAR prior

  // Parameters for the correlation structure
  vector[S] alpha0;             // Intercept(s) for correlation strength (one per time step)
  real alpha1;                  // Coefficient for land-sea mask
  vector[M] phi;                // Spatial random effect for correlation (CAR model)
}

transformed parameters {
  // Scale node-level effects for outcomes
  vector[2 * N] phi_scaled = phi_raw * sigma_phi;
  vector[N] phi1 = phi_scaled[1:N];
  vector[N] phi2 = phi_scaled[(N+1):(2*N)];
  
  // Compute the linear predictor for correlation strength.
  // alpha0 is time-dependent; alpha1 and phi are added as before.
  vector[M] rho_lin = alpha0[time_idx] + alpha1 * land_fraction[time_idx] + phi;
  
  // Apply the tanh link to constrain the correlation to (-1, 1)
  vector[M] rho_s = tanh(rho_lin);
}

model {
  //--- Likelihoods with exchanged distributions ---
  // y1:    log(y1) ~ Student-t( nu = nu_1, mu = beta0_1 + phi1, sigma = sigma_1 )
  //           ⇨ equivalent to: f(y1) ∝ Student_t_lpdf(log(y1) | …) * (1/y1)
  {
    vector[N] mu_y1 = beta0_1 + phi1;
    for (i in 1:N) {
      target += student_t_lpdf(log(y1[i]) | nu_1, mu_y1[i], sigma_1);
      target += -log(y1[i]);
    }
  }

  // y2:    y2 ~ Lognormal( mu = beta0_2 + phi2, sigma = sigma_2 )
  y2 ~ lognormal(beta0_2 + phi2, sigma_2);

  // Priors for intercepts and noise parameters
  beta0_1 ~ normal(0, 1);
  beta0_2 ~ normal(0, 1);
  sigma_1 ~ normal(1, 0.5);
  sigma_2 ~ normal(1, 0.5);
  nu_1    ~ gamma(2, 0.1);      // Prior for degrees of freedom

  // Priors for the outcome-level spatial effects
  phi_raw   ~ normal(0, 1);
  sigma_phi ~ exponential(10); 
  tau_phi   ~ gamma(2, 0.5);

  // Priors for the correlation structure parameters
  alpha0 ~ normal(0, 1);
  alpha1 ~ normal(0,1);

  // ICAR prior for the correlation random effect φ
  target += icar_normal_lpdf(phi | M, E_edges, e1, e2, tau_phi);
  
  // Bivariate CAR prior for the outcome spatial effects,
  // using the new correlation strength (rho_s) computed above.
  target += car_normal_edgecorr_lpdf(phi_scaled | N, N_edges, node1, node2, rho_s);

  // Sum-to-zero constraints for identifiability of the outcome fields
  sum(phi1) ~ normal(0, 0.001 * N);
  sum(phi2) ~ normal(0, 0.001 * N);
}

generated quantities {
  // Posterior fitted (additive) predictors for the outcomes
  vector[N] mu1 = beta0_1 + phi1;    // On log-scale for y1 (Student-t)
  vector[N] mu2 = beta0_2 + phi2;    // On log-scale for y2 (Lognormal)

  // Output the correlation strength (after applying tanh)
  vector[M] rho_s_out = tanh(alpha0[time_idx] + alpha1 * land_fraction[time_idx] + phi);
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
model_compiled_cmdstanr <- cmdstanr::cmdstan_model(model_filename)


# Load necessary libraries
library(dplyr)
library(readr)
library(terra)
library(tidyr)  # Load tidyr for gather function


# Load the data
df <- read_csv("/Users/jyx/M4R/data_combined/CHARLEY_2004_08_combined.csv")
# Load the mask CSV file
mask <- read.csv("/Users/jyx/M4R/land_sea_mask.csv", header = TRUE)


# Compute wind speed and remove original u10, v10 columns
df$w10 <- sqrt(df$u10^2 + df$v10^2)
df <- df %>% dplyr::select(-u10, -v10)

unique_times <- sort(unique(df$valid_time))

# Filter data using quantile thresholds for tp and w10
tp_quantile  <- quantile(df$tp,  0.90, na.rm = TRUE)
w10_quantile <- quantile(df$w10, 0.90, na.rm = TRUE)
df <- df %>% filter(tp >= tp_quantile, w10 >= w10_quantile)

# Sort data by time, then longitude, then latitude
df <- df %>% arrange(valid_time, longitude, latitude)
time_steps <- unique_times[c(112, 114, 118,124,125,126)]
S <- length(time_steps)


# Keep only those time steps
df_filtered <- df %>%
  filter(valid_time %in% time_steps)


# 7) For safety, select only columns we need (lon, lat, mask_value)
mask_distinct <- mask %>%
  dplyr::select(longitude, latitude, lsm) %>%
  distinct()

# 8) Merge df_filtered with mask_distinct by location
df_filtered_mask <- df_filtered %>%
  left_join(mask_distinct, by = c("longitude" = "longitude", "latitude" = "latitude"))

df_mask_counts <- df_filtered_mask %>%
  group_by(valid_time) %>%
  summarise(
    n_land = sum(lsm > 0.95, na.rm = TRUE),
    n_sea  = sum(lsm == 0, na.rm = TRUE),
    total  = n()
  )

# 'df_mask_counts' contains: valid_time, n_land, n_sea, total
# We compute the fraction of land for each valid_time
df_mask_counts <- df_mask_counts %>%
  mutate(fraction_land = n_land / total)

# Ensure it is in the same order as your 'time_steps' vector
df_mask_counts_ordered <- df_mask_counts %>%
  filter(valid_time %in% time_steps) %>%
  arrange(match(valid_time, time_steps))

# This should yield a data frame in the order of time_steps
fraction_vec <- df_mask_counts_ordered$fraction_land





#-------------------------------------------------------------------
# 3. Define Helper Functions for Spatial Adjacency
#-------------------------------------------------------------------
# This function finds 4-neighbors (N, S, E, W) given a grid (local nodes)
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

# This function builds the adjacency (local node IDs) for one grid
build_adjacency <- function(grid) {
  neighbors_list <- lapply(grid$id, function(i) {
    nbrs <- find_neighbors(i, grid)
    if (length(nbrs) == 0) {
      return(NULL)
    } else {
      return(list(node = i, neighbors = nbrs))
    }
  })
  neighbors_list <- Filter(Negate(is.null), neighbors_list)
  
  adjacency_list <- do.call(rbind, lapply(neighbors_list, function(x) {
    data.frame(node1 = x$node, node2 = x$neighbors)
  }))
  
  # Enforce node1 < node2 to avoid duplicate edges
  filtered_list <- adjacency_list %>%
    mutate(smaller_node = pmin(node1, node2),
           larger_node  = pmax(node1, node2)) %>%
    dplyr::select(node1 = smaller_node, node2 = larger_node) %>%
    distinct()
  
  return(filtered_list)
}


# Flatten data into a single spatio-temporal graph
df_flat <- data.frame()      # To hold all nodes across time steps
spatial_edges_list <- list() # To store spatial edges from each time step
offset <- 0                  # Offset to shift local node IDs to global

# Offsets[s] = the global ID that starts time step s
offsets <- numeric(S+1)
offsets[1] <- 1       # Time step 1 starts at global ID = 1

for (s in seq_len(S)) {
  # Subset data for time step s
  df_s <- df %>% filter(valid_time == time_steps[s])
  
  # Build a unique grid (local nodes) for time step s
  grid <- df_s %>%
    distinct(latitude, longitude) %>%
    arrange(-latitude, longitude) %>%
    mutate(id = row_number())  # Local node IDs from 1 to N_s
  
  # Number of nodes at time s
  N_s <- nrow(grid)
  
  # Merge grid back to assign each observation its local node ID
  df_s <- df_s %>% inner_join(grid, by = c("latitude", "longitude"))
  
  # Assign a unique global node ID: global_id = local id + offset
  df_s <- df_s %>% mutate(global_id = id + offset, time_id = s)
  
  # Append to the flattened node data frame
  df_flat <- bind_rows(df_flat, df_s)
  
  # Build spatial adjacency for this time step (using local node IDs)
  adj_local <- build_adjacency(grid)
  # Convert local node IDs to global node IDs by adding offset
  if(nrow(adj_local) > 0){
    adj_local <- adj_local %>% 
      mutate(node1 = node1 + offset, node2 = node2 + offset)
  }
  spatial_edges_list[[s]] <- adj_local
  
  # Update offset for the next time step
  offset <- offset + N_s
  offsets[s+1] <- offset + 1  # Next block starts at offset+1
}


df_flat <- df_flat %>%
  dplyr::select(-sst, -msl, -u100, -v100)

# Total number of spatio-temporal nodes
N_global <- max(df_flat$global_id)

# Combine spatial edges from all time steps
spatial_edges_df <- bind_rows(spatial_edges_list)

########################
## Assign time_id to Each Edge
########################
# Each edge is within one time slice, so node1 and node2 must match
# Assign time_id to each edge
# Create a lookup table for node time assignment
node_time_lookup <- df_flat %>% 
  distinct(global_id, time_id) %>% 
  rename(node = global_id)

# Join the lookup table to assign time_ids to edges (vectorized)
spatial_edges_df <- spatial_edges_df %>%
  left_join(node_time_lookup, by = c("node1" = "node")) %>% 
  rename(node1_time = time_id) %>%
  left_join(node_time_lookup, by = c("node2" = "node")) %>% 
  rename(node2_time = time_id)

# Verify that both nodes in each edge belong to the same time step
stopifnot(all(spatial_edges_df$node1_time == spatial_edges_df$node2_time))

# Now assign the time_id for the edge (using one of the two, e.g., node1_time)
spatial_edges_df <- spatial_edges_df %>%
  mutate(time_id = node1_time)


# Build edge-level adjacency
edges_df <- spatial_edges_df %>%
  mutate(edge_id = row_number())

# Map nodes to their associated edges
node_to_edges <- edges_df %>%
  gather(key = "end", value = "node", node1, node2) %>%
  group_by(node) %>%
  summarize(edge_ids = list(edge_id), .groups = "drop")

# Map each edge to its neighboring edges
edge_neighbors <- edges_df %>%
  rowwise() %>%
  mutate(
    neighbors = list(unique(c(
      unlist(node_to_edges$edge_ids[node_to_edges$node == node1]),
      unlist(node_to_edges$edge_ids[node_to_edges$node == node2])
    ))),
    neighbors = list(setdiff(neighbors, edge_id))
  ) %>%
  ungroup()

# Convert neighbors to a tidy (long) format
edge_adjacency <- edge_neighbors %>%
  dplyr::select(edge_id, neighbors) %>%
  unnest_longer(neighbors) %>%
  filter(!is.na(neighbors)) %>%  # Remove NAs
  distinct()

# Ensure edge_id < neighbors and remove duplicates
edge_adjacency <- edge_adjacency %>%
  mutate(smaller = pmin(edge_id, neighbors),
         larger  = pmax(edge_id, neighbors)) %>%
  dplyr::select(edge_id = smaller, neighbors = larger) %>%
  distinct() %>%
  arrange(edge_id, neighbors)

# Total number of edges
M <- nrow(edges_df)



data_stan <- list(
  N = N_global,
  N_edges = nrow(edges_df),
  node1   = edges_df$node1,
  node2   = edges_df$node2,
  y1 = df_flat$tp,
  y2 = df_flat$w10,
  M = M,
  E_edges = nrow(edge_adjacency),
  e1      = edge_adjacency$edge_id,
  e2      = edge_adjacency$neighbors,
  time_idx = edges_df$time_id,
  S        = S,                # number of time steps
  land_fraction = fraction_vec   # fraction_vec from above
)



##############################################
### Sampling

# Updated initialization function for the model with an ICAR prior on psi
init_function <- function() {
  list(
    beta0_1  = rnorm(1, 0, 0.5),
    beta0_2  = rnorm(1, 0, 0.5),
    sigma_1  = runif(1, 0.5, 1.5),     # scale for log-Student-t on y₁
    nu_1     = runif(1, 2, 10),        # degrees of freedom for log-Student-t on y₁
    sigma_2  = runif(1, 0.5, 1.5),     # scale for lognormal on y₂
    
    phi_raw  = rnorm(2 * N_global, 0, 0.05),  # Non-centered spatial effects
    sigma_phi = runif(1, 0.01, 0.1),
    tau_phi  = runif(1, 0.1, 10),             # ICAR precision
    
    alpha0   = rnorm(S, 0, 0.5),
    alpha1   = rnorm(1,0, 1),
    phi      = rnorm(M, 0, 0.3)               # length = M
  )
}


# Run the sampler
model_fit_m6 <- model_compiled_cmdstanr$sample(
  data = data_stan,
  seed = 3,
  chains = 2, 
  parallel_chains = 2,
  iter_warmup = 800,
  iter_sampling = 1400,
  refresh = 200, 
  save_warmup = TRUE,
  save_cmdstan_config = TRUE,
  adapt_delta = 0.99,  
  max_treedepth = 15,
  init = init_function
)

# Save output to RDS
model_fit_m6$save_object(
  file = file.path(out.dir, "model_compiled_cmdstanr.rds")
)


# Diagnostic plots
draws_array_m6 <- model_fit_m6$draws(format = "array")
mcmc_trace(draws_array_m6, pars = c("beta0_1", "lp__"))



summary_subset_m6 <- model_fit_m6$summary(
  variables = c("beta0_1", "beta0_2", "sigma_1", "sigma_2","alpha1")
)
print(summary_subset_m6)


# Directly extract as matrix
draws_mat <- model_fit_m6$draws("rho_s_out", format = "draws_matrix")

# Compute mean per edge
rho_means <- colMeans(draws_mat)

# Combine with edge metadata
rho_df <- tibble(
  edge_id = seq_along(rho_means),
  rho_mean = rho_means,
  time_id = edges_df$time_id,
  land_frac = fraction_vec[edges_df$time_id]
)

library(ggplot2)
library(tibble)
library(dplyr)

# Extract draws and compute posterior mean
draws_mat <- model_fit_m6$draws("rho_s_out", format = "draws_matrix")
rho_means <- colMeans(draws_mat)

# Combine with edge metadata
rho_df <- tibble(
  edge_id   = seq_along(rho_means),
  rho_mean  = rho_means,
  time_id   = edges_df$time_id,
  land_frac = fraction_vec[edges_df$time_id]
)

rho_df <- rho_df %>%
  mutate(time_label = factor(time_id, labels = paste("Time", time_steps)))

# Plot: color by time step
ggplot(rho_df, aes(x = land_frac, y = rho_mean, color = time_label)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, size = 1.1) +
  labs(
    x = "Land Fraction",
    y = expression(hat(rho)[s]),
    color = "Time Step",
    title = "Edge-level Correlation vs. Land Fraction",
    subtitle = "Colored by Time Step"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggplot(rho_df, aes(x = land_frac, y = rho_mean, color = time_label)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, color = "black", size = 1.2) +
  labs(
    x = "Land Fraction",
    y = expression(hat(rho)[s]),
    color = "Time Step",
    title = "Edge-level Correlation vs. Land Fraction",
    subtitle = "Colored by Time Step with Global Fit"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

