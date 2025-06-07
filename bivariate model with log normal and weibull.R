out.dir <- "/Users/jyx/M4R"
in.dir  <- "/Users/jyx/M4R/"
model_text <-"
//------------------------------------------------------------
// Stan model: Bivariate Spatio‐Temporal Model with ICAR on ψ
//    – y₁ (precipitation) ∼ Log‐Normal
//    – y₂ (wind speed) ∼ Weibull
//------------------------------------------------------------

functions {
  // A bivariate CAR prior using edge‐based correlation (theta_corr).
  //    φ is a 2*N vector: first N entries = φ₁; next N entries = φ₂.
  //    node1, node2 define adjacency among the N “nodes”.
  //    theta_corr[e] is the edge‐specific correlation parameter in (−1,1).
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

    // Edge‐based differences (standard CAR penalty)
    vector[N_edges] diff1 = phi1[node1] - phi1[node2];
    vector[N_edges] diff2 = phi2[node1] - phi2[node2];
    real edge_term = dot_self(diff1) + dot_self(diff2);

    // Cross‐term over edges
    real cross_term = -2 * dot_product(phi1[node1] .* phi2[node2], theta_corr);

    // Node‐based “ρ‐sum” terms
    vector[N] rho_sum_1 = rep_vector(0, N);
    vector[N] rho_sum_2 = rep_vector(0, N);
    rho_sum_1[node1] += abs(theta_corr);
    rho_sum_2[node2] += abs(theta_corr);

    real node_term_1 = dot_product(phi1 .* phi1, rho_sum_1);
    real node_term_2 = dot_product(phi2 .* phi2, rho_sum_2);
    real node_term = node_term_1 + node_term_2;

    return -0.5 * (edge_term + cross_term + node_term);
  }

  // Standard ICAR prior on an edge‐graph for θ
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
  // 1) Node‐based spatial graph (size = N)
  int<lower=1> N;                     // number of spatial nodes
  int<lower=1> N_edges;               // number of edges among nodes
  array[N_edges] int node1;           // each edge’s first endpoint in [1..N]
  array[N_edges] int node2;           // each edge’s second endpoint in [1..N]

  // 2) Observations at those N nodes (filtered data)
  vector<lower=0>[N] y1;              // precipitation (we want to fit Log‐Normal)
  vector<lower=0>[N] y2;              // wind speed (we fit Weibull)

  // 3) Edge‐based graph for θ (size = M = N_edges)
  int<lower=1> M;                     // total number of edges (should equal N_edges)
  int<lower=1> E_edges;               // number of adjacency‐pairs among edges
  array[E_edges] int e1;              // each adjacency’s first edge index
  array[E_edges] int e2;              // each adjacency’s second edge index
}

parameters {
  // 1) Intercepts
  real beta0_1;                       // precipitation intercept (on log scale)
  real beta0_2;                       // wind intercept (on Weibull’s log‐scale)

  // 2) Dispersions / shape parameters
  real<lower=0> sigma1;               // log‐scale SD of y1
  real<lower=0> k_wind;               // Weibull shape for y2

  // 3) Non‐centered spatial effects
  vector[2 * N] phi_raw;              
  real<lower=0> sigma_phi;            // scale for phi_raw

  // 4) Edge‐level latent for correlation
  vector[M] theta_raw;                
  real mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> tau_theta;            // precision for edge‐ICAR
}

transformed parameters {
  // 1) Re‐scale φ
  vector[2 * N] phi = phi_raw * sigma_phi;
  vector[N] phi1 = phi[1:N];
  vector[N] phi2 = phi[(N+1):(2*N)];

  // 2) Re‐scale θ and map to (−1,1)
  vector[M] theta = mu_theta + theta_raw * sigma_theta;
  vector[M] theta_corr = tanh(theta);
}

model {
  // ------------------------------------------------------------
  // 1. Likelihoods
  // ------------------------------------------------------------

  // 1a. Precipitation y1 ~ LogNormal( log‐mean = beta0_1 + φ₁[i], scale = σ₁ )
  for (i in 1:N) {
    real mu1_i = beta0_1 + phi1[i];
    y1[i] ~ lognormal(mu1_i, sigma1);
  }

  // 1b. Wind y2 ~ Weibull( shape = k_wind, scale = λ_i ), with
  //     log(λ_i) = beta0_2 + φ₂[i]  ⇒  λ_i = exp(beta0_2 + φ₂[i]).
  for (i in 1:N) {
    real lambda_i = exp(beta0_2 + phi2[i]);
    y2[i] ~ weibull(k_wind, lambda_i);
  }

  // ------------------------------------------------------------
  // 2. Priors for intercepts / dispersions
  // ------------------------------------------------------------
  beta0_1   ~ normal(0, 1);            // center on zero
  beta0_2   ~ normal(0, 1);            // center on zero

  sigma1    ~ normal(0, 1);            // log‐scale SD > 0
  k_wind    ~ gamma(2, 0.1);           // weak prior for Weibull shape

  // ------------------------------------------------------------
  // 3. Priors for spatial effects
  // ------------------------------------------------------------
  phi_raw   ~ normal(0, 1);
  sigma_phi ~ normal(0, 1);

  // ------------------------------------------------------------
  // 4. Priors for edge‐correlation parameters θ
  // ------------------------------------------------------------
  theta_raw  ~ student_t(3, 0, 1);
  sigma_theta ~ normal(0, 1);
  tau_theta   ~ gamma(2, 0.5);         // precision for GMRF on edges
  mu_theta    ~ normal(0, 0.5);

  // ------------------------------------------------------------
  // 5. CAR/ICAR priors
  // ------------------------------------------------------------
  // 5a. Bivariate CAR on φ = (φ₁, φ₂) using edge‐wise theta_corr
  target += car_normal_edgecorr_lpdf(
    phi             | 
    N, N_edges,     // # of nodes & node‐adjacency edges 
    node1, node2,   // adjacency lists for the node‐graph 
    theta_corr      // in (−1,1)
  );

  // 5b. ICAR on θ over the edge‐graph
  target += icar_normal_lpdf(
    theta         | 
    M,             // # edges in node‐graph
    E_edges,       // # adjacencies among those edges
    e1, e2,        // each adjacency = (e1[k], e2[k])
    tau_theta      // precision for edge‐ICAR
  );

  // ------------------------------------------------------------
  // 6. Soft sum‐to‐zero identifiability constraints
  // ------------------------------------------------------------
  sum(phi1)  ~ normal(0, 0.001 * N);
  sum(phi2)  ~ normal(0, 0.001 * N);
  sum(theta) ~ normal(0, 0.001 * M);
}

generated quantities {
  // 1) Posterior fitted means (for later LOO or posterior predictive)
  vector[N] mu1 = beta0_1 + phi1;       // log‐mean for y1
  vector[N] mu2 = beta0_2 + phi2;       // log‐scale for λ of y2

  // 2) Posterior theta_corr on (−1,1)
  vector[M] theta_corr_out = theta_corr;

  // 3) Pointwise log‐lik for LOO / WAIC
  vector[N] log_lik_y1;
  vector[N] log_lik_y2;

  for (i in 1:N) {
    log_lik_y1[i] = lognormal_lpdf(y1[i] 
                          | mu1[i], sigma1);
    real lambda_i = exp(mu2[i]); 
    log_lik_y2[i] = weibull_lpdf(y2[i] 
                          | k_wind, lambda_i);
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
  
  M = M,
  E_edges  = nrow(edge_adjacency),  # or whatever you call your edge-adjacency df
  e1       = edge_adjacency$edge_id,
  e2       = edge_adjacency$neighbors
)
##############################################
### Sampling

# Updated initialization function for the model with an ICAR prior on psi
init_function <- function() {
  list(
    beta0_1     = rnorm(1, 0, 0.5),
    beta0_2     = rnorm(1, 0, 0.5),
    sigma1      = runif(1, 0.1, 1.0),   # start log‐SD in (0.1, 1)
    k_wind      = runif(1, 1.0, 5.0),   # start Weibull shape between 1 and 5
    
    phi_raw     = rnorm(2 * N, 0, 0.1), 
    sigma_phi   = runif(1, 0.1, 1.0),
    
    theta_raw   = rnorm(M, 0, 0.1),
    sigma_theta = runif(1, 0.1, 1.0),
    tau_theta   = runif(1, 0.5, 5.0),
    mu_theta    = rnorm(1, 0, 0.2)
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


##############################################
# Compute posterior mean for each edge's correlation
theta_corr_summary <- model_fit_nodis$summary(variables = "theta_corr_out", mean)
# The result is a data frame with one row per edge, containing a 'mean' column.
library(dplyr)
# Add edge_id based on the parameter index and rename the mean column
theta_corr_summary <- theta_corr_summary %>%
  mutate(edge_id = as.integer(gsub("theta_corr_out\\[|\\]", "", variable)),
         corr_mean = mean) %>%
  dplyr::select(edge_id, corr_mean)
# Join edge endpoints with their coordinates and attach correlation means
edges_plot_data <- edges_df %>%
  left_join(theta_corr_summary, by = "edge_id") %>%           # add corr_mean for each edge
  left_join(grid, by = c("node1" = "id")) %>%                 # join to get node1 coordinates
  rename(lon1 = longitude, lat1 = latitude) %>%
  left_join(grid, by = c("node2" = "id")) %>%                 # join to get node2 coordinates
  rename(lon2 = longitude, lat2 = latitude)
library(ggplot2)

ggplot(edges_plot_data, aes(x = lon1, y = lat1, xend = lon2, yend = lat2)) +
  # Draw each edge as a line segment between the two node coordinates
  geom_segment(aes(color = corr_mean, size = abs(corr_mean)), 
               lineend = "round", alpha = 0.8) +
  # Diverging color scale: blue (negative) to white (zero) to red (positive)
  scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                        name = "Correlation") +
  # Scale line width by absolute correlation (optional tuning of range for aesthetics)
  scale_size_continuous(range = c(0.5, 2), name = "|Correlation|") +
  # Use an appropriate coordinate system for maps (fixed aspect ratio for lat/lon)
  coord_quickmap() +  # preserves relative distances in lat-lon:contentReference[oaicite:4]{index=4}
  theme_minimal() +
  labs(x = "Longitude", y = "Latitude",
       title = "Posterior Mean Edge Correlations",
       subtitle = "Blue = negative correlation, Red = positive correlation")

##############################################
library(cmdstanr)
library(dplyr)
library(tidyr)

# Get posterior draws for all edge correlations
draws_theta <- model_fit_nodis$draws(variables = "theta_corr_out", format = "draws_matrix")
# Convert to tidy long format
draws_long <- draws_theta %>%
  as.data.frame() %>%
  pivot_longer(cols = everything(), names_to = "param", values_to = "value") %>%
  mutate(edge_id = as.integer(gsub("theta_corr_out\\[|\\]", "", param)))
theta_summary <- draws_long %>%
  group_by(edge_id) %>%
  summarise(
    mean_corr = mean(value),
    lower_95 = quantile(value, 0.025),
    upper_95 = quantile(value, 0.975),
    .groups = "drop"
  )
# Top 3 positive and top 3 negative edges
top_edges <- theta_summary %>%
  arrange(desc(mean_corr)) %>%
  slice(1:3) %>%
  bind_rows(
    theta_summary %>%
      arrange(mean_corr) %>%
      slice(1:3)
  )
library(ggplot2)

# Add edge labels for plot
top_edges <- top_edges %>%
  mutate(edge_label = paste("Edge", edge_id))

ggplot(top_edges, aes(x = edge_label, y = mean_corr)) +
  geom_point(color = "black") +
  geom_errorbar(aes(ymin = lower_95, ymax = upper_95), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "95% Credible Intervals for Selected Edge Correlations",
    x = "Edge ID",
    y = "Posterior Mean Correlation"
  ) +
  theme_minimal()


##############################################
##############################################
# ──────────────────────────────────────────────────────────────────────────────
# 0) Load required libraries
# ──────────────────────────────────────────────────────────────────────────────
library(ggplot2)
library(posterior)   # for summarizing posterior draws
library(dplyr)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Extract posterior‐mean estimates from the CmdStanMCMC object
# ──────────────────────────────────────────────────────────────────────────────

# (Assuming you've already run:)
#    model_fit_nodis <- model_compiled_cmdstanr_nodis$sample( ... )
# and that object is still in your R session.

# (a) Summarize all draws to a data.frame:
post_summary <- model_fit_nodis$summary()  
# This returns a data.frame with columns: variable, mean, sd, quantile2.5, quantile97.5, etc.

# (b) Grab the posterior mean of sigma1 and k_wind:
sigma1_hat <- post_summary %>% filter(variable == "sigma1") %>% pull(mean)
k_wind_hat <- post_summary %>% filter(variable == "k_wind") %>% pull(mean)

# (c) Extract the posterior mean of each mu1[i] and mu2[i].  
#     The `variable` names look like "mu1[1]", "mu1[2]", …, "mu2[1]", … etc.
mu1_df <- post_summary %>% 
  filter(grepl("^mu1\\[", variable)) %>% 
  arrange(as.integer(gsub("mu1\\[|\\]", "", variable))) %>%
  select(variable, mean)

mu2_df <- post_summary %>% 
  filter(grepl("^mu2\\[", variable)) %>% 
  arrange(as.integer(gsub("mu2\\[|\\]", "", variable))) %>%
  select(variable, mean)

# Now pull out just the numeric vector of means, in order:
mu1_hat <- mu1_df$mean   # length = N
mu2_hat <- mu2_df$mean   # length = N

# ──────────────────────────────────────────────────────────────────────────────
# 2) Compute “quantile‐residuals” for both y1 (LogNormal) and y2 (Weibull)
# ──────────────────────────────────────────────────────────────────────────────
# We define
#    r1[i] = Φ^{-1}(  P(Y1[i] ≤ y1[i] | mu1_hat[i], sigma1_hat) ),   Y1 ~ LogNormal
#    r2[i] = Φ^{-1}(  P(Y2[i] ≤ y2[i] | k_wind_hat,    λ=exp(mu2_hat[i])) ),  Y2 ~ Weibull
#
# If the model is correct, {r1[i]} and {r2[i]} should each be approximately standard‐Normal.

# (a) Make sure y1 and y2 are > 0 and have the same order as mu1_hat / mu2_hat.
#     (You said earlier that you replaced NA or < 0 with 0, but the lognormal/weibull CDF
#      only makes sense > 0.  We assume here y1>0, y2>0.)
#     If you have zeros, you can add a tiny positive constant (e.g. +1e-6) before taking P(Y ≤ y).
#
y1_pos <- ifelse(y1 <= 0, 1e-9, y1)
y2_pos <- ifelse(y2 <= 0, 1e-9, y2)

# (b) Compute the CDF value for each i, then transform by the Normal‐inverse‐CDF:
cdf1 <- plnorm(y1_pos, meanlog = mu1_hat, sdlog = sigma1_hat)  
cdf2 <- pweibull(
  y2_pos,
  shape = k_wind_hat,
  scale = exp(mu2_hat)
)

r1 <- qnorm(p = cdf1)   # “quantile residual” for precipitation
r2 <- qnorm(p = cdf2)   # “quantile residual” for wind speed

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build a single data.frame for each outcome containing:
#      • observed y
#      • fitted linear predictor (mu_hat)
#      • quantile‐residual (r)
# ──────────────────────────────────────────────────────────────────────────────

df_y1 <- tibble(
  y        = y1_pos,
  fitted   = mu1_hat,        # log‐scale “fitted” for lognormal
  resid    = r1
)

df_y2 <- tibble(
  y        = y2_pos,
  fitted   = mu2_hat,        # log‐scale “fitted” for Weibull’s λ
  resid    = r2
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Define plotting functions for the three diagnostics
# ──────────────────────────────────────────────────────────────────────────────

# (a) Histogram of residuals
plot_hist_resid <- function(df, resid_col = "resid", title = "") {
  ggplot(df, aes_string(x = resid_col)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "grey80", color = "black") +
    stat_function(fun = dnorm, args = list(mean = 0, sd = 1),
                  color = "steelblue", size = 1) +
    labs(title = title,
         x = "Quantile residual",
         y = "Density") +
    theme_minimal()
}

# (b) Q–Q plot of residuals vs. standard Normal
plot_qq_resid <- function(df, resid_col = "resid", title = "") {
  ggplot(df, aes_string(sample = resid_col)) +
    stat_qq(color = "darkred", size = 1) +
    stat_qq_line(color = "black") +
    labs(title = title,
         x = "Theoretical Normal quantiles",
         y = "Sample quantiles of residuals") +
    theme_minimal()
}

# (c) Residual vs. Fitted
plot_resid_vs_fitted <- function(df, resid_col = "resid", fitted_col = "fitted", title = "") {
  ggplot(df, aes_string(x = fitted_col, y = resid_col)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = title,
         x = "Fitted (linear predictor scale)",
         y = "Quantile residual") +
    theme_minimal()
}

# ──────────────────────────────────────────────────────────────────────────────
# 5) Draw the six plots: three for y1 and three for y2
# ──────────────────────────────────────────────────────────────────────────────

# 5a) Outcome 1: precipitation (LogNormal model)
p1_hist   <- plot_hist_resid(df_y1, resid_col = "resid",
                             title = "Histogram of Quantile Residuals (tp)")
p1_qq     <- plot_qq_resid  (df_y1, resid_col = "resid",
                             title = "Q–Q Plot of Quantile Residuals (tp)")
p1_resfit <- plot_resid_vs_fitted(df_y1, resid_col = "resid", fitted_col = "fitted",
                                  title = "Residual vs. Fitted (tp)")

# 5b) Outcome 2: wind speed (Weibull model)
p2_hist   <- plot_hist_resid(df_y2, resid_col = "resid",
                             title = "Histogram of Quantile Residuals (ws)")
p2_qq     <- plot_qq_resid  (df_y2, resid_col = "resid",
                             title = "Q–Q Plot of Quantile Residuals (ws)")
p2_resfit <- plot_resid_vs_fitted(df_y2, resid_col = "resid", fitted_col = "fitted",
                                  title = "Residual vs. Fitted (ws)")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Print them (or arrange with gridExtra/cowplot/etc. if you prefer)
# ──────────────────────────────────────────────────────────────────────────────
print(p1_hist)
print(p1_qq)
print(p1_resfit)

print(p2_hist)
print(p2_qq)
print(p2_resfit)

combined_plots <- 
  (p1_hist | p1_qq | p1_resfit) /  # First row: Outcome 1
  (p2_hist | p2_qq | p2_resfit)     # Second row: Outcome 2

# Add global title and adjust spacing
combined_plots + 
  plot_layout(guides = "collect")  # Sync legends if needed
