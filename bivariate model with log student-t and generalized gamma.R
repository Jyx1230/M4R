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
// Stan model: Bivariate spatio model
//
//   • tp (precipitation) ∼ Log‐Student-t(ν_tp, μ1, σ1)
//   • ws (wind speed)   ∼ Generalized‐Gamma(k_gg, scale = exp(μ2), b_gg)
//   • Bivariate CAR prior on φ = (φ₁, φ₂) with edge‐wise correlations θ_corr
//   • ICAR prior on θ over the edge‐graph
//------------------------------------------------------------

functions {
  // A bivariate CAR prior using edge-based correlation (theta_corr).
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

  // Standard ICAR prior on an edge-graph for θ
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

  // Log-density of y ~ Generalized-Gamma(k, scale=a, b):
  //   f(y) = (b / Gamma(k)) * (y/a)^(k*b - 1) * exp(-(y/a)^b) * (1/a)
  real generalized_gamma_lpdf(real y, real k, real a, real b) {
    // y > 0, k>0, a>0, b>0
    return log(b) 
           - lgamma(k)
           + (k * b - 1) * (log(y) - log(a))
           - pow(y / a, b)
           - log(a);
  }
}

data {
  // 1) Node-based spatial graph (size = N)
  int<lower=1> N;                     // number of spatial nodes
  int<lower=1> N_edges;               // number of edges among nodes
  array[N_edges] int node1;           // each edge’s first endpoint in [1..N]
  array[N_edges] int node2;           // each edge’s second endpoint in [1..N]

  // 2) Observations at those N nodes (filtered data)
  vector<lower=0>[N] y1;              // precipitation (tp)
  vector<lower=0>[N] y2;              // wind speed (ws)

  // 3) Edge-based graph for θ (size = M = N_edges)
  int<lower=1> M;                     // total number of edges
  int<lower=1> E_edges;               // number of adjacency-pairs among edges
  array[E_edges] int e1;              // each adjacency’s first edge index
  array[E_edges] int e2;              // each adjacency’s second edge index
}

parameters {
  // 1) Intercepts
  real beta0_1;                       // precipitation intercept (log-scale)
  real beta0_2;                       // wind intercept (log-scale)

  // 2) Dispersion / shape parameters for tp
  real<lower=0>   sigma1;             // scale of Student‐t on log(y1)
  real<lower=2>   nu_tp;              // degrees of freedom for log(y1)

  // 3) Generalized‐Gamma parameters for ws
  real<lower=0>   k_gg;               // “shape‐1” for Gen-Gamma
  real<lower=0>   b_gg;               // “shape‐2” (Weibull‐type) for Gen-Gamma

  // 4) Spatial random effects (non-centered)
  vector[2 * N] phi_raw;              
  real<lower=0>  sigma_phi;           // scale for phi_raw

  // 5) Edge‐CAR latent for θ
  vector[M] theta_raw;                
  real      mu_theta;
  real<lower=0> sigma_theta;
  real<lower=0> tau_theta;            // precision for edge‐ICAR
}

transformed parameters {
  // 1) Re-scale φ
  vector[2 * N] phi = phi_raw * sigma_phi;
  vector[N] phi1 = phi[1:N];
  vector[N] phi2 = phi[(N+1):(2*N)];

  // 2) Re-scale θ and map to (−1,1)
  vector[M] theta = mu_theta + theta_raw * sigma_theta;
  vector[M] theta_corr = tanh(theta);
}

model {
  // ============================================================
  // 1. Likelihoods
  // ============================================================
  for (i in 1:N) {
    // 1a) tp: Log-Student-t
    real mu1_i = beta0_1 + phi1[i];
    target += student_t_lpdf(log(y1[i]) | nu_tp, mu1_i, sigma1)
              - log(y1[i]);  // Jacobian for log→y
  }

  for (i in 1:N) {
    // 1b) ws: Generalized‐Gamma(k_gg, a=exp(mu2_i), b_gg)
    real mu2_i = beta0_2 + phi2[i];
    real a_i   = exp(mu2_i);
    target += generalized_gamma_lpdf(y2[i] | k_gg, a_i, b_gg);
  }

  // ============================================================
  // 2. Priors for intercepts / dispersions
  // ============================================================
  beta0_1 ~ student_t(3, 0, 2.5);
  beta0_2 ~ student_t(3, 0, 2.5);

  sigma1 ~ student_t(3, 0, 2.5);
  nu_tp  ~ gamma(2, 0.1);

  k_gg ~ gamma(0.1, 0.1);   // very weak over (0,∞)
  b_gg ~ gamma(0.1, 0.1);   // very weak over (0,∞)

  // ============================================================
  // 3. Priors for spatial effects
  // ============================================================
  phi_raw   ~ normal(0, 1);
  sigma_phi ~ cauchy(0, 2.5);

  // ============================================================
  // 4. Priors for edge-correlation θ
  // ============================================================
  theta_raw   ~ student_t(3, 0, 1);
  sigma_theta ~ cauchy(0, 2.5);
  mu_theta    ~ student_t(3, 0, 1);
  tau_theta   ~ gamma(0.1, 0.1);

  // ============================================================
  // 5. CAR/ICAR priors
  // ============================================================
  target += car_normal_edgecorr_lpdf(
    phi             |
    N, N_edges,     
    node1, node2,   
    theta_corr      
  );
  target += icar_normal_lpdf(
    theta         | 
    M,             
    E_edges,       
    e1, e2,        
    tau_theta      
  );

  // ============================================================
  // 6. Soft sum-to-zero constraints
  // ============================================================
  sum(phi1)  ~ normal(0, 0.001 * N);
  sum(phi2)  ~ normal(0, 0.001 * N);
  sum(theta) ~ normal(0, 0.001 * M);
}

generated quantities {
  // 1) Posterior‐fitted linear predictors (on log scale) for diagnostics
  vector[N] mu1 = beta0_1 + phi1;   // log-location for y1
  vector[N] mu2 = beta0_2 + phi2;   // log-location for y2

  // 2) (Optional) You can also compute pointwise log‐lik here if you like:
  // vector[N] log_lik_y1;
  // vector[N] log_lik_y2;
  // for (i in 1:N) {
  //   log_lik_y1[i] = student_t_lpdf(log(y1[i]) | nu_tp, mu1[i], sigma1) - log(y1[i]);
  //   log_lik_y2[i] = generalized_gamma_lpdf(y2[i] | k_gg, exp(mu2[i]), b_gg);
  // }
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
init_function <- function() {
  list(
    # ─ Intercepts (Student-t(3,0,2.5)) ─
    beta0_1 = rnorm(1, 0, 2.0),
    beta0_2 = rnorm(1, 0, 2.0),
    
    # ─ tp: Log-Student-t ─
    sigma1 = runif(1, 0.1, 5.0),
    nu_tp  = runif(1, 2.1, 30.0),
    
    # ─ ws: Generalized-Gamma parameters ─
    #    k_gg ~ Gamma(0.1,0.1)  (very flat)  
    #    b_gg ~ Gamma(0.1,0.1)  (very flat) 
    k_gg = runif(1, 0.1, 10.0),
    b_gg = runif(1, 0.1, 10.0),
    
    # ─ Spatial random effects (non-centered) ─
    phi_raw   = rnorm(2 * N, 0, 1.0),
    sigma_phi = runif(1, 0.1, 2.5),
    
    # ─ Edge‐CAR parameters ─
    theta_raw   = rnorm(M, 0, 1.0),
    sigma_theta = runif(1, 0.1, 2.5),
    mu_theta    = rnorm(1, 0, 1.0),
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



# ──────────────────────────────────────────────────────────────────────────────
# 1) Extract posterior summaries
# ──────────────────────────────────────────────────────────────────────────────
post_summary <- model_fit_nodis$summary()

sigma1_hat <- post_summary %>% filter(variable == "sigma1") %>% pull(mean)
nu_tp_hat  <- post_summary %>% filter(variable == "nu_tp")  %>% pull(mean)

k_gg_hat   <- post_summary %>% filter(variable == "k_gg")   %>% pull(mean)
b_gg_hat   <- post_summary %>% filter(variable == "b_gg")   %>% pull(mean)

# Extract mu1[i] and mu2[i]
mu1_df <- post_summary %>%
  filter(grepl("^mu1\\[", variable)) %>%
  arrange(as.integer(gsub("mu1\\[|\\]", "", variable))) %>%
  select(variable, mean)

mu2_df <- post_summary %>%
  filter(grepl("^mu2\\[", variable)) %>%
  arrange(as.integer(gsub("mu2\\[|\\]", "", variable))) %>%
  select(variable, mean)

mu1_hat <- mu1_df$mean   # length = N
mu2_hat <- mu2_df$mean   # length = N

# ──────────────────────────────────────────────────────────────────────────────
# 2) Observed data in the same order: y1, y2
# ──────────────────────────────────────────────────────────────────────────────
y1_pos <- ifelse(y1 <= 0, 1e-9, y1)
y2_pos <- ifelse(y2 <= 0, 1e-9, y2)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Compute Dunn–Smyth (quantile) residuals
# ──────────────────────────────────────────────────────────────────────────────

# 3a) For y1 ~ Log-Student-t(ν_tp, μ1, σ1):
z1 <- (log(y1_pos) - mu1_hat) / sigma1_hat
u1 <- pt(z1, df = nu_tp_hat)
r1 <- qnorm(u1)

# 3b) For y2 ~ GenGamma(k_gg, a = exp(mu2), b_gg):
#     CDF of Generalized-Gamma:  F(y; k,a,b) = lower_incomplete_gamma(k, (y/a)^b) / Γ(k)
#     Stan does not automatically give us that, so we approximate via p_gengamma from “flexsurv” or implement our own.
#
# However, a quick way is:
#   t_i = (y2_pos / exp(mu2_hat))^b_gg_hat
#   Then   F2_i =  gamma_p(k_gg_hat, t_i)   [regularized lower‐gamma]
#
# In R, the regularized lower‐gamma CDF is pgamma(t_i, shape = k_gg_hat, rate = 1).
# So:
t2 <- (y2_pos / exp(mu2_hat))^b_gg_hat
u2 <- pgamma(t2, shape = k_gg_hat, rate = 1)  # this is P( T ≤ t2 ) for T~Gamma(k_gg,1)

r2 <- qnorm(u2)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Build tibbles for diagnostics
# ──────────────────────────────────────────────────────────────────────────────
df_y1 <- tibble(
  y      = y1_pos,
  fitted = mu1_hat,   # log-scale
  resid  = r1
)

df_y2 <- tibble(
  y      = y2_pos,
  fitted = mu2_hat,   # log-scale
  resid  = r2
)

stopifnot(nrow(df_y1) == length(r1))
stopifnot(nrow(df_y2) == length(r2))

# ──────────────────────────────────────────────────────────────────────────────
# 5) Define the plotting functions 
# ──────────────────────────────────────────────────────────────────────────────
plot_hist_resid <- function(df, resid_col = "resid", title = "") {
  ggplot(df, aes(x = !!sym(resid_col))) +
    geom_histogram(aes(y = ..density..),
                   bins  = 30,
                   fill  = "grey80",
                   color = "black") +
    stat_function(fun = dnorm,
                  args = list(mean = 0, sd = 1),
                  color = "steelblue", size = 1) +
    labs(title = title, x = "Quantile residual", y = "Density") +
    theme_minimal()
}

plot_qq_resid <- function(df, resid_col = "resid", title = "") {
  ggplot(df, aes(sample = !!sym(resid_col))) +
    stat_qq(color = "darkred", size = 1) +
    stat_qq_line(color = "black") +
    labs(title = title,
         x     = "Theoretical Normal quantiles",
         y     = "Sample quantiles of residuals") +
    theme_minimal()
}

plot_resid_vs_fitted <- function(df, resid_col = "resid", fitted_col = "fitted", title = "") {
  ggplot(df, aes(x = !!sym(fitted_col), y = !!sym(resid_col))) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = title,
         x     = "Fitted (log-scale)",
         y     = "Quantile residual") +
    theme_minimal()
}

# ──────────────────────────────────────────────────────────────────────────────
# 6) Create the six diagnostic plots
# ──────────────────────────────────────────────────────────────────────────────
p1_hist   <- plot_hist_resid(df_y1, title = "Histogram of Quantile Residuals (tp)")
p1_qq     <- plot_qq_resid  (df_y1, title = "Q–Q Plot of Quantile Residuals (tp)")
p1_resfit <- plot_resid_vs_fitted(df_y1, title = "Residual vs. Fitted (tp)")

p2_hist   <- plot_hist_resid(df_y2, title = "Histogram of Quantile Residuals (ws)")
p2_qq     <- plot_qq_resid  (df_y2, title = "Q–Q Plot of Quantile Residuals (ws)")
p2_resfit <- plot_resid_vs_fitted(df_y2, title = "Residual vs. Fitted (ws)")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Print or arrange
# ──────────────────────────────────────────────────────────────────────────────
print(p1_hist); print(p1_qq); print(p1_resfit)
print(p2_hist); print(p2_qq); print(p2_resfit)

library(patchwork)
( p1_hist   | p1_qq   | p1_resfit ) /
  ( p2_hist   | p2_qq   | p2_resfit ) +
  plot_layout(guides = "collect")

