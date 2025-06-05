
library(readr)
library(dplyr)
library(ggplot2)
library(viridis)
library(tidyverse)
library(maps)
library(patchwork)  
library(extRemes)  


# 1) Read the CSV file
df_full <- read_csv("/Users/jyx/M4R/data_combined/ANDEREW_1992_08_combined.csv")
# 2) Compute wind speed and remove original u10, v10 columns

df_full <- df_full %>%
  mutate(w10 = sqrt(u10^2 + v10^2)) %>%
  dplyr::select(-u10, -v10)  # Ensure it's using dplyr's select

# Sort the unique times
unique_times <- sort(unique(df_full$time))


df_full <- df_full %>%
  mutate(time_index = as.integer(factor(time)))

# Calculate the thresholds
tp_quantile  <- quantile(df_full$tp,  0.92, na.rm = TRUE)
w10_quantile <- quantile(df_full$w10, 0.90, na.rm = TRUE)

# Filter out rows where tp or w10 are below thresholds
df <- df_full %>%
  filter(tp >= tp_quantile , w10 >= w10_quantile)

# Sort the filtered data
df <- df[order(df$valid_time, df$longitude, df$latitude), ]

# 1) Calculate the average w10 for each time index (using FILTERED data)
df_avg_w10 <- df_full %>%  # Use the filtered data, not df_full
  group_by(time_index) %>%
  summarise(avg_w10 = mean(w10, na.rm = TRUE))

# 2) Plot the average w10 over filtered time indices
ggplot(df_avg_w10, aes(x = time_index, y = avg_w10)) +
  geom_line(color = "blue", linewidth = 1) +          # Line plot
  geom_point(color = "red", size = 3) +               # Points for emphasis
  labs(
    title = "Average Wind Speed (w10) during High-Impact Events",
    x = "Time Index (Filtered)",
    y = "Average Wind Speed (m/s)",
    caption = "Filtered for top 10% tp and w10 values"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )


plot_map_time_ws10 <- function(data, tindex_vec,
                             long_range = c(-95, -70),
                             lat_range = c(22, 32)) {
  # Filter the data for the chosen time indexes
  data_time <- data %>% 
    filter(time_index %in% tindex_vec)
  
  if(nrow(data_time) == 0) {
    stop("No data found for the specified time indexes.")
  }
  
  # Plot all chosen time indexes in one figure, faceted by time_index
  ggplot(data_time, aes(x = longitude, y = latitude, fill = w10)) +
    geom_tile() +
    borders("world",
            xlim = range(data_time$longitude),
            ylim = range(data_time$latitude),
            colour = "black") +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    # Facet by time_index so each subplot is a different time index
    facet_wrap(~ time_index, nrow = 2, ncol = 2) +
    labs(
      title = paste("Total wind speed at time indexes:", paste(tindex_vec, collapse = ", ")),
      x = "Longitude",
      y = "Latitude",
      fill = "Wind speed"
    ) +
    theme_minimal() +
    coord_quickmap(xlim = long_range, ylim = lat_range, expand = FALSE) +
    theme(
      axis.title = element_text(color = 'black', size = 10),
      legend.position = "right",    # put the legend on the right, for example
      plot.margin = margin(t = 0, r = 5, b = 0, l = 0, unit = "mm")
    )
}


plot_map_time_tp <- function(data, tindex_vec,
                               long_range = c(-95, -70),
                               lat_range = c(22, 32)) {
  # Filter the data for the chosen time indexes
  data_time <- data %>% 
    filter(time_index %in% tindex_vec)
  
  if(nrow(data_time) == 0) {
    stop("No data found for the specified time indexes.")
  }
  
  # Plot all chosen time indexes in one figure, faceted by time_index
  ggplot(data_time, aes(x = longitude, y = latitude, fill = tp)) +
    geom_tile() +
    borders("world",
            xlim = range(data_time$longitude),
            ylim = range(data_time$latitude),
            colour = "black") +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    # Facet by time_index so each subplot is a different time index
    facet_wrap(~ time_index, nrow = 2, ncol = 2) +
    labs(
      title = paste("Total precipitation at time indexes:", paste(tindex_vec, collapse = ", ")),
      x = "Longitude",
      y = "Latitude",
      fill = "Total precipitation"
    ) +
    theme_minimal() +
    coord_quickmap(xlim = long_range, ylim = lat_range, expand = FALSE) +
    theme(
      axis.title = element_text(color = 'black', size = 10),
      legend.position = "right",    # put the legend on the right, for example
      plot.margin = margin(t = 0, r = 5, b = 0, l = 0, unit = "mm")
    )
}

plot_compare_ws10 <- function(df_full, df_filtered, tindex_vec,
                              long_range = c(-95, -70),
                              lat_range = c(22, 32)) {
  if (length(tindex_vec) != 3) {
    stop("Please provide exactly three time indices.")
  }
  
  # Tag the data
  df_full_tagged <- df_full %>%
    filter(time_index %in% tindex_vec) %>%
    mutate(Source = "Unfiltered")
  
  df_filtered_tagged <- df_filtered %>%
    filter(time_index %in% tindex_vec) %>%
    mutate(Source = "Filtered")
  
  # Combine both datasets
  df_combined <- bind_rows(df_full_tagged, df_filtered_tagged)
  
  # Make Source a factor so "Unfiltered" is always top row
  df_combined$Source <- factor(df_combined$Source, levels = c("Unfiltered", "Filtered"))
  
  # Plot
  ggplot(df_combined, aes(x = longitude, y = latitude, fill = w10)) +
    geom_tile() +
    borders("world",
            xlim = long_range,
            ylim = lat_range,
            colour = "black") +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    facet_grid(Source ~ time_index) +
    labs(
      x = "Longitude",
      y = "Latitude",
      fill = "Wind Speed"
    ) +
    theme_minimal() +
    coord_quickmap(xlim = long_range, ylim = lat_range, expand = FALSE) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      strip.text = element_text(size = 10),
      axis.title = element_text(size = 10),
      legend.position = "right"
    )
}

plot_compare_tp <- function(df_full, df_filtered, tindex_vec,
                            long_range = c(-95, -70),
                            lat_range = c(22, 32)) {
  if (length(tindex_vec) != 3) {
    stop("Please provide exactly three time indices.")
  }
  
  # Tag the data
  df_full_tagged <- df_full %>%
    filter(time_index %in% tindex_vec) %>%
    mutate(Source = "Unfiltered")
  
  df_filtered_tagged <- df_filtered %>%
    filter(time_index %in% tindex_vec) %>%
    mutate(Source = "Filtered")
  
  # Combine both datasets
  df_combined <- bind_rows(df_full_tagged, df_filtered_tagged)
  
  # Make Source a factor so "Unfiltered" is always the top row
  df_combined$Source <- factor(df_combined$Source, levels = c("Unfiltered", "Filtered"))
  
  # Plot
  ggplot(df_combined, aes(x = longitude, y = latitude, fill = tp)) +
    geom_tile() +
    borders("world",
            xlim = long_range,
            ylim = lat_range,
            colour = "black") +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    facet_grid(Source ~ time_index) +
    labs(
      title = "Total Precipitation (TP) Before and After Filtering: Spatial Comparison at Selected Time Indices",
      x = "Longitude",
      y = "Latitude",
      fill = "Precipitation"
    ) +
    theme_minimal() +
    coord_quickmap(xlim = long_range, ylim = lat_range, expand = FALSE) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      strip.text = element_text(size = 10),
      axis.title = element_text(size = 10),
      legend.position = "right"
    )
}


# 1. Read in each hurricane file
df_full1 <- read_csv("/Users/jyx/M4R/data_combined/CHARLEY_2004_08_combined.csv")
df_full2 <- read_csv("/Users/jyx/M4R/data_combined/WILMA_2005_10_combined.csv")
df_full3 <- read_csv("/Users/jyx/M4R/data_combined/GABRIELLE_2001_09_combined.csv")
df_full4 <- read_csv("/Users/jyx/M4R/data_combined/ANDEREW_1992_08_combined.csv")

# 2. Define a function that: computes w10, filters by quantiles,
#    computes grid‐wise correlations, and returns a ggplot
make_corr_plot <- function(df, storm_name,
                           tp_q = 0.94, w10_q = 0.91) {
  # compute wind speed & drop components
  df2 <- df %>%
    mutate(w10 = sqrt(u10^2 + v10^2)) %>%
    dplyr::select(-u10, -v10) %>%
    # (optional) create an integer time index
    mutate(time_index = as.integer(factor(time)))
  
  # quantiles
  tp_thres  <- quantile(df2$tp,  tp_q, na.rm = TRUE)
  w10_thres <- quantile(df2$w10, w10_q, na.rm = TRUE)
  
  # filter high‐tp & high‐w10
  df_filt <- df2 %>%
    filter(tp  >= tp_thres,
           w10 >= w10_thres) %>%
    arrange(valid_time, longitude, latitude)
  
  # compute correlation per grid cell
  cor_res <- df_filt %>%
    group_by(longitude, latitude) %>%
    summarise(
      cor_tp_w10 = cor(tp, w10, use = "complete.obs"),
      .groups = "drop"
    )
  
  # build the plot
  ggplot(cor_res, aes(longitude, latitude, fill = cor_tp_w10)) +
    geom_tile() +
    scale_fill_viridis_c(option = "plasma", na.value = "white") +
    labs(
      title = paste0(storm_name),
      fill  = "r"
    ) +
    coord_fixed() +
    theme_minimal()
}

# 3. Generate each storm’s plot
p1 <- make_corr_plot(df_full1, "Charley (Aug 2004)")
p2 <- make_corr_plot(df_full2, "Wilma (Oct 2005)")
p3 <- make_corr_plot(df_full3, "Gabrielle (Sep 2001)")
p4 <- make_corr_plot(df_full4, "Anderew (Aug 1992)")

# 4. Combine into a 2×2 panel
combined_panel <- (p1 + p2) / (p3 + p4) +
  plot_annotation(
    theme = theme(plot.title = element_text(size = 16))
  )

# Print it
print(combined_panel)



# Load hurricane data
df_charley <- read_csv("/Users/jyx/M4R/data_combined/CHARLEY_2004_08_combined.csv")

# Compute correlation over time (using filtered data)
cor_filtered_time <- compute_filtered_spatial_correlation_over_time(df_charley)

# Filter to time range 105 to 134
cor_filtered_trimmed <- cor_filtered_time %>%
  filter(time_index >= 95, time_index <= 134)

# Plot with vertical dashed lines
p1<-ggplot(cor_filtered_trimmed, aes(x = time_index, y = cor_tp_w10)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "orange", size = 2) +
  geom_vline(xintercept = c(111, 125), linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = 112, y = max(cor_filtered_trimmed$cor_tp_w10, na.rm = TRUE),
           label = "Landfall", vjust = -0.5, hjust = 1.1, color = "red", size = 3) +
  annotate("text", x = 125, y = max(cor_filtered_trimmed$cor_tp_w10, na.rm = TRUE),
           label = "Departure", vjust = -0.5, hjust = -0.1, color = "red", size = 3) +
  labs(
    title = "Charley",
    subtitle = "Time indices 195–134, with landfall (112) and departure (125) markers",
    x = "Time Index",
    y = "Pearson Correlation"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(size = 12)
  )


# Function to compute spatial correlation at each time index
compute_spatial_correlation_over_time <- function(df) {
  df <- df %>%
    mutate(w10 = sqrt(u10^2 + v10^2)) %>%
    drop_na(tp, w10) %>%
    mutate(time_index = as.integer(factor(time)))
  
  df %>%
    group_by(time_index) %>%
    summarise(
      cor_tp_w10 = cor(tp, w10, method = "pearson", use = "complete.obs"),
      .groups = "drop"
    )
}

# Load Wilma data
df_wilma <- read_csv("/Users/jyx/M4R/data_combined/WILMA_2005_10_combined.csv")

# Compute correlation
cor_wilma_time <- compute_spatial_correlation_over_time(df_wilma)

# Filter time index range 210–243
cor_wilma_trimmed <- cor_wilma_time %>%
  filter(time_index >= 208, time_index <= 240)

# Plot
p2<-ggplot(cor_wilma_trimmed, aes(x = time_index, y = cor_tp_w10)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "darkred", size = 2) +
  geom_vline(xintercept = c(217, 234), linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = 217, y = max(cor_wilma_trimmed$cor_tp_w10, na.rm = TRUE),
           label = "Landfall", vjust = -0.5, hjust = 1.1, color = "red", size = 3) +
  annotate("text", x = 234, y = max(cor_wilma_trimmed$cor_tp_w10, na.rm = TRUE),
           label = "Departure", vjust = -0.5, hjust = -0.1, color = "red", size = 3) +
  labs(
    title = "Wilma",
    subtitle = "Time indices 208–240, with landfall (217) and departure (234) markers",
    x = "Time Index",
    y = "Pearson Correlation"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(size = 12)
  )


# Parameter Stability Method for Threshold Selection
# -------------------------------------------------
# 1) Read and prepare data
df_full <- read_csv("/Users/jyx/M4R/data_combined/CHARLEY_2004_08_combined.csv")

# 2) Compute wind speed and remove original u10, v10 columns
df_full <- df_full %>%
  mutate(w10 = sqrt(u10^2 + v10^2)) %>%
  dplyr::select(-u10, -v10)

# Sort the unique times
unique_times <- sort(unique(df_full$time))

df_full <- df_full %>%
  mutate(time_index = as.integer(factor(time)))


# Choose which variable to analyze (precipitation or wind speed)
variable <- "w10"  # or "tp" for precipitation

# Extract the variable of interest
x <- na.omit(df_full[[variable]])

# Create a sequence of potential thresholds (using 90th-99th percentiles as candidates)
threshold_candidates <- quantile(x, probs = seq(0.90, 0.99, by = 0.01))

# Function to fit GPD and return parameters for each threshold
get_gpd_parameters <- function(threshold) {
  fit <- fevd(x, threshold = threshold, type = "GP", method = "MLE")
  c(threshold = threshold, 
    scale = fit$results$par[1], 
    shape = fit$results$par[2])
}

# Calculate parameters for each candidate threshold
param_stability <- map_dfr(threshold_candidates, get_gpd_parameters)

ggplot(param_stability, aes(x = threshold)) +
  geom_line(aes(y = scale.scale, color = "Scale parameter")) +  # Use scale.scale
  geom_line(aes(y = shape.shape, color = "Shape parameter")) +  # Use shape.shape
  geom_vline(
    xintercept = threshold_candidates[which.min(abs(diff(param_stability$scale.scale)))],  # Use scale.scale
    linetype = "dashed",
    color = "red"
  ) +
  labs(
    title = "Parameter Stability Plot for Wind speed",
    x = "Threshold",
    y = "Parameter Value",
    color = "Parameter"
  ) +
  theme_minimal()

