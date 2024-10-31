#!/usr/bin/env Rscript

# The R code is using to show plot id arrangement
# count values in each plot and count values distribution

# No warnings
options(warn = -1)

# Load necessary libraries
library(sf)
library(ggplot2)
library(ggspatial)

# Check if the correct number of arguments are given
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript result_show.R shapefile_path countfile_path output_dir")
}

# Read arguments from command line
shapefile_path <- args[1]
countfile_path <- args[2]
output_dir <- args[3]

# Check if output directory exists, if not create it
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Read the shapefile
corn_plot <- tryCatch({
  st_read(shapefile_path, quiet = TRUE)
}, error = function(e) {
  stop(paste("Error reading shapefile:", e$message), call. = FALSE)
})

# Calculate the centroid of each plot and bind it to the data frame
corn_plot <- cbind(corn_plot, st_coordinates(st_centroid(corn_plot)))

# Create the plot using ggplot2
plot_id_map_path <- file.path(output_dir, "tassel_count_plot.svg")
p1 <- ggplot() +
  geom_sf(data = corn_plot, alpha = 0.3) +
  geom_sf_label(data = corn_plot, aes(X, Y, label = ID)) + 
  scale_fill_viridis_c() +
  annotation_north_arrow(location = "tr", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.05, "in"),
                         style = north_arrow_fancy_orienteering) +
  ggtitle('Map of Plot ID') +
  theme_void()

# Save the plot as SVG
ggsave(filename = plot_id_map_path, plot = p1, height = 5, width = 10, dpi = 300, units = "in")

# Read the count data
df_count <- tryCatch({
  read.csv(countfile_path)
}, error = function(e) {
  stop(paste("Error reading count file:", e$message), call. = FALSE)
})
names(df_count)[1] <- "ID"

# Merge count data with the plot data
corn_plot2 <- merge(corn_plot, df_count)

# Show the count in plot vis
count_plot_path <- file.path(output_dir, "count_plot.svg")
p2 <- ggplot() +
  geom_sf(data = corn_plot2, aes(fill = pdcount)) +
  geom_sf_label(data = corn_plot2, aes(X, Y, label = pdcount)) + 
  annotation_north_arrow(location = "tr", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.05, "in"),
                         style = north_arrow_fancy_orienteering) +
  scale_fill_gradientn(colors = c("#9DBF9E", "#FCB97D", "#A84268"), na.value = "grey80") + 
  ggtitle('Map of Count in Plot') +
  theme_void()

# Save the count plot as SVG
ggsave(filename = count_plot_path, plot = p2, height = 5, width = 10, dpi = 300, units = "in")

# Create a histogram of the counts
histogram_path <- file.path(output_dir, "tassel_count_histogram.svg")
p3 <- ggplot(df_count, aes(x = ID, y = pdcount)) +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.68) +
  xlab("plot_id") + ylab("Count") +
  scale_x_continuous(breaks = df_count$ID, labels = df_count$ID) +
  theme(text = element_text(size = 20), axis.text.x = element_text(size = 8)) + 
  ggtitle("Tassel Count")

# Save the histogram as SVG
ggsave(filename = histogram_path, plot = p3, height = 5, width = 10, dpi = 300, units = "in")

# Return the paths of the generated files
cat("Generated files:\n")
cat("Plot ID Map: ", plot_id_map_path, "\n")
cat("Count Plot: ", count_plot_path, "\n")
cat("Histogram: ", histogram_path, "\n")