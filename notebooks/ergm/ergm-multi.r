install.packages("languageserver")
install.packages("ergm")
install.packages("ergm.multi")

library(ergm)
library(ergm.multi)


# Set the directory paths where adjacency matrices are stored for each treatment
treatment1_dir <- "/home/milky/sna/data/processed/1_3_create_adj_matrix/Cs_5DIZ/count"
treatment2_dir <- "/home/milky/sna/data/processed/1_3_create_adj_matrix/Cs_5DIZ/count"

# Function to read adjacency matrices from a directory
read_matrices_from_directory <- function(directory) {
  files <- list.files(directory, full.names = TRUE)
  matrices <- lapply(files, read.csv, header = FALSE)
  return(matrices)
}

# Read adjacency matrices for each treatment
treatment1_matrices <- read_matrices_from_directory(treatment1_dir)
treatment2_matrices <- read_matrices_from_directory(treatment2_dir)

# Convert matrices to network objects (assuming they are symmetric)
treatment1_networks <- lapply(treatment1_matrices, function(mat) as.network(mat + t(mat)))
treatment2_networks <- lapply(treatment2_matrices, function(mat) as.network(mat + t(mat)))

# Construct ergm.multi object
multi_ergm <- ergm.multi(list(treatment1_networks, treatment2_networks))

# Fit the multi-ERGM model
fit_multi_ergm <- ergm.multi(fit = multi_ergm)

# Perform comparisons or extract results from the fitted multi-ERGM model
# For instance, you can use summary, plot, or other functions to explore and compare the models
summary(fit_multi_ergm)
