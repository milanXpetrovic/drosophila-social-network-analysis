library(parallel)
library(goldfish)
library(jsonlite)
library(dplyr)

getActors <- function(df) {
  unique_actors <- sort(unique(c(df$sender, df$receiver)))
  unique_actors <- unique_actors[unique_actors != "all"]
  unique_actors_df <- data.frame(label = unique_actors)
  return(unique_actors_df)
}

convert_to_json <- function(result_element) {
  # Extract relevant fields
  parameters <- result_element$parameters
  standardErrors <- result_element$standardErrors
  logLikelihood <- result_element$logLikelihood
  finalScore <- result_element$finalScore
  finalInformationMatrix <- result_element$finalInformationMatrix
  convergence <- result_element$convergence
  nIterations <- result_element$nIterations
  nEvents <- result_element$nEvents
  formula <- as.character(result_element$formula)  # Convert formula to character
  model <- result_element$model
  subModel <- result_element$subModel
  rightCensored <- result_element$rightCensored
  nParams <- result_element$nParams
  call <- as.character(result_element$call)  # Convert call to character
  coefMat <- result_element$coefMat
  AIC <- result_element$AIC
  BIC <- result_element$BIC

  # Create the JSON-friendly list
  json_data <- list(
    parameters = parameters,
    standardErrors = standardErrors,
    logLikelihood = logLikelihood,
    finalScore = finalScore,
    finalInformationMatrix = finalInformationMatrix,
    convergence = convergence,
    nIterations = nIterations,
    nEvents = nEvents,
    formula = formula,
    model = model,
    subModel = subModel,
    rightCensored = rightCensored,
    nParams = nParams,
    call = call,
    coefMat = coefMat,
    AIC = AIC,
    BIC = BIC
  )

  # Convert the list to JSON string
  json_string <- toJSON(json_data, pretty = TRUE)
  
  return(json_string)
}

read_and_clean <- function(group_path, filename) {
  data <- read.csv(paste0(group_path, filename))
  colnames(data) <- c("X", "time", "node", "replace")
  subset(data, select = -X)
}


rate_model_summaries_CNTRL <- list()
choice_model_summaries_CNTRL <- list()
treatment_name <- 'Cs_5DIZ'
folder_path <- paste0("/home/milky/sci/drosophila-isolation/r-scripts/data/", treatment_name, "/")
csv_edgelists <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)

for (i in seq_along(csv_edgelists)) {
  file_path <- csv_edgelists[[i]]
  df <- read.csv(file_path)

  interaction_data <- df[, c('time', 'sender', 'receiver', 'increment')]
  flies <- getActors(interaction_data)
  flies <- getActors(interaction_data)
  flies$bursty <- rep(0, 12)
  flies$influence_pos <- rep(0, 12)
  flies$influence_neg <- rep(0, 12)
  flies$activity <- rep(0, 12)
  flies$popularity <- rep(0, 12)
  group_name <- basename(file_path)
  group_name <- gsub("\\.csv$", "", group_name)
  group_path = paste0('/srv/milky/drosophila-datasets/drosophila-isolation/data/dynam_data','/',treatment_name ,'/', group_name , '/')
  cov_data_bursty <- read_and_clean(group_path, "burstines.csv")
  cov_data_influence_pos <- read_and_clean(group_path, "positive_influence.csv")
  cov_data_influence_neg <- read_and_clean(group_path, "negative_influence.csv")
  cov_data_activity <- read_and_clean(group_path, "out_degree.csv")
  cov_data_popularity <- read_and_clean(group_path, "in_degree.csv")
  nodesAttr <- defineNodes(flies)
  interaction_network <- defineNetwork(nodes = nodesAttr, directed = TRUE)
  interaction_network <- linkEvents(x = interaction_network, changeEvent = interaction_data, nodes = nodesAttr)
  dependent_val <- defineDependentEvents(events = interaction_data, nodes = nodesAttr, defaultNetwork = interaction_network)

  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_bursty, attribute = "bursty")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_influence_pos, attribute = "influence_pos")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_influence_neg, attribute = "influence_neg")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_activity, attribute = "activity")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_popularity, attribute = "popularity")
  
  rateFormula <- 
    dependent_val ~ 1 +
    indeg(interaction_network, weighted = TRUE, window = 864) +
    outdeg(interaction_network, weighted = TRUE, window = 864) +
    ego(nodesAttr$bursty) +
    ego(nodesAttr$influence_pos) +
    ego(nodesAttr$influence_neg)

  rate_model <- estimate(
    rateFormula,
    model = "DyNAM", subModel = "rate"
  )

  print("Here!")
  result_element <- summary(rate_model)
  json_string <- convert_to_json(result_element)
  save_path <- paste0("/home/milky/sci/drosophila-isolation/r-scripts/res/rate_model/", group_name, '.json')
  writeLines(json_string, save_path)

  choice_model <- estimate(dependent_val ~ inertia 
                           + recip(interaction_network, weighted = FALSE, window = 288)
                           + indeg(interaction_network, weighted = TRUE, window = 864)
                           + outdeg(interaction_network, weighted = TRUE, window = 864)
                           + trans(interaction_network, weighted = FALSE, window = 864)
                           + sim(nodesAttr$activity)
                           + sim(nodesAttr$popularity),
                           model = "DyNAM", subModel = "choice",
                           estimationInit = list(returnIntervalLogL = TRUE, maxIterations = 1000))
  result_element <- summary(choice_model)
  json_string <- convert_to_json(result_element)
  save_path <- paste0("/home/milky/sci/drosophila-isolation/r-scripts/res/choice_model/", group_name, '.json')
  writeLines(json_string, save_path)
  print(group_name)
}
