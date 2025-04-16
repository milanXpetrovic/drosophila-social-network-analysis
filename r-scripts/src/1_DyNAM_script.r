#%%
# install.packages("jsonlite")
# install.packages("dplyr")
library(goldfish)
library(jsonlite)
library(dplyr)
# getActors function
getActors <- function(df) {
  unique_actors <- sort(unique(c(df$sender, df$receiver)))
  unique_actors <- unique_actors[unique_actors != "all"]
  
  # Convert to a data frame with column name 'label'
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

read_and_clean <- function(filename) {
  data <- read.csv(paste0(group_path, filename))
  colnames(data) <- c("X", "time", "node", "replace")
  subset(data, select = -X)
}

# Initialize lists to store model summaries
rate_model_summaries_CNTRL <- list()
choice_model_summaries_CNTRL <- list()

treatment_name <- 'CS_10D'
folder_path <- paste0("/home/milky/sci/drosophila-isolation/r-scripts/data/", treatment_name, "/")
csv_edgelists <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)

for (file_path in csv_edgelists){
  # csv_edgelists
  df <- read.csv(file_path)
  interaction_data <- df[, c('time', 'sender', 'receiver', 'increment')]
  flies <- getActors(interaction_data)
  flies$bursty <- rep(0,12)
  flies$influence_pos <- rep(0,12)
  flies$influence_neg <- rep(0,12)
  flies$activity <- rep(0,12)
  flies$popularity <- rep(0,12)

  group_name <- basename(file_path)
  group_name <- gsub("\\.csv$", "", group_name)
  group_path = paste0('/srv/milky/drosophila-datasets/drosophila-isolation/data/dynam_data','/',treatment_name ,'/', group_name , '/')

  cov_data_bursty <- read_and_clean("burstines.csv")
  cov_data_influence_pos <- read_and_clean("positive_influence.csv")
  cov_data_influence_neg <- read_and_clean("negative_influence.csv")
  cov_data_activity <- read_and_clean("out_degree.csv")
  cov_data_popularity <- read_and_clean("in_degree.csv")
  
  nodesAttr <- defineNodes(flies)
  interaction_network <- defineNetwork(nodes = nodesAttr, directed = TRUE)
  interaction_network <- linkEvents(x = interaction_network, changeEvent = interaction_data, nodes = nodesAttr)
  dependent <- defineDependentEvents(events = interaction_data, nodes = nodesAttr,
                                     defaultNetwork = interaction_network)
  
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_bursty, attribute = "bursty")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_influence_pos, attribute = "influence_pos")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_influence_neg, attribute = "influence_neg")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_activity, attribute = "activity")
  nodesAttr <- linkEvents(x = nodesAttr, changeEvent = cov_data_popularity, attribute = "popularity")
  
  # Rate submodel
  rate_model <- estimate(dependent ~ 1
                         + indeg(interaction_network, weighted = TRUE, window = 864)
                         + outdeg(interaction_network, weighted = TRUE, window = 864)
                         + ego(nodesAttr$bursty)
                         + ego(nodesAttr$influence_pos)
                         + ego(nodesAttr$influence_neg),
                         model = "DyNAM", subModel = "rate")
  
  result_element <- summary(rate_model)
  json_string <- convert_to_json(result_element)

  save_path <- paste0("/home/milky/sci/drosophila-isolation/r-scripts/res/rate_model/", group_name, '.json')
  writeLines(json_string, save_path)

  # Choice submodel
  choice_model <- estimate(dependent ~
                             inertia
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
# Creating vectors to store data for meta-analysis
# effects <- c("rate_intercept", "rate_indeg", "rate_outdeg", "rate_bursty", "rate_influence_pos", "rate_influence_neg",
#              "choice_inertia", "choice_recip", "choice_indeg", "choice_outdeg", "choice_trans", "choice_sim_activity", "choice_sim_popularity")

##
# result table
# network - group name
# type - Treatment
# each effect has estimate and std err
