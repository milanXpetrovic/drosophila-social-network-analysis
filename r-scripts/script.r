#%%
# install.packages("goldfish")
library(goldfish)

folder_path <- "/home/milky/drosophila-isolation/r-scripts/data/CS_10D_csv/"
csv_files <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)

results_list <- list()

for (file_path in csv_files){
  df <- read.csv(file_path)
  df <- df[, c('time', 'sender', 'receiver', 'increment')]
  node_list <- unique(df$sender)
  node_list = defineNodes(data.frame(label=node_list))
  interactionNetwork <- defineNetwork(nodes=node_list, directed = TRUE)
  interactionNetwork <- linkEvents(x=interactionNetwork, changeEvent=df, nodes=node_list)
  interactionDependent <- defineDependentEvents(
    events = df, nodes = node_list,
    defaultNetwork = interactionNetwork
    )
  window_size <- 480
  modelRem <- estimate(
    interactionDependent ~ 
    indeg(interactionNetwork, window = window_size) + outdeg(interactionNetwork, window = window_size) + 
    inertia(interactionNetwork, window = window_size) + recip(interactionNetwork, window = window_size) +
    trans(interactionNetwork, window = window_size),
      model = "REM"
    )
  analysis_result <- summary(modelRem)  
  results_list[[file_path]] <- analysis_result
}

# install.packages("jsonlite")
# install.packages("dplyr")
library(jsonlite)
library(dplyr)

library(jsonlite)

json_list <- list()
for (i in seq_along(results_list)) {
  result_element <- results_list[[i]]
  print(result_element$ parameters)
  str(result_element)
  parameters <- result_element$parameters
  standardErrors <- result_element$standardErrors
  logLikelihood <- result_element$logLikelihood
  finalScore <- result_element$finalScore
  finalInformationMatrix <- result_element$finalInformationMatrix
  convergence <- result_element$convergence
  nIterations <- result_element$nIterations
  nEvents <- result_element$nEvents
  names <- result_element$names
  formula <- as.character(result_element$formula)  # Convert formula to character
  model <- result_element$model
  subModel <- result_element$subModel
  rightCensored <- result_element$rightCensored
  nParams <- result_element$nParams
  call <- as.character(result_element$call)  # Convert call to character
  coefMat <- result_element$coefMat
  AIC <- result_element$AIC
  BIC <- result_element$BIC
  # formula <- as.character(formula)
  # call <- as.character(call)

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
  json_data$formula <- as.character(json_data$formula)
  json_data$call <- as.character(json_data$call)
  json_list[[i]] <- json_data
}
json_string <- toJSON(json_list, pretty = TRUE)
writeLines(json_string, "output_all.json")
print("done")
