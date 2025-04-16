# Initialize an empty dataframe to store the results
meta_data <- data.frame(effect_name=character(), 
                        estimate=numeric(), 
                        std_error=numeric(), 
                        stringsAsFactors=FALSE)

# Looping through all sheets
for (i in 1:XXX) {
  
  sheet_name <- paste("Sheet", i)
  
  # Extracting parameters and standard errors for rate model
  rate_params <- rate_model_summaries_CNTRL[[sheet_name]]$parameters
  rate_se     <- rate_model_summaries_CNTRL[[sheet_name]]$standardErrors
  
  # Extracting parameters and standard errors for choice model
  choice_params <- choice_model_summaries_CNTRL[[sheet_name]]$parameters
  choice_se     <- choice_model_summaries_CNTRL[[sheet_name]]$standardErrors
  
  # Combining all parameters and standard errors into a dataframe
  all_params <- c(rate_params, choice_params)
  all_se     <- c(rate_se, choice_se)
  
  temp_data  <- data.frame(effect_name = effects,
                           estimate = all_params,
                           std_error = all_se)
  
  # Binding the data into the meta_data dataframe
  meta_data <- rbind(meta_data, temp_data)
}
