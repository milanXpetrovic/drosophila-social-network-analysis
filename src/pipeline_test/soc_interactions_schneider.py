## Implementaion of algorithm for automatic detection of social interactions described in
## J. Schneider and J. D. Levine : "Automated identification of social interaction criteria in Drosophila melanogaster"






## STEPS
## Each treatment consists of n populations (trials) with m individuals. 


# STEP 1.1: 
# From every population (trials) in given treatment,
# randomly pick one individual and take whole trajectory data
# Individual data: pos x, pos y, orientation

# STEP 1.2:
# These trajectories were normalized in space and combined


# STEP 1.3:
# The angle and distance between each fly and all other flies' 
# centre of mass is established for all trials within a treatment
# and 500 'null' trials of the same treatment. 
# The maximum distance was set to 20 body lengths