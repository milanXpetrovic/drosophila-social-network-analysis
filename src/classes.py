class Population():
	"""
    A class to represent a population.

    ...

    Attributes
    ----------
    path : str
        path to data

    Methods
    -------
    social_interactions():
        Returns adjacency matrix.
    """

		
	def __init__():
		load_config()
		preproc_data()


	def social_interactions():
		"""
		A method for determining the edges between nodes in a graph. Determines if fly is doing social interaction, based on algorithm from Schneider et al. 2014.

		Parameters
        ----------
        variable : type
        	variable description

        Returns
        -------
        variable : type
        	variable description
        """

		pass


	def global_graph_measures():
		"""
		Calculates global measures for population graph.
		"""
		pass


	def retention_heatmap():
		"""
		Returns heatmap of population retention in arena. 
		"""
		pass


	def edges_timestamps_graph():
		"""
		Returns edges with values of start and end of interaction.
		
		edge = source, target, time_start, time_end

		"""
		pass


	def get_path_features():
		"""
		Calculation the path features from raw data. Features are calculated depending on the default segment (step) size and the average value which is calculated on set window size 
		"""
		pass


	def individual_graph_measures():
		"""
		Returns local measures in the graph. The result is returned in the form of a dictionary containing the measure value for each node.
		"""
		pass


	def distance_traveled():
		"""
		Returns a dictionary with total Euclidean distance traveled. 
		The distance between all position points in the 2d coordinate system is calculated for every individual.
		"""
		pass


	def measures_correlation():
		"""
		Return correlation matrix between given dictionary values.
		"""
		pass


	def measures_distribution():
		"""
		Plots value distribution from given dictionary
		"""
		pass






