class Evaluator:
    def __init__(self, sktime_performance_metric):
        '''
        Initializes the Evaluator object.

        Args:
            sktime_performance_metric: The sktime performance metric object to use for evaluation.
        '''
        self.sktime_performance_metrics = sktime_performance_metric

    def evaluate(self, model):
        '''
        Evaluates the model on a dataset (the dataset is loaded in this function)

        Args:
            model: The model to evaluate. Must provide a .predict() method.
        
        Returns:
            performance: The performance of the model based on the provided performance metric object.
        '''
        # TODO: load evaluation dataset
        return None
