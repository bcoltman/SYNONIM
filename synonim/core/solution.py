import pandas as pd

class Solution:
    def __init__(self, X_opt, objective, genome_names, method=None, selection_order=None, details=None):
        """
        Parameters:
            X_opt (array-like): The binary vector indicating the selected candidates.
            objective (float): The achieved objective value.
            genome_names (list of str): The names of all genomes/isolates corresponding to the indices in X_opt.
            selection_order (list of str, optional): 
                For optimizers (e.g. the heuristic) where the order of selection is important, 
                pass in the ordered list of selected genome names. 
                If not provided, the selected names will be derived from X_opt in the natural order.
            details (dict, optional): Additional metadata (e.g. runtime, logging info).
        """
        self.X_opt = X_opt
        self.objective = objective
        
        self.selection_order = selection_order
        self.method = method
        self.details = details if details is not None else {}
        
        # Derive the selected names:
        # If an order is provided, we trust that to be the intended order.
        # Otherwise, simply pick names based on the binary vector order.
        if self.selection_order is None:
            self.selected_names = [name for name, flag in zip(genome_names, X_opt) if flag]
        else:
            self.selected_names = self.selection_order
            
    def __repr__(self):
        return f"Solution(objective={self.objective}, selected_names={self.selected_names})"