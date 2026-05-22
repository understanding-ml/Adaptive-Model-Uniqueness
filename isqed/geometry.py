import cvxpy as cp
import numpy as np

# isqed/geometry.py
import cvxpy as cp
import numpy as np

class DISCOSolver:
    """
    Solves the convex projection problem: 
    Find w in Simplex that minimizes || target - peers @ w ||_2
    """
    
    @staticmethod
    def solve_weights_and_distance(target_vec: np.ndarray, peer_matrix: np.ndarray):
        """
        Args:
            target_vec: (D,) or (D, 1)
            peer_matrix: (D, N_peers)
            
        Returns:
            distance: float
            weights: np.ndarray (N_peers,)
        """
        # --- CRITICAL FIX: Flatten inputs to prevent broadcasting explosion ---
        # Ensure target is (D,)
        target_vec = target_vec.flatten()
        # Ensure peers are (D, N) - usually already is, but safety first
        # (Assuming input structure is correct, just shapes need aligning)
        
        D, N_peers = peer_matrix.shape
        if len(target_vec) != D:
             raise ValueError(f"Shape mismatch: Target {target_vec.shape} vs Peers {peer_matrix.shape}")

        # Define variable
        w = cp.Variable(N_peers)
        
        # Objective: Minimize L2 distance
        # Now both sides are (D,), so subtraction works element-wise correctly.
        objective = cp.Minimize(cp.norm(target_vec - peer_matrix @ w, 2))
        
        # Constraints: Simplex
        constraints = [w >= 0, cp.sum(w) == 1]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        
        # Use robust solvers
        try:
            prob.solve(solver=cp.ECOS)
        except:
            try:
                prob.solve(solver=cp.SCS)
            except:
                prob.solve() # Auto-select
            
        if w.value is None:
            # Fallback if solver fails completely (rare)
            return float('inf'), np.ones(N_peers)/N_peers
            
        return prob.value, w.value

    @staticmethod
    def compute_pier(target_vec, peer_matrix, w):
        """Helper to compute residual vector given weights"""
        target_vec = target_vec.flatten()
        y_hat = peer_matrix @ w
        return target_vec - y_hat