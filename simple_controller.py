# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import time
from scipy.optimize import minimize


# ══════════════════════════════════════════════════════════════════════════════
# Model definition and MPC implementation
# ══════════════════════════════════════════════════════════════════════════════

class Model:
    
    def __init__(self, start_temp = 25.0):

        # ----------------------------------------------------------------------
        # Define system matrices (with placeholder values)
        # ----------------------------------------------------------------------

        # Internal Model - Equations III.1
        self.A = np.array([[0.98]])
        self.B = np.array([[10.0, -15.0]])  # Control input matrix with two inputs (resistor adds heat, fan removes heat))
        self.C = np.array([[1.0]])

        # Measured Perturbance Model - Equations III.2 (External factors)
        self.A_p = np.array([[0.99]])
        self.B_p = np.array([[0.01]])
        self.C_p = np.array([[1.0]])

        # Servo Reference Model - Equations III.3 (Desired temperature trajectory)
        self.A_ra = np.array([[0.2]])
        self.B_ra = np.array([[0.8]])
        self.C_ra = np.array([[1.0]])

        # Regulation Reference Model - Equations III.4 (Perturbance rejection)
        self.A_rr = np.array([[0.85]])
        self.B_rr = np.array([[0.15]])
        self.C_rr = np.array([[1.0]])


        # ----------------------------------------------------------------------
        # Tuning parameters
        # ----------------------------------------------------------------------
        self.N = 8  # Prediction horizon
        self.Q = np.array([[1.0]])  # Error weighting matrix
        self.R = np.array([[0.01, 0.0], [0.0, 0.5]])  # Control effort weighting matrix (R_11 for resistor, R_22 for fan)


        # ----------------------------------------------------------------------
        # Constraints
        # ----------------------------------------------------------------------
        self.u_min = np.array([0.0, 0.0])  # Minimum control inputs (resistor off, fan off)
        self.u_max = np.array([1.0, 1.0])  # Maximum control inputs (resistor full power, fan full power)


        # ----------------------------------------------------------------------
        # Initialize states and control inputs
        # ----------------------------------------------------------------------
        self.x = np.array([[start_temp]])  # Initial state of the system (temperature)
        self.x_p = np.zeros((1, 1))  # Initial state of the perturbance model
        self.x_ra = np.array([[start_temp]]) # Initial state of the servo reference model
        self.x_rr = np.zeros((1, 1))  # Initial state of the regulation reference model
        self.u_prev = np.zeros((2, 1))  # Previous control input (resistor, fan)

    
    def cost_function(self, u_next_flat, y_target):
        """
        Calculates J_k for the proposed future commands

        u_next_flat is a flattened array of control inputs for the entire horizon (2*N as expected by scipy)
        y_target is the desired temperature output (y_d in the equations)
        """
        cost = 0.0
        u_prev = np.copy(self.u_prev)  # Start with the previous control input
        x_prev = np.copy(self.x)  # Start with the current state

        u_next = u_next_flat.reshape(self.N, 2, 1)  # Reshape flat 1D array into (N, 2, 1) sequence of vectors

        for i in range(self.N):

            # Get the control input for this step
            u_curr = u_next[i]

            # State prediction using the internal model
            x_pred = self.A @ x_prev + self.B @ u_curr  # Predicted state: x_{k+1} = A * x_k + B * u_k
            y_pred = self.C @ x_pred  # Predicted output: y_k = C * x_k

            # Errors
            e_i = y_target - y_pred  # Tracking error: e_i = y_d - y_i
            delta_u_i = u_curr - u_prev  # Control input change: Δ u_i = u_i - u_{i-1}

            # Cost accumulation
            error_cost = (e_i.T @ self.Q @ e_i).item()  # Quadratic error cost: e_i^T * Q * e_i
            control_cost = (delta_u_i.T @ self.R @ delta_u_i).item()  # Quadratic control effort cost: Δ u_i^T * R * Δ u_i
            cost += error_cost + control_cost

            u_prev = np.copy(u_curr)  # Update previous control input for the next iteration

        return cost
    

    def control_step(self, y_actual, u_pert, z):
        """
        y_actual: actual hotplate temperature (y_s in the equations)
        u_pert: measured perturbance (u_p in the equations)
        z: target setpoint for the hotplate temperature (z_k in the equations)
        """

        # ----------------------------------------------------------------------
        # Calculate current model outputs
        # ----------------------------------------------------------------------
        y_internal = self.C @ self.x  # Internal model output: y_k = C * x_k
        y_perturbance = self.C_p @ self.x_p  # Perturbance model output: y_p_k = C_p * x_p_k


        # ----------------------------------------------------------------------
        # Calculate unmeasured disturbance
        # ----------------------------------------------------------------------
        e = y_actual - (y_internal[0, 0] + y_perturbance[0, 0])  # Unmeasured disturbance: e_k = y_s_k - y_k - y_p_k


        # ----------------------------------------------------------------------
        # Update model states
        # ----------------------------------------------------------------------
        self.x_p = self.A_p @ self.x_p + self.B_p @ np.array([[u_pert]])  # Perturbance model state update: x_p_{k+1} = A_p * x_p_k + B_p * u_p_k
        self.x_ra = self.A_ra @ self.x_ra + self.B_ra @ np.array([[z]])  # Servo reference model state update: x_ra_{k+1} = A_ra * x_ra_k + B_ra * z_k
        self.x_rr = self.A_rr @ self.x_rr + self.B_rr @ np.array([[e]])  # Regulation reference model state update: x_rr_{k+1} = A_rr * x_rr_k + B_rr * e_k

        y_ra = self.C_ra @ self.x_ra  # Servo reference model output: y_ra_k = C_ra * x_ra_k
        y_rr = self.C_rr @ self.x_rr  # Regulation reference model output: y_rr_k = C_rr * x_rr_k
        y_pert_next = self.C_p @ self.x_p  # Next perturbance model output: y_p_{k+1} = C_p * x_p_{k+1}


        # ----------------------------------------------------------------------
        # Target trajectory
        # ----------------------------------------------------------------------
        y_target = np.array([[y_ra[0, 0] - y_rr[0, 0] - y_pert_next[0, 0]]]) # Target trajectory: y_d_k = y_ra_k - y_rr_k - y_p_{k+1}


        # ----------------------------------------------------------------------
        # Optimize control inputs over the prediction horizon
        # ----------------------------------------------------------------------

        # Create bounds for 2 variables (resistor and fan) over N steps, resulting in 2*N bounds for the optimization
        bounds = []
        for _ in range(self.N):
            bounds.append((self.u_min[0], self.u_max[0]))  # Resistor bounds
            bounds.append((self.u_min[1], self.u_max[1]))  # Fan bounds

        u_guess = np.tile(self.u_prev.flatten(), self.N)  # Initial guess: repeat the last control input for the entire horizon 

        result = minimize(self.cost_function, 
                          u_guess, 
                          args = (y_target,),
                          bounds = bounds,
                          method = 'SLSQP',
                          options = {'ftol': 1e-3, 'disp': False} # ftol speeds up optimization by allowing a less precise solution, which is often sufficient for control applications
                          )
        
        # Safety check in case the optimization fails
        if not result.success:
            print("Optimization failed:", result.message)
            # Fall back to a safe control input (e.g., no change from previous)
            return self.u_prev[0, 0], self.u_prev[1, 0]

        # Extract the optimal sequence and reshape
        u_optimal_seq = result.x.reshape(self.N, 2, 1)  # Reshape flat array back to (N, 2, 1)


        # ----------------------------------------------------------------------
        # Receding horizon control
        # ----------------------------------------------------------------------

        # Control input to apply at this step (first in the sequence)
        u_curr = u_optimal_seq[0]

        # Update the internal model state with the applied control input
        self.x = self.A @ self.x + self.B @ u_curr  # Internal model state update: x_{k+1} = A * x_k + B * u_k

        # Save the current control input for the next iteration
        self.u_prev = np.copy(u_curr)

        # Return the control input to apply to the system (resistor and fan)
        return u_curr[0, 0], u_curr[1, 0]  # Return as scalar values (resistor power, fan power)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation loop
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Starting conditions
    y_actual = 25.0  # Initial hotplate temperature (°C)
    y_pert = 25.0  # Initial perturbance (°C)

    # Create model instance
    model = Model(start_temp = y_actual)

    print("--- Starting simulation ---")
    print("Goal 1: Heat up to 100°C")
    y_target = 50.0  # Target temperature (°C)


    y_curr = y_actual  # Initialize current temperature for simulation
    for k in range(15):

        # Ask controller what to do
        u_resistor, u_fan = model.control_step(y_curr, y_pert, y_target)

        # Simulate the system response (for testing purposes, use the internal model directly)
        heat_added = 10.0 * u_resistor  # Resistor adds heat
        heat_removed = -15.0 * u_fan  # Fan removes heat
        natural_cooling = (y_curr - y_pert) * 0.02  # Natural cooling effect
        y_curr += heat_added + heat_removed - natural_cooling  # Update current temperature based on the model

        print(f"Step {k:02d} | Target: {y_target}°C | Actual: {y_curr:.2f}°C | Resistor: {u_resistor*100:5.1f}% | Fan: {u_fan*100:5.1f}%")

        # Suddenly change the goal at step 7 to require aggressive cooling
        if k == 6:
            print("\n>>> USER DROPPED SETPOINT! Goal 2: Cool to 40°C <<<")
            y_target = 40.0
            
        time.sleep(0.1) # Artificially slow down the loop for readability