import jax
import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedQuadraticProblem


class JUNKTURN(AbstractConstrainedQuadraticProblem):
    """The spacecraft orientation problem by Junkins and Turner.
    
    This is a nonlinear optimal control problem.
    The problem is not convex.
    
    Source:
    A.I Tyatushkin, A.I. Zholudev and N. M. Erinchek,
    "The gradient method for solving optimal control problems with phase
    constraints", 
    in "System Modelling and Optimization", P. Kall, ed., pp. 456--464,
    Springer Verlag, Lecture Notes in Control and Information Sciences 180, 1992.
    This reference itself refers to:
    I.L. Junkins and I.D. Turner,
    "Optimal continuous torque attitude maneuvers",
    AIAA/AAS Astrodynamics Conference, Palo Alto, 1978.
    
    SIF input: Ph. Toint, February 1994.
    
    Classification: QQR2-MN-V-V
    """
    
    N: int = 1000  # Default value
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    
    @property
    def n(self):
        """Number of variables: 10 * (N + 1)"""
        return 10 * (self.N + 1)
    
    @property
    def m(self):
        """Number of constraints: 7 * N"""
        return 7 * self.N
    
    def objective(self, y, args):
        """Compute the objective function."""
        del args
        
        N = self.N
        h = inexact_asarray(100.0 / N)
        h_2 = h / 2.0
        h_4 = h / 4.0
        
        # Variables ordered as per SIF: X(I,T) with I outer, T inner
        # X(1,0)...X(1,N), X(2,0)...X(2,N), ..., X(7,0)...X(7,N)
        # Then U(1,0)...U(1,N), U(2,0)...U(2,N), U(3,0)...U(3,N)
        
        state_size = 7 * (N + 1)
        
        # Extract state variables - no reshaping needed for objective
        # Extract control variables - reshape as (3, N+1)
        u = y[state_size:].reshape(3, N + 1)  # u[i-1, t] = U(i,t)
        
        # Objective from SIF file GROUP USES section:
        # ZE OBJ       U1S(0)                   H/4
        # ZE OBJ       U1S(T)                   H/2  (for T=1..N-1)
        # ZE OBJ       U2S(T)                   H/2  (for T=1..N-1)
        # ZE OBJ       U3S(T)                   H/2  (for T=1..N-1)
        # ZE OBJ       U1S(N)                   H/4
        # where UiS(t) = U(i,t)^2
        
        # First term: h/4 * u1(0)^2
        obj = h_4 * (u[0, 0] * u[0, 0])
        
        # Middle terms: h/2 * sum of all three u^2 for t=1..N-1
        if N > 1:
            u_squared_middle = jnp.sum(u[:, 1:N] * u[:, 1:N])
            obj = obj + h_2 * u_squared_middle
        
        # Last term: h/4 * u1(N)^2 (note: only u1, not all three)
        obj = obj + h_4 * (u[0, N] * u[0, N])
        
        return obj
    
    def constraint(self, y):
        """Compute the constraints."""
        N = self.N
        h = inexact_asarray(100.0 / N)
        h_2 = h / 2.0
        h_10 = h / 10.0
        h6_5 = h * 1.2
        sh = h * 1.0909
        s1h = h * (-0.08333)
        s2h = h * 0.18182
        
        # Extract variables using SIF ordering: X(I,T) with I outer, T inner
        x_flat = y[:7 * (N + 1)]
        x = x_flat.reshape(7, N + 1)  # x[i-1, t] = X(i,t)
        
        u_flat = y[7 * (N + 1):]
        u = u_flat.reshape(3, N + 1)  # u[i-1, t] = U(i,t)
        
        # Initialize constraint array
        c = jnp.zeros(7 * N)
        
        def compute_constraints_at_t(t):
            """Compute 7 constraints for time step t."""
            t_idx = t - 1  # Python 0-indexing
            
            # Products needed for constraints
            x1t = x[0, t]
            x2t = x[1, t]
            x3t = x[2, t]
            x4t = x[3, t]
            x5t = x[4, t]
            x6t = x[5, t]
            x7t = x[6, t]
            
            # Element products
            p15 = x1t * x5t
            p16 = x1t * x6t
            p17 = x1t * x7t
            p25 = x2t * x5t
            p26 = x2t * x6t
            p27 = x2t * x7t
            p35 = x3t * x5t
            p36 = x3t * x6t
            p37 = x3t * x7t
            p45 = x4t * x5t
            p46 = x4t * x6t
            p47 = x4t * x7t
            p56 = x5t * x6t
            p57 = x5t * x7t
            p67 = x6t * x7t
            
            # C(1,t): X(1,t) - X(1,t-1) - h/2*(products) = 0
            c1 = x[0, t] - x[0, t-1] + h_2 * (p25 + p36 + p47)
            
            # C(2,t): X(2,t) - X(2,t-1) - h/2*(products) = 0
            c2 = x[1, t] - x[1, t-1] - h_2 * (p15 + p37 - p46)
            
            # C(3,t): X(3,t) - X(3,t-1) - h/2*(products) = 0
            c3 = x[2, t] - x[2, t-1] - h_2 * (p16 - p27 + p45)
            
            # C(4,t): X(4,t) - X(4,t-1) - h/2*(products) = 0
            c4 = x[3, t] - x[3, t-1] - h_2 * (p17 + p26 - p35)
            
            # C(5,t): X(5,t) - X(5,t-1) - h*U(1,t) - s1h*p67 = 0
            c5 = x[4, t] - x[4, t-1] - h * u[0, t] - s1h * p67
            
            # C(6,t): X(6,t) - X(6,t-1) - 6h/5*U(2,t) + h/10*p57 = 0
            c6 = x[5, t] - x[5, t-1] - h6_5 * u[1, t] + h_10 * p57
            
            # C(7,t): X(7,t) - X(7,t-1) - sh*U(3,t) - s2h*p56 = 0
            c7 = x[6, t] - x[6, t-1] - sh * u[2, t] - s2h * p56
            
            return -jnp.array([c1, c2, c3, c4, c5, c6, c7])
        
        # Vectorize over all time steps t=1..N
        all_constraints = jax.vmap(compute_constraints_at_t)(jnp.arange(1, N + 1))
        c = all_constraints.reshape(-1)  # Flatten to 7*N
        
        return c, None
    
    def equality_constraints(self):
        """All constraints are equalities."""
        return jnp.ones(self.m, dtype=bool)
    
    @property
    def y0(self):
        """Initial guess."""
        N = self.N
        
        # Initialize all to 1.0 as per SIF DEFAULT
        y = jnp.ones(self.n)
        
        # Variables use SIF ordering: X(I,T) with I outer, T inner
        # Create array and set specific values from SIF START POINT
        
        # Set X(1,0) = 1.0 at position (1-1)*(N+1) + 0 = 0
        y = y.at[0].set(1.0)
        
        # Set X(5,0) = 0.01 at position (5-1)*(N+1) + 0 = 4*(N+1)
        y = y.at[4 * (N + 1)].set(0.01)
        
        # Set X(6,0) = 0.005 at position (6-1)*(N+1) + 0 = 5*(N+1)
        y = y.at[5 * (N + 1)].set(0.005)
        
        # Set X(7,0) = 0.001 at position (7-1)*(N+1) + 0 = 6*(N+1)
        y = y.at[6 * (N + 1)].set(0.001)
        
        # Set final conditions
        # X(1,N) = 0.43047 at position (1-1)*(N+1) + N = N
        y = y.at[N].set(0.43047)
        
        # X(2,N) = 0.70106 at position (2-1)*(N+1) + N = (N+1) + N = 2*N + 1
        y = y.at[(N + 1) + N].set(0.70106)
        
        # X(3,N) = 0.0923 at position (3-1)*(N+1) + N = 2*(N+1) + N
        y = y.at[2 * (N + 1) + N].set(0.0923)
        
        # X(4,N) = 0.56098 at position (4-1)*(N+1) + N = 3*(N+1) + N  
        y = y.at[3 * (N + 1) + N].set(0.56098)
        
        return inexact_asarray(y)
    
    @property
    def args(self):
        """Additional arguments (none for this problem)."""
        return None
    
    @property
    def bounds(self):
        """Variable bounds."""
        N = self.N
        
        # Most variables are free (-inf, inf)
        lower = jnp.full(self.n, -jnp.inf)
        upper = jnp.full(self.n, jnp.inf)
        
        # Fix bounds using SIF ordering: X(I,T) with I outer, T inner
        # Initial conditions X(I,0) at position (I-1)*(N+1) + 0
        # X(1,0) at position 0
        lower = lower.at[0].set(1.0)
        upper = upper.at[0].set(1.0)
        
        # X(2,0) at position (N+1)
        lower = lower.at[N + 1].set(0.0)
        upper = upper.at[N + 1].set(0.0)
        
        # X(3,0) at position 2*(N+1)
        lower = lower.at[2 * (N + 1)].set(0.0)
        upper = upper.at[2 * (N + 1)].set(0.0)
        
        # X(4,0) at position 3*(N+1)
        lower = lower.at[3 * (N + 1)].set(0.0)
        upper = upper.at[3 * (N + 1)].set(0.0)
        
        # X(5,0) at position 4*(N+1)
        lower = lower.at[4 * (N + 1)].set(0.01)
        upper = upper.at[4 * (N + 1)].set(0.01)
        
        # X(6,0) at position 5*(N+1)
        lower = lower.at[5 * (N + 1)].set(0.005)
        upper = upper.at[5 * (N + 1)].set(0.005)
        
        # X(7,0) at position 6*(N+1)
        lower = lower.at[6 * (N + 1)].set(0.001)
        upper = upper.at[6 * (N + 1)].set(0.001)
        
        # Final conditions - X(I,N) at position (I-1)*(N+1) + N
        # X(1,N) at position N
        lower = lower.at[N].set(0.43047)
        upper = upper.at[N].set(0.43047)
        
        # X(2,N) at position (N+1) + N
        lower = lower.at[(N + 1) + N].set(0.70106)
        upper = upper.at[(N + 1) + N].set(0.70106)
        
        # X(3,N) at position 2*(N+1) + N
        lower = lower.at[2 * (N + 1) + N].set(0.0923)
        upper = upper.at[2 * (N + 1) + N].set(0.0923)
        
        # X(4,N) at position 3*(N+1) + N
        lower = lower.at[3 * (N + 1) + N].set(0.56098)
        upper = upper.at[3 * (N + 1) + N].set(0.56098)
        
        # X(5,N) at position 4*(N+1) + N
        lower = lower.at[4 * (N + 1) + N].set(0.0)
        upper = upper.at[4 * (N + 1) + N].set(0.0)
        
        # X(6,N) at position 5*(N+1) + N
        lower = lower.at[5 * (N + 1) + N].set(0.0)
        upper = upper.at[5 * (N + 1) + N].set(0.0)
        
        # X(7,N) at position 6*(N+1) + N  
        lower = lower.at[6 * (N + 1) + N].set(0.0)
        upper = upper.at[6 * (N + 1) + N].set(0.0)
        
        return lower, upper
    
    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF)."""
        return None
    
    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        # From SIF file comments:
        # SOLTN(1000) = 1.224842784D-5
        if self.N == 1000:
            return 1.224842784e-5
        return None