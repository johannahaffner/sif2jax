import jax
import jax.numpy as jnp
from functools import partial

from ..._problem import AbstractUnconstrainedMinimisation


class JIMACK(AbstractUnconstrainedMinimisation):
    """Simplified nonlinear elasticity finite element problem.
    
    Source: Parallel optimization for a finite element problem from nonlinear 
    elasticity, P. K. Jimack, School of Computer Studies, U. of Leeds, 
    report 91.22, July 1991.
    
    Classification: OUR2-AN-3549-0
    
    TODO: Human review needed
    Attempts made: 
    - Fixed variable ordering to match SIF loop structure
    - Corrected objective function with volume integration factor  
    - Moved to unconstrained_minimisation category
    - Implemented vectorized finite element with JAX
    - Attempted to replicate SIF Gaussian quadrature coordinate bug
    Suspected issues: 
    - Still 2/9 test failures (objective and gradient at starting point)
    - May need exact replication of FORTRAN numerical precision
    - Possible remaining issues in Gauss quadrature or element parameter handling
    Resources needed: 
    - Detailed analysis of FORTRAN IPHI function implementation
    - Verification of element parameter usage (RI, RJ, RK scaling by 0.1)
    - Possible numerical precision differences between JAX and FORTRAN
    """
    
    # Grid parameters (must match SIF file)
    M: int = 6   # Grid lines in Z (points = M+1 = 7)
    N: int = 12  # Grid lines in X,Y (points = N+1 = 13)
    
    # Physical parameters
    DELTA: float = 0.1   # Domain height in Z direction
    EPS: float = 0.01    # Material parameter
    
    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    
    @property
    def n(self):
        """Total number of variables: (N+1)*(N+1)*(M+1)*3."""
        return (self.N + 1) * (self.N + 1) * (self.M + 1) * 3
    
    @partial(jax.jit, static_argnums=(0,))
    def objective(self, y, args):
        """Simplified but efficient objective function."""
        del args
        
        # Convert from SIF ordering (AX,AY,AZ triplets) to grid arrays  
        n_points = self.N + 1
        m_points = self.M + 1
        
        # Reshape from flat triplets to (n_points*n_points*m_points, 3)
        triplets = y.reshape((-1, 3))  # Shape: (3549, 3)
        
        # Extract components and reshape to grid
        ax_flat = triplets[:, 0]  # AX values
        ay_flat = triplets[:, 1]  # AY values  
        az_flat = triplets[:, 2]  # AZ values
        
        # Reshape to grid following SIF loop order: I→J→K
        ax_grid = ax_flat.reshape((n_points, n_points, m_points))
        ay_grid = ay_flat.reshape((n_points, n_points, m_points))
        az_grid = az_flat.reshape((n_points, n_points, m_points))
        
        # Create vars_grid for compatibility with existing code
        vars_grid = jnp.stack([ax_grid, ay_grid, az_grid], axis=0)
        
        # Grid parameters
        dx = 2.0 / self.N      
        dy = dx                
        dz = self.DELTA / self.M  
        
        # Compute energy by summing over all elements
        total_energy = 0.0
        
        # Use JAX loops for performance
        def element_energy(carry, element_idx):
            i, j, k = element_idx
            
            # Extract element corners (8 corners of hexahedral element)
            corners = jnp.array([
                [vars_grid[0, i, j, k], vars_grid[1, i, j, k], vars_grid[2, i, j, k]],          # (0,0,0)
                [vars_grid[0, i+1, j, k], vars_grid[1, i+1, j, k], vars_grid[2, i+1, j, k]],    # (1,0,0)
                [vars_grid[0, i, j+1, k], vars_grid[1, i, j+1, k], vars_grid[2, i, j+1, k]],    # (0,1,0)
                [vars_grid[0, i+1, j+1, k], vars_grid[1, i+1, j+1, k], vars_grid[2, i+1, j+1, k]],  # (1,1,0)
                [vars_grid[0, i, j, k+1], vars_grid[1, i, j, k+1], vars_grid[2, i, j, k+1]],    # (0,0,1)
                [vars_grid[0, i+1, j, k+1], vars_grid[1, i+1, j, k+1], vars_grid[2, i+1, j, k+1]],  # (1,0,1)
                [vars_grid[0, i, j+1, k+1], vars_grid[1, i, j+1, k+1], vars_grid[2, i, j+1, k+1]],  # (0,1,1)
                [vars_grid[0, i+1, j+1, k+1], vars_grid[1, i+1, j+1, k+1], vars_grid[2, i+1, j+1, k+1]]  # (1,1,1)
            ])
            
            # Element bounds (ensure float types)
            xl = -1.0 + dx * i.astype(jnp.float64)
            xu = xl + dx
            yl = -1.0 + dy * j.astype(jnp.float64)
            yu = yl + dy
            zl = dz * k.astype(jnp.float64)
            zu = zl + dz
            
            # Gauss quadrature (8 points)
            w = 1.0 / jnp.sqrt(3.0)
            delx = w / self.N
            dely = dx  
            delz = 0.5 * w * self.DELTA / self.M
            
            xmid, ymid, zmid = 0.5 * (xl + xu), 0.5 * (yl + yu), 0.5 * (zl + zu)
            
            # 8 Gauss points - replicating SIF FORTRAN loop structure with coordinate bug
            # Original SIF has bug: X,Y coordinates determined by K1 instead of I1,J1
            # We must replicate this to match pycutest results
            gauss_pts = []
            for k1 in range(2):  # K1 = 1, 2
                z_coord = zmid - delz if k1 == 0 else zmid + delz
                for j1 in range(2):  # J1 = 1, 2  
                    # SIF bug: Y coordinate determined by K1, not J1!
                    y_coord = ymid - dely if k1 == 0 else ymid + dely
                    for i1 in range(2):  # I1 = 1, 2
                        # SIF bug: X coordinate determined by K1, not I1!
                        x_coord = xmid - delx if k1 == 0 else xmid + delx
                        gauss_pts.append([x_coord, y_coord, z_coord])
            gauss_pts = jnp.array(gauss_pts)
            
            # Compute element energy by summing over Gauss points
            # Each Gauss point has weight 1 (for 2-point Gauss quadrature in each direction)
            # The element volume is dx*dy*dz
            element_volume = dx * dy * dz
            
            element_sum = 0.0
            for g in range(8):
                point_energy = self._gauss_point_energy(gauss_pts[g], corners, xl, xu, yl, yu, zl, zu, dx, dy, dz)
                # Each Gauss point has weight 1, and we need to account for element volume
                element_sum += point_energy
            
            # Multiply by element volume and divide by 8 (since we're using 2^3 Gauss points with weight 1 each)
            # Actually, for 2-point Gauss quadrature, total weight per direction is 2, so total weight is 2^3 = 8
            element_energy = element_sum * element_volume / 8.0
            
            return carry + element_energy, None
        
        # Create all element indices
        element_indices = []
        for k in range(self.M):
            for j in range(self.N):
                for i in range(self.N):
                    element_indices.append([i, j, k])
        element_indices = jnp.array(element_indices)
        
        # Use scan for efficiency
        total_energy, _ = jax.lax.scan(element_energy, 0.0, element_indices)
        
        # Scale factor from SIF file  
        scale = 2.0 * (self.M * self.N * self.N) / self.DELTA
        return scale * total_energy
    
    def _gauss_point_energy(self, gauss_pt, corners, xl, xu, yl, yu, zl, zu, dx, dy, dz):
        """Compute energy density at a Gauss point."""
        
        # Normalized coordinates
        xi = (2.0 * gauss_pt[0] - (xl + xu)) / (xu - xl)
        eta = (2.0 * gauss_pt[1] - (yl + yu)) / (yu - yl)
        zeta = (2.0 * gauss_pt[2] - (zl + zu)) / (zu - zl)
        
        # Trilinear shape function derivatives
        dN_dxi = jnp.array([
            -(1-eta)*(1-zeta), (1-eta)*(1-zeta), -(1+eta)*(1-zeta), (1+eta)*(1-zeta),
            -(1-eta)*(1+zeta), (1-eta)*(1+zeta), -(1+eta)*(1+zeta), (1+eta)*(1+zeta)
        ]) / 8.0
        
        dN_deta = jnp.array([
            -(1-xi)*(1-zeta), -(1+xi)*(1-zeta), (1-xi)*(1-zeta), (1+xi)*(1-zeta),
            -(1-xi)*(1+zeta), -(1+xi)*(1+zeta), (1-xi)*(1+zeta), (1+xi)*(1+zeta)
        ]) / 8.0
        
        dN_dzeta = jnp.array([
            -(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1-xi)*(1+eta), -(1+xi)*(1+eta),
            (1-xi)*(1-eta), (1+xi)*(1-eta), (1-xi)*(1+eta), (1+xi)*(1+eta)
        ]) / 8.0
        
        # Transform to physical derivatives
        dN_dx = dN_dxi * (2.0 / (xu - xl))
        dN_dy = dN_deta * (2.0 / (yu - yl))
        dN_dz = dN_dzeta * (2.0 / (zu - zl))
        
        # Compute deformation gradient F = [∂u/∂X] where u = (ax, ay, az)
        # F[i,j] = ∂u_i/∂X_j
        F = jnp.zeros((3, 3))
        F = F.at[0, 0].set(jnp.sum(dN_dx * corners[:, 0]))  # ∂ax/∂X
        F = F.at[0, 1].set(jnp.sum(dN_dy * corners[:, 0]))  # ∂ax/∂Y
        F = F.at[0, 2].set(jnp.sum(dN_dz * corners[:, 0]))  # ∂ax/∂Z
        F = F.at[1, 0].set(jnp.sum(dN_dx * corners[:, 1]))  # ∂ay/∂X
        F = F.at[1, 1].set(jnp.sum(dN_dy * corners[:, 1]))  # ∂ay/∂Y
        F = F.at[1, 2].set(jnp.sum(dN_dz * corners[:, 1]))  # ∂ay/∂Z
        F = F.at[2, 0].set(jnp.sum(dN_dx * corners[:, 2]))  # ∂az/∂X
        F = F.at[2, 1].set(jnp.sum(dN_dy * corners[:, 2]))  # ∂az/∂Y
        F = F.at[2, 2].set(jnp.sum(dN_dz * corners[:, 2]))  # ∂az/∂Z
        
        # Right Cauchy-Green tensor C = F^T @ F following FORTRAN computation
        # C[i,j] = sum_k F[k,i] * F[k,j] = sum_k (∂u_k/∂X_i) * (∂u_k/∂X_j)
        C = jnp.dot(F.T, F)
        
        # Principal invariants as computed in FORTRAN
        # INV1 = trace(C)
        # INV2 = 0.5 * (trace(C)^2 - trace(C^2))  
        # INV3 = det(C)
        inv1 = jnp.trace(C)
        inv2 = 0.5 * (inv1**2 - jnp.trace(C @ C))
        inv3 = jnp.maximum(jnp.linalg.det(C), 1e-12)
        
        # Material factor
        z = gauss_pt[2]
        v = 1.0 - self.EPS * z / self.DELTA
        v2 = v * v
        
        # Energy density
        bv2 = 4.5 / v2
        cv2 = 3.1 * v2
        
        energy = (0.65 * inv1 + 
                 bv2 * inv2 - 
                 cv2 * (jnp.log(inv3) - 3.0 * jnp.log(v2)))
        
        return energy
    
    @property  
    def y0(self):
        """Initial guess: identity mapping as specified in SIF file.
        
        The SIF file loops DO I, DO J, DO K and sets:
        AX(I,J,K) = RX, AY(I,J,K) = RY, AZ(I,J,K) = RZ
        where RX, RY, RZ are updated after each loop.
        """
        n_points = self.N + 1  # 13 points  
        m_points = self.M + 1  # 7 points
        dx = 2.0 / self.N      # Grid spacing in X,Y
        dz = self.DELTA / self.M  # Grid spacing in Z
        
        y0 = []
        
        # Follow exact SIF loop structure
        rx = -1.0
        for i in range(n_points):  # DO I = 0 to N
            ry = -1.0  
            for j in range(n_points):  # DO J = 0 to N
                rz = 0.0
                for k in range(m_points):  # DO K = 0 to M
                    # Set AX(I,J,K), AY(I,J,K), AZ(I,J,K)
                    y0.extend([rx, ry, rz])
                    
                    # R+ RZ        RZ                       DZ
                    rz += dz
                # End K loop
                
                # R+ RY        RY                       DX  
                ry += dx
            # End J loop
            
            # R+ RX        RX                       DX
            rx += dx
        # End I loop
        
        return jnp.array(y0)
    
    @property
    def args(self):
        """No additional arguments needed."""
        return None
    
    @property
    def expected_result(self):
        """Expected optimal solution (not provided in SIF file)."""
        return None
    
    @property  
    def expected_objective_value(self):
        """Expected optimal objective value (not provided in SIF file)."""
        return None