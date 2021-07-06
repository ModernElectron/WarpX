import numpy as np
from scipy.sparse import csc_matrix, linalg as sla

class PoissonSolverPseudo1D(object):

    def __init__(self, nx, ny, dx):
        self.nz = nx
        self.nx = ny
        self.dx = dx

        self.nxguardrho = 2
        self.nzguardrho = 2
        self.nxguardphi = 1
        self.nzguardphi = 1

        self.decompose_matrix()

    def decompose_matrix(self):

        self.nzsolve = self.nz + 1
        self.nxsolve = self.nx + 3

        # Set up the computation matrix in order to solve A*phi = rho
        A = np.zeros(
            (self.nzsolve*self.nxsolve, self.nzsolve*self.nxsolve)
        )
        kk = 0
        for ii in range(self.nxsolve):
            for jj in range(self.nzsolve):
                temp = np.zeros((self.nxsolve, self.nzsolve))

                if jj == 0 or jj == self.nzsolve - 1:
                    temp[ii, jj] = 1.
                elif jj == 1:
                    temp[ii, jj] = -2.0 #+ self.left_frac
                    temp[ii, jj-1] = 1.
                    temp[ii, jj+1] = 1. #- self.left_frac
                elif jj == self.nzsolve - 2:
                    temp[ii, jj] = -2.0 #+ self.right_frac
                    temp[ii, jj+1] = 1.
                    temp[ii, jj-1] = 1. #- self.right_frac
                elif ii == 0:
                    temp[ii, jj] = 1.0
                    temp[-3, jj] = -1.0
                elif ii == self.nxsolve - 1:
                    temp[ii, jj] = 1.0
                    temp[2, jj] = -1.0
                else:
                    temp[ii, jj] = -4.0
                    temp[ii+1, jj] = 1.0
                    temp[ii-1, jj] = 1.0
                    temp[ii, jj-1] = 1.0
                    temp[ii, jj+1] = 1.0

                A[kk] = temp.flatten()
                kk += 1

        A = csc_matrix(A, dtype=np.float32)
        self.lu = sla.splu(A)

    def solve(self, rho, rightvoltage, leftvoltage=0.0):

        phi = np.zeros(
            (self.nx + 1 + 2*self.nxguardphi,
            self.nz + 1 + 2*self.nzguardphi)
        )

        rho = rho[
            self.nxguardrho:-self.nxguardrho, self.nzguardrho:-self.nzguardrho
        ]

        # Construct b vector
        nx, nz = np.shape(rho)
        source = np.zeros((nx+2, nz), dtype=np.float32)
        source[1:-1,:] = rho * self.dx**2

        source[:,0] = leftvoltage
        source[:,-1] = rightvoltage

        # Construct b vector
        b = source.flatten()

        flat_phi = self.lu.solve(b)
        phi[:, self.nzguardphi:-self.nzguardphi] = (
            flat_phi.reshape(np.shape(source))
        )

        phi[:,:self.nzguardphi] = leftvoltage
        phi[:,-self.nzguardphi:] = rightvoltage

        # the electrostatic solver in WarpX keeps the ghost cell values as 0
        phi[:self.nzguardphi,:] = 0
        phi[-self.nzguardphi:,:] = 0

        return phi.T
