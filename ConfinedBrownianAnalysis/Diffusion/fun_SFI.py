import numpy as np
from ConfinedBrownianAnalysis.Diffusion.StochasticForceInference import *




def Compute_diffusion(pos,dt,z_min=None,z_max=None,N=20, ordre = 4, method = "Vestergaard"):
    """
    Function using the SFI from Ronseray et al. to return the diffusion alors the the different axis.
    Inputs:
        pos - np.array (len(pos),3) 3 dimensinal trajectory of the particle
        t - time step in microns
        z_min, z_max - z bounds to get Di from the computed basis using SFI
        N - length of the returned diffusion arrays
    """

    tlist = np.arange(len(pos)) * dt
    xlist = np.ones((len(pos),1,3))
    xlist[:,0,:] = pos
    data = StochasticTrajectoryData(xlist,tlist)

    S = StochasticForceInference(data)
    S.compute_diffusion(method = method, basis = {"type" : "polynomial", "order" : ordre })

    dir1 = np.zeros(3); dir1[0] = 1
    dir2 = np.zeros(3); dir2[1] = 1
    dir3 = np.zeros(3); dir3[2] = 1

    Rmin=data.X_ito.min(axis=(0,1))
    Rmax=data.X_ito.max(axis=(0,1))

    xbin=np.linspace(Rmin[0],Rmax[0],N)
    ybin=np.linspace(Rmin[1],Rmax[1],N)

    if z_min == None:
        z_min = Rmin[2]
    if z_max == None:
        z_max = Rmax[2]

    zbin = np.linspace(z_min, z_max, N)

    positions = [a * dir1 + b * dir2  + c * dir3 for a in xbin for b in ybin for c in zbin]

    NN = len(positions)
    gridX,gridY,gridZ = np.zeros(NN),np.zeros(NN),np.zeros(NN)
    Dx,Dy,Dz = np.zeros(NN),np.zeros(NN),np.zeros(NN)


    for n, pos in enumerate(positions):

        gridX[n] = dir1.dot(pos)
        gridY[n] = dir2.dot(pos)
        gridZ[n] = dir3.dot(pos)



        tensor = S.D_ansatz(pos.reshape((1,3)))
        Dx[n], Dy[n], Dz[n] = np.squeeze(tensor.diagonal(axis1 = 2))


    inflate = lambda a : np.reshape(a,(N,N,N))

    to_inflate = Dx, Dy, Dz, gridZ
    Dx, Dy, Dz, zz = map(inflate, to_inflate)
    del to_inflate

    to1d = lambda a : np.mean(np.mean(a, axis = 0),0)
    to_1d = Dx, Dy, Dz, zz

    Dxm, Dym, Dzm, z = map(to1d,to_1d)

    return  Dxm, Dym, Dzm, z
