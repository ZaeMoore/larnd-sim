"""
Module to implement the quenching of the ionized electrons
through the detector
"""

from math import log, isnan
from numba import cuda

from .consts import detector, physics, light

@cuda.jit
def quench(tracks, mode):
    """
    This CUDA kernel takes as input an array of track segments and calculates
    the number of electrons and photons that reach the anode plane after recombination.
    It is possible to pick among two models: Box (Baller, 2013 JINST 8 P08005) or
    Birks (Amoruso, et al NIM A 523 (2004) 275).

    Args:
        tracks (:obj:`numpy.ndarray`): array containing the tracks segment information
        mode (int): recombination model (physics.BOX or physics.BIRKS).
    """
    itrk = cuda.grid(1)

    if itrk < tracks.shape[0]:
        dEdx = tracks[itrk]["dEdx"]
        dE = tracks[itrk]["dE"]
        track = tracks[itrk]
        pixel_plane = detector.DEFAULT_PLANE_INDEX

        recomb = 0
        if mode == physics.BOX:
            # Baller, 2013 JINST 8 P08005
            csi = physics.BOX_BETA * dEdx / (detector.E_FIELD * detector.LAR_DENSITY)
            recomb = max(0, log(physics.BOX_ALPHA + csi)/csi)
        elif mode == physics.BIRKS:
            # Amoruso, et al NIM A 523 (2004) 275
            recomb = physics.BIRKS_Ab / (1 + physics.BIRKS_kb * dEdx / (detector.E_FIELD * detector.LAR_DENSITY))
        elif mode == physics.DATA:
            for ip, plane in enumerate(TPC_BORDERS):
                if plane[0][0]-2e-2 <= track["x"] <= plane[0][1]+2e-2 and \
                   plane[1][0]-2e-2 <= track["y"] <= plane[1][1]+2e-2 and \
                   min(plane[2][1]-2e-2,plane[2][0]-2e-2) <= track["z"] <= max(plane[2][1]+2e-2,plane[2][0]+2e-2):
                    pixel_plane = ip
                    break
            track["pixel_plane"] = pixel_plane

            if pixel_plane != detector.DEFAULT_PLANE_INDEX:
                z_anode = TPC_BORDERS[pixel_plane][2][0]
                drift_distance = abs(track["z"] - z_anode)
                drift_time = drift_distance / detector.V_DRIFT
                lifetime_red = exp(drift_time / detector.ELECTRON_LIFETIME) #removed negative to flip to anode->volume
                recomb = lifetime_red
        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BIRKS'")

        if isnan(recomb):
            raise RuntimeError("Invalid recombination value")

        tracks[itrk]["n_electrons"] = recomb * dE / physics.W_ION
        tracks[itrk]["n_photons"] = (dE/light.W_PH - tracks[itrk]["n_electrons"]) * light.SCINT_PRESCALE
