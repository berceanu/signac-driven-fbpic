import math
from prepic import Plasma, lwfa
import unyt as u

SQRT_FACTOR = math.sqrt(2 * math.log(2))


def main():
    laser = lwfa.Laser.from_a0(
        a0=2.4 * u.dimensionless,
        τL=25.0e-15 / SQRT_FACTOR * u.second,
        beam=lwfa.GaussianBeam(w0=22.0e-6 / SQRT_FACTOR * u.meter, λL=0.8e-6 * u.meter),
    )
    plasma = Plasma(n_pe=8.0e18 * 1.0e6 * u.meter ** (-3))

    print(laser)
    print(plasma)


if __name__ == "__main__":
    main()
