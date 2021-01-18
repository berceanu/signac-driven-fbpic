import math
from prepic import Plasma, lwfa
import unyt as u

SQRT_FACTOR = math.sqrt(2 * math.log(2))


def main():
    laser = lwfa.Laser.from_a0(
        a0=2.4 * u.dimensionless,
        τL=27.8 / SQRT_FACTOR * u.femtosecond,
        beam=lwfa.GaussianBeam(
            w0=22.0 / SQRT_FACTOR * u.micrometer, λL=815 * u.nanometer
        ),
    )
    plasma = Plasma(n_pe=8.0e18 * u.cm ** (-3))

    print(laser)
    print(plasma)


if __name__ == "__main__":
    main()
