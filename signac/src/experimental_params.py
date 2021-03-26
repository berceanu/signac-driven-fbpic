import math
from prepic import Plasma, lwfa
import unyt as u

SQRT_FACTOR = math.sqrt(2 * math.log(2))


def main():
    beam = lwfa.GaussianBeam(w0=120 * u.micrometer, λL=800 * u.nanometer)
    laser = lwfa.Laser(ɛL=8 * u.milijoule, τL=50 * u.femtosecond, beam=beam)


    plasma = Plasma(n_pe=2.65e+18 * u.cm ** (-3))

    print(laser)
    print(plasma)


if __name__ == "__main__":
    main()
