import numpy as np
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
import unyt as u
import signac
import electron_spectrum as es
import logging

logger = logging.getLogger(__name__)
log_file_name = "overlay.log"

def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    spectra = es.multiple_jobs_single_iteration(
        jobs=proj.find_jobs(),
        label=es.SpectrumLabel(
            key="zfoc_from_nozzle_center",
            name=r"$x$",
            unit=r"$\mathrm{\mu m}$",
            get_value=lambda job, key: job.sp[key] * 1.0e+6,
        ),
    )
    spectra.plot()
    spectra.plot_quantity("peak_position", ylabel="E (MeV)")
    spectra.plot_quantity("total_charge", ylabel="Q (pC)")


if __name__ == "__main__":
    logging.basicConfig(
        filename=log_file_name,
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
