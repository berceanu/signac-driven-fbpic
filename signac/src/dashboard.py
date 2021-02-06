from signac_dashboard import Dashboard
from signac_dashboard.modules import (
    StatepointList,
    DocumentList,
    ImageViewer,
    VideoViewer,
    FileList,
    Notes,
)
import unyt as u


class MyDashboard(Dashboard):
    def job_sorter(self, job):
        return (job.sp.z_rezolution_factor, job.sp.Nr, job.sp.rmax)

    def job_title(self, job):
        n = f"{job.sp.z_rezolution_factor:.1f}"
        rmax = (job.sp.rmax * u.meter).to(u.micrometer)
        return f"Δz = λ / {n}, Nr = {job.sp.Nr}, rmax = {rmax:.1f}; {job.sp.r_boundary_conditions} BC"


# To use multiple workers, a single shared key must be used. By default, the
# secret key is randomly generated at runtime by each worker. Using a provided
# shared key allows sessions to be shared across workers. This key was
# generated with os.urandom(16)

config = {
    "DASHBOARD_PATHS": ["."],
    "SECRET_KEY": b"\x99o\x90'/\rK\xf5\x10\xed\x8bC\xaa\x03\x9d\x99",
}

modules = [
    StatepointList(name="Parameters", enabled=False),
    DocumentList(name="Derived parameters", enabled=False),
    FileList(enabled=False),
    # Notes(),
    ImageViewer(name="All figures", enabled=False),
    ImageViewer(name="2D Histogram", img_globs=["hist2d.png"], enabled=False),
    ImageViewer(name="Electron Spectrum", img_globs=["final_histogram.png"]),
    VideoViewer(name="Evolution movies", enabled=False),
]

dashboard = MyDashboard(config=config, modules=modules)

if __name__ == "__main__":
    dashboard.main()
