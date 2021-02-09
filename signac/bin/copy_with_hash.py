import pathlib
import signac
import unyt as u

project = signac.get_project()
project.write_statepoints()

def copy_with_hash(path_to_fname, dst_dir, signac_job):
    fname = path_to_fname.stem
    ext = path_to_fname.suffix

    dst_fname = dst_dir / f"{fname}_{signac_job.id:.6}{ext}"
    dst_fname.write_bytes(path_to_fname.read_bytes())


if __name__ == "__main__":
    for job in sorted(project, key=lambda job: job.doc.x):
        x = (job.doc.x * u.meter).to(u.micrometer)
        print(f"{x:.1f} -> {job.id:.6}")

    out_path = pathlib.Path.cwd() / "runs"
    out_path.mkdir(parents=True, exist_ok=True)

    statepoints = pathlib.Path.cwd() / "signac_statepoints.json"
    p = out_path / "signac_statepoints.json"
    p.write_bytes(statepoints.read_bytes())

    for job in project:
        for f_name in ("final_histogram.png", "rho.mp4", "phasespace.mp4"):
            src = pathlib.Path(job.fn(f_name))
            copy_with_hash(src, out_path, job)
