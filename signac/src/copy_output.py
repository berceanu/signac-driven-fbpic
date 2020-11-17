import pathlib
import signac

project = signac.get_project()


def copy_append_hash(path_to_fname, dst_dir):
    fname = path_to_fname.stem
    ext = path_to_fname.suffix

    dst_fname = dst_dir / f"{fname}_{job.id:.6}{ext}"
    dst_fname.write_bytes(path_to_fname.read_bytes())


if __name__ == "__main__":
    out_path = pathlib.Path.cwd() / "runs"
    out_path.mkdir(parents=True, exist_ok=True)

    statepoints = pathlib.Path.cwd() / "signac_statepoints.json"
    p = out_path / "signac_statepoints.json"
    p.write_bytes(statepoints.read_bytes())

    for job in project:
        for fname in ("rho.mp4", "centroid.mp4"):
            src = pathlib.Path(job.fn(fname))
            copy_append_hash(src, out_path)
