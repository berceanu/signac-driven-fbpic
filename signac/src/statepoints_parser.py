"""
Module for parsing data out of signac_statepoints.json files.
These files are generated via init.py by the signac project.
"""
import json
import pathlib
import unyt as u


def to_cm_minus_three(n_e):
    """Converts density from m^(-3) to cm^(-3)."""
    return (n_e / u.meter ** 3).to(1 / u.centimeter ** 3)


def load_statepoints(statepoints_path):
    json_path = statepoints_path / "signac_statepoints.json"
    statepoints = json.loads(json_path.read_bytes())
    return statepoints


def shorten_id(d):
    return {f"{k:.6}": v for k, v in d.items()}


def change_density_units(d):
    return {k: to_cm_minus_three(v) for k, v in d.items()}


def map_id_to_density(statepoints):
    id_to_density = {}
    for signac_job_id, job_state_point in statepoints.items():
        id_to_density[signac_job_id] = job_state_point["n_e"]
    return id_to_density


def sort_by_value(d):
    return dict(sorted(d.items(), key=lambda item: item[1]))


def main():
    """Main entry point."""
    statepoints = load_statepoints(pathlib.Path.home() / "tmp" / "runs")
    id_to_density = map_id_to_density(statepoints)

    d = shorten_id(id_to_density)
    short_id_to_density_cm3 = change_density_units(d)

    bunch_fn_to_density = {
        f"final_bunch_{id}.txt": density
        for id, density in short_id_to_density_cm3.items()
    }
    sorted_bunch_fn_to_density = sort_by_value(bunch_fn_to_density)

    # #
    for fn, n_e in sorted_bunch_fn_to_density.items():
        print(f"{n_e:.1e} -> {fn}")

    print()  #
    runs_dir = pathlib.Path.home() / "tmp" / "runs"
    txt_files = runs_dir.glob("final_bunch_*.txt")

    for fn in txt_files:
        print(f"{fn.name}: {sorted_bunch_fn_to_density[fn.name]:.1e}")


if __name__ == "__main__":
    main()
