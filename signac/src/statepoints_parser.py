"""
Module for parsing data out of signac_statepoints.json files.
These files are generated via init.py by the signac project.
"""
import json
import pathlib


def main():
    """Main entry point."""
    statepoints_path = pathlib.Path.home() / "tmp" / "runs" / "signac_statepoints.json"
    print(statepoints_path)


if __name__ == "__main__":
    main()
