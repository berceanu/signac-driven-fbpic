import pandas as pd
import pathlib


def main():
    p = pathlib.Path.cwd() / "nvml.csv"
    df = pd.read_csv(p)
    print(df)


if __name__ == "__main__":
    main()
