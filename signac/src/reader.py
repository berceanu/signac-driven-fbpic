import pandas as pd


def read_density(txt_file, every_nth=20, offset=True):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["position_mm", "density_cm_3", "error_density_cm_3"],
    )

    # convert to meters
    df["position_m"] = df.position_mm * 1e-3

    # substract offset
    if offset:
        df.position_m = df.position_m - df.position_m.iloc[0]

    # normalize density
    df["norm_density"] = df.density_cm_3 / df.density_cm_3.max()
    # check density values between 0 and 1
    if not df.norm_density.between(0, 1).any():
        raise ValueError("The density contains values outside the range [0,1].")

    # return every nth item
    df = df.iloc[::every_nth, :]

    # return data as numpy arrays
    return df.position_m.to_numpy(), df.norm_density.to_numpy()


if __name__ == "__main__":
    position_m, norm_density = read_density("../density_1_inlet_spacers.txt")

    print(position_m*1e6)
    # print(norm_density)
