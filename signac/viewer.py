import streamlit as st
from PIL import Image
import pathlib
import json
from operator import itemgetter
import unyt as u
import collections

current_dir = pathlib.Path.cwd()

statepoints_file = current_dir / "signac_statepoints.json"
statepoints_bytes = statepoints_file.read_bytes()
statepoints_dict = json.loads(statepoints_bytes)


d = dict()
for job_id in statepoints_dict:
    tau_fs = (statepoints_dict[job_id]["tau"] * u.second).to_value("fs")
    d[f"{tau_fs:.1f}"] = f"{job_id}"


tau = st.sidebar.select_slider("Pulse duration [fs]", options=sorted(d))

png_file = current_dir / f"hist2d_{d[tau]:.6}.png"
image_bytes = Image.open(png_file)
st.image(image_bytes)

video_file = current_dir / f"rho_{d[tau]:.6}.mp4"
video_bytes = video_file.read_bytes()
st.video(video_bytes)

st.json(statepoints_dict[d[tau]])

# for png_file in current_dir.glob('*.png'):
#     image_bytes = Image.open(png_file)
#     st.image(image_bytes)
