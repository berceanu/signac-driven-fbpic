# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import holoviews as hv

hv.extension('bokeh')

# %%
from holoviews import streams

# %%
from holoviews import opts

# %% [markdown]
# ## 2D plots with interactive slicing

# %%
# ls = np.linspace(0, 10, 200)
xx, yy = np.meshgrid(ls, ls)
bounds=(0,0,10,10)   # Coordinate system: (left, bottom, right, top)

# %%
energy = hv.Dimension('energy', label='E', unit='MeV')
distance = hv.Dimension('distance', label='d', unit='m')
charge = hv.Dimension('charge', label='Q', unit='pC')

# %%
image = hv.Image(np.sin(xx)*np.cos(yy), bounds=bounds, kdims=[energy, distance], vdims=charge)
pointer = streams.PointerXY(x=5,y=5, source=image)


dmap = hv.DynamicMap(lambda x, y: hv.VLine(x) * hv.HLine(y), streams=[pointer])
x_sample = hv.DynamicMap(lambda x, y: image.sample(energy=x).opts(color='darkred'), streams=[pointer])
y_sample = hv.DynamicMap(lambda x, y: image.sample(distance=y).opts(color='lightsalmon'), streams=[pointer])

pointer_dmap = hv.DynamicMap(lambda x, y: hv.Points([(x, y)]), streams=[pointer])
pointer_x_sample = hv.DynamicMap(lambda x, y: hv.Points([(y, image[x,y])]), streams=[pointer])
pointer_y_sample = hv.DynamicMap(lambda x, y: hv.Points([(x, image[x,y])]), streams=[pointer])

layout = (image * dmap * pointer_dmap) + ((x_sample * pointer_x_sample) + (y_sample * pointer_y_sample))

layout.opts(
    opts.Image(cmap='Viridis', aspect='square', frame_width=300, colorbar=True, tools=['hover']),
    opts.Curve(framewise=False, ylim=(-1, 1)),
    opts.VLine(color='darkred'),
    opts.HLine(color='lightsalmon'),
    opts.Points(color='red', marker='o', size=10)
).cols(3)

# %% [markdown]
# ## Interactive integration

# %%
xs = np.linspace(-3, 3, 400)


# %%
def function(xs, time):
    "Some time varying function"
    return np.exp(np.sin(xs+np.pi/time))


# %%
def integral(limit_a, limit_b, y, time):
    limit_a = -3 if limit_a is None else np.clip(limit_a,-3,3)
    limit_b = 3 if limit_b is None else np.clip(limit_b,-3,3)
    curve = hv.Curve((xs, function(xs, time)))
    area  = hv.Area ((xs, function(xs, time)))[limit_a:limit_b]
    summed = area.dimension_values('y').sum() * 0.015  # Numeric approximation
    return (area * curve * hv.VLine(limit_a) * hv.VLine(limit_b) * hv.Text(limit_b - 0.8, 2.0, '%.2f' % summed))


# %%
integral_streams = [
    streams.Stream.define('Time', time=1.0)(),
    streams.PointerX().rename(x='limit_b'),
    streams.Tap().rename(x='limit_a')
]

# %%
integral_dmap = hv.DynamicMap(integral, streams=integral_streams)

# %%
integral_dmap.opts(
    opts.Area(color='#fff8dc', line_width=2),
    opts.Curve(color='black'),
    opts.VLine(color='red'))

# %%
image.dimensions()

# %%
dim('charge').min().apply(image)

# %%
dim('charge').max().apply(image)

# %%
