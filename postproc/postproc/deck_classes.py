from collections import namedtuple
import numpy as np
import math
import pint

ureg = pint.UnitRegistry()
ureg.load_definitions('./units/lwfa_def.txt')

Point = namedtuple('Point', ['x', 'y', 'z'])
NrGridPoints = namedtuple('NrGridPoints', ['x', 'y', 'z'])
WindowProp = namedtuple('WindowProp', ['v_x','t_start','t_end'])


class Grid:
    def __init__(self, left_edge, right_edge, nr_points):
        self._le = left_edge
        self._re = right_edge
        self._np = nr_points
        
        # box axes
        Box = namedtuple('Box', ['x', 'y', 'z'])
        self.box = Box(x = self._linspace('x'),
                       y = self._linspace('y'),
                       z = self._linspace('z'))
        
        # box size
        Width = namedtuple('Width', ['x', 'y', 'z'])
        self.width = Width(x = self._width('x'),
                          y = self._width('y'),
                          z = self._width('z'))
        
    def _linspace(self, axis):
        return np.linspace(getattr(self._le, axis), getattr(self._re, axis), getattr(self._np,axis))
    
    def _width(self, axis):
        return getattr(self._re, axis) - getattr(self._le, axis)


class MovingWindow(Grid):
    def __init__(self, win_prop, *args, **kwargs):
        self._prop = win_prop
        super().__init__(*args, **kwargs)

class Domain(Grid):
    def __init__(self, moving_window):
        self.mw = moving_window        
        redge = self._get_right_edge()
        self.npoints = self._get_nr_points(redge)

        super().__init__(self.mw._le, redge, self.mw._np)
    
    def _get_right_edge(self):
        # total moving time of the window
        delta_t = self.mw._prop.t_end - self.mw._prop.t_start
        # corresponding distance
        dist = delta_t * self.mw._prop.v_x
        # new x limit of domain
        x_max = self.mw._re.x + dist
        return Point(x_max, self.mw._re.y, self.mw._re.z)
    
    def _get_nr_points(self, re):
        # resolution along x direction
        delta_x = self.mw.width.x / (self.mw._np.x - 1)
        # width of the full domain along x
        width_x = re.x - self.mw._le.x
        n = width_x / delta_x
        assert n.dimensionless, 'length units error!'
        nx = math.trunc(n.magnitude + 1)
        return NrGridPoints(nx, self.mw._np.y, self.mw._np.z)


class Plasma:
    def __init__(self, plasma_skin_depth, up_ramp_length, flat_top_length):
        self.skin_depth = plasma_skin_depth
        self.url = up_ramp_length
        self.ftl = flat_top_length        
        self.time_unit = self.skin_depth.to('femtosecond', 'lwfa')
        self.frequency = self.skin_depth.to('terahertz', 'lwfa')
        self.density = self.skin_depth.to('1/cm**3', 'lwfa')
        
    def __repr__(self):
        return 'Plasma(plasma_skin_depth={}, up_ramp_length={}, flat_top_length={})'.format(self.skin_depth, self.url, self.ftl)
    
    def __str__(self):
        return 'Plasma with skin depth {}, time unit {}, frequency {} and density {}.'.format(self.skin_depth, self.time_unit, self.frequency, self.density)


class Species:
    def __init__(self, domain, plasma, n):
        self._dom = domain
        self._pl = plasma
        self.rho_min = n * self._pl.density
        self.rho = self.build_density(self._dom.box, 
                                      self._pl.url.magnitude, 
                                      self._pl.ftl.magnitude,
                                      self._dom.width.y.magnitude / 2,
                                      self._pl.density.magnitude)
        
    
    def build_density(self, box, url, ftl, half_width, density_unit):
        x, y, z = tuple(getattr(box, ax) for ax in ('x', 'y', 'z'))        
        shape = tuple(v.shape[0] for v in (y, x, z))  
        rho = np.zeros(shape)

        rho[...] = np.where(np.logical_or(x <= -url, x >= ftl + url), 0, 1)[np.newaxis, :, np.newaxis]

        cond = (x > -url) & (x < 0)
        rho[:, cond, :] = ((1 + np.cos(x * math.pi / url)) / 2)[np.newaxis, cond, np.newaxis]

        cond = (x > ftl) & (x < ftl + url)
        rho[:, cond, :] = ((1 + np.cos((x - ftl) * math.pi / url)) / 2)[np.newaxis, cond, np.newaxis]

        cond_y = np.abs(y) > 0.99 * half_width
        cond_z = np.abs(z) > 0.99 * half_width
        rho[cond_y, :, :] = 0
        rho[:, :, cond_z] = 0
        
        rho *= density_unit
        return rho
