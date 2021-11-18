from os import stat
from re import U
import sys
from astropy.units.function import units
import pandas
from poliastro.core.propagation import func_twobody

from poliastro.twobody.propagation import cowell
sys.path.append("./modules")
import numpy as np
import pandas as pd
from numpy.linalg import inv, norm
from astropy import units as u
from datetime import datetime
from numba import jit
from scipy.optimize import minimize, fsolve, newton
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import *
from poliastro.constants import J2000
from poliastro.util import time_range
from datetime import datetime, timedelta
import pdb
import plotly.io as pio
pio.renderers.default = "vscode"
from orbit_module import get_orbit, tle_data, create_rso, rv_sample, access_tle
from observation import rv_observation, collision_model, objective, mass_constraint1, \
    mass_constraint2, velocity_constraint, partial_Lagrangian, hessian_lagrangian, svdinv, lagrangian
from astropy.constants import G, M_earth
import matplotlib.pyplot as plt
from astropy import time
from astropy.coordinates import EarthLocation, AltAz
import astropy.coordinates as coord
from tqdm import tqdm

from astropy.coordinates import solar_system_ephemeris
from poliastro.ephem import build_ephem_interpolant
from poliastro.bodies import Sun,Earth
from poliastro.core.elements import rv2coe
from poliastro.core.perturbations import atmospheric_drag_exponential, atmospheric_drag_model, third_body, J2_perturbation, J3_perturbation, radiation_pressure
from poliastro.core.propagation import func_twobody
from poliastro.earth.atmosphere import COESA76 
from poliastro.constants import rho0_earth, H0_earth

mu = G*M_earth
type = 'testing'

if type == 'training':
    sat_Data = pd.read_csv('./data/star_link.csv', header = None)
    location = './data/training_data.csv'
    sat_list = sat_Data.loc[:,0]
else :
    sat_Data = pd.read_csv('./data/sat_list.csv', header = None)
    location = './data/test_data.csv'
    sat_list = sat_Data.loc[:,0] + "-2-" + sat_Data.loc[:,2]

epoch = time.Time('2021-06-7 12:00:00', format='iso', scale='utc')
sat_no = len(sat_list)

m_rso = (0.5 + 0.2*np.random.randn(sat_no))*u.kg
m_sat = (2 + 0.5*np.random.randn(sat_no))*u.kg
# Creating a perturbation function to add to RSO and Satellite Orbit

# For knowing details of Perturbation models use go to link:- 
# https://github.com/poliastro/poliastro/blob/main/src/poliastro/core/perturbations.py

# The Perturbations that will be added are described below:-
#  1. Atmoshperic Drag Model:- If orbits involved are withing 1000 Km of altitude from Earth surface then COESCA76 Model can be used
#                               , Here I have used Exponential Decay Model
#  2. J2 Acceleration
#  3. J3 Acceleration
#  4. Radiation Perturbation

# Creating a Sun Ephemeris interpolater function
solar_system_ephemeris.set("de432s") # database keeping positions of bodies in Solar system over time
epoch = time.Time(epoch.jd, format="jd", scale="tdb") # conversion of epoch into julian Date
#body_s = build_ephem_interpolant(Sun, 28 * u.day, (epoch.value * u.day, epoch.value * u.day + 5 * u.day), rtol=1e-5) 
# body_s is a CartesianRepresentation of the barycentric position of a body(Sun in our case) (i.e., in the ICRS frame)
# Orbital period of satellite in build_ephem_interpolant ?(Taken 28 days)

def perturb_function(t0, state, k): # t0 is simulation time step, state is of the body present in the orbit, 
                                    # k = gravitational constant
    # Keplerian Acceleration(Ideal Motion)
    du_kep = func_twobody(t0, u_=state, k=k)

    # Atmoshperic Drag Acceleration
    # COESCA76 is a more sophisticated Atmosphere Model, but it's valid upto 1000Km above earth surface
    #ax, ay, az = atmospheric_drag_model(t0,state,k,Earth.R.to(u.km).value,C_D=2.2,A_over_m=A_over_m,model=COESA76)
    #du_atm = [0,0,0,ax,ay,az]

    A_over_m = ((np.pi / 4.0) * (u.m ** 2) / (100 * u.kg)).to_value(u.km**2/u.kg) # effective spacecraft area/mass of the spacecraft (km^2/kg)
    rho0 = rho0_earth.to(u.kg / u.km ** 3).value
    H0 = H0_earth.to(u.km).value
    ax, ay, az = atmospheric_drag_exponential(t0,state,k,R=Earth.R.to(u.km).value,C_D=2.2,A_over_m=A_over_m,H0=H0,rho0=rho0)
    du_atm = [0,0,0,ax,ay,az]

    # J2 Acceleration
    ax, ay, az = J2_perturbation(t0, state, k, J2=Earth.J2.value,R=Earth.R.to(u.km).value)
    du_J2 = [0,0,0,ax,ay,az]

    # J3 pertubation haven't been validated yet in official poliastro source code,
    ax, ay, az = J3_perturbation(t0, state, k, J3=Earth.J3.value,R=Earth.R.to(u.km).value)
    du_J3 = [0,0,0,ax,ay,az]

    # Radiation Pressure
    C_R = 1.5
    Wdivc_s = 3.9 # total star emitted power divided by the speed of light (W * s / km)
    # Some value is going below interpolation range ?
    # Error:- A value in x_new is below the interpolation range. (x_new for our case is body_s)
    #ax, ay, az = 10*radiation_pressure(t0,state,k,R=Earth.R.to(u.km).value,C_R=C_R,A_over_m=A_over_m,Wdivc_s=Wdivc_s,star=body_s)
    #du_rad = [0,0,0,ax,ay,az]

    return du_kep + du_J3 + du_J2+ du_atm #+ du_rad


Z = np.zeros([sat_no, 9])
rso_data = np.zeros([sat_no, 7])
TLE = access_tle()
Start = datetime.now()
print(f'Total satellites are:-{sat_no}')
for i in tqdm(range(0, sat_no)): # range(0, sat_no)
    active_sat_tle = get_orbit(sat_list[i], TLE)
    delt = epoch - active_sat_tle.epoch # epoch = collision time, active_sat_tle = current epoch of satellite
    active_sat = active_sat_tle.propagate(delt, method=cowell,f=perturb_function) # In Satellite Orbit perturbations have been added (,method=cowell,f=perturb_function)
    rso_bc = create_rso(active_sat, 0.06*np.random.randn(3)*u.km/u.s) # RSO Orbit created(as it is created using active sat so same perturbations apply here), Noise added to velocity of RSO due to collision
    collision = collision_model(m_sat[i], m_rso[i], active_sat, rso_bc)
    r_z = collision.sat.r.value + 0.1*np.random.randn(3)
    v_z = collision.sat.v.value + 0.05*np.random.randn(3)
    delv_z = collision.delta_v_sat.value + 0.1*np.random.randn(3)
    Z[i, :] = np.hstack((r_z, v_z, delv_z))
    rso_data[i, :] = np.hstack((rso_bc.r.value, rso_bc.v.value, m_rso[i].value))
End = datetime.now()
#print(f'Time taken without any perturbation {End-Start}')
# Saveing training data    
collision_Data = pd.DataFrame(np.hstack((Z, rso_data)), columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'delvx', 'delvy', 'delvz', \
    'rso_x', 'rso_y', 'rso_z', 'rso_vx', 'rso_vy', 'rso_vz', 'rso_m'])
    # x,y,z,vx,vy,vz,delvx,delvy,delvz are active satellite parameters.
    # rso_x,, rso_y, rso_z, rso_vx, rso_vy, rso)vz, rso_m are resident space object(debris) parameter.
location = './data/collision_perturbed_motion.csv'
collision_Data.to_csv(location, index = False)