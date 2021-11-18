import urllib.request
import requests
import pandas as pd
import json
import getpass
from astropy import time
import astropy.coordinates as coords
import multiprocessing as mp
from numba import jit, njit
from astropy.constants import G, M_earth
from datetime import datetime, timedelta
from poliastro.util import time_range
from astropy import units as u
from datetime import datetime
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import J2000
from scipy.optimize import fsolve
from observation import collision_model
import math
import numpy as np
import pdb
mu = G*M_earth

def remove_space(str):
    str = str.replace(" ", "")
    str = str.replace("-", "")
    return 

def access_tle():
    url = "https://www.celestrak.com/NORAD/elements/active.txt"
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    data_content = response.read()
    return data_content.decode("utf-8")

class tle_data:
    def __init__(self, satellite_name, TLE):
        indx = TLE.find(satellite_name)
        #print(TLE[indx:indx+24])
        #print(TLE[indx+26 : indx+95])
        #print(TLE[indx+97 : indx+166] + '\n')
        # epoch = float(TLE[indx+49 : indx+58])
        self.epoch = tleepoch_to_datestr(TLE[indx+44 : indx+58])
        #self.B_star = int(remove_space(TLE[indx+79 : indx+85]))* (10**(-5))* \
        #    (10**(int(remove_space(TLE[indx+85 : indx+87]))))
        self.i = float(TLE[indx+106 : indx+113])
        self.RA = float(TLE[indx+114 : indx+122])
        self.e = float(TLE[indx+123 : indx+130])*(10**(-7))
        self.w = float(TLE[indx+131 : indx+139])
        self.Me = np.deg2rad(float(TLE[indx+140 : indx+148]))
        self.Mm = float(TLE[indx+149 : indx+160])
        self.rev_num = int(TLE[indx+160 : indx+165])
        self.T = 86400/self.Mm*u.s
        self.a = (mu/(2*np.pi)**2*self.T**2)**(1/3)
        E, nu = anomalies(self.Me, self.e)
        self.E = E
        self.nu = nu

def anomalies(Me, e):
    E = fsolve(kepler_eq,0.6,args=[Me,e]) ## Eccentric anomaly
    nu = 2*math.atan2(math.tan(E/2)*np.sqrt((1+e)/(1-e)),1)  ## True Anomaly (TA)
    return np.rad2deg(E), np.rad2deg(nu)


def kepler_eq(x,argx):
    Me = argx[0]
    e = argx[1]
    return x - e*math.sin(x) - Me

def get_orbit(sat_name, TLE, **kwrgs):
    """
    Generates an Ideal Orbit of desired satellite
    INPUT Parameters:-
     sat_name:- satellite name
     TLE:- Two Line element 
    OUTPUT:-
     Orbit Module with Ideal keplerian motion only 
    """
    sat = tle_data(sat_name, TLE)
    a = sat.a
    ecc = sat.e * u.one
    if kwrgs:
        inc = kwrgs['inc'] * u.deg
        raan = kwrgs['raan'] * u.deg
    else:
        inc = sat.i * u.deg
        raan = sat.RA * u.deg
    
    argp = sat.w * u.deg
    nu = sat.nu * u.deg
    epoch = time.Time(sat.epoch, format='iso', scale='utc')
    sat_orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)
    return sat_orbit

def create_rso(sat_orbit, delv):
    """
    An unperturbed is created with noise added to velocity of the RSO object in GCRS Frame
    """
    r = sat_orbit.r
    v = sat_orbit.v + delv
    rso = Orbit.from_vectors(Earth, r, v, sat_orbit.epoch)
    return rso    
    
def tleepoch_to_datestr(epoch):
    y_d, nbs = epoch.split('.')
    y_d = y_d.replace(" ", "")
    ep_t = datetime.strptime(y_d, '%y%j') + timedelta(seconds=float("." + nbs) * 24 * 60 * 60)
    return ep_t.strftime("%Y-%m-%d %H:%M:%S")

def constellation_prop(sat_list, epoch, k, *inc_raan):
    sat = sat_list[k]
    if inc_raan:
        inc = inc_raan[0]
        raan = inc_raan[1]
        #pdb.set_trace()
        orbit = get_orbit(sat, inc = inc[k], raan = raan[k])
    else:
        orbit = get_orbit(sat)
    delt = epoch - orbit.epoch
    orbit_prop = orbit.propagate(delt)
    sat_loc = coords.CartesianRepresentation(orbit_prop.r.value[0], orbit_prop.r.value[1], orbit_prop.r.value[2], unit='km')
    orbit_prop.gcrs = coords.GCRS(sat_loc, obstime = epoch)
    return orbit_prop


def constellation(sat_list, epoch, **angls):
    pool = mp.Pool(mp.cpu_count())
    if angls:
        inc = angls['inc']
        raan = angls['raan']
        constltn = pool.starmap(constellation_prop, [(sat_list, epoch, k, inc, raan) for k in range(0,7)])
    else:
        constltn = pool.starmap(constellation_prop, [(sat_list, epoch, k) for k in range(0,7)])
    pool.close()    
    return constltn

class rv_sample:
    def __init__(self, orbit, start_time, end_time, period, **kwargs):
        if kwargs:
            t_span_bc = time_range(orbit.epoch + start_time, periods = int(period/2), end = orbit.epoch)
            t_span_ac = time_range(orbit.epoch, periods = int(period/2), end = orbit.epoch + end_time)
            orbit_ac = kwargs['orbit_ac']
            l_bc = len(t_span_bc)
            l_ac = len(t_span_ac)
            
            self.r = np.empty((l_bc + l_ac, 3))
            self.r[:] = np.nan
            self.v = np.empty((l_bc + l_ac, 3))
            self.v[:] = np.nan
            
            for i, t in zip(range(0, l_bc), t_span_bc):
                orbit_prop = orbit.propagate(orbit.epoch - t)
                self.r[i,:] = orbit_prop.r.value
                self.v[i,:] = orbit_prop.v.value
            
            for i, t in zip(range(l_bc, l_bc + l_ac), t_span_ac):
                orbit_prop = orbit_ac.propagate(t - orbit_ac.epoch)
                self.r[i,:] = orbit_prop.r.value
                self.v[i,:] = orbit_prop.v.value
                
        else:
            t_span = time_range(orbit.epoch + start_time, periods= period, end= orbit.epoch + end_time)
            self.r = np.empty((len(t_span), 3))
            self.r[:] = np.nan
            self.v = np.empty((len(t_span), 3))
            self.v[:] = np.nan
        
            for i, t in zip(range(0, len(t_span)), t_span):
                orbit_prop = orbit.propagate(orbit.epoch - t)
                self.r[i,:] = orbit_prop.r.value
                self.v[i,:] = orbit_prop.v.value
                
def state_dynamics(X):
    rd = X[0:3]
    vd = X[3:6]
    md = X[6]
    
    X_dot = np.zero([7,1])
    X_dot[0:3] = vd
    X_dot[3:6] = mu/np.linalg.norm(rd)**3*rd #! mu in meter unit
    X_dot[6] = 0
    return X_dot


def tle_hist():
    # Getting user inputs
    username = input("Enter username: ")
    password = getpass.getpass(prompt = "Enter password: ")
    rso_fileName = input("Enter name of space object list file: ")
    startingDate = input("Enter starting epoch date (YYYY-MM-DD): ")
    terminalDate = input("Enter terminal epoch date (YYYY-MM-DD): ")
    
    # Converting date string to datetime object
    startingDate_obj = datetime.strptime(startingDate, '%Y-%m-%d')
    terminalDate_obj = datetime.strptime(terminalDate, '%Y-%m-%d')
    duration = (terminalDate_obj-startingDate_obj).days
    
    # Loading Data
    rso_list = pd.read_csv(rso_fileName, header = None, index_col = False) # Reading objest list file into dataframe.
    rso_list = rso_list.transpose()[0]  # Converting dataframe to series.
    rso_list = rso_list.astype(int)     # Converting to int to remove space or other characters and for later use.
    rso_list_string = ", ".join([str(rso) for rso in rso_list]) # converting again to string and joining for accessing api.
    
    print("\nThe following NORAD_CAT_ID's are being searched: \n" + rso_list_string + ".\n")
    
    class MyError(Exception):
        def __init___(self,args):
            Exception.__init__(self,"my exception was raised with arguments {0}".format(args))
            self.args = args

    # Credentials for using the site
    siteCred = {'identity': username, 'password': password}
    
    # Forming REST query for accessing https://www.space-track.org
    uriBase                   = "https://www.space-track.org"
    requestLogin              = "/ajaxauth/login"
    requestControlerAndAction = "/basicspacedata/query"
    requestTleData = ( "/class/gp_history/NORAD_CAT_ID/"
                  + rso_list_string + 
                  "/orderby/TLE_LINE1%20ASC/EPOCH/"
                  + startingDate + "--" + terminalDate +
                  "/format/json" )
    
    # Using requests package to drive the RESTful session with space-track.org
    with requests.Session() as session:
        # run the session in a with block to force session to close if we exit

        # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in
        resp = session.post(uriBase + requestLogin, data = siteCred)
        if resp.status_code != 200:
            raise MyError(resp, "POST fail on login")
        
        # This query picks up all the objects' TLEs from the catalog for the specified dates.
        # Note - a 401 failure shows you have bad credentials 
        resp = session.get(uriBase + requestControlerAndAction + requestTleData)
        if resp.status_code != 200:
            print(resp)
            raise MyError(resp, "GET fail on request for satellites")
    
        tleData_df = pd.read_json(resp.text)
    
        session.close()
        
        relevantCol = ['NORAD_CAT_ID', 'CREATION_DATE',
                   'TLE_LINE0', 'TLE_LINE1', 'TLE_LINE2',
                   'ECCENTRICITY', 'SEMIMAJOR_AXIS', 'RA_OF_ASC_NODE',
                   'ARG_OF_PERICENTER', 'INCLINATION', 'PERIOD', 'EPOCH']
        tleData_df = tleData_df[relevantCol]
        tleData_df['EPOCH'] = pd.to_datetime(tleData_df['EPOCH'], format='%Y-%m-%d')
        tleData_df.set_index(['NORAD_CAT_ID','CREATION_DATE'], inplace = True)
        tleData_df.sort_index(inplace = True)
        
        return tleData_df