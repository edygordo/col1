import urllib.request
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
from autograd import grad
import math
import numpy as np
import pdb
mu = G*M_earth

class rv_observation:
    def __init__(self, sim_data, r_sigma, v_sigma):
        self.r = sim_data.r + r_sigma*np.random.randn(sim_data.r.shape[0], sim_data.r.shape[1])
        self.v = sim_data.v + v_sigma*np.random.randn(sim_data.v.shape[0], sim_data.v.shape[1])
        
class collision_model:
    def __init__(self, m_sat, m_rso, sat, rso):
        self.m_sat = m_sat
        self.m_rso = m_rso
        self.sat = sat
        self.rso = rso
        
        m_d = m_sat + m_rso
        m_n1 = m_sat - m_rso
        m_n2 = m_rso - m_sat

        self.v_sat = m_n1/m_d*sat.v + 2*m_rso/m_d*rso.v
        self.v_rso = m_n2/m_d*rso.v + 2*m_sat/m_d*sat.v
        self.delta_v_sat = self.v_sat - sat.v
        self.delta_v_rso = self.v_rso - rso.v
    
    def jacobian(self, X):
        rd = X[0:3]
        vd = X[3:6]
        md = X[6]
        H = np.zeros([6,7])
        H[0:3,0:3] = np.eye(3)
        H[3:6,3:6] = 2*md/(md + self.m_sat.value)*np.eye(3)
        H[3:6,6:7] = 2*self.m_sat/(md + self.m_sat.value)**2*vd
        return H
    
    def collision_obs(self, X):
        rd = X[0:3]
        vd = X[3:6]
        md = X[6]
        v_sat = self.sat.v.value
        
        delv = vd.flatten() - v_sat
        m = 2*md/(self.m_sat.value + md)
        Z = np.hstack((rd.flatten(), m*delv))
        #pdb.set_trace()
        return Z.flatten()
    
def objective(X, Z, collision):
    rsd = Z - collision.collision_obs(X)
    return rsd.T.dot(rsd).item(0)

def mass_constraint1(X):
    cof = np.zeros([7,1])
    cof[6] = 1
    return  1 - cof.T.dot(X).item(0)

def mass_constraint2(X):
    cof = np.zeros([7,1])
    cof[6] = 1
    return  cof.T.dot(X).item(0) - 0.01

def velocity_constraint(X):
    rd = X[0:3]*1000
    vd = X[3:6]*1000
    speed = np.linalg.norm(vd)
    return np.sqrt(mu.value/np.linalg.norm(rd)) - speed

    
def lagrangian(L, Z, collision):    
    X = L[0:7]
    lmbda = L[7:10]
    return objective(X, Z, collision) + lmbda[0]*mass_constraint1(X) + \
            lmbda[1]*mass_constraint2(X) #+ lmbda[2]*velocity_constraint(X)
 
           
def partial_Lagrangian(L, Z, collision):
    n = L.shape[0]
    element = np.finfo(float).eps
    #element = 0.000001
    del_mat = element*np.eye(n)
    partial = np.zeros([n, 1])
    for i in range(0, n):
        partial[i] = (lagrangian(L + del_mat[:,[i]], Z, collision) - lagrangian(L, Z, collision))/element
    return partial

def hessian_lagrangian(L, Z, collision):
    element = 0.1
    del_mat = element*np.eye(10)
    hess = np.zeros([10,10])
    
    for i in range(0,10):
        hess[:, i] = np.array((partial_Lagrangian(L + del_mat[:,[i]], Z, collision) \
            - partial_Lagrangian(L, Z, collision))/element).T
    return hess

def svdinv(mat):
    U, D, VT = np.linalg.svd(mat)
    D_inv = np.zeros(mat.shape)
    eps = 10**-6
    for i in range(0, mat.shape[0]):
        if D[i] > eps:
            D_inv[i, i] = 1/D[i]
        else:
            D_inv[i, i] = 0
    
    return (VT.T.dot(D_inv)).dot(U)