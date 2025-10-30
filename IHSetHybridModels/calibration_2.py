import numpy as np
import xarray as xr
import fast_optimization as fo
from IHSetUtils.CoastlineModel import CoastlineModel
from .hybrid_builder import hybrid_y09, hybrid_ShoreFor, hybrid_md04
import pandas as pd
import json

class cal_Hybrid_2(CoastlineModel):

    """
    cal_Hybrid_2

    Configuration to calibrate and run the Hybrid Model.
    
    This class reads input datasets, performs its calibration.
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Oneline and cross-shore coupled',
            mode='calibration',
            model_type='HY',
            model_key='Hybrid'
        )

        self.setup_forcing()
        self._set_upper_lowers()

    def setup_forcing(self):
        self.switch_Kal = self.cfg['switch_Kal']
        self.cs_model = self.cfg['cs_model']
        self.dSdt = self.cfg['dSdt']
        if self.cs_model != 'Yates et al. (2009)':
            self.D50 = self.cfg['D50']
        self.y_ini = np.zeros_like(self.Obs_splited_[0,:])
        for i in range(self.ntrs):
            self.y_ini[i] = np.nanmean(self.Obs_splited_[:, i])
        if self.switch_Kal == 0:
            if self.cs_model == 'Yates et al. (2009)':
                self.nn = 6
                self.idx_list = [range(0,1),
                                 range(1,2),
                                 range(2,3),
                                 range(3,4),
                                 range(4,5),
                                 range(5,6)]
            elif self.cs_model == 'Davidson et al. (2013)':
                self.nn = 5
                self.idx_list = [range(0,1),
                                 range(1,2),
                                 range(2,3),
                                 range(3,4),
                                 range(4,5)]
            elif self.cs_model == 'Miller and Dean (2004)':
                self.nn = 5
                self.idx_list = [range(0,1),
                                 range(1,2),
                                 range(2,3),
                                 range(3,4),
                                 range(4,5)]
        else:
            if self.cs_model == 'Yates et al. (2009)':
                self.nn = (self.ntrs * 6) + 1
                self.idx_list = [range(0, self.ntrs),
                                 range(self.ntrs, 2*self.ntrs),
                                 range(2*self.ntrs, 3*self.ntrs),
                                 range(3*self.ntrs, 4*self.ntrs),
                                 range(4*self.ntrs, 5*self.ntrs+1),
                                 range(5*self.ntrs+1, 6*self.ntrs+1)]
            elif self.cs_model == 'Davidson et al. (2013)':
                self.nn = (self.ntrs * 5) + 1
                self.idx_list = [range(0, self.ntrs),
                                 range(self.ntrs, 2*self.ntrs),
                                 range(2*self.ntrs, 3*self.ntrs),
                                 range(3*self.ntrs, 4*self.ntrs+1),
                                 range(4*self.ntrs, 5*self.ntrs+1)]
            elif self.cs_model == 'Miller and Dean (2004)':
                self.nn = (self.ntrs * 5) + 1
                self.idx_list = [range(0, self.ntrs),
                                 range(self.ntrs, 2*self.ntrs),
                                 range(2*self.ntrs, 3*self.ntrs),
                                 range(3*self.ntrs, 4*self.ntrs+1),
                                 range(4*self.ntrs, 5*self.ntrs+1)]
        if self.cs_model == 'Miller and Dean (2004)':
            self.sl_s = self.tide_s + self.surge_s
            self.sl = self.tide + self.surge

    def _set_upper_lowers(self):
        if self.switch_Kal == 0:
            if self.cs_model == 'Yates et al. (2009)':
                lowers = np.array([np.log(self.lb[0]), self.lb[1], np.log(self.lb[2]), np.log(self.lb[3])])
                uppers = np.array([np.log(self.ub[0]), self.ub[1], np.log(self.ub[2]), np.log(self.ub[3])])
                if self.is_exp:
                    lowers = np.hstack((lowers,np.log(self.lb[4])))
                    uppers = np.hstack((uppers,np.log(self.ub[4])))
                else:
                    lowers = np.hstack((lowers,self.lb[4]))
                    uppers = np.hstack((uppers,self.ub[4]))
                lowers = np.hstack((lowers, self.lb[5]))
                uppers = np.hstack((uppers, self.ub[5]))
            elif self.cs_model == 'Davidson et al. (2013)':
                lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2]), -1])
                uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2]), 1])
                if self.is_exp:
                    lowers = np.hstack((lowers,np.log(self.lb[3])))
                    uppers = np.hstack((uppers,np.log(self.ub[3])))
                else:
                    lowers = np.hstack((lowers,self.lb[3]))
                    uppers = np.hstack((uppers,self.ub[3]))
                lowers = np.hstack((lowers, self.lb[4]))
                uppers = np.hstack((uppers, self.ub[4]))
            elif self.cs_model == 'Miller and Dean (2004)':
                lowers = np.array([np.log(self.lb[0]), np.log(self.lb[1]), self.lb[2]])
                uppers = np.array([np.log(self.ub[0]), np.log(self.ub[1]), self.ub[2]])
                if self.is_exp:
                    lowers = np.hstack((lowers,np.log(self.lb[3])))
                    uppers = np.hstack((uppers,np.log(self.ub[3])))
                else:
                    lowers = np.hstack((lowers,self.lb[3]))
                    uppers = np.hstack((uppers,self.ub[3]))
                lowers = np.hstack((lowers, self.lb[4]))
                uppers = np.hstack((uppers, self.ub[4]))
            
        if self.switch_Kal == 1:
            # each parameters is defined for each transect
            if self.cs_model == 'Yates et al. (2009)':
                lowers = np.array([np.log(self.lb[0])])
                uppers = np.array([np.log(self.ub[0])])
                for _ in range(1,self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[0])))
                    uppers = np.hstack((uppers, np.log(self.ub[0])))
                for _ in range(self.ntrs, 2*self.ntrs):
                    lowers = np.hstack((lowers, self.lb[1]))
                    uppers = np.hstack((uppers, self.ub[1]))
                for _ in range(2*self.ntrs, 3*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[2])))
                    uppers = np.hstack((uppers, np.log(self.ub[2])))
                for _ in range(3*self.ntrs, 4*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[3])))
                    uppers = np.hstack((uppers, np.log(self.ub[3])))
                for _ in range(4*self.ntrs, 5*self.ntrs+1):
                    if self.is_exp:
                        lowers = np.hstack((lowers, np.log(self.lb[4])))
                        uppers = np.hstack((uppers, np.log(self.ub[4])))
                    else:
                        lowers = np.hstack((lowers, self.lb[4]))
                        uppers = np.hstack((uppers, self.ub[4]))
                for _ in range(5*self.ntrs+1, 6*self.ntrs+1):
                    lowers = np.hstack((lowers, self.lb[5]))
                    uppers = np.hstack((uppers, self.ub[5]))
            elif self.cs_model == 'Davidson et al. (2013)':
                lowers = np.array([self.lb[0]])
                uppers = np.array([self.ub[0]])
                for _ in range(1,self.ntrs):
                    lowers = np.hstack((lowers, self.lb[0]))
                    uppers = np.hstack((uppers, self.ub[0]))
                for _ in range(self.ntrs, 2*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[1])))
                    uppers = np.hstack((uppers, np.log(self.ub[1])))
                for _ in range(2*self.ntrs, 3*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[2])))
                    uppers = np.hstack((uppers, np.log(self.ub[2])))
                for _ in range(3*self.ntrs, 4*self.ntrs):
                    lowers = np.hstack((lowers, -1))
                    uppers = np.hstack((uppers, 1))
                for _ in range(4*self.ntrs, 5*self.ntrs+1):
                    if self.is_exp:
                        lowers = np.hstack((lowers, np.log(self.lb[3])))
                        uppers = np.hstack((uppers, np.log(self.ub[3])))
                    else:
                        lowers = np.hstack((lowers, self.lb[3]))
                        uppers = np.hstack((uppers, self.ub[3]))
                for _ in range(5*self.ntrs+1, 6*self.ntrs+1):
                    lowers = np.hstack((lowers, self.lb[4]))
                    uppers = np.hstack((uppers, self.ub[4]))
            elif self.cs_model == 'Miller and Dean (2004)':
                lowers = np.array([self.lb[0]])
                uppers = np.array([self.ub[0]])
                for _ in range(1,self.ntrs):
                    lowers = np.hstack((lowers, self.lb[0]))
                    uppers = np.hstack((uppers, self.ub[0]))
                for _ in range(self.ntrs, 2*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[1])))
                    uppers = np.hstack((uppers, np.log(self.ub[1])))
                for _ in range(2*self.ntrs, 3*self.ntrs):
                    lowers = np.hstack((lowers, np.log(self.lb[2])))
                    uppers = np.hstack((uppers, np.log(self.ub[2])))
                for _ in range(3*self.ntrs, 4*self.ntrs+1):
                    if self.is_exp:
                        lowers = np.hstack((lowers, np.log(self.lb[3])))
                        uppers = np.hstack((uppers, np.log(self.ub[3])))
                    else:
                        lowers = np.hstack((lowers, self.lb[3]))
                        uppers = np.hstack((uppers, self.ub[3]))
                for _ in range(4*self.ntrs+1, 5*self.ntrs+1):
                    lowers = np.hstack((lowers, self.lb[4]))
                    uppers = np.hstack((uppers, self.ub[4]))

        self.lowers = lowers
        self.uppers = uppers

    def init_par(self, population_size: int):

        pop = np.zeros((population_size, self.nn))
        for i in range(self.nn):
            pop[:, i] = np.random.uniform(self.lowers[i], self.uppers[i], population_size)

        return pop, self.lowers, self.uppers



    def model_sim(self, par: np.ndarray) -> np.ndarray:

        if self.cs_model == 'Yates et al. (2009)':
            a = -np.exp(par[self.idx_list[0]])
            b = par[self.idx_list[1]]
            cacr = -np.exp(par[self.idx_list[2]])
            cero = -np.exp(par[self.idx_list[3]])
            K = par[self.idx_list[4]]
            vlt = par[self.idx_list[5]]
        elif self.cs_model == 'Davidson et al. (2013)':
            phi = par[self.idx_list[0]]
            cp = np.exp(par[self.idx_list[1]])
            cm = np.exp(par[self.idx_list[2]])
            b = par[self.idx_list[3]]
            K = par[self.idx_list[4]]
            vlt = par[self.idx_list[5]]
        elif self.cs_model == 'Miller and Dean (2004)':
            kacr = np.exp(par[self.idx_list[0]])
            kero = np.exp(par[self.idx_list[1]])
            Y0 = par[self.idx_list[2]]
            K = par[self.idx_list[3]]
            vlt = par[self.idx_list[4]]

        if self.is_exp:
            K = np.exp(K)

        if self.cs_model == 'Yates et al. (2009)':
            Ymd, _ = hybrid_y09(self.y_ini,
                                self.dt,
                                self.hs_s,
                                self.tp_s,
                                self.dir_s,
                                self.depth,
                                self.doc,
                                K,
                                self.X0,
                                self.Y0,
                                self.phi,
                                self.bctype,
                                self.Bcoef,
                                self.mb,
                                self.D50,
                                a,
                                b,
                                cacr,
                                cero,
                                vlt,
                                self.dSdt,
                                self.lst_f)
        elif self.cs_model == 'Davidson et al. (2013)':
            Ymd, _ = hybrid_ShoreFor(self.y_ini,
                                      self.dt,
                                      self.hs_s,
                                      self.tp_s,
                                      self.dir_s,
                                      self.depth,
                                      self.doc,
                                      K,
                                      self.X0,
                                      self.Y0,
                                      self.phi,
                                      self.bctype,
                                      self.Bcoef,
                                      self.mb,
                                      self.D50,
                                      phi,
                                      cp, 
                                      cm,
                                      b,
                                      vlt,
                                      self.dSdt,
                                      self.lst_f)
        elif self.cs_model == 'Miller and Dean (2004)':
            Ymd, _ = hybrid_md04(self.y_ini,
                                 self.dt,
                                 self.hs_s,
                                 self.tp_s,
                                 self.dir_s,
                                 self.depth,
                                 self.doc,
                                 K,
                                 self.X0,
                                 self.Y0,
                                 self.phi,
                                 self.bctype,
                                 self.Bcoef,
                                 self.mb,
                                 self.D50,
                                 self.sl_s,
                                 self.Hberm,
                                 Y0,
                                 kacr,
                                 kero,
                                 vlt,
                                 self.dSdt,
                                 self.lst_f)
            
        return Ymd[self.idx_obs_splited].flatten()

    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.cs_model == 'Yates et al. (2009)':
            a = par[self.idx_list[0]]
            b = par[self.idx_list[1]]
            cacr = par[self.idx_list[2]]
            cero = par[self.idx_list[3]]
            K = par[self.idx_list[4]]
            vlt = par[self.idx_list[5]]

            Ymd, _ = hybrid_y09(self.y_ini,
                                self.dt,
                                self.hs,
                                self.tp,
                                self.dir,
                                self.depth,
                                self.doc,
                                K,
                                self.X0,
                                self.Y0,
                                self.phi,
                                self.bctype,
                                self.Bcoef,
                                self.mb,
                                self.D50,
                                a,
                                b,
                                cacr,
                                cero,
                                vlt,
                                self.dSdt,
                                self.lst_f)
        elif self.cs_model == 'Davidson et al. (2013)':
            phi = par[self.idx_list[0]]
            cp = par[self.idx_list[1]]
            cm = par[self.idx_list[2]]
            b = par[self.idx_list[3]]
            K = par[self.idx_list[4]]
            vlt = par[self.idx_list[5]]
            Ymd, _ = hybrid_ShoreFor(self.y_ini,
                                    self.dt,
                                    self.hs,
                                    self.tp,
                                    self.dir,
                                    self.depth,
                                    self.doc,
                                    K,
                                    self.X0,
                                    self.Y0,
                                    self.phi,
                                    self.bctype,
                                    self.Bcoef,
                                    self.mb,
                                    self.D50,
                                    phi,
                                    cp, 
                                    cm,
                                    b,
                                    vlt,
                                    self.dSdt,
                                    self.lst_f)
        elif self.cs_model == 'Miller and Dean (2004)':
            kacr = par[self.idx_list[0]]
            kero = par[self.idx_list[1]]
            Y0 = par[self.idx_list[2]]
            K = par[self.idx_list[3]]
            vlt = par[self.idx_list[4]]
            Ymd, _ = hybrid_md04(self.y_ini,
                                 self.dt,
                                 self.hs,
                                 self.tp,
                                 self.dir,
                                 self.depth,
                                 self.doc,
                                 K,
                                 self.X0,
                                 self.Y0,
                                 self.phi,
                                 self.bctype,
                                 self.Bcoef,
                                 self.mb,
                                 self.D50,
                                 self.sl,
                                 self.Hberm,
                                 Y0,
                                 kacr,
                                 kero,
                                 vlt,
                                 self.dSdt,
                                 self.lst_f)

        return Ymd

    def _set_parameter_names(self):
        if self.cs_model == 'Yates et al. (2009)':
            self.par_names = []
            if self.switch_Kal == 0:
                for i, par in enumerate(['a', 'b', 'cacr', 'cero', 'K', 'vlt']):
                    self.par_names.append(f'{par}')
            else:
                for i, par in enumerate(['a', 'b', 'cacr', 'cero', 'K', 'vlt']):
                    trs = 0
                    for j in self.idx_list[i]:
                        if par == 'K':
                            self.par_names.append(f'{par}_trs_{trs+0.5}')
                            if j == self.ntrs + 1:
                                self.par_names.append(f'{par}_trs_{trs+0.5}')
                        else:
                            self.par_names.append(f'{par}_trs_{trs+1}')
                        trs += 1                        

            self.par_values[self.idx_list[0]] = -np.exp(self.par_values[self.idx_list[0]])
            self.par_values[self.idx_list[2]] = -np.exp(self.par_values[self.idx_list[2]])
            self.par_values[self.idx_list[3]] = -np.exp(self.par_values[self.idx_list[3]])
            if self.is_exp:
                self.par_values[self.idx_list[4]] = np.exp(self.par_values[self.idx_list[4]])

        elif self.cs_model == 'Davidson et al. (2013)':
            self.par_names = []
            if self.switch_Kal == 0:
                for i, par in enumerate(['phi', 'cp', 'cm', 'b', 'K', 'vlt']):
                    self.par_names.append(f'{par}')
            else:
                for i, par in enumerate(['phi', 'cp', 'cm', 'b', 'K', 'vlt']):
                    trs = 0
                    for j in self.idx_list[i]:
                        if par == 'K':
                            self.par_names.append(f'{par}_trs_{trs+0.5}')
                            if j == self.ntrs + 1:
                                self.par_names.append(f'{par}_trs_{trs+0.5}')
                        else:
                            self.par_names.append(f'{par}_trs_{trs+1}')
                        trs += 1
            self.par_values[self.idx_list[1]] = np.exp(self.par_values[self.idx_list[1]])
            self.par_values[self.idx_list[2]] = np.exp(self.par_values[self.idx_list[2]])
            if self.is_exp:
                self.par_values[self.idx_list[3]] = np.exp(self.par_values[self.idx_list[3]])

        elif self.cs_model == 'Miller and Dean (2004)':
            self.par_names = []
            for i, par in enumerate(['kacr', 'kero', 'Y0', 'K', 'vlt']):
                for j in range(len(self.idx_list[i])):
                    self.par_names.append(f'{par}_{j+1}')

            self.par_values[self.idx_list[0]] = np.exp(self.par_values[self.idx_list[0]])
            self.par_values[self.idx_list[1]] = np.exp(self.par_values[self.idx_list[1]])

            if self.is_exp:
                self.par_values[self.idx_list[3]] = np.exp(self.par_values[self.idx_list[3]])
            self.par_values[self.idx_list[4]] = np.exp(self.par_values[self.idx_list[4]])

