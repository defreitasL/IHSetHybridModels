import numpy as np
from typing import Any, List
from IHSetUtils.CoastlineModel import CoastlineModel
from .hybrid_builder import hybrid_y09, hybrid_ShoreFor, hybrid_md04

class assimilate_Hybrid(CoastlineModel):
    """
    Hybrid one-line + cross-shore coupled model — parameter EnKF assimilation.

    • Supports cs_model in {'Yates et al. (2009)', 'Davidson et al. (2013)', 'Miller and Dean (2004)'}
    • Handles switch_Kal == 1 (global params) or 0 (per-transect params) using the
      same idx_list/nn logic as your calibration class.
    • model_step/model_step_batch return a VECTOR (len = n_trs): the last shoreline
      state within the current assimilation window, Ymd[-1, :].
    """

    def __init__(self, path):
        super().__init__(
            path=path,
            model_name='Oneline and cross-shore coupled',
            mode='assimilation',
            model_type='HY',
            model_key='Hybrid'
        )
        self.setup_forcing()
        self._set_upper_lowers()  # reuses your logic

    # -------------------------
    # Forcings & bookkeeping
    # -------------------------
    def setup_forcing(self):
        self.switch_Kal = self.cfg['switch_Kal']
        self.cs_model = self.cfg['cs_model']
        self.dSdt = self.cfg['dSdt']
        if self.cs_model != 'Yates et al. (2009)':
            self.D50 = self.cfg['D50']
        if self.cs_model == 'Miller and Dean (2004)':
            self.Hberm = self.cfg['Hberm']
        self.Yini = np.zeros_like(self.Obs_splited_[0,:])
        for i in range(self.ntrs):
            self.Yini[i] = np.nanmean(self.Obs_splited_[:, i])
        if self.switch_Kal == 1:
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
                                 range(4*self.ntrs+1, 5*self.ntrs+1)]
            elif self.cs_model == 'Miller and Dean (2004)':
                self.nn = (self.ntrs * 5) + 1
                self.idx_list = [range(0, self.ntrs),
                                 range(self.ntrs, 2*self.ntrs),
                                 range(2*self.ntrs, 3*self.ntrs),
                                 range(3*self.ntrs, 4*self.ntrs+1),
                                 range(4*self.ntrs+1, 5*self.ntrs+1)]
        if self.cs_model == 'Miller and Dean (2004)':
            self.sl_s = self.tide_s + self.surge_s
            self.sl = self.tide + self.surge

    # -------------------------
    # Bounds & init pop (reuse)
    # -------------------------
    def _set_upper_lowers(self):
        # Exactly your original method content (unchanged)
        # — I’m just reusing it here
        # BEGIN COPY (same as you posted) ------------------------------------
        if self.switch_Kal == 1:
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
                lowers = np.array([self.lb[0], np.log(self.lb[1]), np.log(self.lb[2])])
                uppers = np.array([self.ub[0], np.log(self.ub[1]), np.log(self.ub[2])])
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
        if self.switch_Kal == 0:
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
        # END COPY ------------------------------------------------------------

    def init_par(self, population_size: int):
        pop = np.zeros((population_size, self.nn))
        for i in range(self.nn):
            pop[:, i] = np.random.uniform(self.lowers[i], self.uppers[i], population_size)
        return pop, self.lowers, self.uppers

    # -------------------------
    # Helpers: map θ -> physical
    # -------------------------
    def _decode_params(self, par: np.ndarray):
        """
        Decode a single member 'par' (in transform space) into physical parameters
        according to cs_model and switch_Kal / idx_list.
        Returns a dict with keys among:
        a,b,cacr,cero,K,vlt,phi,cp,cm,b_md,Y0,kacr,kero  (subset depending on model)
        """
        idx = self.idx_list

        if self.cs_model == 'Yates et al. (2009)':
            a    = -np.exp(par[idx[0]])
            b    = par[idx[1]]
            cacr = -np.exp(par[idx[2]])
            cero = -np.exp(par[idx[3]])
            K    = par[idx[4]]
            vlt  = par[idx[5]]
            if self.is_exp:
                K = np.exp(K)
            return dict(a=a, b=b, cacr=cacr, cero=cero, K=K, vlt=vlt)

        if self.cs_model == 'Davidson et al. (2013)':
            phi = par[idx[0]]
            cp  = np.exp(par[idx[1]])
            cm  = np.exp(par[idx[2]])
            # bmd = par[idx[3]]
            K   = par[idx[3]]
            vlt = par[idx[4]]
            if self.is_exp:
                K = np.exp(K)
            return dict(phi=phi, cp=cp, cm=cm, b=bmd, K=K, vlt=vlt)

        # Miller & Dean (2004)
        kacr = np.exp(par[idx[0]])
        kero = np.exp(par[idx[1]])
        Y0   = par[idx[2]]
        K    = par[idx[3]]
        vlt  = par[idx[4]]
        if self.is_exp:
            K = np.exp(K)
        return dict(kacr=kacr, kero=kero, Y0=Y0, K=K, vlt=vlt)

    # -------------------------
    # One EnKF step (VECTOR obs)
    # -------------------------
    def model_step(self, par: np.ndarray, t_idx: int, context: Any | None = None):
        """
        Run between two consecutive observation indices and return the
        shoreline vector at the LAST time in that window: (n_trs,) — NO flattening.
        """
        pars = self._decode_params(par)
        i0, i1 = self.idx_obs_splited[t_idx-1], self.idx_obs_splited[t_idx]

        # starting shoreline vector for this window
        if context is None or ('y_old' not in context):
            y0 = self.Yini.copy()
        else:
            y0 = np.asarray(context['y_old'], dtype=float)

        if self.cs_model == 'Yates et al. (2009)':
            Ymd, _, _ = hybrid_y09(
                y0,
                self.dt_s[i0:i1],
                self.hs_s[i0:i1],
                self.tp_s[i0:i1],
                self.dir_s[i0:i1],
                self.depth,
                self.doc,
                pars['K'],
                self.X0, self.Y0, self.phi, self.bctype, self.Bcoef, self.mb, self.D50,
                pars['a'], pars['b'], pars['cacr'], pars['cero'],
                pars['vlt'],
                self.dSdt,
                self.lst_f
            )
        elif self.cs_model == 'Davidson et al. (2013)':
            Ymd, _ = hybrid_ShoreFor(
                y0,
                self.dt_s[i0:i1],
                self.hs_s[i0:i1],
                self.tp_s[i0:i1],
                self.dir_s[i0:i1],
                self.depth,
                self.doc,
                pars['K'],
                self.X0, self.Y0, self.phi, self.bctype, self.Bcoef, self.mb, self.D50,
                pars['phi'], pars['cp'], pars['cm'],
                pars['vlt'],
                self.dSdt,
                self.lst_f
            )
        else:  # Miller & Dean (2004)
            Ymd, _, _ = hybrid_md04(
                y0,
                self.dt_s[i0:i1],
                self.hs_s[i0:i1],
                self.tp_s[i0:i1],
                self.dir_s[i0:i1],
                self.depth,
                self.doc,
                pars['K'],
                self.X0, self.Y0, self.phi, self.bctype, self.Bcoef, self.mb, self.D50,
                self.sl_s[i0:i1],
                self.Hberm,
                pars['Y0'],
                pars['kacr'], pars['kero'],
                pars['vlt'],
                self.dSdt,
                self.lst_f
            )

        y_last = np.asarray(Ymd[-1, :], dtype=float)   # shape (n_trs,)
        return y_last, {'y_old': y_last}

    # -------------------------
    # Batched EnKF step (faster)
    # -------------------------
    def model_step_batch(self, pop: np.ndarray, t_idx: int, contexts: List[dict] | None):
        N = pop.shape[0]
        y_out   = np.empty((N, self.ntrs), dtype=float)
        new_ctx = [None] * N

        i0, i1 = self.idx_obs_splited[t_idx-1], self.idx_obs_splited[t_idx]

        for j in range(N):
            pars = self._decode_params(pop[j])
            y0 = self.Yini if (contexts is None or contexts[j] is None or 'y_old' not in contexts[j]) \
                 else np.asarray(contexts[j]['y_old'], dtype=float)

            if self.cs_model == 'Yates et al. (2009)':
                Ymd, _, _ = hybrid_y09(
                    y0, self.dt_s[i0:i1], self.hs_s[i0:i1], self.tp_s[i0:i1], self.dir_s[i0:i1],
                    self.depth, self.doc, pars['K'], self.X0, self.Y0, self.phi, self.bctype,
                    self.Bcoef, self.mb, self.D50, pars['a'], pars['b'], pars['cacr'], pars['cero'],
                    pars['vlt'], self.dSdt, self.lst_f
                )
            elif self.cs_model == 'Davidson et al. (2013)':
                Ymd, _ = hybrid_ShoreFor(
                    y0, self.dt_s[i0:i1], self.hs_s[i0:i1], self.tp_s[i0:i1], self.dir_s[i0:i1],
                    self.depth, self.doc, pars['K'], self.X0, self.Y0, self.phi, self.bctype,
                    self.Bcoef, self.mb, self.D50, pars['phi'], pars['cp'], pars['cm'],
                    pars['vlt'], self.dSdt, self.lst_f
                )
            else:
                Ymd, _, _ = hybrid_md04(
                    y0, self.dt_s[i0:i1], self.hs_s[i0:i1], self.tp_s[i0:i1], self.dir_s[i0:i1],
                    self.depth, self.doc, pars['K'], self.X0, self.Y0, self.phi, self.bctype,
                    self.Bcoef, self.mb, self.D50, self.sl_s[i0:i1], self.Hberm, pars['Y0'],
                    pars['kacr'], pars['kero'], pars['vlt'], self.dSdt, self.lst_f
                )

            y_last = np.asarray(Ymd[-1, :], dtype=float)  # (n_trs,)
            y_out[j, :] = y_last
            new_ctx[j]  = {'y_old': y_last}

        # EnKF expects (N, p) with p == n_trs
        return y_out, new_ctx

    # -------------------------
    # Full forward run (for plotting after EnKF)
    # -------------------------
    def run_model(self, par: np.ndarray) -> np.ndarray:
        if self.cs_model == 'Yates et al. (2009)':
            a = par[self.idx_list[0]]
            b = par[self.idx_list[1]]
            cacr = par[self.idx_list[2]]
            cero = par[self.idx_list[3]]
            K = par[self.idx_list[4]]
            vlt = par[self.idx_list[5]]

            Ymd, _, self.ylt = hybrid_y09(self.Yini,
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
            K = par[self.idx_list[3]]
            vlt = par[self.idx_list[4]]
            Ymd, _, self.ylt = hybrid_ShoreFor(self.Yini,
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
                                    vlt,
                                    self.dSdt,
                                    self.lst_f)
        elif self.cs_model == 'Miller and Dean (2004)':
            kacr = par[self.idx_list[0]]
            kero = par[self.idx_list[1]]
            Y0 = par[self.idx_list[2]]
            K = par[self.idx_list[3]]
            vlt = par[self.idx_list[4]]
            Ymd, _ = hybrid_md04(self.Yini,
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

    # -------------------------
    # Names for pretty output
    # -------------------------
    def _set_parameter_names(self):
        if self.cs_model == 'Yates et al. (2009)':
            self.par_names = []
            if self.switch_Kal == 1:
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
            if self.switch_Kal == 1:
                for i, par in enumerate(['phi', 'cp', 'cm', 'K', 'vlt']):
                    self.par_names.append(f'{par}')
            else:
                for i, par in enumerate(['phi', 'cp', 'cm', 'K', 'vlt']):
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
            self.par_values[self.idx_list[4]] = self.par_values[self.idx_list[4]]