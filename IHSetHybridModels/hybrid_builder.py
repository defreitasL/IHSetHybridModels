import numpy as np
from numba import njit
# from datetime import datetime
from IHSetUtils.libjit.geometry import nauticalDir2cartesianDir, abs_pos, shore_angle
from IHSetUtils.libjit.waves import BreakingPropagation
from IHSetUtils.libjit.morfology import wast, BruunRule_2, wMOORE
import math

@njit(cache=True)
def extrapolate_baseline(X0, Y0, dx):
    """
    Recebe:
      X0, Y0 : arrays (n1,) da linha base original
      dx     : array (n1,) dos comprimentos de segmento
    Devolve:
      X0_p, Y0_p : arrays (n1+2,) com um ponto fantasma em cada extremidade,
                   extrapolado ao longo da linha base.
    """
    n1 = X0.shape[0]
    X0_p = np.empty(n1+1, dtype=np.float64)
    Y0_p = np.empty(n1+1, dtype=np.float64)

    # 1) Primeiro segmento
    dx0 = X0[1] - X0[0]
    dy0 = Y0[1] - Y0[0]
    L0  = math.hypot(dx0, dy0)
    tx0 = dx0 / L0
    ty0 = dy0 / L0
    # ponto fantasma “antes” de X0[0]
    X0_p[0] = X0[0] - tx0 * dx[0]/2
    Y0_p[0] = Y0[0] - ty0 * dx[0]/2

    # 2) interior = segmentos intermediários
    for i in range(n1-1):
        dxi = X0[i+1] - X0[i]
        dyi = Y0[i+1] - Y0[i]
        Li  = math.hypot(dxi, dyi)
        txi = dxi / Li
        tyi = dyi / Li
        X0_p[i+1] = X0[i] + txi * dx[i]/2 
        Y0_p[i+1] = Y0[i] + tyi * dx[i]/2

    # 3) Último segmento
    dxE = X0[-1] - X0[-2]
    dyE = Y0[-1] - Y0[-2]
    LE  = math.hypot(dxE, dyE)
    txE = dxE / LE
    tyE = dyE / LE
    # ponto fantasma “após” X0[-1]
    X0_p[-1] = X0[-1] + txE * dx[-1]/2
    Y0_p[-1] = Y0[-1] + tyE * dx[-1]/2

    return X0_p, Y0_p


@njit(fastmath=True, cache=True)
def _compute_normals(X, Y, phi):
    """
    Devolve vetor de ângulos **normais** (perpendiculares) à costa, em graus.

    Para cada segmento entre os nós i‑i+1 calcula-se duas normais (θ±90°) e escolhe-se
    aquela cuja diferença angular circular em relação a phi[i] seja mínima.

    Entrada:
        X, Y : coordenadas dos nós (len = N)
        phi  : orientação local fornecida pelo usuário (len = N)

    Saída:
        alfas (len = N‑1) – ângulo normal para cada segmento
    """
    n = X.shape[0] - 1
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        # orientação da costa (°)
        theta = np.arctan2(Y[i+1] - Y[i], X[i+1] - X[i]) * 180.0 / np.pi
        # duas normais candidatas
        n1 = (theta + 90.0) % 360.0
        n2 = (theta - 90.0) % 360.0
        # diferença angular circular mínima
        # δ = (a - b + 180) % 360 - 180 ∈ (−180, 180]
        d1 = (n1 - phi[i] + 180.0) % 360.0 - 180.0
        d2 = (n2 - phi[i] + 180.0) % 360.0 - 180.0
        # escolhe a normal de menor desvio absoluto
        out[i] = n1 if abs(d1) <= abs(d2) else n2

    return out

@njit(fastmath=True, cache=True)
def one_step_ls(X0: np.ndarray, Y0: np.ndarray, phi: float, phi_rad: float,
                hs: np.ndarray, tp: np.ndarray, dire: np.ndarray,
                depth: np.ndarray, doc: np.ndarray, dt: float,
                bctype: np.ndarray, Bcoef: float, mb: float,
                D50: float, yold: np.ndarray,
                lstf: callable, alfas: np.ndarray, Ylt: np.ndarray,
                n2: int, kal: np.ndarray) -> tuple:
    """
    Propagate waves and compute sediment transport for one time step.
    X0, Y0:   baseline coordinates
    phi:      orientation of the baseline in nautical degrees
    phi_rad:  orientation of the baseline in radians
    hs:       significant wave height (m)
    tp:       wave period (s)
    dire:     wave direction (°)
    depth:    water depth (m)
    doc:      depth of closure (m)
    dt:       time step (s)
    bctype:   boundary condition type (0 = no flux, 1 = no gradient)
    Bcoef:    breaking coefficient
    mb:       beach slope (tanB)
    D50:      mean grain size (m)
    yold:     previous shoreline position
    t:        current time step index
    lstf:     function to compute longshore sediment transport
    alfas:    array to store normals (buffer)
    Ylt:      array to store longshore transport (buffer)
    n2:       number of segments (length of tp, hs, dire)
    """
    # compute alfas once per time step
    XN, YN  = abs_pos(X0, Y0, phi_rad, yold)
    normals = _compute_normals(XN, YN, phi)
    dx      = ((XN[1:] - XN[:-1])**2 + (YN[1:] - YN[:-1])**2)**0.5

    alfas[1:-1] = normals
    alfas[0]    = normals[0]
    alfas[-1]   = normals[-1]
    
    # propagate waves and compute transport
    hb, dirb, depthb = BreakingPropagation(hs, tp, dire, depth, alfas, Bcoef)
    
    q_now, _         = lstf(hb, tp, dirb, depthb, alfas, kal, mb, D50)

    # apply boundary conditions
    if bctype[0]  == 0:
        q_now[0]  = 0.0
    else:
        q_now[0]  = q_now[1]
    if bctype[1]  == 0:
        q_now[-1] = 0.0
    else:
        q_now[-1] = q_now[-2]

    # diffusion midpoints
    dc = 0.5 * (doc[1:] + doc[:-1])

    inv_i = dt / dc[0]  # inverse of dc[1] for first element
    Ylt[0] = - inv_i * (q_now[1] - q_now[0]) / dx[0] 

    inv_i = dt / dc[-1]  # inverse of dc[-1] for last element
    Ylt[-1] = - inv_i * (q_now[-1] - q_now[-2]) / dx[-1]

    for i in range(1,n2-2):
        inv_i   = dt / dc[i]  # inverse of dx[i+1] for current element
        Ylt[i] = - inv_i * (q_now[i+1] - q_now[i]) / dx[i-1]

    return Ylt, q_now, hb, depthb

@njit(fastmath=True, cache=True)
def hybrid_y09(yi, dt,  hs, tp, dire, depth, doc, kal,
               X0, Y0, phi, bctype, Bcoef, mb, D50, 
               a_y09, b_y09, cacr, cero, vlt, dSdt, lstf):

    n1 = len(X0)
    mt, n2 = tp.shape

    # preallocate output and buffers
    ysol  = np.zeros((mt, n1))
    q     = np.zeros((mt, n2))
    alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
    Ylt   = np.empty(n1, dtype=np.float64)      # buffer for longshore transport
    
    ysol[0, :] = yi

    phi_rad = np.empty(n1, dtype=np.float64)

    for i in range(n1):
        phi_rad[i] = phi[i] * np.pi / 180.0
    
    for t in range(1, mt):

        # longshore step
        Ylt, q_now, hb, _ = one_step_ls(X0, Y0, phi[t-1], phi_rad[t-1],
                                        hs[t-1,:], tp[t-1,:], dire[t-1,:],
                                        depth, doc[t-1,:], dt[t-1],
                                        bctype, Bcoef, mb, D50, ysol[t-1,:],
                                        lstf, alfas, Ylt, n2, kal)
        
        hb_ = (hb[1:-1] + hb[:-2]) / 2.0  # midpoints for yates09
        
        # Yates 2009 model for each segment
        Yst = yates09_onestep(hb_**2, dt[t-1], a_y09, b_y09, cacr, cero, ysol[t-1,:])
        
        Yvlt = dt[t-1] * vlt[t-1]

        Ybru = dt[t-1] * BruunRule_2(1, mb, dSdt)

        ysol[t,:]  = ysol[t-1,:] + Ylt + Yst + Yvlt + Ybru

        q[t,:]     = q_now  

    return ysol, q


@njit(cache=True, fastmath=True)
def yates09_onestep(E, dt, a, b, cacr, cero, y_old):
    """
    Yates et al. 2009 model for one step
    """
    n = E.shape[0]
    
    Seq = (E - b) / a
    E_eq = a * y_old + b
    deltaE = E - E_eq
    sE = math.sqrt(E)

    delta_y = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if y_old < Seq[i]:
            delta_y[i] = dt * cacr * sE[i] * deltaE[i]
        else:
            delta_y[i] = dt * cero * sE[i] * deltaE[i]
    
    return delta_y


@njit(fastmath=True, cache=True)
def hybrid_md04(yi, dt,  hs, tp, dire, depth, doc, kal,
               X0, Y0, phi, bctype, Bcoef, mb, D50, 
               sl, Hberm, DY0, kero, kacr,
               vlt, dSdt, lstf):


    n1 = len(X0)
    mt, n2 = tp.shape

    # preallocate output and buffers
    ysol  = np.zeros((mt, n1))
    q     = np.zeros((mt, n2))
    alfas = np.empty(n2, dtype=np.float64)      # reuse buffer for alfas
    Ylt   = np.empty(n1, dtype=np.float64)      # buffer for longshore transport
    Yst  = np.empty(n1, dtype=np.float64)       # buffer for millerdean transport
    dY0 = np.zeros((mt, n1))                    # buffer for millerdean delta_y0
    Y0md = np.zeros(n1, dtype=np.float64) + DY0 # initial baseline for millerdean

    ysol[0, :] = yi

    phi_rad = np.empty(n1, dtype=np.float64)

    for i in range(n1):
        phi_rad[i] = phi[i] * np.pi / 180.0
    
    for t in range(1, mt):

        # longshore step
        Ylt, q_now, hb, depthb = one_step_ls(X0, Y0, phi[t-1], phi_rad[t-1],
                                             hs[t-1,:], tp[t-1,:], dire[t-1,:],
                                             depth, doc[t-1,:], dt[t-1],
                                             bctype, Bcoef, mb, D50, ysol[t-1,:],
                                             lstf, alfas, Ylt, n2, kal)
        
        hb_ = (hb[1:-1] + hb[:-2]) / 2.0  # midpoints for millerdean
        depthb_ = (depthb[1:-1] + depthb[:-2]) / 2.0  # midpoints for millerdean

        wast_ = wast(hb_, D50)
        
        # Yates 2009 model for each segment
        Yst, Y0md = millerdean2004_onestep(hb_, depthb_, sl[t-1, :], wast_,
                                          dt[t-1], Hberm, Y0md, kero, kacr,
                                          ysol[t-1,:], Yst, dY0[t-1,:])
            
        Yvlt      = dt[t-1] * vlt[t-1]

        Ybru      = dt[t-1] * BruunRule_2(1, mb, dSdt)

        ysol[t,:] = ysol[t-1,:] + Ylt + Yst + Yvlt + Ybru

        dY0[t,:]  = Ylt + Yvlt + Ybru

        q[t,:]    = q_now  

    return ysol, q


@njit(cache=True, fastmath=True)
def millerdean2004_onestep(Hb, depthb, sl, wast, dt, Hberm, DY0, kero, kacr, yold, delta_y, dY0):
    """
    Miller and Dean 2004 model for one step
    Hb:      wave height (m)
    depthb:  water depth (m)
    sl:      sea level (m)
    wast:    width of active surf zone (m)
    dt:      time step (s)
    Hberm:   berm height (m)
    DY0:      bseline (m)
    kero:    erosion coefficient
    kacr:    accretion coefficient
    yold:    previous shoreline position
    Y:       output array to store results
    """
    n = Hb.shape[0]
    wl = 0.106 * Hb + sl
    DY = DY0 + dY0 

    yeq = DY - wast * wl / (Hberm + depthb)


    for i in range(n):
        if yold < yeq[i]:
            delta_y[i] = kacr * dt * (yeq[i] - yold)
        else:
            delta_y[i] = kero * dt * (yeq[i] - yold)

    return delta_y, DY


@njit(fastmath=True, cache=True)
def hybrid_ShoreFor(yi, dt,  hs, tp, dire, depth, doc, kal,
                    X0, Y0, phi, bctype, Bcoef, mb, D50,
                    phi_sf, cp, cm, vlt, dSdt, lstf):

    n1 = len(X0)
    mt, n2 = tp.shape

    # preallocate output and buffers
    ysol        = np.zeros((mt, n1))
    q           = np.zeros((mt, n2))
    alfas       = np.empty(n2, dtype=np.float64)       # reuse buffer for alfas
    Ylt         = np.empty(n1, dtype=np.float64)       # buffer for longshore transport
    Omega_eq    = np.zeros((mt, n1), dtype=np.float64) # buffer for equilibrium dimensionless fall velocity
    ysol[0, :] = yi

    phi_rad = np.empty(n1, dtype=np.float64)

    for i in range(n1):
        phi_rad[i] = phi[i] * np.pi / 180.0

    # compute ShoreFor constants
    dt_ = 0.0
    for j in range(dt.shape[0]):
        dt_ += dt[j]
    dt_ /= dt.shape[0]

    tau = phi_sf * 24.0
    alpha = math.exp(-math.log(10.0) * dt_ / tau)
    
    diff_cm_cp = cm - cp

    ws = wMOORE(D50)

    tp_ = (tp[t-1, 1:-1] + tp[t-1, :-2]) / 2.0
    hb_ = (hs[t-1, 1:-1] + hs[t-1, :-2]) / 2.0

    Omega_eq[0, :] = hb_ / (ws * tp_[0, :])  # initial equilibrium fall velocity
    
    for t in range(1, mt):

        # longshore step
        Ylt, q_now, hb, _ = one_step_ls(X0, Y0, phi[t-1], phi_rad[t-1],
                                             hs[t-1,:], tp[t-1,:], dire[t-1,:],
                                             depth, doc[t-1,:], dt[t-1],
                                             bctype, Bcoef, mb, D50, ysol[t-1,:],
                                             lstf, alfas, Ylt, n2, kal)
        
        hb_ = (hb[1:-1] + hb[:-2]) / 2.0  # midpoints for ShoreFor
        tp_ = (tp[t-1, 1:-1] + tp[t-1, :-2]) / 2.0  # midpoints for ShoreFor

        omega = hb_ / (ws * tp_)  # dimensionless fall velocity
        P   = hb_ ** 2 * tp_  # wave power
        
        # ShoreFor model for each segment
        Yst, Omega_eq[t, :] = ShoreFor_onestep(P, Omega_eq[t-1, :], omega, alpha,
                                  diff_cm_cp, cp, dt[t-1])
        
        Yvlt = dt[t-1] * vlt[t-1]

        Ybru = dt[t-1] * BruunRule_2(1, mb, dSdt)

        ysol[t,:]  = ysol[t-1,:] + Ylt + Yst + Yvlt + Ybru

        q[t,:]     = q_now  

    return ysol, q

@njit(cache=True, fastmath=True)
def ShoreFor_onestep(P, Omega_eq_old, omega, alpha, diff_cm_cp, cp, dt):
    """
    ShoreFor model for one step
    P:       wave power (W/m)
    Omega_eq_old: previous equilibrium dimensionless fall velocity
    omega:   current dimensionless fall velocity
    alpha:   decay factor
    diff_cm_cp: difference between cm and cp
    cp:      accretion rate
    dt:      time step (s)
    """
    
    n = P.shape[0]
    S = np.zeros(n, dtype=np.float64)
    OmegaEQ = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        OmegaEQ[i] = alpha * Omega_eq_old + (1.0 - alpha) * omega[i]
        sP = math.sqrt(P[i])
        F = sP * (OmegaEQ[i] - omega[i])
        cond_neg = 1.0 if F < 0.0 else 0.0
        inc = F * (diff_cm_cp * cond_neg + cp)
        S[i] = dt * inc

    return S, OmegaEQ
