#%%% Procesamiento de templogs de CPA enfriados en LN2/ vapor de LN2
import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    
    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Transicion de fase
def detectar_TF_y_plot(t,T,T_central=0,delta_T=0.2,umbral_dTdt=0.15,min_puntos=5,plot=True,identif=None):
    """
    Detecta mesetas de transición de fase en una curva Temperatura vs Tiempo
    y opcionalmente genera un gráfico con la región identificada.

    La meseta se define como una región donde:
    - La temperatura se mantiene dentro de un intervalo alrededor de T_central
    - La derivada temporal |dT/dt| es menor que un umbral dado
    - Los puntos cumplen continuidad temporal (segmentos consecutivos)
    - La longitud del segmento supera un mínimo de puntos (min_puntos)

    Parámetros
    ----------
    t : array_like
        Tiempo [s]
    T : array_like
        Temperatura [°C]
    T_central : float, opcional
        Temperatura central de la transición (default: 0 °C)
    delta_T : float, opcional
        Tolerancia en temperatura (± delta_T) (default: 0.2 °C)
    umbral_dTdt : float, opcional
        Umbral máximo para |dT/dt| [°C/s] (default: 0.15 °C/s)
    min_puntos : int, opcional
        Número mínimo de puntos consecutivos para validar una meseta (default: 5)
    plot : bool, opcional
        Si True, genera la figura con los resultados (default: True)

    Retorna
    -------
    mesetas : list of dict
        Lista de mesetas detectadas. Cada elemento contiene:
        - "t_inicio" : tiempo inicial [s]
        - "t_fin"    : tiempo final [s]
        - "duracion" : duración de la meseta [s]
        - "T_media"  : temperatura media en la meseta [°C]

    fig : matplotlib.figure.Figure o None
        Figura generada (si plot=True)
    ax : matplotlib.axes.Axes o None
        Eje de temperatura
    ax2 : matplotlib.axes.Axes o None
        Eje de derivada dT/dt

    Notas
    -----
    - La derivada dT/dt se calcula mediante diferencias finitas (np.gradient).
    - La segmentación en bloques continuos evita identificar puntos aislados
      (ruido) como mesetas físicas.
    - El método es especialmente útil en experimentos térmicos donde la
      transición de fase se manifiesta como una meseta (ej: fusión del agua).
    - Para datos ruidosos se recomienda suavizar previamente la señal de temperatura."""
    dT_dt = np.gradient(T, t) # --- Derivada ---


    mask = ((T > (T_central - delta_T)) & (T < (T_central + delta_T)) & (np.abs(dT_dt) < umbral_dTdt))     # --- Filtro ---

    idx = np.where(mask)[0]

    if len(idx) == 0:
        return [], None, None, None

    segmentos = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)     # --- Segmentos continuos ---

    mesetas = []
    for seg in segmentos:
        if len(seg) >= min_puntos:
            t_ini = t[seg[0]]
            t_fin = t[seg[-1]]

            mesetas.append({"t_inicio": t_ini,"t_fin": t_fin,
                "duracion": t_fin - t_ini,"T_media": np.mean(T[seg])})

    if plot:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10,7),sharex=True, constrained_layout=True)

        ax.plot(t, T, '.-', label='Temperatura')
        ax2.plot(t, dT_dt, '.-', label='dT/dt')

        # Umbrales derivada
        ax2.axhline(umbral_dTdt, color='k', ls='--')
        ax2.axhline(-umbral_dTdt, color='k', ls='--')

        # --- Mesetas ---
        for i, m in enumerate(mesetas):
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            label = f'T. Fase ({m["duracion"]:.1f} s)' if i == 0 else None

            # Curva resaltada
            ax.plot(t[mask_m], T[mask_m], 'g-', lw=3, label=label)

            # Sombreado en ambos plots
            ax.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)
            ax2.axvspan(m["t_inicio"], m["t_fin"], color='g', alpha=0.2)

        # --- Labels ---
        ax.set_ylabel('T (°C)')
        ax2.set_ylabel('dT/dt (°C/s)')
        ax2.set_xlabel('t (s)')
        ax.set_title(identif+'\nTransición de fase S-L')

        for a in (ax, ax2):
            a.grid()
            a.legend()

        # --- Insets (primera meseta) ---
        if mesetas:
            m = mesetas[0]
            mask_m = (t >= m["t_inicio"]) & (t <= m["t_fin"])

            # -------- Inset en Temperatura --------
            axin = ax.inset_axes([0.5, 0.1, 0.45, 0.45])
            axin.plot(t, T, 'k-')
            axin.plot(t[mask_m], T[mask_m], 'g-', lw=2)

            axin.axhline(T_central - delta_T, ls='--', color='k')
            axin.axhline(T_central + delta_T, ls='--', color='k')

            axin.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)
            axin.set_ylim(T_central - 2*delta_T, T_central + 2*delta_T)

            axin.grid()
            ax.indicate_inset_zoom(axin)

            # -------- Inset en dT/dt --------
            ax2in = ax2.inset_axes([0.5, 0.1, 0.45, 0.45])
            ax2in.plot(t, dT_dt, 'k-')
            ax2in.plot(t[mask_m], dT_dt[mask_m], 'g-', lw=2)

            ax2in.axhline(umbral_dTdt, ls='--', color='k')
            ax2in.axhline(-umbral_dTdt, ls='--', color='k')

            ax2in.set_xlim(m["t_inicio"] - 5, m["t_fin"] + 5)

            # zoom vertical más ajustado a la derivada en la meseta
            dT_local = dT_dt[mask_m]
            margen = 0.1 * (np.max(np.abs(dT_local)) + 1e-6)
            ax2in.set_ylim(np.min(dT_local) - margen, np.max(dT_local) + margen)

            ax2in.grid()
            ax2.indicate_inset_zoom(ax2in)

        return mesetas, fig, ax, ax2

    return mesetas, None, None, None
#%% Levanto los archivos de templog
paths_152 = glob('CPA-400uL_FF-100uL/*152dA*templog*',recursive=True)
paths_152.sort()
paths_125 = glob('CPA-400uL_FF-100uL/*125dA*templog*',recursive=True)
paths_125.sort()
paths_100 = glob('CPA-400uL_FF-100uL/*100dA*templog*',recursive=True)
paths_100.sort()
paths_RT = glob('CPA-400uL_FF-100uL/*sincampo*templog*',recursive=True)



#%%
%matplotlib 

fig00, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=True,sharex=True
                                  )

_,t_RT,T_RT,_ = lector_templog(paths_RT[0])

for i,r in enumerate(paths_152):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_125):
    _,t,T, _ = lector_templog(r)
    ax2.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_100):
    _,t,T, _ = lector_templog(r)
    ax3.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax3.plot(t_RT,T_RT,'.-',label='RT')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='lower right')
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,)
ax3.set_xlabel('t (s)')
ax.set_title('58 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')
ax3.set_title('38 kA/m',loc='left')
plt.suptitle(f'Templogs\nCPA: 400 uL - NF@cit_13h: 100 uL')
plt.savefig('comparativa_templogs_152_125_100_RT_misma_escala.png',dpi=300)



#%%


fig00, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=True,sharex=False)

_,t_RT,T_RT,_ = lector_templog(paths_RT[0])

for i,r in enumerate(paths_152):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_125):
    _,t,T, _ = lector_templog(r)
    ax2.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_100):
    _,t,T, _ = lector_templog(r)
    ax3.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax3.plot(t_RT,T_RT,'.-',label='RT')

for a in ax,ax2,ax3:
    a.grid()
    a.legend(loc='lower right')
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,)
ax3.set_xlabel('t (s)')
ax.set_title('58 kA/m',loc='left')
ax2.set_title('47 kA/m',loc='left')
ax3.set_title('38 kA/m',loc='left')
plt.suptitle(f'Templogs\nCPA: 400 uL - NF@cit_13h: 100 uL')
plt.savefig('comparativa_templogs_152_125_100_RT_distinta_escala.png',dpi=300)















#%%
_,t_100,T_100,_ = lector_templog(path_1)
_,t_050,T_050,_ = lector_templog(path_2)
_,t_agua,T_agua,_ = lector_templog(path_3)


fig00, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en LN2 - expuesto a BT1',loc='left')
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)
ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()
#%% 4,5,6 CPA100/050/000 - 500 uL - BT1 Vapor  
path_4 = '260325_121904_CPA100_BT1_500uL_vapor.csv'
path_5 = '260325_123339_CPA050_BT1_500uL_vapor.csv'
path_6 = '260325_124612_agua_BT1_500uL_vapor.csv'

_,t_100,T_100,_ = lector_templog(path_4)
_,t_050,T_050,_ = lector_templog(path_5)
_,t_agua,T_agua,_ = lector_templog(path_6)

#%
fig01, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en vapor LN2 - expuesto a BT1',loc='left')
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)
ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)

plt.show()
#%% 7,8,9 CPA100/050/000 - 500 uL - BT1 repeticion 1,2,3 
   
path_7 = '260325_154359_CPA100_BT1_500uL.csv'
path_8 = '260325_155106_CPA050_BT1_500uL.csv'
path_9 = '260325_160313_agua_BT1_500uL.csv'

_,t_100,T_100,_ = lector_templog(path_7)
_,t_050,T_050,_ = lector_templog(path_8)
_,t_agua,T_agua,_ = lector_templog(path_9)

#%
fig03, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en LN2 - expuesto a BT1',loc='left')
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)
ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()

#%% 10,11,12 CPA100/050/000 - 500 uL - BT1 Vapor repeticion 4,5,6 
   
path_10 = '260325_161242_CPA100_BT1_500uL.csv'
path_11 = '260325_161814_CPA050_BT1_500uL.csv'
path_12 = '260325_162259_agua_BT1_500uL.csv'

_,t_100,T_100,_ = lector_templog(path_10)
_,t_050,T_050,_ = lector_templog(path_11)
_,t_agua,T_agua,_ = lector_templog(path_12)

#%
fig04, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% 50% y 0%  - enfriado en vapor LN2 - expuesto a BT1',loc='left')
ax.plot(t_agua,T_agua,'.-',label='Agua',alpha=0.8)
ax.plot(t_050,T_050,'.-',label='CPA050',alpha=0.8)
ax.plot(t_100,T_100,'.-',label='CPA100',alpha=0.8)

ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()

#%% Separo en LN2/ vapor LN2
en_LN2 = glob('*500uL.csv')
en_LN2.sort()
en_vapor = glob('*500uL_vapor.csv')
en_vapor.sort()
#%% ploteo en LN2 CPA 100%
fig100, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% - enfriado en LN2 - expuesto a BT1',loc='center',fontsize=14)

for i,p in enumerate(en_LN2):
    if 'CPA100' in p:
        _,t_100,T_100,_ = lector_templog(p)
        ax.plot(t_100,T_100,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,175)
# ax.set_ylim(-2,30)
plt.show()

# %% ploteo en LN2 CPA 50%
fig050, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 50% - enfriado en LN2 - expuesto a BT1',loc='center',fontsize=14)

for i,p in enumerate(en_LN2):
    if 'CPA050' in p:
        _,t_050,T_050,_ = lector_templog(p)
        ax.plot(t_050,T_050,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
# ax.set_ylim(-2,30)
plt.show()

#%% Ploteo agua en LN2 
fig000, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('Agua - enfriado en LN2 - expuesto a BT1',loc='center',fontsize=14)

for i,p in enumerate(en_LN2):
    if 'agua' in p:
        print(' -',p)
        _,t_agua,T_agua,_ = lector_templog(p)
        ax.plot(t_agua,T_agua,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
# ax.set_ylim(-2,30)
plt.show()

#%% calculo TFase en cada caso
# for i,p in enumerate(en_LN2):
#     if 'agua' in p:
#         _,t_agua,T_agua,_ = lector_templog(p)
#         meseta,_,_,_ =detectar_TF_y_plot(t_agua,T_agua,plot=True,identif=p[7:-4],delta_T=0.7,umbral_dTdt=0.3,min_puntos=5)
# %% Ahora ploteo en vapor LN2 CPA 100%

fig100_vap, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 100% - enfriado en vapor de LN2 - expuesto a BT1',loc='center',fontsize=14)

for i,p in enumerate(en_vapor):
    if 'CPA100' in p:
        print(' -',p)
        _,t_100,T_100,_ = lector_templog(p)
        ax.plot(t_100,T_100,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,700)
plt.show()
#%% vapor LN2 CPA 50%
fig050_vap, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('CPA 50% - enfriado en vapor de LN2 - expuesto a BT1',loc='center',fontsize=14)

print('ploteando: ')
for i,p in enumerate(en_vapor):
    if 'CPA050' in p:
        print(' -',p)
        _,t_100,T_100,_ = lector_templog(p)
        ax.plot(t_100,T_100,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()
#%% Vapor LN2 Agua 
fig000_vap, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

ax.set_title('Agua - enfriado en vapor de LN2 - expuesto a BT1',loc='center',fontsize=14)

for i,p in enumerate(en_vapor):
    if 'agua' in p:
        print(' -',p)
        _,t_agua,T_agua,_ = lector_templog(p)
        ax.plot(t_agua,T_agua,'.-',label=p[7:-4],alpha=0.8)
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')    
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.legend(loc='best',ncol=2)
ax.set_xlim(0,)
plt.show()   

# %% salvo figuras
figs= [fig100,fig050,fig000,fig100_vap,fig050_vap,fig000_vap]
names= ['CPA100','CPA050','agua','CPA100_vapor','CPA050_vapor','agua_vapor']

for i,f in enumerate(figs):
    f.savefig(f'{i}_'+names[i]+'.png',dpi=300)
#%% Comparativas

figLN2, (ax1,ax2,ax3) =plt.subplots(3,1,figsize=(12,10),constrained_layout=True,sharex=True)

ax1.set_title('CPA 100%',loc='left')
for i,p in enumerate(en_LN2):
    if 'CPA100' in p:
        print(' -',p)
        _,t_100,T_100,_ = lector_templog(p)
        ax1.plot(t_100,T_100,'.-',label=p[7:-4],alpha=0.8)

ax2.set_title('CPA 50%',loc='left')
for i,p in enumerate(en_LN2):
    if 'CPA050' in p:
        print(' -',p)
        _,t_050,T_050,_ = lector_templog(p)
        ax2.plot(t_050,T_050,'.-',label=p[7:-4],alpha=0.8)
        
ax3.set_title('Agua',loc='left')
for i,p in enumerate(en_LN2):
    if 'agua' in p:
        print(' -',p)
        _,t_agua,T_agua,_ = lector_templog(p)
        ax3.plot(t_agua,T_agua,'.-',label=p[7:-4],alpha=0.8)
for ax in [ax1,ax2,ax3]:
    ax.grid()
    ax.set_ylabel('T (°C)')
    ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    ax.legend(loc='best',ncol=2)

ax3.set_xlabel('t (s)')    
ax3.set_xlim(0,200)
plt.suptitle('CPA 100% 50% y 0%  - enfriado en LN2 - expuesto a BT1',fontsize=14)
plt.savefig('comparativas_100_050_000_en_LN2.png',dpi=300)
plt.show()   

#%%
figvapor, (ax1,ax2,ax3) =plt.subplots(3,1,figsize=(12,10),constrained_layout=True,sharex=True)

ax1.set_title('CPA 100%',loc='left')
for i,p in enumerate(en_vapor):
    if 'CPA100' in p:
        print(' -',p)
        _,t_100,T_100,_ = lector_templog(p)
        ax1.plot(t_100,T_100,'.-',label=p[7:-4],alpha=0.8)

ax2.set_title('CPA 50%',loc='left')
for i,p in enumerate(en_vapor):
    if 'CPA050' in p:
        print(' -',p)
        _,t_050,T_050,_ = lector_templog(p)
        ax2.plot(t_050,T_050,'.-',label=p[7:-4],alpha=0.8)
        
ax3.set_title('Agua',loc='left')
for i,p in enumerate(en_vapor):
    if 'agua' in p:
        print(' -',p)
        _,t_agua,T_agua,_ = lector_templog(p)
        ax3.plot(t_agua,T_agua,'.-',label=p[7:-4],alpha=0.8)
for ax in [ax1,ax2,ax3]:
    ax.grid()
    ax.set_ylabel('T (°C)')
    ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    ax.legend(loc='best',ncol=2)

ax3.set_xlabel('t (s)')    
ax3.set_xlim(0,700)
plt.suptitle('CPA 100% 50% y 0%  - enfriado en vapor de LN2 - expuesto a BT1',fontsize=14)
plt.savefig('comparativas_100_050_000_en_vapor_LN2.png',dpi=300)
plt.show()   
