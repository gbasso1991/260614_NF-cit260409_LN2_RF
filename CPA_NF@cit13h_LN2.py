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
#%% 400 CPA - 100 FF - NF@cit_13h diluido a la mitad
dir_1 =  '1_CPA-400uL_FF-100uL_diluida'
paths_152_1 = glob(dir_1+'/*152dA*templog*',recursive=True)
paths_152_1.sort()
paths_125_1 = glob(dir_1+'/*125dA*templog*',recursive=True)
paths_125_1.sort()
paths_100_1 = glob(dir_1+'/*100dA*templog*',recursive=True)
paths_100_1.sort()
paths_RT_1 = glob(dir_1+'/*sincampo*templog*',recursive=True)

fig00, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=True,sharex=True
                                  )
_,t_RT,T_RT,_ = lector_templog(paths_RT_1[0])

for i,r in enumerate(paths_152_1):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_125_1):
    _,t,T, _ = lector_templog(r)
    ax2.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_100_1):
    _,t,T, _ = lector_templog(r)
    ax3.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax3.plot(t_RT,T_RT,'.-',label='RT')

for a in ax,ax2,ax3:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,)

ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',loc='lower right',frameon=True,shadow=True)
ax2.legend(title='$f$= 300 kHz - $H_0$=47 kA/m',loc='lower right',frameon=True,shadow=True)
ax3.legend(title='$f$= 300 kHz - $H_0$=38 kA/m',loc='lower right',frameon=True,shadow=True)

ax3.set_xlabel('t (s)')
# ax.set_title('58 kA/m',loc='left')
# ax2.set_title('47 kA/m',loc='left')
# ax3.set_title('38 kA/m',loc='left')
plt.suptitle(f'CPA: 400 uL - NF@cit_13h: 100 uL')
plt.savefig('1_templogs_152_125_100_RT_400_100_misma_escala.png',dpi=300)

fig01, (ax,ax2,ax3) =plt.subplots(3,1,figsize=(12,8),constrained_layout=True,sharey=True,sharex=False)

_,t_RT,T_RT,_ = lector_templog(paths_RT_1[0])

for i,r in enumerate(paths_152_1):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_125_1):
    _,t,T, _ = lector_templog(r)
    ax2.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_100_1):
    _,t,T, _ = lector_templog(r)
    ax3.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax3.plot(t_RT,T_RT,'.-',label='RT')

for a in [ax,ax2,ax3]:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,)
ax3.set_xlabel('t (s)')

ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',loc='lower right',frameon=True,shadow=True)
ax2.legend(title='$f$= 300 kHz - $H_0$=47 kA/m',loc='lower right',frameon=True,shadow=True)
ax3.legend(title='$f$= 300 kHz - $H_0$=38 kA/m',loc='lower right',frameon=True,shadow=True)

# ax.set_title('58 kA/m',loc='left')
# ax2.set_title('47 kA/m',loc='left')
# ax3.set_title('38 kA/m',loc='left')
plt.suptitle(f'CPA: 400 uL - NF@cit_13h: 100 uL')
plt.savefig('1_templogs_152_125_100_RT_400_100_distinta_escala.png',dpi=300)

#%% 425 CPA - 75 FF - NF@cit_13h diluido a la mitad

dir_2 =  '2_CPA-425uL_FF-75uL_concentrada'
paths_152_2 = glob(dir_2+'/*152dA*templog*',recursive=True)
paths_152_2.sort()

fig10, ax =plt.subplots(figsize=(10,5),constrained_layout=True)

for i,r in enumerate(paths_152_2):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',frameon=True,shadow=True)
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_xlabel('t (s)')
plt.suptitle(f'CPA: 425 uL - NF@cit_13h: 75 uL')
plt.savefig('2_templogs_152dA_425-75.png',dpi=300)

#%% 450 CPA - 50 FF - NF@cit_13h diluido a la mitad
dir_3 =  '3_CPA-450uL_FF-50uL_concentrada'
paths_152_3 = glob(dir_3+'/*152dA*templog*',recursive=True)
paths_152_3.sort()

paths_125_3 = glob(dir_3+'/*125dA*templog*',recursive=True)
paths_125_3.sort()

fig30, (ax,ax2) =plt.subplots(2,1,figsize=(10,7),constrained_layout=True,sharex=True)

for i,r in enumerate(paths_152_3):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for i,r in enumerate(paths_125_3):
    _,t,T, _ = lector_templog(r)
    ax2.plot(t,T,'.-',label=r.split('_')[-1][:-4])

for a in ax,ax2:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,)

ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',frameon=True,shadow=True)
ax2.legend(title=' $f$= 300 kHz - $H_0$=47 kA/m',frameon=True,shadow=True)
ax2.set_xlabel('t (s)')

plt.suptitle(f'CPA: 450 uL - NF@cit_13h: 50 uL')
plt.savefig('3_templogs_152_125dA_450-50.png',dpi=300)
# %%
dir_4 = '4_CPA-475uL_FF-25uL_concentrada'
paths_152_4 = glob(dir_4+'/*152dA*templog*',recursive=True)
paths_152_4.sort()

fig,ax =plt.subplots(figsize=(10,5),constrained_layout=True)

for i,r in enumerate(paths_152_4):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',frameon=True,shadow=True)
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_xlabel('t (s)')
plt.suptitle(f'CPA: 475 uL - NF@cit_13h: 25 uL')
plt.savefig('4_templogs_152dA_475-25.png',dpi=300)

#%% 400 - 100 concentrada
dir_5 =  '5_CPA-400uL_FF-100uL_concentrada'
paths_152_5 = glob(dir_5+'/*152dA*templog*',recursive=True)
paths_152_5.sort()

fig,ax =plt.subplots(figsize=(10,5),constrained_layout=True)

for i,r in enumerate(paths_152_5):
    _,t,T, _ = lector_templog(r)
    ax.plot(t,T,'.-',label=r.split('_')[-1][:-4])

ax.grid()
ax.legend(title='$f$= 300 kHz - $H_0$=58 kA/m',frameon=True,shadow=True)
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,)
ax.set_xlabel('t (s)')
plt.suptitle(f'CPA: 400 uL - NF@cit_13h: 100 uL (concentrada)')
plt.savefig('5_templogs_152dA_400-100_concentrada.png',dpi=300)
# %% Ahora comparo los calentamientos a RT
path_RT_CPA500 = '260415_050229_CPA_puro.csv'
path_RT_CPA400_FF100 = '1_CPA-400uL_FF-100uL_diluida/260415_124355_sincampo_templog10.csv'
path_RT_CPA400_FF100_conc = '5_CPA-400uL_FF-100uL_concentrada/260416_103137_RT_templog04.csv'

_,t_RT_CPA500,T_RT_CPA500, _ = lector_templog(path_RT_CPA500)
_,t_RT_CPA400_FF100,T_RT_CPA400_FF100, _ = lector_templog(path_RT_CPA400_FF100)
_,t_RT_CPA400_FF100_conc,T_RT_CPA400_FF100_conc, _ = lector_templog(path_RT_CPA400_FF100_conc)

fig, ax =plt.subplots(figsize=(12,5),constrained_layout=True)
ax.plot(t_RT_CPA500,T_RT_CPA500,'.-',label='100% CPA')
ax.plot(t_RT_CPA400_FF100,T_RT_CPA400_FF100,'.-',label='80% CPA - 20% FF (diluida)')
ax.plot(t_RT_CPA400_FF100_conc,T_RT_CPA400_FF100_conc,'.-',label='80% CPA - 20% FF (concentrada)')
ax.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
ax.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
ax.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
ax.grid()
ax.legend(ncol=2,frameon=True,shadow=True)
ax.set_ylabel('T (ºC)')
ax.set_xlim(0,1000)
ax.set_xlabel('t (s)')
plt.suptitle('comparacion calentamientos a RT')
plt.savefig('comparacion_calentamientos_RT_CPA500_400-100_diluida_400-100_concentrada.png',dpi=300)
# %% Comparo lo expuesto a 152dA

paths_400_100 = glob(dir_5+'/*152dA*templog*',recursive=True)
paths_425_75  = glob(dir_2+'/*152dA*templog*',recursive=True)
paths_450_50  = glob(dir_3+'/*152dA*templog*',recursive=True)
paths_475_25  = glob(dir_4+'/*152dA*templog*',recursive=True)


fig, axs =plt.subplots(3,1,figsize=(12,9),constrained_layout=True,sharex=True)

for j in range(3):
    _,t,T, _ = lector_templog(paths_400_100[j])
    axs[j].plot(t,T,'.-',label='CPA 400 uL - FF 100 uL')


    _,t,T, _ = lector_templog(paths_425_75[j])
    axs[j].plot(t,T,'.-',label='CPA 425 uL - FF 75 uL')

    
    _,t,T, _ = lector_templog(paths_450_50[j])
    axs[j].plot(t,T,'.-',label='CPA 450 uL - FF 50 uL')

    _,t,T, _ = lector_templog(paths_475_25[j])
    axs[j].plot(t,T,'.-',label='CPA 475 uL - FF 25 uL')


for a in axs:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,250)
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(ncol=2,frameon=True,shadow=True)
ax3.set_xlabel('t (s)')
plt.suptitle('Comparación de calentamientos a $H_0$ = 58 kA/m (152 dA) - $f$ = 300 kHz')
plt.savefig('comparacion_calentamientos_152dA_400-100_425-75_450-50_475-25.png',dpi=300)
# %% Idem pero cambio labels

fig2, axs =plt.subplots(3,1,figsize=(12,9),constrained_layout=True,sharex=True)

for j in range(3):
    _,t,T, _ = lector_templog(paths_400_100[j])
    axs[j].plot(t,T,'.-',label='80% CPA - 20% FF')


    _,t,T, _ = lector_templog(paths_425_75[j])
    axs[j].plot(t,T,'.-',label='85% CPA - 15% FF')

    
    _,t,T, _ = lector_templog(paths_450_50[j])
    axs[j].plot(t,T,'.-',label='90% CPA - 10% FF')

    _,t,T, _ = lector_templog(paths_475_25[j])
    axs[j].plot(t,T,'.-',label='95% CPA -  5% FF')


for a in axs:
    a.grid()
    a.set_ylabel('T (ºC)')
    a.set_xlim(0,250)
    a.axhline(y=0,c='k',lw=0.8,label='T = 0°C')
    a.axhline(-43,c='k',ls='--',lw=0.8,label='T$_m$ = -43°C')
    a.axhline(-121,c='k',ls='-.',lw=0.8,label='T$_g$ = -121°C')
    a.legend(ncol=2,frameon=True,shadow=True)
ax3.set_xlabel('t (s)')
plt.suptitle('Comparación de calentamientos a $H_0$ = 58 kA/m (152 dA) - $f$ = 300 kHz')
plt.savefig('comparacion_calentamientos_152dA_400-100_425-75_450-50_475-25_bis.png',dpi=300)
# %%