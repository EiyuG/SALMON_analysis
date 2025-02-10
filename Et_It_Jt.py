# Calculate the external electric field, laser intensity, and current density


########################################## Modules
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MultipleLocator


########################################## Numeric constant
pi        = np.pi            #円周率
Bohr      = 0.5292           #[Å]
Hartree   = 27.21            #[eV]
AtomCurr  = 0.6623           #[A]
AtomTime  = 0.02419          #[fs]
AtomField = 51.422067        #[V/Å]
Plank     = 4.13566777       #[eV・fs]
e         = 1.602176634e-19  #電気素量[C]
c         = 2.9979e10        #光速[cm/s]
epsilon0  = 8.8542e-14       #真空の誘電率[F/cm]=[C/(V・cm)]


########################################## Function
def calc_Et_It_Jt(directory, file, unit):
    #外部電場,レーザー強度の概形,電流密度を計算
    #引数 {unit:'a.u.'or'A_eV_fs'}
    
    #データ読込
    head_s = 7  #ヘッダーの行数
    file_path = f'../../{directory}/{file}'
    data = pd.read_csv(file_path, sep='\s+', skiprows=head_s, header=None, dtype=np.float64).values
    
    #時間、外部電場、電流密度を取得
    #t:Time[fs], E_ext_x[V/Å], E_ext_y[V/Å], E_ext_z[V/Å], Jm_x[1/(fs*Å^2)], Jm_y[1/(fs*Å^2)], Jm_z[1/(fs*Å^2)]
    t, Etx, Ety, Etz, Jtx, Jty, Jtz = data[:, 0], data[:, 4], data[:, 5], data[:, 6], data[:, 13], data[:, 14], data[:, 15]

    #電流密度
    conversion_factor_J = -e * 1e15  # [C/(s*Å^2)] = [A/Å^2]
    Jtx *= conversion_factor_J
    Jty *= conversion_factor_J
    Jtz *= conversion_factor_J

    #原子単位系からの単位変換
    if(unit == 'a.u.'):
        t *= AtomTime     #[fs]
        Etx *= AtomField  #[V/Å]
        Ety *= AtomField 
        Etz *= AtomField
        conversion_factor_J_au = -e * 1e15 / (AtomTime * Bohr**2)   #[A/Å^2] #ここ見直す
        Jtx *= conversion_factor_J_au
        Jty *= conversion_factor_J_au
        Jtz *= conversion_factor_J_au

    #強度(電場の2乗の包絡線)を計算
    Itx, Ity, Itz = Etx**2, Ety**2, Etz**2  #[V^2/Å^2]
    order=10 #包絡線の滑らかさ
    for I, max_ids in zip([Itx, Ity, Itz], [argrelextrema(It, np.greater, order=order)[0] for It in [Itx, Ity, Itz]]):
        if max_ids.size > 0:
            I[:] = CubicSpline(t[max_ids], I[max_ids])(t)  # スプライン補間

    #強度に関する単位変換(I=cε0E^2/2)
    conversion_factor_I = c * epsilon0 * 1e16 / 2
    Itx *= conversion_factor_I #[W/cm^2]
    Ity *= conversion_factor_I
    Itz *= conversion_factor_I

    return t, Etx, Ety, Etz, Itx, Ity, Itz, Jtx, Jty, Jtz
