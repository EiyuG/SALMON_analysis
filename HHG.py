########################################## Modules
import numpy as np
import pandas as pd


########################################## Numeric constant
pi        = np.pi            #円周率
Bohr      = 0.5292           #[Å]
Hartree   = 27.21            #[eV]
AtomCurr  = 0.6623           #[A]
AtomTime  = 0.02419          #[fs]
AtomField = 51.422067        #[V/Å]
Plank     = 4.13566777       #[eV・fs]
e         = 1.602176634e-19  #電気素量[C]
c         = 299.792458       #光速 [nm/fs]
epsilon0  = 8.8542e-14       #真空の誘電率[F/cm]=[C/(V・cm)]


########################################## Function

def calc_HHG(directory, file, wavelength, Nelctron, T, unit):
    #外部電場,レーザー強度の概形,電流密度を計算
    #引数 {wavelength:波長[nm], Nelectron:系内の電子数, T:pulse duration [fs], unit:'a.u.' or 'A_eV_fs'}
    
    #データ読込
    head_s = 7  #ヘッダーの行数
    file_path = f'../../{directory}/{file}'
    data = pd.read_csv(file_path, sep='\s+', skiprows=head_s, header=None, dtype=np.float64).values
    
    #時間、外部電場、電流密度を取得
    #t:Time[fs], E_ext_x[V/Å], E_ext_y[V/Å], E_ext_z[V/Å], Jm_x[1/(fs*Å^2)], Jm_y[1/(fs*Å^2)], Jm_z[1/(fs*Å^2)]
    t, Jtx, Jty, Jtz = data[:, 0], data[:, 13], data[:, 14], data[:, 15]
    Jtx *= -1
    Jty *= -1
    Jtz *= -1
    
    #原子単位系からの単位変換
    if(unit == 'a.u.'):
        t *= AtomTime     #[fs]
        conversion_factor_J_au = -1 / (AtomTime * Bohr**2)   #[1/(fs*Å^2)]
        Jtx *= conversion_factor_J_au
        Jty *= conversion_factor_J_au
        Jtz *= conversion_factor_J_au

    #smoothing function
    def smoothing_func_1(x):
        fx = 1 - 3 * x**2 + 2 * x**3
        return fx

    def smoothing_func_2(x):
        a=pi * (x - 1/2)
        fx=(np.cos(a))**4
        return fx

    #パルスdurationを抽出
    Nh = 0
    for i in range(len(Jtx) - 1):
        judge = (t[i] - T) * (t[i+1] - T)
        if(judge <= 0): 
            Nh=i+1
            break

    #フーリエ変換に用いる電流密度を取得。
    #その際smoothing functionをかける
    smfunc = 1  #1:smoothing_func_1, 2:smoothing_func_2
    Jhx=np.zeros(Nh,dtype='float64')   
    Jhy=np.zeros(Nh,dtype='float64')
    Jhz=np.zeros(Nh,dtype='float64')   
    for i in range(Nh):
        if(smfunc == 1):
            Jhx[i] = smoothing_func_1(t[i] / T) * Jtx[i]
            Jhy[i] = smoothing_func_1(t[i] / T) * Jty[i]
            Jhz[i] = smoothing_func_1(t[i] / T) * Jtz[i]
        if(smfunc == 2):
            Jhx[i] = smoothing_func_2(t[i] / T) * Jtx[i]
            Jhy[i] = smoothing_func_2(t[i] / T) * Jty[i]
            Jhz[i] = smoothing_func_2(t[i] / T) * Jtz[i]

    #高速離散フーリエ変換
    Jhx_fft=np.fft.fft(Jhx)
    Jhy_fft=np.fft.fft(Jhy)
    Jhz_fft=np.fft.fft(Jhz)
    dt=t[1]-t[0]

    #強度を計算
    Sx = abs(Jhx_fft)**2  #[1/Å^4]
    Sy = abs(Jhy_fft)**2
    Sz = abs(Jhz_fft)**2

    #基本周波数
    w0 = 2 * pi * c / wavelength  #[/fs]

    #サイクル数
    Ncycle = T * w0 / (2 * pi)

    #横軸(harmonic order)
    harmonic_order = np.arange(len(Jtx)) / Ncycle
    harmonic_order = harmonic_order[:Nh//2]

    #強度を電子数で規格化
    for i in range(Nh):
        factor = 1 / Nelctron
        Sx[i] *= factor
        Sy[i] *= factor
        Sz[i] *= factor
    Sx = Sx[:Nh//2]
    Sy = Sy[:Nh//2]
    Sz = Sz[:Nh//2]

    return t, harmonic_order, Jtx, Jty, Jtz, Sx, Sy, Sz
