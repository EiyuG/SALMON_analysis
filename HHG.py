########################################## Function

def calc_HHG(directory, file, wavelength, Nelctron, T, unit):
    #電流密度からHHGを計算
    #引数 {wavelength:波長[nm], Nelectron:系内の電子数, T:pulse duration [fs], unit:'a.u.' or 'A_eV_fs'}
    
    #データ読込
    head_s = 7  #ヘッダーの行数
    file_path = f'../../{directory}/{file}'
    data = pd.read_csv(file_path, sep='\s+', skiprows=head_s, header=None, dtype=np.float64).values
    
    #時間、外部電場、電流密度を取得
    #t:Time[fs], E_ext_x[V/Å], E_ext_y[V/Å], E_ext_z[V/Å], Jm_x[1/(fs*Å^2)], Jm_y[1/(fs*Å^2)], Jm_z[1/(fs*Å^2)]
    t, Jtx, Jty, Jtz = data[:, 0], -data[:, 13], -data[:, 14], -data[:, 15]
    
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
    Nh = np.argmax(t >= T)  #t[i]が初めてT以上になるインデックスを取得

    # フーリエ変換に用いる電流密度の取得とスムージング
    smfunc = 1  # 1: smoothing_func_1, 2: smoothing_func_2
    smoothing_function = smoothing_func_1 if smfunc == 1 else smoothing_func_2
    t_norm = t[:Nh] / T
    smoothing_values = smoothing_function(t_norm)
    Jhx, Jhy, Jhz = smoothing_values * Jtx[:Nh], smoothing_values * Jty[:Nh], smoothing_values * Jtz[:Nh]

    #高速離散フーリエ変換
    Jhx_fft=np.fft.fft(Jhx)
    Jhy_fft=np.fft.fft(Jhy)
    Jhz_fft=np.fft.fft(Jhz)
    dt=t[1]-t[0]

    #強度を計算
    Sx, Sy, Sz = abs(Jhx_fft)**2, abs(Jhy_fft)**2, abs(Jhz_fft)**2  #[1/Å^4]

    #基本周波数
    w0 = 2 * pi * c / wavelength  #[/fs]

    #サイクル数
    Ncycle = T * w0 / (2 * pi)

    #横軸(harmonic order)
    harmonic_order = np.arange(len(Jtx)) / Ncycle
    harmonic_order = harmonic_order[:Nh // 2]

    #強度を電子数で規格化
    factor = 1 / Nelctron
    Sx, Sy, Sz = Sx[:Nh // 2] * factor, Sy[:Nh // 2] * factor, Sz[:Nh // 2] * factor
    
    return t, harmonic_order, Jtx, Jty, Jtz, Sx, Sy, Sz

print('done')
