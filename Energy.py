# Calculate the amount of energy absorbed from the output of the external electric field and current density


########################################## Modules
import numpy as np
import pandas as pd

########################################## Numeric constant
pi       =np.pi            #円周率
Bohr     =0.5292           #[Å]
Hartree  =27.21            #[eV]
AtomCurr =0.6623           #[A]
AtomTime =0.02419          #[fs]
AtomField=51.422067        #[V/Å]
Plank    =4.13566777       #[eV・fs]
e        =1.602176634e-19  #電気素量[C]


########################################## Function

def calc_energy(directory,file,volume,Natom,unit):
    #Optical energy absorption
    #引数 {volume:セル体積[Å^3], Natom:セルに含まれる原子数, unit:'a.u.'or'A_eV_fs'}
    
    #データ読込
    head_s = 7  #ヘッダーの行数
    file_path = f'../../{directory}/{file}' #適宜変更
    data = pd.read_csv(file_path, sep='\s+', skiprows=head_s, header=None, dtype=np.float64).values
    
    #時間、外部電場、電流密度を取得
    #t:Time[fs], E_ext_x[V/Å], E_ext_y[V/Å], E_ext_z[V/Å], Jm_x[1/(fs*Å^2)], Jm_y[1/(fs*Å^2)], Jm_z[1/(fs*Å^2)]
    t, Etx, Ety, Etz, Jtx, Jty, Jtz = data[:, 0], data[:, 4], data[:, 5], data[:, 6], data[:, 13], data[:, 14], data[:, 15]

    #出力されるJに-e倍をすることで電流密度J(t)[C/(fs・Å^2)]になる。
    #吸収エネルギー量の計算（累積積分）W(t)=∫_0^t E(t)*J(t)dt
    #W(t)[C・V/Å^3]=[J/Å^3]を電気素量eで割って[eV/Å^3]に変換する。
    #以上の手続きは結局×-1をすることになる.
    dt = t[1] - t[0]
    Wtx = np.concatenate(([0], np.cumsum((Etx[1:] * Jtx[1:] + Etx[:-1] * Jtx[:-1]) / 2) * dt * (-1)))
    Wty = np.concatenate(([0], np.cumsum((Ety[1:] * Jty[1:] + Ety[:-1] * Jty[:-1]) / 2) * dt * (-1)))
    Wtz = np.concatenate(([0], np.cumsum((Etz[1:] * Jtz[1:] + Etz[:-1] * Jtz[:-1]) / 2) * dt * (-1)))
    
    #合計エネルギー
    Wt_volume = Wtx + Wty + Wtz           #[eV/Å^3]           
    Wt_atom = Wt_volume * volume / Natom  #[eV/atom]

    #原子単位系の場合
    if(unit == 'a.u.'):
        t *= AtomTime #[fs]
        Wt_volume = (Wtx + Wty + Wtz) * Hartree  #[eV/Å^3]
        Wt_atom = Wt_volume * volume / Natom     #[eV/atom]

    #最終的なエネルギー吸収量
    final_Wt_volume, final_Wt_atom = Wt_volume[-1], Wt_atom[-1]

    print(directory, f"{final_Wt_volume:.16f} [eV/Å^3], {final_Wt_atom:.16f} [eV/atom]")

    return t, Wt_atom, Wt_volume

print('done')
