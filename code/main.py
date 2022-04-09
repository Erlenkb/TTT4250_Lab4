import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as tbl

###### GLOBAL ###########################################


third_octave = [100, 125, 160, 200, 250, 315, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]

x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000]#[100, 125, 160, 200, 250, 315, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
x_ticks_third_octave_Bands = ["100", "200", "500", "1000", "2000", "5000"]#['100', '125', '160', '200', '250', '315', '400', '500', '640', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000']

third_octave_entire_range = [50, 63, 80,100, 125, 160, 200, 250, 315, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000]
x_ticks_entire_range = [63, 125, 250, 500, 1000, 2000, 4000]
x_ticks_entire_range_label = ["63", "125", "250", "500", "1000", "2000", "4000"]



Background = [10, 13]
Source_room_S1 = [0,2,4,6,8]
Source_room_S2 = [1,3,5,7,9]
Receiver_room_S1 = [12,15,17,19,21]
Receiver_room_S2 = [11,14,16,18,20]

#### Font details #######################################
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
#########################################################

room_size_source = [8.501, 6.042, 5.174]
room_size_receiver = [4.11, 5.58, 4.58]

######## Define what to plot ############################



#########################################################

def _calculate_unfavorable(array):
    steps = [3,3,3,3,3,3,1,1,1,1,1,0,0,0,0,0,0]
    start = 33
    ref_curve = [33]
    for i in range(len(steps)):
        start += steps[i]
        ref_curve.append(start)


    unfav_dist = 100
    ones = [1]*len(ref_curve)
    cnt = 0

    while(unfav_dist > 32):
        unfav = []
        for i in range(len(array)):
            if ref_curve[i] > array[i]: unfav.append(round(ref_curve[i] - array[i],1))
            else: unfav.append(0)
        new_unfav = sum(unfav)
        if new_unfav < unfav_dist: unfav_dist =  new_unfav

        if unfav_dist > 32 :
            for i, val in enumerate(ref_curve):
                ref_curve[i] = val - 1
            cnt += 1

    print("R --- Reference curve --- Unfavorable Deviation")
    for i in range(len(array)): print("{0} \t {1} \t {2}".format(array[i], ref_curve[i], unfav[i]))
    print("################################################")
    print(unfav_dist, "\t", cnt)
    return unfav, cnt, ref_curve

def _split_array(array):
    """

    :param array:
    :return: R_S1, R_S2, S_S1, S_S2, b
    """
    R_S1 = []
    R_S2 = []
    S_S1 = []
    S_S2 = []
    b = []
    for i in Source_room_S1 : S_S1.append(array[i])
    for i in Source_room_S2 : S_S2.append(array[i])
    for i in Receiver_room_S1 : R_S1.append(array[i])
    for i in Receiver_room_S2: R_S2.append(array[i])
    for i in Background : b.append(array[i])
    return np.array(R_S1), np.array(R_S2), np.array(S_S1), np.array(S_S2), np.array(b)

def _LeqArray_Lab4(file):
    df = pd.read_csv(file, sep=";")
    array = df.to_numpy()
    Data_Array = array[:,28:46].astype(np.float)
    return Data_Array

def _calculate_log_mean(lst):
    avg = 0
    for i in lst:
        avg += 10**(i / 10)
    return round(10*np.log10(avg / len(lst)),1)

def _create_L_together(array):
    lst = np.transpose(array)
    temp = []
    for i in lst : temp.append(_calculate_log_mean(i))
    return np.array(temp)

def _create_b(b):
    b = np.transpose(np.array(b))
    temp = []
    for i in b: temp.append(_calculate_log_mean(i))
    b = np.array(temp)
    #LpiBLP = _calculate_log_mean(noise)
    #print("########",b)
    return b

def _calculate_SPL(array):
    temp = 0
    for i in array:
        temp = temp + 10**(0.1*i)
    return round((10*np.log10(temp)),1)

def _two_into_one_array(arr1, arr2):
    arr3 = np.column_stack((arr1, arr2))
    temp = []
    for i in arr3: temp.append(_calculate_log_mean(i))
    return np.array(temp)

def _plot_Semilogx(R_S1, R_S2, S_S1, S_S2, background, title1, title2, S1andS2 = False):
    fig = plt.figure(figsize=(17,7))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    receiver = _two_into_one_array(R_S1, R_S2)
    source = _two_into_one_array(S_S1, S_S2)

    back = np.array(background)


    if S1andS2:
        ax1.step(third_octave, R_S1, where="mid", color="dimgray", label="$L_{sb}$ for S1")
        ax1.step(third_octave, R_S1 - back,where="mid", color="blue", label="$L_{sb}$ - $L_b$  for S1")
        ax1.axline((100, 10), (5000, 10), linestyle="--", linewidth=0.8, color="r", label="Seperation line for correction")
        ax1.step(third_octave, R_S2,where="mid", color="goldenrod", label="$L_{sb}$ for S2")
        ax1.step(third_octave, R_S2 - back,where="mid", color="blueviolet", label="$L_{sb}$ - $L_b$  for S2")

        ax2.step(third_octave, S_S1,where="mid", color="dimgray", label="$L_{sb}$ for S1")
        ax2.step(third_octave, S_S1 - back,where="mid", color="blue", label="$L_{sb}$ - $L_b$  for S1")
        ax2.step(third_octave, S_S2,where="mid", color="goldenrod", label="$L_{sb}$ for S2")
        ax2.step(third_octave, S_S2 - back,where="mid", color="blueviolet", label="$L_{sb}$ - $L_b$  for S2")
        ax2.axline((100, 10), (5000, 10), linestyle="--", linewidth=0.8, color="r", label="Seperation line for correction")

    else:
        ax1.step(third_octave, receiver,where="mid", color="dimgray", label="$L_{sb}$")
        ax1.step(third_octave, receiver - back,where="mid", color="goldenrod",linewidth=2, label="$L_{sb}$ - $L_b$")
        ax1.axline((100, 10), (5000, 10), linestyle="--", linewidth=0.8, color="r", label="Seperation line for correction")

        ax2.step(third_octave, source,where="mid", color="dimgray", label="$L_{sb}$")
        ax2.step(third_octave, source - back,where="mid", color="goldenrod", linewidth=2, label="$L_{sb}$ - $L_b$")
        ax2.axline((100, 10), (5000, 10), linestyle="--", linewidth=0.8, color="r", label="Seperation line for correction")

        ax1.step(third_octave, back, where="mid", color="forestgreen", label="$L_b$")
        ax2.step(third_octave, back, where="mid", color="forestgreen", label="$L_b$")

    ax1.set_xscale('log')
    ax1.grid(which="major")
    ax1.grid(which="minor", linestyle=":")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_xticks(x_ticks_third_octave)
    ax1.set_xticklabels(x_ticks_third_octave_Bands)
    ax1.set_title(title1)

    ax2.set_xscale('log')
    ax2.grid(which="major")
    ax2.grid(which="minor", linestyle=":")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.set_xticks(x_ticks_third_octave)
    ax2.set_xticklabels(x_ticks_third_octave_Bands)
    ax2.set_title(title2)

    ax1.legend()
    ax2.legend()
    fig.savefig("Lab4Pressureplot.png")
    plt.show()

def _plot_DnT_ref(DnT, Dn, ref_curve, R_prime, tab, unfav_dist_sum):
    fig = plt.figure(figsize=(20, 8.5))
    fig.tight_layout()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    Header = ["Frequency [Hz]", "$R`$ [dB]", "Shifted Reference\nCurve [dB]", "Unfavorable \nDeviation [dB]"]
    Header = ["$\\bf{Frequency [Hz]}$","$\\bf{Unfavorable}$ \n $\\bf{Deviation [dB]}$" ]
    table = tbl.table(ax1, cellText=tab, colLabels=Header, cellLoc="center",loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    #table = ax1.add_table(cellText=table, cellLoc="center", loc="center",colLabels=Header,)
    ax1.add_table(table)


    table.scale(0.7,2.5)
    ax1.set_axis_off()

    ax2.step(third_octave, DnT,where="mid", color="dimgray", label="$D_{nT}$")
    ax2.step(third_octave, Dn,where="mid", color="black", label="$D_{n}$")
    ax2.step(third_octave, ref_curve,where="mid",linestyle="--", color="forestgreen", label="Shifted Reference Curve")
    #ax.axline((100,0), (100, 80), linestyle="..", linewidth=0.8, color="black", label="Frequency range according to the lab")
    #ax.axline((5000, 0), (5000, 80), linestyle="..", linewidth=0.8, color="black")
    ax2.step(third_octave, R_prime,where="mid", color="goldenrod", label="$R_{prime}$")

    ax2.set_xscale('log')
    ax2.grid(which="major")
    ax2.grid(which="minor", linestyle=":")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.set_xticks(x_ticks_third_octave)
    ax2.set_xticklabels(x_ticks_third_octave_Bands)

    title = str("$D_{nT}$, $D_n$, $R`$ and the Shifted Reference Curve. \n") + str("Sum of Unfavorable Deviation: {0}dB \n Weighted Sound Reduction Index: {1}dB".format(round(unfav_dist_sum,1),ref_curve[7]))

    #title = str("$D_{nT}$, $R`$ and the Shifted Reference Curve \n Sum of Unfavorable Deviation: ", str(unfav_dist_sum) ,str("dB \n Weighted Sound Reduction Index: ", str(ref_curve[7]),"dB"))

    ax2.set_title(title)

    ax2.legend()
    fig.savefig("Lab4DnT.png")
    plt.show()

def _calculate_D(receiver, source):
    return source - receiver

def _calculate_Dn(D, A):
    temp = []
    A0 = 10
    for i in range(len(D)) : temp.append(D[i] - 10*np.log10(A[i] / A0))
    return np.array(temp)

def _calculate_Dnt(D, T):
    temp = []
    T0 = 0.5
    for i in range(len(D)) : temp.append(D[i] + 10*np.log10(T[i] / T0))
    return np.array(temp)

def _surface_seperate():
    return round(1.18 * 1.21,2)

def _calculate_R_prime(D, A):
    temp = []
    S = _surface_seperate()
    for i in range(len(D)): temp.append(D[i] + 10*np.log10(S / A[i]))
    return np.array(temp)

def _calculate_T(file):
    df = pd.read_csv(file, sep=";")
    array = df.to_numpy()
    data = array[4:22, 4].astype(np.float)
    for i in range(len(data)) : data[i] = data[i]*2
    return data

def _calculate_A(T, V):
    return 0.16 * (V / T)

def _calculate_V(vol_array):
    x = vol_array[0]
    y = vol_array[1]
    z = vol_array[2]
    return round(x * y * z, 2)

def _create_table_data(R, ref_curve, unfav_dist):
    #temp = [["Frequency [Hz]","R` [dB]","Shifted Reference Curve [dB]","Unfavorable Deviation [dB]"]]
    temp = []
    for i in range(len(R)):
        #temp1 = [third_octave[i],round(R[i],1),round(ref_curve[i],1),round(unfav_dist[i],1)]
        temp1 = [third_octave[i], round(unfav_dist[i],1)]
        temp.append(temp1)
    return temp

def _create_R0():
    w = []
    R0 = []
    rhos = (15.33 + 10.08) / 2
    rho  = 1.225 * 343.2
    for i in third_octave : w.append(i*2*np.pi)
    for i in w : R0.append(10*np.log10(1 + ((i * rhos) / (2 * rho))**2 ))

    return R0


def _create_R_random(R0):
    R_random = []
    for i in R0 : R_random.append(i - 10*np.log10(0.23 * i))
    return R_random

def _create_R_field(R0):
    R_field = []
    for i in R0 : R_field.append(i - 5)
    return R_field


def _plot_R_R_field_R_random(R_prime, R_field, R_random):
    fig, ax = plt.subplots(figsize=(8,7))

    ax.step(third_octave, R_prime,where="mid", color="dimgray", label="$R_{prime}$")
    ax.step(third_octave, R_field,where="mid", color="forestgreen", label="$R_{field}$")
    ax.step(third_octave, R_random,where="mid", color="goldenrod", label="$R_{random}$")
    ax.set_xscale('log')
    ax.grid(which="major")
    ax.grid(which="minor", linestyle=":")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xticks(x_ticks_third_octave)
    ax.set_xticklabels(x_ticks_third_octave_Bands)

    title = "$R_{prime}$, $R_{field}$ and $R_{random}$."
    ax.set_title(title)

    ax.legend()
    fig.savefig("LabRs.png")
    plt.show()

def _calculate_Corrected(arr, b):
    temp = []
    for i in range(len(arr)) : temp.append(10*np.log10(10**(0.1 * arr[i]) - 10**(0.1 * b[i])))
    return temp

############# CODE RUNS FROM HERE ###############################

array = _LeqArray_Lab4("lab4.csv")

R_S1, R_S2, S_S1, S_S2, b = _split_array(array)

###################################
R_S1_avg = _create_L_together(R_S1)
R_S2_avg = _create_L_together(R_S2)
S_S1_avg = _create_L_together(S_S1)
S_S2_avg = _create_L_together(S_S2)
##################################

b = _create_b(b)
R_S1 = _create_L_together(R_S1)

title1 = str("Receiving room")
title2 = str("Source room")

_plot_Semilogx(R_S1_avg, R_S2_avg, S_S1_avg, S_S2_avg ,b, title1, title2)

print(_calculate_SPL(R_S1_avg))
print(_calculate_SPL(R_S2_avg))

receiver = _two_into_one_array(R_S1_avg, R_S2_avg)
source = _two_into_one_array(S_S1_avg, S_S2_avg)

source = _calculate_Corrected(source, b)

####### Values ######################
T = _calculate_T("reverb_time.csv")
V = _calculate_V(room_size_receiver)
A = _calculate_A(T, V)
D = _calculate_D(receiver,source)
R_prime = _calculate_R_prime(D,A)
DnT = _calculate_Dnt(D,T)
Dn = _calculate_Dn(D, A)
####################################

unfav_dist, cnt, ref_curve = _calculate_unfavorable(R_prime)

table = _create_table_data(R_prime,ref_curve,unfav_dist)

_plot_DnT_ref(DnT, Dn, ref_curve, R_prime, table, sum(unfav_dist))

R0 = _create_R0()
R_random = _create_R_random(R0)
R_field = _create_R_field(R0)

_plot_R_R_field_R_random(R_prime,R_field, R_random)

