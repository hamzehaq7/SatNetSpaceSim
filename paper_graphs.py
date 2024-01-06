from math import pi,log10,cos,sin,sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

k = 1.381e-23                               # Boltzmann constant (J/K)
c = 3e8                                     # speed of light (m/s)
Re = 6371000                                # radius of earth (m)
h = 2000e3                                  # height of satellite (m)

def degrees_to_radians(degrees):
    radians = degrees * pi / 180
    return radians

def Ep(h, epsilon,lemda, L_other, Gt, Gr, Pt, T, Rb, k):
  Lfs, L, snr, L_dB, Lfs_dB,snr_dB,Pr_dB,distance = [],[],[],[],[],[],[],[]
  for i in range(len(epsilon)):
    e = degrees_to_radians(epsilon[i])        # convert to radians
    term1 = (h + Re)**2/Re**2
    term2 = (cos(e))**2
    term3 = sqrt(term1 - term2)
    term4 = term3 - sin(e)
    d = Re * term4
    distance.append(d)

    freespace = (4*pi*d/lemda)**2
    Lfs.append(freespace)

    totalloss = freespace*L_other
    L.append(totalloss)

    gamma = Pt*Gt*Gr/(k*T*totalloss*Rb)
    snr.append(gamma)

    gammadB = 10*log10(gamma)
    snr_dB.append(gammadB)

    totallossdb = 10*log10(totalloss)
    L_dB.append(totallossdb)

    freesspacedb = 10*log10(freespace)
    Lfs_dB.append(freesspacedb)

  return Lfs_dB, L_dB, snr_dB, distance


def plotnew(epsilon, snrdBup, snrdBdown, xlabel, uplabel, downlabel, QAM16up, QAM32up, PSK32up, PSK16up, QAM16down, QAM32down, PSK32down, PSK16down, image):
  plt.figure(figsize=(10,8))
  n=3
  m=1
  markertype, n,m, PSK16color, PSK32color, QAM32color, QAM16color = 'x',3, 1, "dimgray", "darkorange", "royalblue", "forestgreen"
  plt.semilogy(epsilon, PSK16down, label = downlabel + '16PSK', markersize = n, color = PSK16color, marker=markertype)
  plt.semilogy(epsilon, PSK16up, label = uplabel + '16PSK', markersize = m, color = PSK16color)
  plt.semilogy(epsilon, PSK32down, label = downlabel + '32PSK', markersize = n, color = PSK32color, marker=markertype)
  plt.semilogy(epsilon, QAM32down, label = downlabel + '32QAM', markersize = n, color = QAM32color, marker=markertype)
  plt.semilogy(epsilon, PSK32up, label = uplabel + '32PSK', markersize = m, color = PSK32color)
  plt.semilogy(epsilon, QAM32up, label = uplabel + '32QAM', markersize = m, color = QAM32color)
  plt.semilogy(epsilon, QAM16down, label = downlabel + '16QAM', markersize = n, color = QAM16color, marker=markertype)
  plt.semilogy(epsilon, QAM16up, label = uplabel + '16QAM', markersize = m, color = QAM16color)

  plt.ylabel('Bit Error Probability (BER)')
  plt.xlabel(xlabel)
  plt.legend()
  plt.grid(True)
  plt.ylim(1e-7,1)
  plt.savefig(image, format='eps', bbox_inches='tight')
  plt.show()


def callnew(epsilon, snrdBup, snrdBdown):
  snrup, QAM16up, QAM32up, PSK32up, PSK16up  = [],[],[],[],[]
  snrdown, QAM16down, QAM32down, PSK32down, PSK16down = [],[],[],[],[]

  QAM16up, QAM32up, PSK32up, PSK16up = calc (snrdBup)
  QAM16down, QAM32down, PSK32down, PSK16down = calc (snrdBdown)

  plotnew(epsilon, snrdBup, snrdBdown, 'Elevation Angle (degrees)', 'Uplink - ', 'Downlink - ', QAM16up, QAM32up, PSK32up, PSK16up, QAM16down, QAM32down, PSK32down, PSK16down, '/content/drive/MyDrive/IEEE_APS/Figures/NewFigure4.eps')

def plot(x, down1, down2, down3, down4, up1, up2, up3, up4, xaxisname, yaxisname, imagename):
    plt.figure(figsize=(8,8))
    color1,color2, color3, color4, index = "blue", 'red', 'black','grey','>'
    plt.plot(x, up1, markersize = 5, color = color1, label= 'Uplink - 1000 km')
    plt.plot(x, up2, markersize = 5, color = color2, label= 'Uplink - 1300 km')
    plt.plot(x, up3, markersize = 5, color = color3, label= 'Uplink - 1600 km')
    plt.plot(x, up4, markersize = 5, color = color4, label= 'Uplink - 2000 km')

    plt.plot(x, down1, markersize = 5, color = color1, label= 'Downlink - 1000 km', marker = index)
    plt.plot(x, down2, markersize = 5, color = color2, label= 'Downlink - 1300 km', marker = index)
    plt.plot(x, down3, markersize = 5, color = color3, label= 'Downlink - 1600 km', marker = index)
    plt.plot(x, down4, markersize = 5, color = color4, label= 'Downlink - 2000 km', marker = index)

    plt.xlabel(xaxisname)
    plt.ylabel(yaxisname)
    plt.grid(True)
    plt.legend()
    plt.savefig(imagename, format='eps', bbox_inches='tight')
    plt.show()


def plotYYY(x, y1, y2, y3, xlabel, ylabel, title, imagename):
    plt.figure(figsize=(9,7))
    plt.plot(x[:4], y1, 'o-', markersize=8, color=c1, label='1 plane')
    plt.plot(x, y2, 'o-', markersize=8, color=c2, label='6 planes')
    plt.plot(x, y3, 'o-', markersize=8, color=c3, label='12 planes')
    for i, (xi, yi) in enumerate(zip(x, y1)):
        if i > 0:
          plt.plot([xi, xi], [yi, yi + 3], color=c1, linestyle='-', linewidth=0)
          if i == 1:
            plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(10,-35), ha='center', fontsize=m, color=c1)
          else:
            plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(0, -30), ha='center', fontsize=m, color=c1)
    for i, (xi, yi) in enumerate(zip(x, y2)):
        if i > 0:
            plt.plot([xi, xi], [yi, yi + 3], color=c2, linestyle='-', linewidth=0)
            if i ==1:
              plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(25,-5), ha='center', fontsize=m, color=c2)
            else:
              plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(0,5), ha='center', fontsize=m, color=c2)
    for i, (xi, yi) in enumerate(zip(x, y3)):
        if i > 0:
            plt.plot([xi, xi], [yi, yi + 3], color=c3, linestyle='-', linewidth=0)
            if i==1:
              plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(-10,0), ha='center', fontsize=m, color=c3)
            else:
              plt.annotate(f'{yi:.2f}', (xi, yi + 3), textcoords="offset points", xytext=(0,5), ha='center', fontsize=m, color=c3)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(imagename, dpi=300)  # Save the figure
    plt.show()

def calc(snr_dB):
  QAM16, QAM32, PSK16, PSK32 = [],[],[],[]
  m, M16, M32 = 4, 16, 32

  for i in range(len(snr_dB)):
    gamma = 10**(snr_dB[i]/10)

    psk16 = erfc(np.sqrt(m*gamma)*sin(pi/M16))/m
    psk32 = erfc(np.sqrt(m*gamma)*sin(pi/M32))/m
    qam16 = (2/m)*(1-(1/np.sqrt(M16)))*erfc(np.sqrt(3*m*gamma/(2*(M16-1))))
    qam32 = (2/m)*(1-(1/np.sqrt(M32)))*erfc(np.sqrt(3*m*gamma/(2*(M32-1))))

    QAM16.append(qam16)
    QAM32.append(qam32)
    PSK32.append(psk32)
    PSK16.append(psk16)

  return QAM16, QAM32, PSK16, PSK32


def plot5Y(x, y1, y2, y3, y4, y5, xaxisname, yaxisname, imagename):
    plt.figure(figsize=(8,6))
    plt.plot(x, y5,markersize = 5, color = "grey", label= 'Uplink - XL')
    plt.plot(x, y4,markersize = 5, color = "green", label= 'Uplink - L')
    plt.plot(x, y3,markersize = 5, color = "black", label= 'Uplink - M')
    plt.plot(x, y2,markersize = 5, color = "red", label= 'Uplink - S')
    plt.plot(x, y1, markersize = 5, color = "blue", label= 'Downlink', marker = 'x')

    plt.xlabel(xaxisname)
    plt.ylabel(yaxisname)
    plt.grid(True)
    plt.legend()
    # plt.savefig(imagename, format='eps', bbox_inches='tight')
    plt.show()


def plot3Y(x, y1, y2, y3, xlabel, ylabel, imagename):
    plt.figure()
    # plt.plot(x, y1, 'o-',markersize = 5, color = "b", label='UHF')
    plt.plot(x, y3,markersize = 5, color = "black", label='L-band (1.2GHz)')
    plt.plot(x, y1,markersize = 5, color = "b", label='UHF (436MHz)')
    plt.plot(x, y2,markersize = 5, color = "r", label='VHF (145MHz)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    # plt.savefig(imagename, format='eps', bbox_inches='tight')
    plt.show()

# ------------------------ General Parameters --------------------- #
## DOWNLINK ANALYSIS
L_other_dB_down = 7                                   # other losses: atmospheric, feeder, polarization, depointing (dB)
Pt_dB_down = 50                                  # transmit power (dB)
Gt_dB_down = 15.5                                # transmitter antenna gain (dB)
Gr_dB_down = 0                               # receiver antenna gain (dB)
T_down = 1378                                     # system temp noise (K)
Rb_down = 1e6                                  # data rate = 1 Mbps
lemda_down = c/437e6                                # frequency assuming VHF = 145MHz = 145e6 Hz
# Conversion from dB to normal
Gr_down = 10**(Gr_dB_down/10)                         # receiver antenna gain
Gt_down = 10**(Gt_dB_down/10)                         # transmitter antenna gain
Pt_down = (10**(Pt_dB_down/10))/1000                         # transmit power (W) antenna gain
L_other_down = 10**(L_other_dB_down/10)               # other losses


# UPLINK ANALYSIS
L_other_dB_up = 4.2                                   # other losses: atmospheric, feeder, polarization, depointing (dB)
Pt_dB_up = 15                                  # transmit power (dBm)
Gt_dB_up = 0                                # transmitter antenna gain (dB)
Gr_dB_up = 12.34                               # receiver antenna gain (dB)
T_up = 1160                                     # system temp noise (K)
Rb_up = 2400                                  # data rate = 2400 bps
lemda_up = c/146e6                               # frequency assuming VHF = 145MHz = 145e6 Hz
# Conversion from dB to normal
Gr_up = 10**(Gr_dB_up/10)                         # receiver antenna gain
Gt_up = 10**(Gt_dB_up/10)                         # transmitter antenna gain
Pt_up = (10**(Pt_dB_up/10))/1000                         # transmit power (W) antenna gain
L_other_up = 10**(L_other_dB_up/10)               # other losses


# --------------------------------- Fig 2  ------------------------------- #
Latm_dB = 0.2802                            # atmospheric loss (dB)
Lpol_dB = 2.31                              # polarization loss (dB)
LD_dB = 0.0012                              # feeder loss (dB)
Lf_dB = 1                                   # depointing loss (dB)
Pt_dB = 10                                  # transmit power (dB)
Gt_dB = 6.56                                # transmitter antenna gain (dB)
Gr_dB = 13.51                               # receiver antenna gain (dB)
T = 300                                     # system temp noise (K) - 290 or 2000??
Rb = 200e6                                  # desired data rate = 200Mbps = 200e6 bps
VHFf = 145e6                                # frequency assuming VHF = 145MHz = 145e6 Hz
UHFf = 436e6                                # frequency assuming UHF = 436MHz = 436e6 Hz
Lbandf = 1.2e9                              # frequency assuming L-band = 1.2GHz

lemda_VHF = c/VHFf
lemda_UHF = c/UHFf
lemda_Lband = c/Lbandf
epsilon = list(range(0,180))               # range of elevation angles (in degrees) to be analyzed

Gr = 10**(Gr_dB/10)                         # receiver antenna gain
Gt = 10**(Gt_dB/10)                         # transmitter antenna gain
Pt = 10**(Pt_dB/10)                         # transmit power (W) antenna gain
Latm = 10**(Latm_dB/10)                     # atmospheric loss
Lpol = 10**(Lpol_dB/10)                     # polarization loss
Lf = 10**(Lf_dB/10)                         # feeder loss
Ld = 10**(LD_dB/10)                         # depointing loss
L_other = Latm*Lpol*Ld*Lf*Ld*Lf

Lfs_dB_UHF, L_dB_UHF, snr_dB_UHF,_ =                  Ep(h, epsilon, lemda_UHF,   L_other, Gt,Gr,Pt,T,Rb,k)
Lfs_dB_VHF, L_dB_VHF, snr_dB_VHF,_  =                 Ep(h, epsilon, lemda_VHF,   L_other, Gt,Gr,Pt,T,Rb,k)
Lfs_dB_Lband, L_dB_Lband, snr_dB_Lband,distance =     Ep(h, epsilon, lemda_Lband, L_other, Gt,Gr,Pt,T,Rb,k)
plot3Y(distance, Lfs_dB_UHF, Lfs_dB_VHF, Lfs_dB_Lband, 'Distance (m)', 'Free Space Loss (dB)', '/content/drive/MyDrive/IEEE APS/Figures/NewFigure2a.png')
plot3Y(epsilon, Lfs_dB_UHF, Lfs_dB_VHF, Lfs_dB_Lband, 'Elevation Angle (degrees)', 'Free Space Loss (dB)', '/content/drive/MyDrive/IEEE APS/Figures/Figure3.eps')



# --------------------------------- Fig 3  ------------------------------- #
epsilon = list(range(9,171,4))               # range of elevation angles (in degrees) to be analyzed
Pt_up_S, Pt_up_M, Pt_up_L, Pt_up_XL = 0.03, 0.1, 0.3, 1 # 15, 20, 25, 30 dBm (respectively)

Lfs_dB_up_S, L_dB_up_S, snr_dB_up_S, distance_up_S =     Ep(h, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up_S, T_up, Rb_up, k)
Lfs_dB_up_M, L_dB_up_M, snr_dB_up_M, distance_up_M =     Ep(h, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up_M, T_up, Rb_up, k)
Lfs_dB_up_L, L_dB_up_L, snr_dB_up_L, distance_up_L =     Ep(h, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up_L, T_up, Rb_up, k)
Lfs_dB_up_XL, L_dB_up_XL, snr_dB_up_XL, distance_up_XL = Ep(h, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up_XL, T_up, Rb_up, k)

Lfs_dB_down_elev, L_dB_down_elev, snr_dB_down_elev, distance = Ep(h, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)

plot5Y(epsilon, snr_dB_down_elev, snr_dB_up_S, snr_dB_up_M, snr_dB_up_L, snr_dB_up_XL, 'Elevation Angle (degrees)', 'SNR (dB)', '/content/drive/MyDrive/IEEE_APS/Figures/NewFigure3.eps')



# --------------------------------- Fig 4  ------------------------------- #
Pt_dB = 15                                  # transmit power (dB)
Pt = (10**(Pt_dB/10))/1000                         # transmit power (W) antenna gain
Lfs_dB_up_elev, L_dB_up_elev, snr_dB_up_elev, distance = Ep(h, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt, T_up, Rb_up, k)
Lfs_dB_down_elev, L_dB_down_elev, snr_dB_down_elev, distance = Ep(h, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)
callnew(epsilon, snr_dB_up_elev, snr_dB_down_elev)


# --------------------------------- Fig 5  ------------------------------- #
epsilon = list(range(9, 172, 4))

Lfs_dB_up_h1, L_dB_up_h1, snr_dB_up_h1, distance_up_h1 = Ep(1000e3, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up, T_up, Rb_up, k)
Lfs_dB_up_h2, L_dB_up_h2, snr_dB_up_h2, distance_up_h2 = Ep(1300e3, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up, T_up, Rb_up, k)
Lfs_dB_up_h3, L_dB_up_h3, snr_dB_up_h3, distance_up_h3 = Ep(1600e3, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up, T_up, Rb_up, k)
Lfs_dB_up_h4, L_dB_up_h4, snr_dB_up_h4, distance_up_h4 = Ep(2000e3, epsilon, lemda_up, L_other_up, Gt_up, Gr_up, Pt_up, T_up, Rb_up, k)

Lfs_dB_down_h1, L_dB_down_h1, snr_dB_down_h1, distance_down_h1 = Ep(1000e3, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)
Lfs_dB_down_h2, L_dB_down_h2, snr_dB_down_h2, distance_down_h2 = Ep(1300e3, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)
Lfs_dB_down_h3, L_dB_down_h3, snr_dB_down_h3, distance_down_h3 = Ep(1600e3, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)
Lfs_dB_down_h4, L_dB_down_h4, snr_dB_down_h4, distance_down_h4 = Ep(2000e3, epsilon, lemda_down, L_other_down, Gt_down, Gr_down, Pt_down, T_down, Rb_down, k)

plot(epsilon,
       snr_dB_down_h1, snr_dB_down_h2, snr_dB_down_h3, snr_dB_down_h4,
       snr_dB_up_h1, snr_dB_up_h2, snr_dB_up_h3, snr_dB_up_h4,
       'Elevation Angle (degrees)', 'SNR (dB)', '/content/drive/MyDrive/IEEE_APS/Figures/NewFigure5.eps')


# --------------------------------- Fig 7  ------------------------------- #
m,c1,c2,c3=9,'m','b','black'
x = [1, 24, 48, 72, 96, 288]
p1 = [1.11034002, 18.87578, 19.56974, 19.70854]
p6 = [0, 21.30464955, 44.136, 59.195, 65.37127, 66.1095]
p12 = [0, 30.46495, 61.1381, 72.44969, 84.59403, 99.2845]

plotYYY(x, p1, p6, p12, 'Number of Satellites', '% Coverage Time Per Day', '% Coverage Time Per Day for Different Number of Planes','/content/drive/MyDrive/IEEE_APS/Figures/NewFigure7.eps')
