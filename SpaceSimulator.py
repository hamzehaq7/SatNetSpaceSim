from math import pi, log10, cos, sin, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc
import sys
import ipywidgets as widgets
from IPython.display import display, HTML
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import Axes3D

# Define colors for styling
primary_color = '#3498db'  # Blue
background_color = '#ecf0f1'  # Light Gray

k = 1.381e-23                               # Boltzmann constant (J/K)
c = 3e8                                     # speed of light (m/s)
Re = 6371000                                # radius of earth (m)

def degrees_to_radians(degrees):
    radians = degrees * pi / 180
    return radians

# Styling for the input textboxes
input_style = {'description_width': 'initial'}

# Create textboxes for manual entering
lat_textbox = widgets.FloatText(description='Latitude (degrees):',                    value=24.10, style=input_style)
long_textbox = widgets.FloatText(description='Longitude (degrees):',                  value=54.75, style=input_style)
height_textbox = widgets.FloatText(description='CubeSat Height (km):',                value=500, style=input_style)
L_atm_dB_textbox = widgets.FloatText(description='Atmospheric Loss (dB):',value=0,style=input_style)
L_pol_dB_textbox = widgets.FloatText(description='Polarization Mismatch Loss (dB):',value=0,style=input_style)
L_fTx_dB_textbox = widgets.FloatText(description='Feeder Loss of the Transmitter (dB):',value=0,style=input_style)
L_fRx_dB_textbox = widgets.FloatText(description='Feeder Loss of the Receiver (dB):',value=0,style=input_style)
L_DTx_dB_textbox = widgets.FloatText(description='Transmitter Depointing Loss (dB):',value=0,style=input_style)
L_DRx_dB_textbox = widgets.FloatText(description='Receiver Depointing Loss (dB):',value=0,style=input_style)
Pt_dB_textbox = widgets.FloatText(description='Transmit Power (dBm):', value=30, style=input_style)
Gt_dB_textbox = widgets.FloatText(description='Transmitter Antenna Gain (dB):', value=15, style=input_style)
Gr_dB_textbox = widgets.FloatText(description='Receiver Antenna Gain (dB):', value=20, style=input_style)
T_textbox = widgets.FloatText(description='System Temperature (K):', value=1000, style=input_style)
Rb_textbox = widgets.FloatText(description='Desired Data Rate (bps):', value=1e6, style=input_style)
f_MHz_textbox = widgets.FloatText(description='Frequency (MHz):', value=1000, style=input_style)
epsilon_degree_textbox = widgets.FloatText(description='Elevation Angle (degrees):', value=45, style=input_style)
M_textbox = widgets.FloatText(description='M Value:', value=16, style=input_style)
modulation_widget = widgets.Dropdown(options=['psk', 'qam'], value='psk', description='Modulation:', style=input_style)

# Output widget for header and calculation result
header_output = widgets.Output()
output_text = widgets.Output()

# Function to plot 3D satellite position
def plot_satellite(lat, longt, height):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    r = Re + height * 1000  # Radius of Earth plus satellite height
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot Satellite
    ax.scatter(longt, lat, Re + height * 1000, c='red', marker='o', label='Satellite')

    # Set labels and title
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title('Satellite Monitoring Earth')

    # Show the plot
    plt.show()

# Function to perform calculations and display results
def calculate_and_display(button):
    with output_text:
        output_text.clear_output(wait=True)
        display(get_calculated_output())
        # Plot 3D satellite position
        plot_satellite(lat_textbox.value, long_textbox.value, height_textbox.value)

def get_calculated_output():
    h = height_textbox.value * 1000
    L_atm_dB = L_atm_dB_textbox.value
    L_pol_dB = L_pol_dB_textbox.value
    L_fTx_dB = L_fTx_dB_textbox.value
    L_fRx_dB = L_fRx_dB_textbox.value
    L_DTx_dB = L_DTx_dB_textbox.value
    L_DRx_dB = L_DRx_dB_textbox.value
    Pt_dB = Pt_dB_textbox.value
    Gt_dB = Gt_dB_textbox.value
    Gr_dB = Gr_dB_textbox.value
    lat = lat_textbox.value
    longt = long_textbox.value
    T = T_textbox.value
    Rb = Rb_textbox.value
    f = f_MHz_textbox.value * 1e6
    epsilon_degree = epsilon_degree_textbox.value
    mod = modulation_widget.value
    M = M_textbox.value

    L_other_dB = L_atm_dB + L_pol_dB + L_fTx_dB + L_fRx_dB + L_DTx_dB + L_DRx_dB
    lemda = c / f
    Gr = 10**(Gr_dB / 10)
    Gt = 10**(Gt_dB / 10)
    Pt = (10**(Pt_dB / 10)) / 1000
    L_other = 10**(L_other_dB / 10)

    e = degrees_to_radians(epsilon_degree)

    term1 = (h + Re)**2 / Re**2
    term2 = (cos(e))**2
    term3 = sqrt(term1 - term2)
    term4 = term3 - sin(e)
    d = Re * term4

    freespace = (4 * pi * d / lemda)**2
    freesspacedb = 10 * log10(freespace)

    totalloss = freespace * L_other
    totallossdb = 10 * log10(totalloss)

    gamma = Pt * Gt * Gr / (k * T * totalloss * Rb)
    gammadB = 10 * log10(gamma)

    # Create a fancy table for output
    table = PrettyTable()
    table.field_names = ["Parameters", "Values"]
    table.add_row(["Latitude", f"{lat} degrees"])
    table.add_row(["Longitude", f"{longt} degrees"])
    table.add_row(["CubeSat Height", f"{h/1000} km"])
    table.add_row(["Atmospheric Loss", f"{L_atm_dB} dB"])
    table.add_row(["Polarization Mismatch Loss", f"{L_pol_dB} dB"])
    table.add_row(["Feeder Loss of the Transmitter", f"{L_fTx_dB} dB"])
    table.add_row(["Feeder Loss of the Receiver", f"{L_fRx_dB} dB"])
    table.add_row(["Transmitter Depointing Loss", f"{L_DTx_dB} dB"])
    table.add_row(["Receiver Depointing Loss", f"{L_DRx_dB} dB"])
    table.add_row(["Transmit Power", f"{Pt_dB} dBm"])
    table.add_row(["Transmitter Antenna Gain", f"{Gt_dB} dB"])
    table.add_row(["Receiver Antenna Gain", f"{Gr_dB} dB"])
    table.add_row(["System Temperature", f"{T} K"])
    table.add_row(["Desired Data Rate", f"{Rb} bps"])
    table.add_row(["Elevation Angle", f"{epsilon_degree} degrees"])
    table.add_row(["Frequency", f"{f/1e6} MHz"])
    table.add_row(["Wavelength", f"{lemda} m"])
    table.add_row(["Distance", f"{d/1000} km"])
    table.add_row(["Free Space Path Loss", f"{freesspacedb} dB"])
    table.add_row(["Total Loss", f"{totallossdb} dB"])
    table.add_row(["SNR", f"{gammadB} dB"])

    if mod == 'psk':
        psk = erfc(np.sqrt(4 * gamma) * sin(pi / M)) / 4
        table.add_row([f"BER (PSK_{M})", f"{psk}"])
    elif mod == 'qam':
        qam = (2 / 4) * (1 - (1 / np.sqrt(M))) * erfc(np.sqrt(3 * 4 * gamma / (2 * (M - 1))))
        table.add_row([f"BER (QAM_{M})", f"{qam}"])
    else:
        table.add_row(["Modulation", "Not a valid technique"])

    return table

# Create a button to trigger calculations
calculate_button = widgets.Button(description='Calculate', style=widgets.ButtonStyle(button_color=primary_color))

# Attach the function to the button click event
calculate_button.on_click(calculate_and_display)

# Header with website-like styling
header_html = HTML("""
    <div style="background-color: {}; padding: 20px; text-align: center; font-size: 24px; color: white; border-radius: 10px;">
        SatNet SpaceSim
    </div>
""".format(primary_color))

# Display header and main layout
with header_output:
    display(header_html)


    L_atm_dB = L_atm_dB_textbox.value
    L_pol_dB = L_pol_dB_textbox.value
    L_fTx_dB = L_fTx_dB_textbox.value
    L_fRx_dB = L_fRx_dB_textbox.value
    L_DTx_dB = L_DTx_dB_textbox.value
    L_DRx_dB = L_DRx_dB_textbox.value

# Styling for the main layout
main_layout = widgets.VBox([
    widgets.HBox([lat_textbox, long_textbox, height_textbox]),
    widgets.HBox([Pt_dB_textbox, Gt_dB_textbox, Gr_dB_textbox]),
    widgets.HBox([T_textbox, Rb_textbox, f_MHz_textbox]),
    widgets.HBox([L_atm_dB_textbox, L_pol_dB_textbox]),
    widgets.HBox([L_fTx_dB_textbox, L_fRx_dB_textbox]),
    widgets.HBox([L_DTx_dB_textbox, L_DRx_dB_textbox]),
    widgets.HBox([epsilon_degree_textbox]),
    widgets.HBox([modulation_widget, M_textbox]),
    calculate_button,
    output_text
])

# Apply styling to the main layout
main_layout.layout.display = 'flex'
main_layout.layout.flex_flow = 'column'
main_layout.layout.align_items = 'center'
main_layout.layout.margin = '20px'

# Display main layout
display(widgets.VBox([header_output, main_layout], layout=widgets.Layout(display='flex', flex_flow='column', align_items='center')))
