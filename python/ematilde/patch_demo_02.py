"""
slot_ring_pyAEDT.py
Parameterized pyAEDT script to build a shared-aperture slot-ring antenna element
(from Shirazi et al., IEEE AWPL 2017) with:
 - Substrate (Rogers RT/5880)
 - Slot-ring geometries (outer L-band ring and four small C-band rings)
 - Microstrip feeds (Port1 for L-band, Ports2-5 for C-band)
 - PIN diode series R-L-C lumped elements (parameterized ON/OFF)
 - Radiation boundary and solution setup covering both bands (L + C)
Notes: tune geometry parameters in the 'params' dict if you need to shift resonances.
"""
# required libraries
import os
import scipy.signal
import socket
from datetime import datetime
import platform
import pyaedt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyaedt import Hfss
from math import pi

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np

import uncc_mts_unit_cell as unit_cell
import uncc_mts_compute_config as compute_config

if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT231"] = "/opt/Ansys/v252/AnsysEM"
else:
    os.environ["ANSYSEM_ROOT231"] = "C:\\Program Files\\AnsysEM\\v231\\Win64\\"

# ---------------------------
# USER / ENVIRONMENT SETTINGS
# ---------------------------
# If AEDT is local and supports non-graphical sessions:
AEDT_VERSION = "2025.2"  # change if needed (e.g., "2022.2", "2024.1")
NON_GRAPHICAL = True  # True -> headless if AEDT supports it

# Project / design names
PROJECT_NAME = "SlotRing_Antenna_pyAEDT"
DESIGN_NAME = "SlotRing_C_and_L"

# ---------------------------
# GEOMETRY / MATERIAL PARAMETERS
# ---------------------------
# These are parameterized for quick tuning
params = {
    # substrate
    "sub_x": 120.0,  # mm substrate dimension in x
    "sub_y": 120.0,  # mm substrate dimension in y
    "h_sub": 0.79,  # mm substrate thickness (paper used 0.79 mm)
    "eps_r": 2.2,
    "tan_delta": 0.0009,

    # copper
    "th_cu": 0.035,  # mm copper thickness

    # outer (L-band) big slot-ring (square) - perimeter chosen by paper relation;
    # user can tweak these to shift L-band resonance.
    "outer_ring_side": 80.0,  # mm outer square side length — tuneable
    "outer_ring_width": 2.0,  # mm slot width (ring trace width)

    # 2x2 small C-band rings inside aperture
    "small_ring_side": 11.3,  # mm (approx — tuneable to hit 5.71 GHz)
    "small_ring_width": 1.2,  # mm

    # spacing & placement
    "small_array_pitch": 30.0,  # mm center-to-center spacing for the 2x2 small rings

    # feed / pad geometry
    "feed_pad_w": 3.0,  # mm microstrip pad width
    "feed_pad_l": 6.0,  # mm pad length from edge

    # PIN diode RLC (series) parameters (paper values)
    "L_pin": 0.15e-9,  # Henries
    "C_on": 8e-15,  # Farads
    "C_off": 32e-15,  # Farads
    "R_on": 3.0,  # Ohms
    "R_off": 15e3,  # Ohms

    # solution frequencies
    "f_center_C": 5.71e9,
    "f_center_L": 1.76e9,
}


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def mm(x):
    """Return value in mm for readability (pyAEDT expects units in mm by default)"""
    return x


# ---------------------------
# START AEDT / HFSS Session
# ---------------------------
print("Starting HFSS session (pyAEDT). This will launch AEDT if needed...")
hfss = Hfss(projectname=PROJECT_NAME,
            designname=DESIGN_NAME,
            specified_version=AEDT_VERSION,
            non_graphical=NON_GRAPHICAL,
            new_desktop_session=True)

# Set unit system (HFSS default is mm)
hfss.modeler.model_units = "mm"

# ---------------------------
# MATERIALS
# ---------------------------
# Add substrate material - RT/Duroid 5880 (approx)
hfss.materials["RT5880"] = {
    "permittivity": params["eps_r"],
    "loss_tangent": params["tan_delta"],
    "conductivity": None,
}

# ---------------------------
# CREATE SUBSTRATE (solid)
# ---------------------------
sub_x = params["sub_x"]
sub_y = params["sub_y"]
h_sub = params["h_sub"]
print("Creating substrate...")
substrate = hfss.modeler.create_box([-sub_x / 2, -sub_y / 2, 0],
                                    [sub_x, sub_y, h_sub],
                                    name="Substrate",
                                    matname="RT5880")

# ---------------------------
# CREATE GROUND PLANE (copper sheet -> then thicken)
# ---------------------------
gnd_th = params["th_cu"]
print("Creating ground plane (copper sheet)...")
# create a thin solid copper ground plane placed on top of substrate bottom (z=0)
gnd = hfss.modeler.create_box([-sub_x / 2, -sub_y / 2, 0],
                              [sub_x, sub_y, gnd_th],
                              name="GroundSolid",
                              matname="copper")

# Move ground to top of substrate thickness if needed (we will place conductor on top of substrate)
# In this geometry: substrate bottom at z=0, substrate top at z = h_sub.
# We'll place ground at top of substrate (typical microstrip: ground on bottom - but slot antennas are planar)
# For the slot-ring in the paper, ground and slot are on a single metal layer around substrate top.
# To keep things simple, place metal on the top of substrate:
hfss.modeler.move(gnd, [0, 0, h_sub])  # now ground sits on top of the substrate

# ---------------------------
# CREATE RADIATOR SHEET (copper) AND SLOT FEATURES
# We'll create a top copper sheet (radiator) then boolean-subtract slot shapes to form slot-rings.
# ---------------------------
print("Creating radiator copper sheet (top layer)...")
rad_th = params["th_cu"]
radiator = hfss.modeler.create_box([-sub_x / 2, -sub_y / 2, h_sub + gnd_th],
                                   [sub_x, sub_y, rad_th],
                                   name="RadiatorSolid",
                                   matname="copper")

# For simplicity we'll create slot-rings as rectangular ring cutouts on the radiator.
# Outer big ring (square donut)
outer_side = params["outer_ring_side"]
outer_w = params["outer_ring_width"]
outer_outer = hfss.modeler.create_rectangle(position=[-outer_side / 2, -outer_side / 2, h_sub + gnd_th + 0.0001],
                                            dimension_list=[outer_side, outer_side],
                                            name="outer_sq",
                                            matname="copper",
                                            is_covered=False)  # will use to cut

outer_inner_side = outer_side - 2 * outer_w
outer_inner = hfss.modeler.create_rectangle(
    position=[-outer_inner_side / 2, -outer_inner_side / 2, h_sub + gnd_th + 0.0001],
    dimension_list=[outer_inner_side, outer_inner_side],
    name="outer_sq_inner",
    matname="copper",
    is_covered=False)

# Extrude (pull) the rectangles through the radiator thickness to create cutting solids
outer_cut_solid = hfss.modeler.create_box([-outer_side / 2, -outer_side / 2, h_sub + gnd_th],
                                          [outer_side, outer_side, rad_th + 0.5],
                                          name="outer_cut_solid",
                                          matname="vacuum")
outer_cut_inner_solid = hfss.modeler.create_box([-outer_inner_side / 2, -outer_inner_side / 2, h_sub + gnd_th],
                                                [outer_inner_side, outer_inner_side, rad_th + 0.6],
                                                name="outer_cut_inner_solid",
                                                matname="vacuum")

# Subtract the outer_cut_solid - outer_cut_inner_solid from radiator to create ring slot
print("Cutting outer ring slot from radiator...")
rad_after1 = hfss.modeler.subtract(radiator, outer_cut_solid)
if isinstance(rad_after1, list):  # sometimes returns a list
    radiator = rad_after1[0]
rad_after2 = hfss.modeler.subtract(radiator, outer_cut_inner_solid)
if isinstance(rad_after2, list):
    radiator = rad_after2[0]

# ---------------------------
# Create 2x2 small C-band slot-rings inside the big ring
# ---------------------------
print("Creating 2x2 small C-band rings inside outer ring...")
sr_side = params["small_ring_side"]
sr_w = params["small_ring_width"]
pitch = params["small_array_pitch"]
# center offsets for 2x2
offsets = [(-pitch / 2, -pitch / 2), (pitch / 2, -pitch / 2), (-pitch / 2, pitch / 2), (pitch / 2, pitch / 2)]
small_cut_solids = []
for i, (ox, oy) in enumerate(offsets):
    out_x = ox - sr_side / 2
    out_y = oy - sr_side / 2
    out_box = hfss.modeler.create_box([out_x, out_y, h_sub + gnd_th],
                                      [sr_side, sr_side, rad_th + 0.6],
                                      name=f"small_out_{i}",
                                      matname="vacuum")
    inner_side = sr_side - 2 * sr_w
    in_x = ox - inner_side / 2
    in_y = oy - inner_side / 2
    in_box = hfss.modeler.create_box([in_x, in_y, h_sub + gnd_th],
                                     [inner_side, inner_side, rad_th + 0.7],
                                     name=f"small_in_{i}",
                                     matname="vacuum")
    # subtract out_box then add back in_box (i.e., create ring)
    radiator = hfss.modeler.subtract(radiator, out_box)
    radiator = hfss.modeler.unite(radiator, hfss.modeler.create_box([in_x, in_y, h_sub + gnd_th],
                                                                    [inner_side, inner_side, rad_th + 0.8],
                                                                    name=f"temp_in_{i}",
                                                                    matname="copper"))
    # Note: above we try to re-add inner copper; depending on AEDT results you may want to use boolean operations differently.

# ---------------------------
# FEEDLINES & PORT PADS (microstrip) - create simple rectangular pads reaching the outer edge
# Port mapping used:
#   Port1 -> L-band: connect to outer big ring aperture (center feed) -- we'll create a feed near center
#   Ports2-5 -> C-band: four microstrip feeds connecting to each small ring
# ---------------------------
print("Creating microstrip feed pads and traces (parameterized).")
pad_w = params["feed_pad_w"]
pad_l = params["feed_pad_l"]


# function to create a microstrip trace and pad from outside toward ring center
def create_pad(cx, cy, name="pad"):
    # create rectangle pad sitting on top of radiator layer (slightly above to ensure connection)
    pad = hfss.modeler.create_rectangle(position=[cx - pad_w / 2, cy - pad_l, h_sub + gnd_th + rad_th + 0.0002],
                                        dimension_list=[pad_w, pad_l],
                                        name=name,
                                        matname="copper")
    # extrude to solid of thickness rad_th (so it joins with radiator)
    pad_solid = hfss.modeler.create_box([cx - pad_w / 2, cy - pad_l, h_sub + gnd_th + rad_th * 0.5],
                                        [pad_w, pad_l, rad_th],
                                        name=f"{name}_solid",
                                        matname="copper")
    # Unite pad with radiator so it's a single conductor (join)
    try:
        newrad = hfss.modeler.unite(radiator, pad_solid)
        return newrad
    except Exception as e:
        print("Warning: unite pad failed:", e)
        return pad_solid


# Create C-band feed pads for each small ring
c_feed_centers = offsets  # same coords for pads
for idx, (cx, cy) in enumerate(c_feed_centers):
    radiator = create_pad(cx, cy - (sr_side / 2 + 2.0), name=f"Cfeed_pad_{idx + 2}")  # offset slightly outside ring

# Create L-band feed pad at center (Port1)
radiator = create_pad(0, 0 - (outer_side / 2 + 2.0), name="Lfeed_pad_1")

# ---------------------------
# PIN Diode modeling (Lumped R-L-C series elements)
# We'll create small gap solids and then assign lumped RLC elements between the two faces.
# For simplicity we will create small gap boxes at each diode location and then remove a tiny chunk from radiator to leave a gap.
# ---------------------------
print("Creating PIN diode gap locations and assigning lumped RLC models...")
# Example diode positions: place 4 around each small ring sides (paper uses 16 diodes total inside slots).
# For simplicity: create 1 diode per small ring side (4 per small ring) — this is a template; user can refine.
pin_locations = []
gap_size = 0.2  # mm gap across which lumped RLC will be placed
# We'll just create one diode per small ring at top side (for example)
for i, (ox, oy) in enumerate(offsets):
    # place gap on top edge of each small ring
    gx = ox
    gy = oy + sr_side / 2  # top center of small ring
    # create two tiny pads representing the two sides of the gap, separated by gap_size
    left_pad = hfss.modeler.create_box([gx - 0.8, gy - 0.2 - gap_size / 2, h_sub + gnd_th + rad_th * 0.1],
                                       [1.6, 0.4, rad_th],
                                       name=f"pin_left_{i}",
                                       matname="copper")
    right_pad = hfss.modeler.create_box([gx - 0.8, gy - 0.2 + gap_size / 2, h_sub + gnd_th + rad_th * 0.1],
                                        [1.6, 0.4, rad_th],
                                        name=f"pin_right_{i}",
                                        matname="copper")
    # cut the small region from radiator so these pads are separate
    radiator = hfss.modeler.subtract(radiator, left_pad)
    radiator = hfss.modeler.subtract(radiator, right_pad)
    pin_locations.append((left_pad, right_pad))

# Now assign lumped R-L-C elements between left_pad and right_pad faces
# We will create parameterized R/L/C values and assign the series lumped elements
R_on = params["R_on"]
R_off = params["R_off"]
L_pin = params["L_pin"]
C_on = params["C_on"]
C_off = params["C_off"]

# For each created gap, assign a lumped R-L-C (series). Use ON for ones that close for C-band, OFF else.
# Here we set them all to ON for C-band operation by default (user can parameterize later).
for i, (left_pad, right_pad) in enumerate(pin_locations):
    # identify the two faces to connect the lumped element across
    # In pyAEDT we must specify faces by object and face index; we attempt to use bounding box center to pick faces.
    # Use the HFSS API call create_lumped_rlc (this API name may vary slightly by pyAEDT version).
    try:
        # Create a series lumped R-L-C element
        lumped_name = f"PIN_Lumped_{i}"
        hfss.create_lumped_rlc(name=lumped_name,
                               entity_list=[left_pad.name, right_pad.name],
                               resistance=R_on,
                               inductance=L_pin,
                               capacitance=C_on,
                               use_s_param=True)
    except Exception as e:
        print("Warning: create_lumped_rlc failed for", lumped_name, " — you may need to adapt API call. Error:", e)
        # fallback: create a small object representing the diode gap and add a boundary -> manual step later.

# ---------------------------
# CREATE RADIATION BOUNDARY (Airbox)
# ---------------------------
print("Creating radiation airbox and assigning radiation boundary...")
air_margin = 30.0  # mm
air = hfss.modeler.create_box([-sub_x / 2 - air_margin, -sub_y / 2 - air_margin, -air_margin],
                              [sub_x + 2 * air_margin, sub_y + 2 * air_margin, h_sub + rad_th + 2 * air_margin],
                              name="Airbox",
                              matname="air")
# assign radiation boundary (radiation) to the outer faces of the airbox
hfss.assign_radiation(air.name)

# ---------------------------
# PORTS
# Create wave or lumped ports on microstrip pads (we'll create lumped ports on each pad as a default)
# ---------------------------
print("Assigning ports on feed pads (lumped ports)...")
# Find the pad solids (we created them with names containing 'Cfeed_pad' and 'Lfeed_pad')
# Create lumped ports across short gaps to ground or reference
try:
    # Loop through objects and add lumped ports
    all_objs = hfss.modeler.object_names
    pad_objs = [name for name in all_objs if "Cfeed_pad" in name or "Lfeed_pad" in name]
    port_index = 1
    for pad_name in pad_objs:
        # Get object and select a face pair for the lumped port assignment.
        # pyaedt has method assign_lumped_port_to_sheet or create_lumped_port. We'll attempt create_lumped_port.
        port_name = f"Port{port_index}"
        hfss.create_lumped_port(name=port_name,
                                object_name=pad_name,
                                axisdir="Z",
                                reference_object=hfss.modeler.find_objects("GroundSolid")[0])
        port_index += 1
except Exception as e:
    print("Warning: automatic port creation failed. Error:", e)
    print("You may need to manually assign ports on the microstrip pads using the HFSS GUI or adapt the API calls.")

# ---------------------------
# SOLUTION SETUP
# ---------------------------
print("Creating solution setup and frequency sweep...")
# Create an initial solution setup near C-band center
f0 = params["f_center_C"]
# create solution setup; many pyAEDT versions use add_setup instead of create_setup.
try:
    setup = hfss.create_setup("setup1")
    setup.props["Frequency"] = f0
    setup.props["PortsOnly"] = False
    setup.props["MaxDeltaS"] = 0.02
    # create sweep 1.0-7.0 GHz to include L and C bands
    hfss.create_linear_count_sweep(setupname="setup1",
                                   unit="GHz",
                                   start_value=1.0,
                                   stop_value=7.0,
                                   num_points=301,
                                   sweep_type="Interpolating")
except Exception as e:
    print("Warning: creating setup failed. Error:", e)
    # Fallback: use high-level API
    try:
        hfss.create_setup("setup1")
        hfss.create_linear_count_sweep("setup1", "GHz", 1.0, 7.0, 201, "Interpolating")
    except Exception as ee:
        print("Second attempt also failed:", ee)

# ---------------------------
# SAVE & RUN (optional)
# ---------------------------
# Save project to disk
project_path = os.path.join(os.getcwd(), PROJECT_NAME + ".aedt")
print("Saving project as:", project_path)
hfss.save_project(project_path)

print("Script completed geometry creation. You can run a solve with hfss.analyze_all() if you wish.")
# Optionally run simulation (uncomment if you want to solve now)
# print("Starting analysis...")
# hfss.analyze_all()

# Close (but keep AEDT alive if needed)
# hfss.release_desktop()
print(
    "Done. If any pyAEDT API call failed you will see warnings — this means the API call needs minor adaptation for your specific pyAEDT/AEDT version.")
