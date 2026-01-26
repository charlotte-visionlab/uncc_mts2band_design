#MTS: Unit Cell Simulation
#Experiment 1
#Wan, X., Cai, B., Li, Y. et al. Dual-channel near-field control by polarizations using isotropic and
#inhomogeneous metasurface. Sci Rep 5, 15853 (2015). https://doi.org/10.1038/srep15853
#-------------------
#This example shows how you can use PyAEDT to create a multipart scenario in HFSS
#and set up a metasurface unit cell dispersion analysis.

#This is an HFSS implementation of dispersion calculation for a unit cell consisting of a small metal patch.

#When the patch size is equal to the cell size the dispersion matches that of a parallel plate waveguide.
#Results for this code are intended to match the design described in:

#Wan, X., Cai, B., Li, Y. et al. Dual-channel near-field control by polarizations using isotropic and
#inhomogeneous metasurface. Sci Rep 5, 15853 (2015). https://doi.org/10.1038/srep15853

#The fill percentage, i.e., ratio of the metal patch dimensions in (X,Y) to the size to the unit cell, is controlled
#via the fill_pct variable which takes on a value (0,0) <= fill_pct <= (1,1) and determines the ratio of the square
#metal patch size in (X,Y) to the unit cell size in (X,Y).


###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import socket
from datetime import datetime
import platform
from ansys.aedt.core import Hfss, launch_desktop

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np
import math 

import uncc_mts_compute_config as compute_config


def sinuous_curve(r, alpha, r_start, tau):
    return alpha * np.sin((np.pi * np.log(r / r_start)) / np.log(tau))

# --- BALUN EXPONENTIAL TAPER FUNCTION ---
def exponential_taper_profile(x_mm, L_mm, W_start_mm, W_end_mm):
    """
    Calculates the half-width (y-profile) of the balun strip at distance x_mm
    along the taper for an exponential profile.
    """
    if W_start_mm <= 0 or W_end_mm <= 0 or L_mm == 0:
        return np.ones_like(x_mm) * W_start_mm / 2.0

    A = W_start_mm / 2.0
    B = np.log(W_end_mm / W_start_mm) / L_mm

    x_clamped = np.clip(x_mm, 0, L_mm)

    return A * np.exp(B * x_clamped)

# ============================================================
#   KLOPFENSTEIN TAPER WIDTH PROFILE (Canonical Implementation)
# ============================================================

def klopf_width_profile(x, L, W1, W2, ripple=0.01):
    """
    Canonical Klopfenstein taper width profile implementation (Option C).
    x : scalar or 1D numpy array of positions along taper [0..L]
    L : total taper length (same units as x)
    W1: start width (full width) at x=0 (mm)
    W2: end width (full width) at x=L (mm)
    ripple: allowable reflection ripple (rho). Typical small value 0.01.
    Returns: width array (same shape as x), mapping Klopfenstein impedance-like curve to width.
    """
    # Ensure numpy array input for vectorized operation
    x_arr = np.array(x, dtype=float)
    # avoid divide by zero
    L_safe = L if np.abs(L) > 1e-12 else 1.0

    # ratio R (map widths as proxy to impedance ratio)
    R = float(W2) / float(W1) if float(W1) != 0 else 1.0
    # G0 parameter (Klopfenstein)
    G0 = 0.5 * np.log(R + 1e-15)

    # guard for small ripple or small G0
    rho = max(1e-6, float(ripple))
    # If G0 is too small (W1 ~= W2), return linear/interpolated width
    if np.abs(G0) < 1e-8:
        # simple linear interpolation
        return W1 + (W2 - W1) * (x_arr / L_safe)

    # A constant (arccosh argument must be >= 1)
    arg = G0 / rho
    # numerical safety
    if arg < 1.0:
        arg = 1.0
    A = np.arccosh(arg)

    # compute C(x)
    C = np.cosh(A * (2.0 * x_arr / L_safe - 1.0)) / np.cosh(A)

    # Klopfenstein-like mapping: exponential mapping of G0*C
    Z_like = np.exp(G0 * C)

    # Normalize Z_like to start at 1 at x=0 and end at R at x=L approximately
    Z0 = np.exp(G0 * (np.cosh(A * (-1.0)) / np.cosh(A)))
    ZL = np.exp(G0 * (np.cosh(A * (1.0)) / np.cosh(A)))
    # avoid division by zero
    if np.abs(ZL - Z0) < 1e-12:
        frac = (x_arr / L_safe)
    else:
        frac = (Z_like - Z0) / (ZL - Z0)

    # map fraction to width
    width = W1 + frac * (W2 - W1)

    return width


cm2mm = 10
frequency_max = 6e9
frequency_min = 0.8e9

plot_antenna = True

copper_thickness = 35.0e-6
radius_start_cm = 2.5 / cm2mm  # initial radius
radius_end_cm = 37.5 / cm2mm
tau = 1.238  # tau is usually between 1.2 and 1.5
alpha = np.deg2rad(45)  # alpha is usually between 30 and 70 degrees
# self complementary delta = 180/(2N)
delta = np.deg2rad(22.5)  # delta is the angular width of the arm
# delta = np.deg2rad(180.0 / (2 * N))  # delta is the angular width of the arm
# R_end = R0 * (tau * N)  # final radius
N = radius_end_cm / (radius_start_cm * tau)
# R_end = np.abs(sinuous_curve(R0 * (tau * N), alpha, R0, tau))  # final radius
# Rogers 5880 standard thicknesses are 0.127, 0.252, 0.508, 0.787, 1.575
dielectric_height_cm = 1.575 / cm2mm  # <=== dielectric slab height in meters
# height = 2.54e-3  # m
arm_contact_diameter_cm = 2 * 0.25 * radius_start_cm
arm_contact_pad_thickness = 0.3
PEC_ANTENNA = True
dielectric_diameter_cm = np.array(1.1 * 2 * radius_end_cm)
antenna_diameter_cm = 2 * radius_end_cm
# ground_plane_diameter_cm = 10 * 20.0e-3
ground_plane_diameter_cm = dielectric_diameter_cm - 0.25
# ground_plane_hole_diameter_cm = 0.5
ground_plane_hole_diameter_cm = 0.25

cavity_wall_thickness_cm = 1.0 / cm2mm
cavity_diameter_cm = dielectric_diameter_cm + 2 * cavity_wall_thickness_cm

cavity_height_cm = 30.0 / cm2mm

machine = "ece-emag1"
# DIELECTRIC MATERIALS

dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "Taconic TLY (tm)"
# dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
# ground_plane_material_name = "pec"
ground_plane_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"


# --- NEW FEATURE: BALUN AND FERRITE ABSORBER PARAMETERS ---

# Balun Parameters (Microstrip to Parallel-Strip Klopfenstein Taper)
Z_unb = 50.0  # Unbalanced input impedance (Ohm)
Z_bal = 150.0 # Balanced output impedance (Ohm, typical for sinuous)
balun_length_cm = 0.8 * radius_end_cm # Shorter for compactness
balun_taper_start_width_mm = 0.5  # Approx. 50 Ohm line width on this substrate, scaled for mm
balun_taper_end_width_mm = cm2mm * 0.5 * arm_contact_diameter_cm # Connects to the arm feed
balun_separation_start_mm = 0.5 # Gap/slot at the start of the taper

# Balun layer setup: on the bottom of the dielectric (Z = -dielectric_height_cm)
balun_z_position_mm = -cm2mm * dielectric_height_cm

# Ferrite Absorber Parameters (Unidirectional Radiation)
ferrite_material_name = "Ferrite_Absorber_UWB"
ferrite_thickness_cm = 5.0 / cm2mm # 5 mm thick ferrite backing
ABSORBER_MR = 2.0 # Relative Permeability (mu_r)
ABSORBER_MLT = 0.8 # Magnetic Loss Tangent (tan(delta_m)) - CRITICAL for absorption


class Point2D:
    def __init__(self, x, y):
        # This assignment calls __setattr__
        self.data = {'x': x, 'y': y}

    def __getattr__(self, name):
        # This allows access like point.x
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            # FIX: Use object.__setattr__ to prevent recursion and correctly
            # set the internal dictionary without triggering the custom logic.
            object.__setattr__(self, name, value)
        else:
            # This handles assignments like point.x = 10
            self.data[name] = value
# --- END OF FIX ---


class SinuousArm:

    def __init__(self):
        self.start = Point2D(0.0, 0.0)
        self.end = Point2D(0.0, 0.0)
        self.inner = Point2D(0.0, 0.0)
        self.outer = Point2D(0.0, 0.0)
        self.contour_mm = Point2D(0.0, 0.0)
        self.contour_position_array_mm = np.empty(0)
        self.contour_position_list_mm = []
        self.contact_position_array_mm = np.empty(0)
        self.contact_position_list_mm = []
        self.contact_center_position_mm = Point2D(0.0, 0.0)


def make_sinuous_arm(r_start, r_end, N, tau, alpha, delta, height_mm=0.0, z_orientation=0.0,
                     num_samples=1000, plot=False):
    angle_offset = np.deg2rad(z_orientation)
    arm = SinuousArm()

    # R_end = R0 * (tau * N)  # final radius
    r = np.linspace(r_start, r_end, num_samples)

    phi_inner = angle_offset + delta + sinuous_curve(r, alpha, r_start, tau)
    phi_outer = angle_offset - delta + sinuous_curve(r, alpha, r_start, tau)

    phi_inner_end = angle_offset + delta + sinuous_curve(r_end, alpha, r_start, tau)
    phi_outer_end = angle_offset - delta + sinuous_curve(r_end, alpha, r_start, tau)

    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)

    t_end = np.linspace(phi_inner_end, phi_outer_end, num_samples)

    arm.end.x = r_end * np.cos(t_end)
    arm.end.y = r_end * np.sin(t_end)
    arm.start.x = np.array([r_start * np.cos(phi_inner_start), r_start * np.cos(phi_outer_start)])
    arm.start.y = np.array([r_start * np.sin(phi_inner_start), r_start * np.sin(phi_outer_start)])

    arm.inner.x = r * np.cos(phi_inner)
    arm.inner.y = r * np.sin(phi_inner)
    arm.outer.x = r * np.cos(phi_outer)
    arm.outer.y = r * np.sin(phi_outer)

    arm.contour_mm.x = np.concatenate((np.flip(arm.start.x), arm.inner.x, arm.end.x, np.flip(arm.outer.x)))
    arm.contour_mm.y = np.concatenate((np.flip(arm.start.y), arm.inner.y, arm.end.y, np.flip(arm.outer.y)))
    num_contour_points = len(arm.contour_mm.x)
    arm.contour_position_array_mm = np.array(
        [arm.contour_mm.x, arm.contour_mm.y, height_mm * np.ones(num_contour_points)]).T
    # z_angle_rad = np.deg2rad(z_orientation)
    # rotation_matrix = [[np.cos(z_angle_rad), -np.sin(z_angle_rad), 0.0],
    #                    [np.sin(z_angle_rad), np.cos(z_angle_rad), 0.0],
    #                    [0.0, 0.0, 1.0]]
    # arm_contour_points_array = rotation_matrix @ arm_contour_points_array
    arm.contour_position_list_mm = [elem.tolist() for elem in arm.contour_position_array_mm]
    if plot:
        plot_colors = {"-b", "-r", "-k", "-c", "-m"}
        plt.figure()
        # plt.plot(arm.contour.x, arm.contour.y, '-b')
        plt.plot(arm.contour_mm.x, arm.contour_mm.y, '-r')
        plt.axis('equal')
        plt.show()
        lim = 1.1 * r_end
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
    return arm


def make_sinuous_arm_circular_terminal(sinuous_arm, radius_mm, r_start, tau, alpha, delta, height_mm=0.0,
                                       z_orientation=0.0, num_samples=50, plot=False):
    angle_offset = np.deg2rad(z_orientation - 22.5)
    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)
    # arm_start_x = np.array([R0 * np.cos(phi_inner_start), R0 * np.cos(phi_outer_start)])
    # arm_start_y = np.array([R0 * np.sin(phi_inner_start), R0 * np.sin(phi_outer_start)])
    # circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    circle_center_radius = 0.75 * r_start
    circle_center = Point2D(circle_center_radius * np.cos(circle_center_phi),
                            circle_center_radius * np.sin(circle_center_phi))
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    feed_x = circle_center.x + radius_mm * np.cos(theta)
    feed_y = circle_center.y + radius_mm * np.sin(theta)
    num_contour_points = len(feed_x)
    contour_position_array_mm = np.array([feed_x, feed_y, height_mm * np.ones(num_contour_points)]).T
    # plt.fill(feed_x, feed_y)
    sinuous_arm.contact_center_position_mm = Point2D(circle_center.x, circle_center.y)
    sinuous_arm.contact_position_array_mm = contour_position_array_mm
    sinuous_arm.contact_position_list_mm = [elem.tolist() for elem in contour_position_array_mm]
    return contour_position_array_mm, circle_center


def make_sinuous_arm_trapezoidal_terminal(sinuous_arm, radius_mm, r_start, tau, alpha, delta, height_mm=0.0,
                                          z_orientation=0.0, num_samples=50, plot=False):
    angle_offset = np.deg2rad(z_orientation)
    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)

    trap_position_list = np.array([[r_start * np.cos(phi_inner_start), r_start * np.sin(phi_inner_start)],
                                   [r_start * np.cos(phi_outer_start), r_start * np.sin(phi_outer_start)],
                                   [radius_mm * np.cos(phi_outer_start), radius_mm * np.sin(phi_outer_start)],
                                   [radius_mm * np.cos(phi_inner_start), radius_mm * np.sin(phi_inner_start)]])
    # circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    trap_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    trap_center_radius = 0.5 * (r_start + radius_mm)
    circle_center = Point2D(trap_center_radius * np.cos(trap_center_phi),
                            trap_center_radius * np.sin(trap_center_phi))
    num_contour_points = trap_position_list.shape[0]
    contour_position_array_mm = np.concatenate((trap_position_list, height_mm * np.ones((num_contour_points, 1))), axis=1)
    # plt.fill(feed_x, feed_y)
    sinuous_arm.contact_center_position_mm = Point2D(circle_center.x, circle_center.y)
    sinuous_arm.contact_position_array_mm = contour_position_array_mm
    sinuous_arm.contact_position_list_mm = [elem.tolist() for elem in contour_position_array_mm]
    return contour_position_array_mm, circle_center


if plot_antenna:
    plt.figure()

sinuous_antenna_arms_list = []
for arm_index in np.arange(4):
    sinuous_antenna_arm = make_sinuous_arm(cm2mm * radius_start_cm, cm2mm * radius_end_cm, N, tau, alpha, delta,
                                           height_mm=0.0, z_orientation=90 * arm_index, plot=False)
    # make_sinuous_arm_circular_terminal(sinuous_antenna_arm, cm2mm * arm_contact_diameter_cm / 2, cm2mm * radius_start_cm,
    #                                    tau, alpha, delta, height_mm=0.0, z_orientation=90 * arm_index, plot=False)
    make_sinuous_arm_trapezoidal_terminal(sinuous_antenna_arm, cm2mm * arm_contact_diameter_cm / 2,
                                          cm2mm * radius_start_cm,
                                          tau, alpha, delta, height_mm=0.0, z_orientation=90 * arm_index, plot=False)
    sinuous_antenna_arms_list.append(sinuous_antenna_arm)
    if plot_antenna:
        plot_colors = {"-b", "-r", "-k", "-c", "-m"}
        # plt.plot(arm.contour.x, arm.contour.y, '-b')
        plt.fill(sinuous_antenna_arm.contour_mm.x, sinuous_antenna_arm.contour_mm.y)
        plt.fill(sinuous_antenna_arm.contact_position_array_mm[:, 0],
                 sinuous_antenna_arm.contact_position_array_mm[:, 1])

if plot_antenna:
    plt.axis('equal')
    lim = 10 * 1.1 * radius_end_cm
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.show()

if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT252"] = "/opt/Ansys/v252/AnsysEM"
else:
    os.environ["ANSYSEM_ROOT231"] = "C:\\Program Files\\AnsysEM\\v231\\Win64\\"

aedt_version = "2025.2"

# Ensure compute_config is available or define a placeholder if not needed for script execution
try:
    solver_configuration = compute_config.SolverConfig().solver_config
except NameError:
    # Placeholder if compute_config is not defined or imported correctly
    solver_configuration = None


###############################################################################
# Define program variables
###############################################################################
speed_of_light = 2.99792458e8
frequency_nominal = 0.5 * (frequency_max + frequency_min)
wavelength_nominal = speed_of_light / frequency_nominal

# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
antenna_coord_origin_xy_cm = 0.5 * np.array([0.0, 0.0])

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
aluminum_color = [143, 143, 175]

radiation_volume_height_cm = dielectric_height_cm + 6 * dielectric_height_cm

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "Sinuous DF Antenna " + current_time_str
design_name = "Sinuous DF Antenna " + current_time_str

save_file_prefix = "sinuous_df_antenna"
save_filename_no_extension = save_file_prefix + "_" + current_time_str + "_" + socket.gethostname()
save_filename_matlab = save_filename_no_extension + ".mat"
save_filename_numpy = save_filename_no_extension + ".npy"


def matplotlib_pyplot_setup():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('x-axis', fontsize = SMALL_SIZE)
    # plt.rc('y-axis', fontsize = SMALL_SIZE)


matplotlib_pyplot_setup()

###############################################################################
# Set non-graphical mode
###############################################################################

aedt_version = "2025.2"
non_graphical = False

###############################################################################
# Launch AEDT
###############################################################################
NewThread = True
desktop = launch_desktop(aedt_version, non_graphical)


# Solution Types are: { "Modal", "Terminal", "Eigenmode", "Transient Network", "SBR+", "Characteristic"}
hfss = Hfss(
    project=project_name,
    design=design_name,
    version=aedt_version,
    solution_type="Terminal",

    non_graphical=non_graphical
)


hfss.modeler.model_units = 'mm'
hfss.autosave_disable()

# --- NEW FEATURE: DEFINE FERRITE ABSORBER MATERIAL ---
hfss.materials.add_material(
    name=ferrite_material_name,
    props={
        "permittivity": 10.0,
        "dielectric_loss_tangent": 0.05,
        "permeability": ABSORBER_MR,
        "magnetic_loss_tangent": ABSORBER_MLT, # Crucial high loss factor
    }
)


###############################################################################
# Define HFSS Variables
###############################################################################
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength_nominal))

###############################################################################
# Define geometries
###############################################################################
dielectric_slab_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                     cm2mm * antenna_coord_origin_xy_cm[1],
                                     cm2mm * -dielectric_height_cm])
dielectric_slab_params = {"name": "dielectric_slab",
                          "cs_axis": "Z",
                          "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
                                                              dielectric_slab_position[1],
                                                              dielectric_slab_position[2]).split(","),
                          "radius": "{}mm".format(cm2mm * dielectric_diameter_cm / 2),
                          "height": "{}mm".format(cm2mm * dielectric_height_cm),
                          "numSides": 0,
                          "matname": dielectric_material_name}
dielectric_slab_geom = hfss.modeler.create_cylinder(**dielectric_slab_params)
dielectric_slab_geom.color = dielectric_color

# --- NEW FEATURE: FERRITE ABSORBER GEOMETRY (Unidirectional Radiation) ---
#ferrite_height_mm = cm2mm * ferrite_thickness_cm
#ferrite_radius_mm = cm2mm * dielectric_diameter_cm / 2  # Match substrate size

#GROUND_Z_MM = -2.083   # <-- your ground plane Z (mm). Make sure the sign matches HFSS!

#EPS_Z = 0.0            # set 0.001 if you want a tiny gap to avoid coincident faces

# We want TOP of ferrite at the ground plane:
#ferrite_top_z_mm   = GROUND_Z_MM - EPS_Z
#ferrite_z_start_mm = ferrite_top_z_mm - ferrite_height_mm   # bottom of ferrite
#ferrite_z_end_mm   = ferrite_top_z_mm                        # top of ferrite


#ferrite_params = {"name": "ferrite_absorber",
#                  "cs_axis": "Z",
#                  "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
 #                                                     dielectric_slab_position[1],
  #                                                    ferrite_z_start_mm).split(","),
   #               "radius": "{}mm".format(ferrite_radius_mm),
    #              "height": "{}mm".format(ferrite_height_mm),
     #             "numSides": 0,
      #            "matname": ferrite_material_name}

#ferrite_geom = hfss.modeler.create_cylinder(**ferrite_params)
#ferrite_geom.color = [100, 50, 50] # Dark color for absorber
#ferrite_geom.transparency = 0.5
# -----------------------------------------------------------------

# FIT ALL
hfss.modeler.fit_all()

ground_plane_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                  cm2mm * antenna_coord_origin_xy_cm[1],
                                  cm2mm * -dielectric_height_cm - 0.035])

# Draw the sinusoidal antenna trace as a polyline
sinuous_arm_polyline_list = []

for arm_index, arm in enumerate(sinuous_antenna_arms_list):
    arm_position_list = arm.contour_position_list_mm
    arm_object_name = "sinuous_arm_" + str(arm_index)
    arm_polyline_params = {
        "position_list": arm_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": arm_object_name,
        "matname": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    sinuous_arm_polyline = hfss.modeler.create_polyline(**arm_polyline_params)
    sinuous_arm_polyline.color = metal_color

    arm_contact_position_list = arm.contact_position_list_mm
    arm_contact_object_name = "sinuous_arm_contact_" + str(arm_index)
    arm_contact_polyline_params = {
        "position_list": arm_contact_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": arm_contact_object_name,
        "matname": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    sinuous_contact_arm_polyline = hfss.modeler.create_polyline(**arm_contact_polyline_params)
    sinuous_contact_arm_polyline.color = metal_color

    unite_params = {
        "unite_list": [arm_object_name, arm_contact_object_name],
        "purge": False,
        "keep_originals": False
    }
    hfss.modeler.unite(**unite_params)

    through_hole_contact_cavity_position = [arm.contact_center_position_mm.x,
                                            arm.contact_center_position_mm.y,
                                            cm2mm * -dielectric_height_cm]

    arm_contact_cavity_name = "contact_through_hole_cavity_" + str(arm_index)
    through_hole_contact_params = {"name": arm_contact_cavity_name,
                                   "cs_axis": "Z",
                                   "position": "{}mm,{}mm,{}mm".format(through_hole_contact_cavity_position[0],
                                                                       through_hole_contact_cavity_position[1],
                                                                       through_hole_contact_cavity_position[2]).split(
                                       ","),
                                   "radius": "{}mm".format(
                                       cm2mm * arm_contact_diameter_cm / 2 - arm_contact_pad_thickness),
                                   "height": "{}mm".format(cm2mm * dielectric_height_cm),
                                   "numSides": 0,
                                   "matname": None}
    through_hole_contact_geom = hfss.modeler.create_cylinder(**through_hole_contact_params)

    subtract_params = {
        # "blank_list": ["dielectric_slab", "ground_plane", arm_object_name],
        "blank_list": ["dielectric_slab", arm_object_name],
        "tool_list": [arm_contact_cavity_name],
        "keep_originals": False
    }
    hfss.modeler.subtract(**subtract_params)

    # Existing port definition logic (kept commented out or minimal to focus on the new balun feed)
    if PEC_ANTENNA:
        hfss.assign_perfecte_to_sheets(
            **{"sheet_list": arm_object_name,
               "sourcename": None,
               "is_infinite_gnd": False})
        sinuous_arm_polyline_list.append(sinuous_arm_polyline)
    # else: [Keeping this block simple for PEC antenna]

# --------------------------------------------------------------------
#substrate block


mdl = hfss.modeler
mdl.model_units = "mm"

try:
    mdl.set_working_coordinate_system("Global")
except Exception:
    pass

# ----------------------------
# Names
# ----------------------------
SUB1_NAME = "Balun_Substrate"
SUB2_NAME = "Balun_Substrate_2"
GND_NAME  = "Balun_Ground"

# ----------------------------
# Radius (use SAME radius for everything)
# ----------------------------
BALUN_RADIUS_MM = float(cm2mm * float(dielectric_diameter_cm) / 2.0)

# ----------------------------
# Thicknesses (Z only)
# ----------------------------
SUB1_THK_MM = 0.254   # thickness of substrate 1 (top substrate)
SUB2_THK_MM = 0.254   # thickness of substrate 2 (bottom substrate)

# ----------------------------
# Materials
# ----------------------------
SUB_MAT    = globals().get("dielectric_material_name", "Rogers RT/duroid 5880 (tm)")
GROUND_MAT = "copper"

# ----------------------------
# Optional XY shifts (do NOT affect thickness)
# ----------------------------
SHIFT1_XY = (0.0, 0.0)
SHIFT2_XY = (0.0, 0.0)
SHIFTG_XY = (0.0, 0.0)


def zmin_zmax(obj_name: str):
    # bounding_box = [xmin,ymin,zmin,xmax,ymax,zmax]
    try:
        bb = mdl.objects[obj_name].bounding_box
    except Exception:
        bb = mdl.get_object_bounding_box(obj_name)
    return float(bb[2]), float(bb[5])

ANT_SUB_NAME = "dielectric_slab"   # <-- change if your antenna substrate name differs
ant_zmin, ant_zmax = zmin_zmax(ANT_SUB_NAME)
print(f"[DEBUG] {ANT_SUB_NAME} zmin={ant_zmin:.6f}  zmax={ant_zmax:.6f}")

# We want Substrate 1 TOP to touch the antenna substrate BOTTOM
ANT_SUB_BOT_Z = ant_zmin

# If faces coincident still looks like overlap, keep a tiny gap:
Z_SHIFT_MM = -10.0
EPS_Z = 0.001  # mm (1 micron). Set to 0.0 to force perfect contact.
SUB1_TOP_Z = float(ANT_SUB_BOT_Z - EPS_Z + Z_SHIFT_MM)

print(f"[DEBUG] Balun Sub1 TOP_Z = {SUB1_TOP_Z:.6f} (touching antenna bottom minus EPS_Z)")

# ----------------------------
# Cleanup old copies if rerunning
# ----------------------------
for nm in [SUB1_NAME, SUB2_NAME, GND_NAME]:
    try:
        if nm in mdl.object_names:
            mdl.delete([nm])
    except Exception:
        pass

# ----------------------------
# Substrate 1 (top)
# spans: [SUB1_BOT_Z .. SUB1_TOP_Z]
# ----------------------------
SUB1_BOT_Z = float(SUB1_TOP_Z - SUB1_THK_MM)

sub1 = mdl.create_cylinder(
    orientation="Z",
    origin=[SHIFT1_XY[0], SHIFT1_XY[1], SUB1_BOT_Z],
    radius=float(BALUN_RADIUS_MM),
    height=float(SUB1_THK_MM),
    name=SUB1_NAME,
    material=SUB_MAT,
    num_sides=0
)
if not sub1:
    raise RuntimeError("Failed to create Balun_Substrate")

# ----------------------------
# Substrate 2 (below substrate 1)
# TOP touches Substrate 1 BOTTOM
# spans: [SUB2_BOT_Z .. SUB2_TOP_Z]
# ----------------------------

#SUB2_TOP_Z = float(SUB1_BOT_Z)
#SUB2_BOT_Z = float(SUB2_TOP_Z - SUB2_THK_MM)

#sub2 = mdl.create_cylinder(
    orientation="Z",
    origin=[SHIFT2_XY[0], SHIFT2_XY[1], SUB2_BOT_Z],
    radius=float(BALUN_RADIUS_MM),
    height=float(SUB2_THK_MM),
    name=SUB2_NAME,
    material=SUB_MAT,
    num_sides=0
#)
#if not sub2:
 #   raise RuntimeError("Failed to create Balun_Substrate_2")

#hfss.assign_perfecte_to_sheets(
 #   assignment="Balun_Ground",
  #  name="PerfE_Balun_Ground",
   # is_infinite_ground=False
#)


# -----------------------------
# GROUND PLANE (FLAT, XY PLANE)
# -----------------------------
# Put it wherever you want in Z. If you want it "attached" to a substrate,
# set z_ground = substrate_bottom_z (or whatever reference you use).
#gnd_radius = 41.25
#z_ground = -2.083

#gnd = mdl.create_circle(
#    orientation="XY",             # <-- THIS is the key (flat)
 #   origin=[0.0, 0.0, z_ground],  # center of the ground disk
  #  radius=gnd_radius,            # same as substrate radius
   # num_sides=0,                  # 0 = true circle
 #   is_covered=True,              # filled sheet (disk)
  #  name="Balun_Ground",
   # material="copper"
#)

# If you're modeling metals as PerfectE sheets (common for thin copper),
 #assign PerfectE boundary:
#hfss.assign_perfecte_to_sheets(
 #   assignment=gnd.name,
  #  name="PerfE_Balun_Ground",
   # is_infinite_ground=False
#)


# ----------------------------
# CREATE BALUN_GROUND (needed for lumped ports reference)
# ----------------------------
#GND_NAME = "Balun_Ground"   # must match your ports block name exactly

# place ground at bottom of Substrate 1
#GND_Z = float(SUB1_BOT_Z)

# delete old one if rerun
#try:
  #  if GND_NAME in mdl.object_names:
 #       mdl.delete([GND_NAME])
#except Exception:
 #   pass

#gnd = mdl.create_circle(
 #   orientation="XY",
  #  origin=[0.0, 0.0, GND_Z],
 #   radius=float(BALUN_RADIUS_MM),
  #  num_sides=0,
  #  is_covered=True,
 #   name=GND_NAME,
#    material="copper"
#)

#if not gnd:
  #  raise RuntimeError("Failed to create Balun_Ground")

#hfss.assign_perfecte_to_sheets(
 #   assignment=GND_NAME,
#    name=f"PerfE_{GND_NAME}",
 #   is_infinite_ground=False
#)

#print(f"[OK] Created {GND_NAME} at z={GND_Z:.6f}")


# ==========================================================
# CURVED S-SHAPED BALUN
# ==========================================================








def _create_sbend_centerline(p_start, p_end, npts=200, s_amplitude=0.5):

    #Create a smooth S-like curve centerline between p_start and p_end.
    #p_start, p_end: (x,y) tuples in mm
    #s_amplitude: lateral amplitude in mm (max offset orthogonal to straight line)
    #returns list of (x,y) points length npts and approximate chord length

    p0 = np.array(p_start, dtype=float)
    p1 = np.array(p_end, dtype=float)
    t = np.linspace(0.0, 1.0, npts)
    # Straight line vector
    v = p1 - p0
    length = np.linalg.norm(v) + 1e-12
    tangent = v / length
    # normal vector for lateral S-offset (rotate tangent by 90 deg)
    normal = np.array([-tangent[1], tangent[0]])
    # Smooth S offset using a sinusoidal shape (one half wave gives single S)
    offsets = s_amplitude * np.sin(np.pi * t)  # single bump
    pts = [p0 + t_i * v + offsets_i * normal for t_i, offsets_i in zip(t, offsets)]
    return pts, length

def _poly_from_centerline_with_width(center_pts, widths):

    #Form a closed polygon from a centerline and local widths (full width).
    #center_pts: list of (x,y)    widths: list of widths (full width) same length as center_pts
    #returns polygon points (list of [x,y,z]) going around perimeter (left side then reversed right side)

    left_pts = []
    right_pts = []
    n = len(center_pts)
    for i in range(n):
        p = np.array(center_pts[i], dtype=float)
        # compute local tangent via neighbor differences
        if i == 0:
            p_next = np.array(center_pts[i+1], dtype=float)
            tangent = p_next - p
        elif i == n-1:
            p_prev = np.array(center_pts[i-1], dtype=float)
            tangent = p - p_prev
        else:
            p_next = np.array(center_pts[i+1], dtype=float)
            p_prev = np.array(center_pts[i-1], dtype=float)
            tangent = p_next - p_prev
        norm = np.linalg.norm(tangent) + 1e-12
        t_hat = tangent / norm
        normal = np.array([-t_hat[1], t_hat[0]])
        half_w = widths[i] / 2.0
        left = p + normal * half_w
        right = p - normal * half_w
        left_pts.append(left)
        right_pts.append(right)
    # build closed polygon: left side then reversed right side
    poly = left_pts + right_pts[::-1]
    # convert to 3D by adding z coordinate (balun_z_position_mm)
    poly3 = [[float(pt[0]), float(pt[1]), float(balun_z_position_mm)] for pt in poly]
    return poly3

# Determine balun pairings: (arms 0 & 2) and (arms 1 & 3)
balun_pairs = [(0,2),(1,3)]

# Define port feed locations: place them just outside substrate along +X and -X for the two baluns
radius_end_mm = cm2mm * radius_end_cm
port_offset_mm = 5.0  # mm beyond substrate edge
port_left = (-radius_end_mm - port_offset_mm, 0.0)
port_right = ( radius_end_mm + port_offset_mm, 0.0)

# for each balun (two total), create two traces connecting port -> each arm contact center
#for idx, pair in enumerate(balun_pairs):
 #   if idx == 0:
  #      port_point = port_left
   # else:
    #    port_point = port_right

    # get arm contact centers
 #   a_idx, b_idx = pair
#    a_center = (sinuous_antenna_arms_list[a_idx].contact_center_position_mm.x,
  #              sinuous_antenna_arms_list[a_idx].contact_center_position_mm.y)
   # b_center = (sinuous_antenna_arms_list[b_idx].contact_center_position_mm.x,
    #            sinuous_antenna_arms_list[b_idx].contact_center_position_mm.y)

    # make S-bend centerlines from port to each arm center
    # we produce an S that curves gently; amplitude chosen relative to substrate radius
#    s_amp = 0.25 * radius_end_mm  # amplitude scaled to substrate radius
#    pts_a, len_a = _create_sbend_centerline(port_point, a_center, npts=240, s_amplitude=s_amp)
#    pts_b, len_b = _create_sbend_centerline(port_point, b_center, npts=240, s_amplitude=s_amp)

#    pts_b, len_b = _create_sbend_centerline(port_point, b_center, npts=240, s_amplitude=s_amp)


    # create length arrays for klopfenstein widths
    # create x positions along length for the klopf function (use arclength param approx by cumulative chord length)
#    def _cumulative_lengths(pts):
 #       cum = [0.0]
  #      for i in range(1, len(pts)):
   #         cum.append(cum[-1] + np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])))
    #    return np.array(cum)
   # cum_a = _cumulative_lengths(pts_a)
   # cum_b = _cumulative_lengths(pts_b)

    # map distances to 0..L and compute widths via klopf_width_profile
    # W1 = start width (near port), W2 = end width (near arm contact)
   # W1 = float(balun_taper_start_width_mm)
   # W2 = float(balun_taper_end_width_mm)
    # Ensure arrays for klopf function are in mm length (use L = cum[-1])
   # x_a = cum_a
   # x_b = cum_b
   # Ltot_a = x_a[-1] if x_a[-1] > 1e-9 else 1.0
   # Ltot_b = x_b[-1] if x_b[-1] > 1e-9 else 1.0

   # widths_a = klopf_width_profile(x_a, Ltot_a, W1, W2, ripple=0.01)
   # widths_b = klopf_width_profile(x_b, Ltot_b, W1, W2, ripple=0.01)

    # Build polygons for each trace
   # poly_a = _poly_from_centerline_with_width(pts_a, widths_a)
   # poly_b = _poly_from_centerline_with_width(pts_b, widths_b)

    # Create polyline objects in HFSS and color them (these will be metal patches)
   # balun_a_name = f"balun_{idx}_trace_A"
   # balun_b_name = f"balun_{idx}_trace_B"

   # balun_a_params = {
    #    "position_list": poly_a,
     #   "segment_type": None,
      #  "cover_surface": True,
       # "close_surface": True,
        #"name": balun_a_name,
       # "matname": None,
       # "xsection_type": None,
       # "xsection_orient": None,
       # "xsection_width": 1,
     #   "xsection_topwidth": 1,
      #  "xsection_height": 1,
       # "xsection_num_seg": 0,
     #   "xsection_bend_type": None,
      #  "non_model": False
    #}
    #balun_b_params = balun_a_params.copy()
    #balun_b_params["position_list"] = poly_b
    #balun_b_params["name"] = balun_b_name

    #balun_a_obj = hfss.modeler.create_polyline(**balun_a_params)
    #balun_b_obj = hfss.modeler.create_polyline(**balun_b_params)

    # color and mark as copper-like metal (but don't change global materials)
    #balun_a_obj.color = [60, 160, 60]  # green tone
    #balun_b_obj.color = [60, 160, 60]

    # Optionally union balun pieces into a single object (not required, but keeps model tidy)
    #try:
     #   hfss.modeler.unite(unite_list=[balun_a_name, balun_b_name], purge=False, keep_originals=False)
    #except Exception:
        # If unite fails (names mismatch or objects not ready), ignore
     #   pass
#---------------------------------------------------------------------------------------


mdl = hfss.modeler
mdl.model_units = "mm"

WIDTH = 400.0
HEIGHT = 130.0
RADIUS = 40.0

START_FRAC = 0.15

# =====================================
# INDEPENDENT TAPER KNOBS
# =====================================
# Exponential taper controls
W0_EXP = 80.0
ALPHA_EXP = 0.014
Z_EXP = -11.829

# Linear taper controls
W0_LIN = 20.0          # <-- adjust linear initial width here (ONLY affects linear)
ALPHA_LIN = 0.010      # <-- used to decide where linear hits MIN (via match-distance logic)
Z_LIN = -11.575      # <-- linear Z height

# =====================================
# POSITION + SIZE KNOBS
# =====================================
ANT_DIAMETER = 82.5
ANT_RADIUS = ANT_DIAMETER / 2.0

BALUN_SCALE = (0.8 * ANT_RADIUS) / (WIDTH / 2.0) * 1.15
BALUN_CENTER_X = 14.0
BALUN_CENTER_Y = -12.3

MIN_WIDTH_MM = 0.65

# opposite copy center
ANT_CENTER_X = 0.0
ANT_CENTER_Y = 0.0

ARC_PTS = 48
LINE_PTS = 80
MAX_POLY_PTS = 1200

# =====================================
# CURVED RECTANGLE CENTERLINE
# =====================================
def rounded_rectangle_path(x0, y0, w, h, r):
    pts = []
    def add(x, y): pts.append([x, y])

    tl = (x0 + r, y0 + r)
    tr = (x0 + w - r, y0 + r)
    br = (x0 + w - r, y0 + h - r)
    bl = (x0 + r, y0 + h - r)

    for t in np.linspace(0, 1, LINE_PTS, endpoint=False):
        add(tl[0] + t * (tr[0] - tl[0]), y0)

    for th in np.linspace(-math.pi/2, 0, ARC_PTS, endpoint=False):
        add(tr[0] + r * math.cos(th), tr[1] + r * math.sin(th))

    for t in np.linspace(0, 1, LINE_PTS, endpoint=False):
        add(x0 + w, tr[1] + t * (br[1] - tr[1]))

    for th in np.linspace(0, math.pi/2, ARC_PTS, endpoint=False):
        add(br[0] + r * math.cos(th), br[1] + r * math.sin(th))

    for t in np.linspace(0, 1, LINE_PTS, endpoint=False):
        add(br[0] - t * (br[0] - bl[0]), y0 + h)

    for th in np.linspace(math.pi/2, math.pi, ARC_PTS, endpoint=False):
        add(bl[0] + r * math.cos(th), bl[1] + r * math.sin(th))

    for t in np.linspace(0, 1, LINE_PTS, endpoint=False):
        add(x0, bl[1] - t * (bl[1] - tl[1]))

    for th in np.linspace(math.pi, 3*math.pi/2, ARC_PTS):
        add(tl[0] + r * math.cos(th), tl[1] + r * math.sin(th))

    return np.array(pts, dtype=float)

# =====================================
# ARC LENGTH + NORMALS
# =====================================
def curve_geometry(pts):
    d = np.diff(pts, axis=0)
    seg = np.linalg.norm(d, axis=1)
    seg[seg == 0] = 1e-12
    s = np.concatenate([[0.0], np.cumsum(seg)])

    t = np.zeros_like(pts)
    t[:-1] = d / seg[:, None]
    t[-1] = t[-2]

    n = np.column_stack([-t[:, 1], t[:, 0]])
    return s, n

# =====================================
# TAPERED POLYGON (EXP)
# =====================================
def tapered_polygon_exp(pts, s, n, start_frac, w0, alpha, min_w_raw):
    s0 = start_frac * s[-1]
    left, right = [], []
    for i in range(len(pts)):
        if s[i] < s0:
            continue
        w_raw = w0 * math.exp(-alpha * (s[i] - s0))
        w_raw = max(w_raw, min_w_raw)
        off = 0.5 * w_raw * n[i]
        left.append(pts[i] + off)
        right.append(pts[i] - off)
    return np.vstack([left, right[::-1]])

# =====================================
# TAPERED POLYGON (LINEAR, independent)
# Uses its OWN (w0, alpha) to decide the distance where it hits MIN
# =====================================
def tapered_polygon_lin_match_self(pts, s, n, start_frac, w0, alpha, min_w_raw):
    s0 = start_frac * s[-1]

    # distance where "its own exponential" would reach min (used only to set linear slope)
    ds_hit = 0.0
    if alpha > 0 and w0 > min_w_raw:
        ds_hit = math.log(w0 / min_w_raw) / alpha
    ds_hit = max(ds_hit, 1e-9)

    slope = (w0 - min_w_raw) / ds_hit

    left, right = [], []
    for i in range(len(pts)):
        if s[i] < s0:
            continue
        ds = (s[i] - s0)
        w_raw = w0 - slope * ds
        w_raw = max(w_raw, min_w_raw)
        off = 0.5 * w_raw * n[i]
        left.append(pts[i] + off)
        right.append(pts[i] - off)

    return np.vstack([left, right[::-1]])

# =====================================
# PROPER POLYGON CLIP x >= xmin  (keeps a valid polygon)
# =====================================
def clip_poly_xmin(poly, xmin):
    p = np.array(poly, dtype=float)
    if len(p) < 3:
        return p

    out = []
    N = len(p)
    for i in range(N):
        A = p[i]
        B = p[(i + 1) % N]
        Ain = (A[0] >= xmin)
        Bin = (B[0] >= xmin)

        if Ain and Bin:
            out.append(B)
        elif Ain and (not Bin):
            dx = B[0] - A[0]
            if abs(dx) < 1e-12:
                continue
            t = (xmin - A[0]) / dx
            out.append(A + t * (B - A))
        elif (not Ain) and Bin:
            dx = B[0] - A[0]
            if abs(dx) < 1e-12:
                out.append(B)
                continue
            t = (xmin - A[0]) / dx
            out.append(A + t * (B - A))
            out.append(B)

    return np.array(out, dtype=float)

# =====================================
# ORIGINAL "keep_right_half" REFERENCE (for positioning only)
# =====================================
def old_keep_right_half_centroid(poly):
    poly = np.array(poly, dtype=float)
    xc = float(np.mean(poly[:, 0]))
    keep = poly[poly[:, 0] >= xc]
    if len(keep) < 3:
        return poly.mean(axis=0), xc
    return keep.mean(axis=0), xc

# =====================================
# CLEANUP / DOWNSAMPLE
# =====================================
def cleanup_poly(poly, tol=1e-6, max_pts=1200):
    p = np.array(poly, dtype=float)
    if len(p) < 3:
        return p

    keep = [p[0]]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - keep[-1]) > tol:
            keep.append(p[i])
    p = np.array(keep, dtype=float)

    if len(p) > max_pts:
        idx = np.linspace(0, len(p) - 1, max_pts, dtype=int)
        p = p[idx]

    return p

# =====================================
# SCALE ABOUT polygon centroid, then TRANSLATE using a reference centroid
# =====================================
def scale_and_move_using_reference(poly, scale_xy, target_xy, ref_centroid_raw):
    p = np.array(poly, dtype=float)
    c_poly = p.mean(axis=0)

    p_scaled = (p - c_poly) * float(scale_xy) + c_poly
    ref_scaled = (ref_centroid_raw - c_poly) * float(scale_xy) + c_poly

    shift = np.array([float(target_xy[0]), float(target_xy[1])]) - ref_scaled
    return p_scaled + shift

# =====================================
# CREATE SHEET + PERFECTE (sheet does not take "material")
# =====================================
def create_sheet(name, poly2d, z, rgb=(255, 0, 0)):
    poly3d = [[float(x), float(y), float(z)] for x, y in poly2d]
    obj = mdl.create_polyline(points=poly3d, close_surface=True, cover_surface=True, name=name)
    if not obj:
        raise RuntimeError(f"{name} failed (create_polyline returned False). Reduce ARC_PTS/LINE_PTS/MAX_POLY_PTS.")
    try:
        obj.color = list(rgb)
    except Exception:
        pass
    hfss.assign_perfecte_to_sheets(assignment=obj.name, name=f"PerfE_{name}", is_infinite_ground=False)
    return obj

# =====================================
# BUILD CENTERLINE
# =====================================
center_pts = rounded_rectangle_path(-WIDTH/2, -HEIGHT/2, WIDTH, HEIGHT, RADIUS)
s, n = curve_geometry(center_pts)

# raw minimum width (before scaling)
min_w_raw = MIN_WIDTH_MM / BALUN_SCALE

# =====================================
# EXP: BUILD, CUT LINE, PLACE
# =====================================
poly_exp_raw = tapered_polygon_exp(center_pts, s, n, START_FRAC, W0_EXP, ALPHA_EXP, min_w_raw)
ref_c_exp_raw, xc_cut = old_keep_right_half_centroid(poly_exp_raw)

poly_exp_raw = clip_poly_xmin(poly_exp_raw, xmin=xc_cut)
poly_exp_raw = cleanup_poly(poly_exp_raw, max_pts=MAX_POLY_PTS)

poly_exp = scale_and_move_using_reference(
    poly_exp_raw, BALUN_SCALE, (BALUN_CENTER_X, BALUN_CENTER_Y), ref_c_exp_raw
)

balun_exp_1 = create_sheet("curved_balun_exp", poly_exp, Z_EXP, (255, 0, 0))

poly_exp_2 = poly_exp.copy()
poly_exp_2[:, 0] = 2*ANT_CENTER_X - poly_exp_2[:, 0]
poly_exp_2[:, 1] = 2*ANT_CENTER_Y - poly_exp_2[:, 1]
balun_exp_2 = create_sheet("curved_balun_exp_2", poly_exp_2, Z_EXP, (0, 0, 255))

# =====================================
# LIN: BUILD using *independent* linear knobs, same cut line, place
# =====================================
poly_lin_raw = tapered_polygon_lin_match_self(center_pts, s, n, START_FRAC, W0_LIN, ALPHA_LIN, min_w_raw)
poly_lin_raw = clip_poly_xmin(poly_lin_raw, xmin=xc_cut)
poly_lin_raw = cleanup_poly(poly_lin_raw, max_pts=MAX_POLY_PTS)

ref_c_lin_raw, _ = old_keep_right_half_centroid(
    tapered_polygon_lin_match_self(center_pts, s, n, START_FRAC, W0_LIN, ALPHA_LIN, min_w_raw)
)

poly_lin = scale_and_move_using_reference(
    poly_lin_raw, BALUN_SCALE, (BALUN_CENTER_X, BALUN_CENTER_Y), ref_c_lin_raw
)

balun_lin_1 = create_sheet("curved_balun_lin", poly_lin, Z_LIN, (200, 200, 0))

poly_lin_2 = poly_lin.copy()
poly_lin_2[:, 0] = 2*ANT_CENTER_X - poly_lin_2[:, 0]
poly_lin_2[:, 1] = 2*ANT_CENTER_Y - poly_lin_2[:, 1]
balun_lin_2 = create_sheet("curved_balun_lin_2", poly_lin_2, Z_LIN, (0, 200, 200))

mdl.fit_all()
print("OK: exp+exp_2 and lin+lin_2 created with independent taper controls.")



balun_lin_1 = create_sheet("curved_balun_lin", poly_lin, Z_LIN, (200, 200, 0))
#balun_lin_2 = create_sheet("curved_balun_lin_2", poly_lin_2, Z_LIN, (0, 200, 200))

#_____________________________________________________________________________


# ==========================================================
# NEW SIMPLE CONNECTION STRIPS (0.65mm) — NO BALUN SHAPE CHANGES
# Creates 2-segment “L” strips when needed (only XY moves), on each layer:
#   - LIN layer z=-1.575:  stop -> end, and (optional) end -> via(0,-1.5)
#   - EXP layer z=-1.625:  stop -> end, and (optional) end -> via(0,+1.5)
#   - EXP straight: (-1.8875,0) -> via(-1.5,0)
#   - LIN straight: ( 1.2375,0) -> via(+1.5,0)
#



TRACE_W = 0.65

# ---- Your points ----
LIN_STOP = (-4.69063162,  0.43427016148, -11.575)
LIN_END  = ( 0.325, -1.8041306092,  -11.575) #0.5979428631

EXP_STOP = (-3.99063162,  0.00027016148, -11.829)
EXP_END  = ( 0.325,        1.6625,        -11.829)

EXP_STRAIGHT_STOP = (-1.8875, 0.0, -11.829)
LIN_STRAIGHT_STOP = ( 1.2375, 0.0, -11.575)

# ---- Via targets (from your earlier via locations) ----
VIA_NY = (0.0, -1.5)   # negative Y
VIA_PY = (0.0,  1.5)   # positive Y
VIA_NX = (-1.5, 0.0)   # negative X
VIA_PX = ( 1.5, 0.0)   # positive X

def _delete_if_exists(name):
    try:
        if name in hfss.modeler.objects:
            hfss.modeler.delete([name])
    except Exception:
        pass

def _make_rect_strip(p0, p1, w, trim_start=0.0, trim_end=0.0):
    """Return a 4-pt polygon (2D) around a segment p0->p1, with optional trimming at ends."""
    a = np.array(p0, dtype=float)
    b = np.array(p1, dtype=float)
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return None

    # unit tangent / normal
    t = v / L
    n = np.array([-t[1], t[0]], dtype=float)

    # clamp trims so we don't invert the segment
    ts = float(max(0.0, trim_start))
    te = float(max(0.0, trim_end))
    if ts + te > 0.95 * L:
        scale = (0.95 * L) / (ts + te)
        ts *= scale
        te *= scale

    # trimmed endpoints
    a2 = a + ts * t
    b2 = b - te * t

    v2 = b2 - a2
    L2 = float(np.linalg.norm(v2))
    if L2 < 1e-9:
        return None

    # keep width sane vs segment length (prevents bow-tie)
    w = float(min(w, 0.80 * L2))
    half = 0.5 * w

    A = a2 + half * n
    B = b2 + half * n
    C = b2 - half * n
    D = a2 - half * n
    return np.array([A, B, C, D], dtype=float)


def _create_sheet_poly(name, poly2d, z, rgb=(220, 220, 220)):
    _delete_if_exists(name)
    pts3d = [[float(x), float(y), float(z)] for x, y in poly2d]
    obj = hfss.modeler.create_polyline(
        points=pts3d,
        close_surface=True,
        cover_surface=True,
        name=name
    )
    if not obj:
        raise RuntimeError(f"{name}: create_polyline returned False")
    try:
        obj.color = list(rgb)
    except Exception:
        pass
    hfss.assign_perfecte_to_sheets(assignment=obj.name, name=f"PerfE_{name}", is_infinite_ground=False)
    return obj


def _trim_segment(p0_xy, p1_xy, trim_start_mm=0.0, trim_end_mm=0.0):
    """Return (p0', p1') shortened along the line by trim_start and trim_end."""
    a = np.array(p0_xy, dtype=float)
    b = np.array(p1_xy, dtype=float)
    v = b - a
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        return p0_xy, p1_xy

    t = v / L
    ts = float(max(0.0, trim_start_mm))
    te = float(max(0.0, trim_end_mm))

    # don’t let it invert
    if ts + te > 0.90 * L:
        scale = (0.90 * L) / (ts + te)
        ts *= scale
        te *= scale

    a2 = a + ts * t
    b2 = b - te * t
    return (float(a2[0]), float(a2[1])), (float(b2[0]), float(b2[1]))




def add_L_strip(name, p0_xyz, p1_xyz, turn_xy=None, width=TRACE_W,
               rgb=(220,220,220), trim_start=0.0, trim_end=0.0):
    """
    Makes either:
      - 1 straight strip p0->p1  (if turn_xy is None)
      - or two straight strips p0->turn->p1  (if turn_xy provided)
    All on same z (XY routing only).
    trim_start trims from the p0 end; trim_end trims from the p1 end (mm).
    """
    x0,y0,z0 = map(float, p0_xyz)
    x1,y1,z1 = map(float, p1_xyz)
    if abs(z0 - z1) > 1e-9:
        raise ValueError(f"{name}: z mismatch ({z0} vs {z1})")

    made = []

    if turn_xy is None:
        (x0t,y0t),(x1t,y1t) = _trim_segment((x0,y0),(x1,y1), trim_start, trim_end)
        poly = _make_rect_strip((x0t,y0t), (x1t,y1t), width)
        if poly is None:
            return None
        made.append(_create_sheet_poly(name, poly, z0, rgb))
        return made

    tx, ty = map(float, turn_xy)

    # segment A: p0 -> turn  (trim_start applies here)
    (ax0,ay0),(ax1,ay1) = _trim_segment((x0,y0),(tx,ty), trim_start, 0.0)
    polyA = _make_rect_strip((ax0,ay0), (ax1,ay1), width)

    # segment B: turn -> p1  (trim_end applies here)
    (bx0,by0),(bx1,by1) = _trim_segment((tx,ty),(x1,y1), 0.0, trim_end)
    polyB = _make_rect_strip((bx0,by0), (bx1,by1), width)

    if polyA is not None:
        made.append(_create_sheet_poly(name + "_A", polyA, z0, rgb))
    if polyB is not None:
        made.append(_create_sheet_poly(name + "_B", polyB, z0, rgb))

    try:
        hfss.modeler.unite(assignment=[name + "_A", name + "_B"], keep_originals=False)
    except Exception:
        pass

    return made


# ----------------------------------------------------------
# 1) LIN: stop -> end (straight)
# ----------------------------------------------------------
add_L_strip(
    name="LIN_stop_to_end",
    p0_xyz=LIN_STOP,
    p1_xyz=LIN_END,
    turn_xy=None,
    width=TRACE_W,
    rgb=(200, 200, 200),

trim_start=0.85,   # <-- pulls the LIN_STOP end inward
    trim_end=0.0

)

# Optional: LIN end -> via (0,-1.5) (short straight)
add_L_strip(
    name="LIN_end_to_viaNY",
    p0_xyz=LIN_END,
    p1_xyz=(VIA_NY[0], VIA_NY[1], LIN_END[2]),
    turn_xy=None,
    width=TRACE_W,
    rgb=(180, 180, 180),


)

# ----------------------------------------------------------
# 2) EXP: stop -> end (straight)
# ----------------------------------------------------------
add_L_strip(
    name="EXP_stop_to_end",
    p0_xyz=EXP_STOP,
    p1_xyz=EXP_END,
    turn_xy=None,
    width=TRACE_W,
    rgb=(200, 200, 200),
    trim_start=0.2
)

# Optional: EXP end -> via (0,+1.5) (short straight)
add_L_strip(
    name="EXP_end_to_viaPY",
    p0_xyz=EXP_END,
    p1_xyz=(VIA_PY[0], VIA_PY[1], EXP_END[2]),
    turn_xy=None,
    width=TRACE_W,
    rgb=(180, 180, 180),
    trim_start=0.8,  # <-- pulls the LIN_STOP end inward
    trim_end=0.0

)

# ----------------------------------------------------------
# 3) EXP straight: stop -> via(-1.5,0)
# ----------------------------------------------------------
add_L_strip(
    name="EXP_straight_to_viaNX",
    p0_xyz=EXP_STRAIGHT_STOP,
    p1_xyz=(VIA_NX[0], VIA_NX[1], EXP_STRAIGHT_STOP[2]),
    turn_xy=None,
    width=TRACE_W,
    rgb=(160, 160, 160),
    trim_start=0.2
)

# ----------------------------------------------------------
# 4) LIN straight: stop -> via(+1.5,0)
# ----------------------------------------------------------
add_L_strip(
    name="LIN_straight_to_viaPX",
    p0_xyz=LIN_STRAIGHT_STOP,
    p1_xyz=(VIA_PX[0], VIA_PX[1], LIN_STRAIGHT_STOP[2]),
    turn_xy=None,
    width=TRACE_W,
    rgb=(160, 160, 160),
    trim_end=0.2
)

hfss.modeler.fit_all()

#______________________________________________________________________________________

# ==========================================================
# BALUN SUBSTRATE CLEARANCE HOLES (between LIN and EXP layers)
# Do this BEFORE creating the vias
# ==========================================================
mdl = hfss.modeler
SUB1_NAME = "Balun_Substrate"  # must match your substrate object name

# Correct via XY positions for the two holes you want:
VIA_PY = (0.0,  1.56)   # EXP +Y
VIA_NX = (-1.56, 0.0)   # EXP -X

# Z span to remove inside Balun_Substrate
Z_LIN_HOLE = -11.575
Z_EXP_HOLE = -11.829

# Use the same drill diameter as via drill (fallback to 0.65 if not defined yet)
CLEAR_D_MM = float(globals().get("DRILL_D_MM", 0.65))
CLEAR_R_MM = 0.5 * CLEAR_D_MM

# Small overcut so subtract doesn’t leave coincident faces
EPS_Z = 0.002  # mm
EPS_R = 0.002  # mm

def _delete_if_exists(name: str):
    try:
        if name in mdl.object_names:
            mdl.delete([name])
    except Exception:
        pass

def subtract_clearance_hole(name, x, y, z1, z2):
    if SUB1_NAME not in mdl.object_names:
        print(f"[WARN] {SUB1_NAME} not found; skipping {name}")
        return

    z0 = float(min(z1, z2) - EPS_Z)
    h  = float(abs(z2 - z1) + 2.0 * EPS_Z)

    _delete_if_exists(name)

    tool = mdl.create_cylinder(
        orientation="Z",
        origin=[float(x), float(y), z0],
        radius=float(CLEAR_R_MM + EPS_R),
        height=float(h),
        name=name,
        material=None,
        num_sides=0
    )
    if not tool:
        raise RuntimeError(f"Failed to create clearance tool: {name}")

    # subtract tool from Balun_Substrate and delete tool afterward
    mdl.subtract(blank_list=[SUB1_NAME], tool_list=[tool.name], keep_originals=False)
    print(f"[OK] Subtracted {name} from {SUB1_NAME} (z {z0:.3f} to {z0+h:.3f})")

# Only the two requested holes:
subtract_clearance_hole("CLR_EXP_PY", VIA_PY[0], VIA_PY[1], Z_LIN_HOLE, Z_EXP_HOLE)
subtract_clearance_hole("CLR_EXP_NX", VIA_NX[0], VIA_NX[1], Z_LIN_HOLE, Z_EXP_HOLE)
#___________________________________________________________________________________________________
# ==========================================================
# V-SPLIT TRIANGLE CUTS + EXTRA LIN VIA-TRIM (curved_balun_lin_3)
# ==========================================================
mdl = hfss.modeler

def _delete_if_exists(nm):
    try:
        if nm in mdl.object_names:
            mdl.delete([nm])
    except Exception:
        pass

def _tri_area_xy(v1, v2, v3):
    return (v2[0] - v1[0])*(v3[1] - v1[1]) - (v3[0] - v1[0])*(v2[1] - v1[1])

def _make_triangle_tool(name, v1, v2, v3):
    _delete_if_exists(name)

    # prevent duplicates
    if (v1[0], v1[1], v1[2]) == (v2[0], v2[1], v2[2]) or \
       (v1[0], v1[1], v1[2]) == (v3[0], v3[1], v3[2]) or \
       (v2[0], v2[1], v2[2]) == (v3[0], v3[1], v3[2]):
        raise ValueError(f"{name}: triangle has duplicate vertices")

    # prevent collinear
    if abs(_tri_area_xy(v1, v2, v3)) < 1e-9:
        raise ValueError(f"{name}: triangle is degenerate/collinear in XY (area~0)")

    tool = mdl.create_polyline(
        points=[
            [float(v1[0]), float(v1[1]), float(v1[2])],
            [float(v2[0]), float(v2[1]), float(v2[2])],
            [float(v3[0]), float(v3[1]), float(v3[2])],
        ],
        close_surface=True,
        cover_surface=True,
        name=name,
        material=None
    )
    if not tool:
        raise RuntimeError(f"{name}: failed to create triangle tool")
    return name

def _subtract_tool_from_targets(tool_name, targets):
    blanks = [t for t in targets if t in mdl.object_names]
    if not blanks:
        print(f"[WARN] No targets found for subtraction with {tool_name}: {targets}")
        return
    mdl.subtract(blank_list=blanks, tool_list=[tool_name], keep_originals=False)

def _mirror_vertices_xy(Vs, flip_x=False, flip_y=False, x_mirror=0.0, y_mirror=0.0):
    out = []
    for x, y, z in Vs:
        if flip_x:
            x = 2.0*x_mirror - x
        if flip_y:
            y = 2.0*y_mirror - y
        out.append([x, y, z])
    return out


# ----------------------------
# REQUIRED COORDINATES
# ----------------------------

# LIN triangle (layer z = -11.575)
LIN_Z      = -11.575
LIN_X_TIP  = -2.116229809
LIN_X_BACK = -3.8
LIN_Y_TOP  =  0.35875
LIN_Y_BOT  = -0.35875

LIN_FLIP_X = False
LIN_FLIP_Y = False
LIN_X_MIRROR = 0.0
LIN_Y_MIRROR = 0.0

# EXP triangle (layer z = -11.829)
EXP_Z      = -11.829
EXP_X_TIP  = -2.116229809
EXP_X_BACK = -3.8
EXP_Y_TOP  =  0.35875
EXP_Y_BOT  = -0.35875

EXP_FLIP_X = False
EXP_FLIP_Y = True
EXP_X_MIRROR = 0.0
EXP_Y_MIRROR = 0.0


# ----------------------------
# Build & mirror vertices
# ----------------------------
V1_LIN = [LIN_X_TIP,  LIN_Y_TOP, LIN_Z]
V2_LIN = [LIN_X_BACK, LIN_Y_TOP, LIN_Z]
V3_LIN = [LIN_X_TIP,  LIN_Y_BOT, LIN_Z]

V1_EXP = [EXP_X_TIP,  EXP_Y_TOP, EXP_Z]
V2_EXP = [EXP_X_BACK, EXP_Y_TOP, EXP_Z]
V3_EXP = [EXP_X_TIP,  EXP_Y_BOT, EXP_Z]

VLIN = _mirror_vertices_xy([V1_LIN, V2_LIN, V3_LIN], LIN_FLIP_X, LIN_FLIP_Y, LIN_X_MIRROR, LIN_Y_MIRROR)
VEXP = _mirror_vertices_xy([V1_EXP, V2_EXP, V3_EXP], EXP_FLIP_X, EXP_FLIP_Y, EXP_X_MIRROR, EXP_Y_MIRROR)

tool_lin = _make_triangle_tool("Tool_Vsplit_LIN", VLIN[0], VLIN[1], VLIN[2])
tool_exp = _make_triangle_tool("Tool_Vsplit_EXP", VEXP[0], VEXP[1], VEXP[2])

LIN_TARGETS = [
    "curved_balun_lin",
    "curved_balun_lin_2",
    "LIN_stop_to_end",
    "LIN_end_to_viaNY",
    "LIN_straight_to_viaPX",
]
EXP_TARGETS = [
    "curved_balun_exp",
    "curved_balun_exp_2",
    "EXP_stop_to_end",
    "EXP_end_to_viaPY",
    "EXP_straight_to_viaNX",
]

_subtract_tool_from_targets(tool_lin, LIN_TARGETS)
_subtract_tool_from_targets(tool_exp, EXP_TARGETS)

print("[OK] LIN/EXP V-split triangle cuts applied.")


# ==========================================================
# SECOND TRIANGLE -> subtract ONLY from curved_balun_lin_3
# ==========================================================
VIA2_V1 = [-3.723,  0.35875,  -11.575]
VIA2_V2 = [-1.921,  0.35875,  -11.575]
VIA2_V3 = [-1.921, -0.35875,  -11.575]

# --- Expand directions you want ---
DX_POS_X = 0.05   # mm: expand to +X (increase right-side x)
DY_NEG_Y = 0.20   # mm: expand to -Y (make bottom y more negative)

# +X expansion: move the right vertical edge (v2 and v3) to the right
VIA2_V2[0] += DX_POS_X
VIA2_V3[0] += DX_POS_X

# -Y expansion: move the bottom point (v3) down
VIA2_V3[1] -= DY_NEG_Y

tool_lin_via2 = _make_triangle_tool("Tool_LIN_ViaTrim_2", VIA2_V1, VIA2_V2, VIA2_V3)
_subtract_tool_from_targets(tool_lin_via2, ["curved_balun_lin_3"])


tool_lin_via2 = _make_triangle_tool("Tool_LIN_ViaTrim_2", VIA2_V1, VIA2_V2, VIA2_V3)

_subtract_tool_from_targets(tool_lin_via2, ["curved_balun_lin_3"])
print("[OK] Second triangle trim subtracted from curved_balun_lin_3 (if it exists).")



# ==========================================================
# RECTANGLE/QUAD CUT on curved_balun_lin_3
# ==========================================================
mdl = hfss.modeler

def _delete_if_exists(nm):
    try:
        if nm in mdl.object_names:
            mdl.delete([nm])
    except Exception:
        pass

def _order_points_clockwise_xy(pts):
    import math
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))

def _make_quad_tool(name, corners):
    _delete_if_exists(name)
    pts = [[float(p[0]), float(p[1]), float(p[2])] for p in corners]
    # reorder so polygon doesn't self-cross
    pts = _order_points_clockwise_xy(pts)
    tool = mdl.create_polyline(points=pts, close_surface=True, cover_surface=True, name=name, material=None)
    if not tool:
        raise RuntimeError(f"{name}: failed to create quad tool")
    return name

# corners (fixed the z typo)
C1 = [-1.88,         0.36,         -11.575]
C2 = [-1.878948683, -0.35875,      -11.575]
C3 = [-2.116229813, -0.3587499983, -11.575]
C4 = [-2.116229809,  0.35875,      -11.575]

tool_rect = _make_quad_tool("Tool_LIN_RectTrim_1", [C1, C2, C3, C4])

if "curved_balun_lin_3" in mdl.object_names:
    mdl.subtract(blank_list=["curved_balun_lin_3"], tool_list=[tool_rect], keep_originals=False)
    print("[OK] Rectangle trim subtracted from curved_balun_lin_3.")
else:
    print("[WARN] curved_balun_lin_3 not found at this point.")

    # ==========================================================
    # CENTER NOTCH TOOL (VISIBLE) + SUBTRACT FROM curved_balun_lin_3
    # ==========================================================
    mdl = hfss.modeler

    TARGET = "curved_balun_lin_3"  # <-- your requested target
    TOOL = "Tool_LIN3_CenterNotch"  # tool will remain visible

    # notch size (your numbers)
    X_HALF = 1.235  # cuts from -1.235 to +1.235
    Y_HALF = 0.7  # cuts from -0.7   to +0.7

    KEEP_TOOL_VISIBLE = True  # keep the tool after subtract so you can SEE it


    def _delete_if_exists(nm):
        try:
            if nm in mdl.object_names:
                mdl.delete([nm])
        except Exception:
            pass


    def _bbox(obj_name):
        try:
            return mdl.objects[obj_name].bounding_box
        except Exception:
            return mdl.get_object_bounding_box(obj_name)


    def _z_of_sheet(obj_name: str) -> float:
        bb = _bbox(obj_name)
        return 0.5 * (float(bb[2]) + float(bb[5]))


    if TARGET not in mdl.object_names:
        raise RuntimeError(f"[ERROR] Target '{TARGET}' not found. Check the name.")

    # Use the REAL z-plane of the target sheet
    z_cut = _z_of_sheet(TARGET)
    print(f"[DEBUG] {TARGET} z = {z_cut:.6f} mm")

    # Make/replace the tool rectangle
    _delete_if_exists(TOOL)
    tool = mdl.create_rectangle(
        orientation="XY",
        origin=[-X_HALF, -Y_HALF, float(z_cut)],
        sizes=[2.0 * X_HALF, 2.0 * Y_HALF],
        name=TOOL,
        material=None
    )
    if not tool:
        raise RuntimeError("[ERROR] Failed to create notch tool rectangle.")

    # Make it obvious in the 3D view
    try:
        mdl.objects[TOOL].color = [255, 0, 255]  # magenta
        mdl.objects[TOOL].transparency = 0.6
    except Exception:
        pass

    # Subtract from target
    mdl.subtract(
        blank_list=[TARGET],
        tool_list=[TOOL],
        keep_originals=KEEP_TOOL_VISIBLE
    )

    # Re-apply PerfectE on the modified sheet (booleans can break old boundaries)
    try:
        hfss.assign_perfecte_to_sheets(
            assignment=TARGET,
            name=f"PerfE_{TARGET}",
            is_infinite_ground=False
        )
    except Exception:
        pass

    mdl.fit_all()
    print(f"[OK] Subtracted {TOOL} from {TARGET} (tool kept={KEEP_TOOL_VISIBLE}).")

# ==========================================================
# PLATED VIA BARRELS (PCB-STYLE): DRILL + COPPER LINING
# Uses your via XY centers; makes hollow copper barrel down to each layer.
# ==========================================================



mdl = hfss.modeler
mdl.model_units = "mm"

# ----------------------------
# Via centers (your coordinates, relative to origin)
# ----------------------------
VIA_NY = (0.0, -1.56)   # negative Y
VIA_PY = (0.0,  1.56)   # positive Y
VIA_NX = (-1.56, 0.0)   # negative X
VIA_PX = ( 1.56, 0.0)   # positive X

# ----------------------------
# Via Z targets (match your copper layers)
# ----------------------------
Z_TOP_CU = 0.0      # antenna arms are drawn at z=0 in your code
Z_LIN    = -11.575   # linear layer
Z_EXP    = -11.829   # exponential layer

# Map each via to which layer it should reach
# (based on your strip connections: LIN goes to NY+PX, EXP goes to PY+NX)
VIA_LIST = [
    ("VIA_LIN_NY", VIA_NY[0], VIA_NY[1], Z_LIN),
    ("VIA_LIN_PX", VIA_PX[0], VIA_PX[1], Z_LIN),
    ("VIA_EXP_PY", VIA_PY[0], VIA_PY[1], Z_EXP),
    ("VIA_EXP_NX", VIA_NX[0], VIA_NX[1], Z_EXP),
]

# ----------------------------
# Geometry knobs (edit these)
# ----------------------------
DRILL_D_MM   = 0.65   # finished drill diameter (typical 0.25~0.40mm)
PLATE_T_MM   = 0.025  # copper plating thickness (25um ~ common)

PAD_TOP_D_MM = 1.10
PAD_BOT_D_MM = 0.80
COPPER_MAT   = "copper"

# Objects that should receive drilled holes (we try these if they exist)
DIELECTRIC_NAMES = [
    "dielectric_slab",
    "Balun_Substrate",
    "Balun_Substrate_2",
]

# Optional: try to "unite" bottom pads with these strip objects if they exist
# (safe: failures are ignored)
BOTTOM_PAD_UNITE_TARGETS = {
    "VIA_LIN_NY": ["LIN_end_to_viaNY", "LIN_stop_to_end", "curved_balun_lin", "curved_balun_lin_2"],
    "VIA_LIN_PX": ["LIN_straight_to_viaPX", "curved_balun_lin", "curved_balun_lin_2"],
    "VIA_EXP_PY": ["EXP_end_to_viaPY", "EXP_stop_to_end", "curved_balun_exp", "curved_balun_exp_2"],
    "VIA_EXP_NX": ["EXP_straight_to_viaNX", "curved_balun_exp", "curved_balun_exp_2"],
}

# ==========================================================
# VIA ANNULAR PADS (donut pads) so the hole is NOT "capped"
# Paste this right after VIA_NY/VIA_PY/VIA_NX/VIA_PX definitions
# ==========================================================

mdl = hfss.modeler

# Use the SAME orientation that works for your ground circle.
# If your circles ever rotate again, switch PAD_ORIENT between "XY" and "Z".
PAD_ORIENT = "XY"

def make_annular_via_pad(name, x, y, z, pad_outer_r_mm, hole_r_mm, mat="copper"):
    # delete if rerunning
    for nm in [name, name + "_outer", name + "_inner"]:
        try:
            if nm in mdl.object_names:
                mdl.delete([nm])
        except Exception:
            pass

    outer = mdl.create_circle(
        orientation=PAD_ORIENT,
        origin=[float(x), float(y), float(z)],
        radius=float(pad_outer_r_mm),
        num_sides=0,
        is_covered=True,      # filled disk (we'll cut the hole out)
        name=name + "_outer",
        material=mat
    )
    if not outer:
        raise RuntimeError(f"Failed outer pad: {name}")

    inner = mdl.create_circle(
        orientation=PAD_ORIENT,
        origin=[float(x), float(y), float(z)],
        radius=float(hole_r_mm),
        num_sides=0,
        is_covered=True,      # filled disk used as the cut tool
        name=name + "_inner",
        material=mat
    )
    if not inner:
        raise RuntimeError(f"Failed inner cut: {name}")

    mdl.subtract(blank_list=[name + "_outer"], tool_list=[name + "_inner"], keep_originals=False)

    # rename result to 'name'
    try:
        mdl.rename(name + "_outer", name)
    except Exception:
        pass

    # optional: PerfectE on pad sheet (only if you want sheets as PEC)
    try:
        hfss.assign_perfecte_to_sheets(
            assignment=name,
            name=f"PerfE_{name}",
            is_infinite_ground=False
        )
    except Exception:
        pass

    return name


# ----------------------------
# Pad sizes (edit these only)
# ----------------------------
VIA_HOLE_DIAM_MM = 0.65
HOLE_R_MM = VIA_HOLE_DIAM_MM / 2.0

# outer pad radius (donut outer edge). Typical: hole radius + 0.25mm..0.75mm
PAD_OUTER_R_MM = HOLE_R_MM    # <-- change this as you like

# ----------------------------
# Create pads on BOTH copper layers at your via XY positions
# ----------------------------
# linear layer pads (z = -10.575)
make_annular_via_pad("Pad_LIN_NY", VIA_NY[0], VIA_NY[1], -11.575, PAD_OUTER_R_MM, HOLE_R_MM)
make_annular_via_pad("Pad_LIN_PX", VIA_PX[0], VIA_PX[1], -11.575, PAD_OUTER_R_MM, HOLE_R_MM)

# exponential layer pads (z = -10.829)
make_annular_via_pad("Pad_EXP_PY", VIA_PY[0], VIA_PY[1], -11.829, PAD_OUTER_R_MM, HOLE_R_MM)
make_annular_via_pad("Pad_EXP_NX", VIA_NX[0], VIA_NX[1], -11.829, PAD_OUTER_R_MM, HOLE_R_MM)



def _delete_if_exists(name: str):
    try:
        if name in mdl.object_names:
            mdl.delete([name])
    except Exception:
        pass

def _safe_subtract(blank, tool):
    try:
        if blank in mdl.object_names and tool in mdl.object_names:
            mdl.subtract(blank_list=[blank], tool_list=[tool], keep_originals=False)
    except Exception:
        pass

def _safe_unite(names):
    try:
        names = [n for n in names if n in mdl.object_names]
        if len(names) >= 2:
            mdl.unite(assignment=names, keep_originals=False)
    except Exception:
        pass

# --- small helper for z of the balun substrate top (inside your via section) ---
def _get_zmax(obj_name: str) -> float:
    try:
        bb = mdl.objects[obj_name].bounding_box
    except Exception:
        bb = mdl.get_object_bounding_box(obj_name)
    return float(bb[5])  # zmax

# (optional but recommended) tiny clearance so surfaces aren't coincident
VIA_HOLE_CLEAR_MM = 0.02   # mm

def make_plated_via(via_name: str, x: float, y: float, z_top: float, z_bot: float):
    """
    Creates (SOLID VIA version):
      1) drill hole cylinder (tool)  [radius = via radius]
      2) subtract hole from any dielectrics it intersects
      3) solid copper via cylinder (NO hollowing)
      4) optional pads at z_top and z_bot
    """
    r_in  = DRILL_D_MM / 2.0
    pad_top_r = PAD_TOP_D_MM / 2.0
    pad_bot_r = PAD_BOT_D_MM / 2.0

    # SOLID via radius: match the drilled hole radius to avoid dielectric overlap
    r_via = r_in

    z0 = float(min(z_top, z_bot))
    h  = float(abs(z_top - z_bot))
    if h < 1e-6:
        raise ValueError(f"{via_name}: z span too small")

    # ---- cleanup if rerun ----
    for suffix in ["_drill", "_solid", "_pad_top", "_pad_bot"]:
        _delete_if_exists(via_name + suffix)

    # ---- drill tool (air / void tool) ----
    drill = mdl.create_cylinder(
        orientation="Z",
        origin=[float(x), float(y), z0],
        radius=float(r_via),
        height=float(h),
        name=via_name + "_drill",
        material=None,
        num_sides=0
    )
    if not drill:
        raise RuntimeError(f"{via_name}: failed to create drill cylinder")

    # ---- subtract drill from dielectrics (whichever exist) ----
    for dname in DIELECTRIC_NAMES:
        _safe_subtract(dname, drill.name)

    # ---- SOLID copper via (NO inner tool, NO subtract) ----
    solid = mdl.create_cylinder(
        orientation="Z",
        origin=[float(x), float(y), z0],
        radius=float(r_via),
        height=float(h),
        name=via_name + "_solid",
        material=COPPER_MAT,
        num_sides=0
    )
    if not solid:
        raise RuntimeError(f"{via_name}: failed to create solid via cylinder")

    # Ensure copper isn't solved inside (typical for conductors)
    try:
        mdl.objects[solid.name].solve_inside = False
    except Exception:
        pass

    # ---- optional pads (sheets) at top & bottom ----
    pad_top = mdl.create_circle(
        orientation="XY",
        origin=[float(x), float(y), float(z_top)],
        radius=float(pad_top_r),
        num_sides=0,
        is_covered=True,
        name=via_name + "_pad_top",
        material=COPPER_MAT
    )
    pad_bot = mdl.create_circle(
        orientation="XY",
        origin=[float(x), float(y), float(z_bot)],
        radius=float(pad_bot_r),
        num_sides=0,
        is_covered=True,
        name=via_name + "_pad_bot",
        material=COPPER_MAT
    )

    try:
        hfss.assign_perfecte_to_sheets(assignment=pad_top.name, name=f"PerfE_{pad_top.name}", is_infinite_ground=False)
        hfss.assign_perfecte_to_sheets(assignment=pad_bot.name, name=f"PerfE_{pad_bot.name}", is_infinite_ground=False)
    except Exception:
        pass

    _safe_unite([pad_bot.name] + BOTTOM_PAD_UNITE_TARGETS.get(via_name, []))

    return solid.name


# ----------------------------
# Build all vias
# ----------------------------
for (nm, x, y, z_target) in VIA_LIST:
    make_plated_via(nm, x, y, Z_TOP_CU, z_target)

mdl.fit_all()
print("OK: SOLID copper vias created at VIA_NY/VIA_PY/VIA_NX/VIA_PX.")

#-------------------------------------------------------------


# ==========================================================
# UNITE BALUN METAL SHEETS PER LAYER (LIN together, EXP together)
# ==========================================================
mdl = hfss.modeler

def unite_keep_first(new_name: str, names):
    # keep only existing objects
    names = [n for n in names if n in mdl.object_names]
    if len(names) == 0:
        print(f"[WARN] Unite '{new_name}': nothing found")
        return None
    if len(names) == 1:
        # rename if requested
        if names[0] != new_name:
            try:
                mdl.rename(names[0], new_name)
                print(f"[OK] Renamed {names[0]} -> {new_name}")
                return new_name
            except Exception:
                return names[0]
        return names[0]

    # Unite keeps the FIRST name; set keep_originals=False to actually merge
    first = names[0]
    try:
        mdl.unite(assignment=names, keep_originals=False)
        # rename merged result to new_name
        if first != new_name and first in mdl.object_names:
            try:
                mdl.rename(first, new_name)
            except Exception:
                pass
        print(f"[OK] United {len(names)} objs -> {new_name}")
        return new_name if new_name in mdl.object_names else first
    except Exception as e:
        print(f"[WARN] Unite '{new_name}' failed: {e}")
        return first

# ---- LIN layer (z = -11.575) ----
LIN_UNITE_LIST = [
    "curved_balun_lin", "curved_balun_lin_2",
    "LIN_stop_to_end", "LIN_end_to_viaNY",
    "LIN_straight_to_viaPX",
    "Pad_LIN_NY", "Pad_LIN_PX",
]
LIN_METAL = unite_keep_first("BALUN_LIN_METAL", LIN_UNITE_LIST)

mdl = hfss.modeler

def _zmid(obj):
    bb = mdl.objects[obj].bounding_box if obj in mdl.objects else mdl.get_object_bounding_box(obj)
    return 0.5 * (float(bb[2]) + float(bb[5]))

a = "curved_balun_lin_3"
b = "VIA_LIN_NY_pad_bot"

if a in mdl.object_names and b in mdl.object_names:
    za = _zmid(a)
    zb = _zmid(b)
    if abs(za - zb) > 1e-6:
        print(f"[WARN] Not uniting {a} and {b}: different Z (za={za}, zb={zb})")
    else:
        # keep the first name (a) as the merged result
        mdl.unite(assignment=[a, b], keep_originals=False)
        print(f"[OK] United {b} into {a}")

        # Optional but recommended: re-assign PerfectE to the merged sheet name
        try:
            hfss.assign_perfecte_to_sheets(
                assignment=a,
                name=f"PerfE_{a}",
                is_infinite_ground=False
            )
        except Exception:
            pass
else:
    print(f"[WARN] Missing object(s): {a} or {b}")



# ---- EXP layer (z = -11.829) ----
EXP_UNITE_LIST = [
    "curved_balun_exp", "curved_balun_exp_2",
    "EXP_stop_to_end", "EXP_end_to_viaPY",
    "EXP_straight_to_viaNX",
    "Pad_EXP_PY", "Pad_EXP_NX",
]
EXP_METAL = unite_keep_first("BALUN_EXP_METAL", EXP_UNITE_LIST)

mdl.fit_all()
print("[DONE] LIN_METAL =", LIN_METAL, " EXP_METAL =", EXP_METAL)






#----------------------------------------------------------------------------------

# ==========================================================
# CENTER NOTCH CUTOUT FOR curved_balun_lin_3
# (x: -1.235..+1.235, y: -0.7..+0.7, centered at origin)
# ==========================================================
mdl = hfss.modeler
mdl.model_units = "mm"

TARGET = "curved_balun_lin_3"

X_HALF = 1.235  # mm
Y_HALF = 0.700  # mm

def _bbox(name):
    try:
        return mdl.objects[name].bounding_box  # [xmin,ymin,zmin,xmax,ymax,zmax]
    except Exception:
        return mdl.get_object_bounding_box(name)

def _sheet_z(name):
    bb = _bbox(name)
    return 0.5 * (float(bb[2]) + float(bb[5]))

if TARGET not in mdl.object_names:
    raise RuntimeError(f"Target '{TARGET}' not found. Check the object name.")

# Put the cut tool exactly on the same Z as the sheet (so subtraction works even if it's at -11.575mm)
z_cut = _sheet_z(TARGET)
print(f"[DEBUG] {TARGET} z_cut = {z_cut:.6f} mm")

tool_name = f"{TARGET}_center_notch_tool"

# cleanup on rerun
try:
    if tool_name in mdl.object_names:
        mdl.delete([tool_name])
except Exception:
    pass

# Rectangle centered at origin
notch = mdl.create_rectangle(
    orientation="XY",
    origin=[-X_HALF, -Y_HALF, z_cut],
    sizes=[2*X_HALF, 2*Y_HALF],
    name=tool_name,
    material=None
)

# Subtract notch from curved_balun_lin_3
mdl.subtract(
    blank_list=[TARGET],
    tool_list=[tool_name],
    keep_originals=False
)

# If tool survived, delete it
try:
    if tool_name in mdl.object_names:
        mdl.delete([tool_name])
except Exception:
    pass

# Re-apply PerfectE if you are using PEC sheets (safe to do even if already assigned)
try:
    hfss.assign_perfecte_to_sheets(
        assignment=TARGET,
        name=f"PerfE_{TARGET}",
        is_infinite_ground=False
    )
except Exception:
    pass

mdl.fit_all()
print("[OK] Center notch cut applied to curved_balun_lin_3.")


# ==========================================================
# END BALUN BLOCK
# ==========================================================



# ==========================================================
# EXCITATIONS: Lumped ports on widest end of each taper sheet
# Reference = LOCAL RETURN PATCH (no Balun_Ground needed)
# ==========================================================
mdl = hfss.modeler

TAPER_SHEETS = [
    "curved_balun_lin",
    "curved_balun_lin_2",
    "curved_balun_exp",
    "curved_balun_exp_2",
]

PORT_Z_IMPEDANCE = 50.0

# --- knobs ---
# Put return patch this far below the signal sheet.
# If you have SUB1_THK_MM defined, use it; otherwise keep 0.254mm.
PORT_RETURN_GAP_MM = float(globals().get("SUB1_THK_MM", 0.254))

# Return patch size (make it small/local)
RETURN_PATCH_W_MM = 6.0
RETURN_PATCH_H_MM = 6.0

def _bbox(obj_name: str):
    try:
        return mdl.objects[obj_name].bounding_box
    except Exception:
        return mdl.get_object_bounding_box(obj_name)

def _z_of_sheet(obj_name: str) -> float:
    bb = _bbox(obj_name)
    return 0.5 * (float(bb[2]) + float(bb[5]))

def _pick_wide_end_xy_from_bbox(obj_name: str):
    bb = _bbox(obj_name)
    xmin, ymin, xmax, ymax = float(bb[0]), float(bb[1]), float(bb[3]), float(bb[4])
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    return max(corners, key=lambda p: p[0]**2 + p[1]**2)

def _delete_if_exists(nm: str):
    try:
        if nm in mdl.object_names:
            mdl.delete([nm])
    except Exception:
        pass

created_ports = []

for sheet in TAPER_SHEETS:
    if sheet not in mdl.object_names:
        print(f"[WARN] Skipping port on '{sheet}' (object not found).")
        continue

    # pick a wide-end point
    x_p, y_p = _pick_wide_end_xy_from_bbox(sheet)
    z_sig = _z_of_sheet(sheet)

    # make local return patch just below the sheet
    z_ref = float(z_sig - PORT_RETURN_GAP_MM)
    ref_name = f"RET_{sheet}"
    _delete_if_exists(ref_name)

    ref_patch = mdl.create_rectangle(
        orientation="XY",
        origin=[float(x_p - RETURN_PATCH_W_MM/2), float(y_p - RETURN_PATCH_H_MM/2), z_ref],
        sizes=[float(RETURN_PATCH_W_MM), float(RETURN_PATCH_H_MM)],
        name=ref_name,
        material="copper"
    )
    if not ref_patch:
        raise RuntimeError(f"Failed to create return patch '{ref_name}'")

    # Make return patch a PEC sheet
    hfss.assign_perfecte_to_sheets(
        assignment=ref_name,
        name=f"PerfE_{ref_name}",
        is_infinite_ground=False
    )

    # unique port name
    base_name = f"LP_{sheet}"
    port_name = base_name
    if port_name in hfss.excitation_names:
        k = 1
        while f"{base_name}_{k}" in hfss.excitation_names:
            k += 1
        port_name = f"{base_name}_{k}"

    # integration line: signal -> return patch
    int_line = [[float(x_p), float(y_p), float(z_sig)], [float(x_p), float(y_p), float(z_ref)]]

    hfss.lumped_port(
        assignment=sheet,
        reference=ref_name,
        create_port_sheet=True,
        integration_line=int_line,
        impedance=PORT_Z_IMPEDANCE,
        name=port_name,
        renormalize=True,
        deembed=0,
        terminals_rename=True
    )

    created_ports.append(port_name)
    print(f"[OK] {port_name}: {sheet} -> {ref_name}  z {z_sig:.3f} -> {z_ref:.3f}")

print("[DONE] Lumped ports created:", created_ports)

# ==========================================================



#----------------------------------------------------------------------

# BALUN CODE

# --- ELECTRICAL PARAMETER DEFINITION ---
#W1_slot = 0.5   # mm (Narrow end gap, reduced for better coupling to 50 Ohm feed)
#W2_slot = 18.0  # mm (Wide end gap, near antenna arms/edge - Tune this based on antenna Z_bal)
#L_taper_mm = 25.0 # mm (Electrically mandated length for 2 GHz operation on Kapton)
#rect_width_mm = 20.0 # Total width of the copper sheet used for initial placement

#W50 = 0.25      # mm (Microstrip line width for 50 Ohm on 0.125mm Kapton)
#Feed_Line_L = 3.0 # mm (Length of the straight 50 Ohm feed section before taper)

# Retrieve key geometric parameters (must be defined previously)
#substrate_radius_mm = cm2mm * dielectric_diameter_cm / 2.0
#balun_z = -cm2mm * dielectric_height_cm # Z-position of the copper layer
#balun_thickness = 0.017 # Copper thickness

# Calculated Dimensions
#L_taper = L_taper_mm # Use the electrically correct length
#half_w_rect = rect_width_mm / 2.0

# Sampling for curve resolution
#samples = 300
#xs = np.linspace(0, L_taper, samples)

# ----------------- 1. CALCULATE EXPONENTIAL CURVATURE -----------------
#W_start = W1_slot
#W_end = W2_slot
#exponent = xs / L_taper
#ws = W_start * np.power((W_end / W_start), exponent)

# Set units
#hfss.modeler.model_units = "mm"

# --- BALUN 1: 0 DEGREES (Points East, Taper runs along -X) ---
#bi = 0
#temp_rect_name = f"temp_removal_rect_{bi}"
#conductor_name = f"balun_strip_line_{bi}"

#balun_center_x = substrate_radius_mm
#balun_center_y = 0.0
#Inner_End_X = balun_center_x - L_taper

# 1. Create the temporary large rectangular sheet
#rect_pts_0 = [
 #   [balun_center_x - L_taper, balun_center_y + half_w_rect, balun_z],
  #  [balun_center_x - L_taper, balun_center_y - half_w_rect, balun_z],
   # [balun_center_x, balun_center_y - half_w_rect, balun_z],
    #[balun_center_x, balun_center_y + half_w_rect, balun_z]
#]
# FIX: Use 'points' instead of 'position_list'
#rect_poly_0 = hfss.modeler.create_polyline(points=rect_pts_0, cover_surface=True,
 #                                           close_surface=True, name=temp_rect_name)

# 2. Define and Create the Exponential Balun Conductor (The desired shape)
#slot_pts_0 = []
#for i in range(samples):
 #   x_pos = Inner_End_X + xs[i]
  #  half_s = ws[i] / 2.0
   # slot_pts_0.append([x_pos, balun_center_y + half_s, balun_z])
#for i in reversed(range(samples)):
 #   x_pos = Inner_End_X + xs[i]
  #  half_s = ws[i] / 2.0
   # slot_pts_0.append([x_pos, balun_center_y - half_s, balun_z])

# FIX: Use 'points' and 'material' arguments
#slot_poly_0 = hfss.modeler.create_polyline(
 #   points=slot_pts_0,
  #  cover_surface=True,
   # close_surface=True,
    #name=conductor_name,
    #material=ground_plane_material_name # Use modern argument
#)
#slot_poly_0.thickness = balun_thickness # Set thickness separately

# DELETE the surrounding rectangle
#hfss.modeler.delete([temp_rect_name])


# --- BALUN 2: 180 DEGREES (Points West, Taper runs along +X) ---
#bi = 1
#temp_rect_name = f"temp_removal_rect_{bi}"
#conductor_name = f"balun_strip_line_{bi}"

#balun_center_x = -substrate_radius_mm
#balun_center_y = 0.0
#Inner_End_X = balun_center_x + L_taper

# 1. Create the temporary large rectangular sheet
#rect_pts_1 = [
 #   [balun_center_x + L_taper, balun_center_y + half_w_rect, balun_z],
  #  [balun_center_x + L_taper, balun_center_y - half_w_rect, balun_z],
   # [balun_center_x, balun_center_y - half_w_rect, balun_z],
    #[balun_center_x, balun_center_y + half_w_rect, balun_z]
#]
# FIX: Use 'points' instead of 'position_list'
#rect_poly_1 = hfss.modeler.create_polyline(points=rect_pts_1, cover_surface=True,
 #                                           close_surface=True, name=temp_rect_name)

# 2. Define and Create the Exponential Balun Conductor (The desired shape)
#slot_pts_1 = []
#for i in range(samples):
 #   x_pos = Inner_End_X - xs[i]
  #  half_s = ws[i] / 2.0
   # slot_pts_1.append([x_pos, balun_center_y + half_s, balun_z])
#for i in reversed(range(samples)):
 #   x_pos = Inner_End_X - xs[i]
  #  half_s = ws[i] / 2.0
   # slot_pts_1.append([x_pos, balun_center_y - half_s, balun_z])

# FIX: Use 'points' and 'material' arguments
#slot_poly_1 = hfss.modeler.create_polyline(
 #   points=slot_pts_1,
  #  cover_surface=True,
   # close_surface=True,
    #name=conductor_name,
    #material=ground_plane_material_name # Use modern argument
#)
#slot_poly_1.thickness = balun_thickness # Set thickness separately

# DELETE the surrounding rectangle
#hfss.modeler.delete([temp_rect_name])

# ----------------- 3. CREATE AND UNITE 50 OHM FEED LINES -----------------

# Feed Line 0 (Connects to balun_strip_line_0)
#Inner_End_X_0 = balun_center_x - L_taper
# FIX: Use explicit orientation and 2D sizes
#feed_rect_0 = hfss.modeler.create_rectangle(
 #   orientation="XY",
  #  position=[Inner_End_X_0 - Feed_Line_L, balun_center_y - W50/2.0, balun_z],
   # sizes=[Feed_Line_L, W50], # 2D sizes for rectangle
    #name="50Ohm_Feed_Line_0"
#)
#feed_rect_0.material_name = ground_plane_material_name
#feed_rect_0.thickness = balun_thickness

# Unite the new feed line to the balun arm
#hfss.modeler.unite(assignment=["50Ohm_Feed_Line_0", "balun_strip_line_0"])

# Feed Line 1 (Connects to balun_strip_line_1)
#Inner_End_X_1 = balun_center_x + L_taper
# FIX: Use explicit orientation and 2D sizes
#feed_rect_1 = hfss.modeler.create_rectangle(
 #   orientation="XY",
  #  position=[Inner_End_X_1, balun_center_y - W50/2.0, balun_z], # NOTE: Position fixed to align correctly
   # sizes=[Feed_Line_L, W50], # 2D sizes for rectangle
    #name="50Ohm_Feed_Line_1"
#)
#feed_rect_1.material_name = ground_plane_material_name
#feed_rect_1.thickness = balun_thickness

# Unite the new feed line to the balun arm
#hfss.modeler.unite(assignment=["50Ohm_Feed_Line_1", "balun_strip_line_1"])
# # ---------- end replacement ----------

# Define vacuum/radiation box
#vacuum_margin_mm = 10  # margin around antenna in mm

# Convert dielectric diameter and height to mm
#antenna_diameter_mm = cm2mm * dielectric_diameter_cm
#substrate_height_mm = cm2mm * dielectric_height_cm
#ferrite_height_mm = cm2mm * ferrite_thickness_cm # Use the new ferrite thickness

#x_size = antenna_diameter_mm + 2 * vacuum_margin_mm
#y_size = antenna_diameter_mm + 2 * vacuum_margin_mm
#z_min_vac = -substrate_height_mm - ferrite_height_mm - vacuum_margin_mm # Extend to below ferrite
#z_max_vac = vacuum_margin_mm
#z_size = z_max_vac - z_min_vac

#x_min = -x_size / 2
#y_min = -y_size / 2
#z_min = z_min_vac

#vacuum = hfss.modeler.create_box([x_min, y_min, z_min],
 #                                [x_size, y_size, z_size],
  #                               name="Vacuum",
  #                               material="air")


###############################################################################
# Solution Setup Section (with solver_setup defined first)
###############################################################################

hfss.create_open_region(frequency=frequency_max)

# 1. DEFINE the solver_setup object (moved here to fix the undefined variable error)
solver_setup = hfss.create_setup(setupname="Setup1", setuptype="HFSSDriven")

# 2. USE the solver_setup object to update its properties
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '6GHz',
                       "MaxDeltaS": 0.02,
                       "PortsOnly": False,
                       "UseMatrixConv": False,
                       "MaximumPasses": 20,
                       "MinimumPasses": 1,
                       "MinimumConvergedPasses": 1,
                       "PercentRefinement": 30,
                       "IsEnabled": True,
                       # ('MeshLink', SetupProps([('ImportMesh', False)])),
                       "BasisOrder": 1,
                       "DoLambdaRefine": True,
                       "DoMaterialLambda": True,
                       "SetLambdaTarget": False,
                       "Target": 0.3333,
                       "UseMaxTetIncrease": False,
                       "PortAccuracy": 2,
                       "UseABCOnPort": False,
                       "SetPortMinMaxTri": False,
                       "UseDomains": False,
                       "UseIterativeSolver": False,
                       "SaveRadFieldsOnly": False,
                       "SaveAnyFields": True,
                       "IESolverType": "Auto",
                       "LambdaTargetForIESolver": 0.15,
                       "UseDefaultLambdaTgtForIESolver": True,
                       "IE Solver Accuracy": 'Balanced'
                       }
solver_setup.props.update(solver_setup_params)

# 3. USE the solver_setup object to create the frequency sweep
frequency_sweep_params = {
    "unit": "GHz",
    "freqstart": 0.8,
    "freqstop": 6,
    "num_of_freq_points": 401,
    "sweepname": "sweep",
    "save_fields": True,
    "save_rad_fields": False,
    "sweep_type": "Interpolating",
    "interpolation_tol": 0.5,
    "interpolation_max_solutions": 250
}
solver_setup.create_frequency_sweep(**frequency_sweep_params)

# --- Save the project ---
project_file_path = os.path.join(os.getcwd(), project_name + ".aedt")
hfss.save_project(project_file_path)
print(f"Project saved to {project_file_path}")

setup_ok = hfss.validate_full_design()
if setup_ok:
    print(f"Design is valid. Setup '{solver_setup.name}' is ready for analysis.")
    # hfss.analyze_setup("Setup1")
else:
    print("Design validation failed. Check geometry, ports, and boundaries.")


# hfss.submit_job(clustername="localhost",
#                 aedt_full_exe_path="/opt/AnsysEM/v231/Linux64/ansysedt.exe",
#                 wait_for_license=True,
#                 setting_file=None)

# plt.savefig(save_filename_no_extension + "_plots" + ".png")
# with open(save_filename_numpy, 'wb') as f:
#     np.save(f, database)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)

hfss.save_project()
# matlab_dict = {
#     # "num_cells": len(database),
#     "global_cell_size": cell_size,
#     # "database": database,
#     "compute_start_timestamp": current_time_str,
#     "compute_host": socket.gethostname(),
#     "compute_duration": time_difference_str
# }

#scipy.io.savemat(save_filename_matlab, matlab_dict)
###############################################################################

###############################################################################
desktop.release_desktop(close_projects=False, close_on_exit=False)
