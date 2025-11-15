"""
MTS: Unit Cell Simulation
Experiment 1
Wan, X., Cai, B., Li, Y. et al. Dual-channel near-field control by polarizations using isotropic and
inhomogeneous metasurface. Sci Rep 5, 15853 (2015). https://doi.org/10.1038/srep15853
-------------------
This example shows how you can use PyAEDT to create a multipart scenario in HFSS
and set up a metasurface unit cell dispersion analysis.

This is an HFSS implementation of dispersion calculation for a unit cell consisting of a small metal patch.

When the patch size is equal to the cell size the dispersion matches that of a parallel plate waveguide.
Results for this code are intended to match the design described in:

Wan, X., Cai, B., Li, Y. et al. Dual-channel near-field control by polarizations using isotropic and
inhomogeneous metasurface. Sci Rep 5, 15853 (2015). https://doi.org/10.1038/srep15853

The fill percentage, i.e., ratio of the metal patch dimensions in (X,Y) to the size to the unit cell, is controlled
via the fill_pct variable which takes on a value  (0,0) <= fill_pct <= (1,1) and determines the ratio of the square
metal patch size in (X,Y) to the unit cell size in (X,Y).
"""

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import socket
from datetime import datetime
import platform
import pyaedt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np

import uncc_mts_compute_config as compute_config


def sinuous_curve(r, alpha, r_start, tau):
    return alpha * np.sin((np.pi * np.log(r / r_start)) / np.log(tau))


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


class Point2D:
    def __init__(self, x, y):
        self.data = {'x': x, 'y': y}

    def __getattr__(self, name):
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value


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
    os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
else:
    os.environ["ANSYSEM_ROOT231"] = "C:\\Program Files\\AnsysEM\\v231\\Win64\\"

aedt_version = "2023.1"

solver_configuration = compute_config.SolverConfig().solver_config

###############################################################################
# Define program variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
non_graphical = False

###############################################################################
# Launch AEDT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NewThread = True
desktop = pyaedt.launch_desktop(specified_version=aedt_version,
                                non_graphical=non_graphical,
                                new_desktop_session=NewThread)

# Solution Types are: { "Modal", "Terminal", "Eigenmode", "Transient Network", "SBR+", "Characteristic"}
hfss = pyaedt.Hfss(
    specified_version=aedt_version,
    solution_type="Terminal",
    new_desktop_session=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical
)

hfss.modeler.model_units = 'mm'
hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength_nominal))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# FIT ALL
hfss.modeler.fit_all()

ground_plane_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                  cm2mm * antenna_coord_origin_xy_cm[1],
                                  cm2mm * -dielectric_height_cm - 0.035])

# ground_plane_params = {"name": "ground_plane",
#                        "cs_plane": "XY",
#                        "position": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                            ground_plane_position[1],
#                                                            ground_plane_position[2]).split(","),
#                        "radius": "{}mm".format(cm2mm * ground_plane_diameter_cm / 2),
#                        "numSides": 0,
#                        "matname": None}
# ground_plane_geom = hfss.modeler.create_circle(**ground_plane_params)
# ground_plane_geom.color = metal_color

# ground_plane_params = {"name": "ground_plane",
#                        "cs_axis": "XY",
#                        "position": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                            ground_plane_position[1],
#                                                            ground_plane_position[2]).split(","),
#                        "radius": "{}mm".format(cm2mm * ground_plane_diameter_cm / 2),
#                        "height": "{}mm".format(0.035),
#                        "numSides": 0,
#                        "matname": "copper"}
# ground_plane_geom = hfss.modeler.create_cylinder(**ground_plane_params)
# ground_plane_geom.color = metal_color

# hfss.assign_perfecte_to_sheets(
#     **{"sheet_list": "ground_plane",
#        "sourcename": None,
#        "is_infinite_gnd": False})

# ground_plane_hole_params = {"name": "ground_plane_hole",
#                             "cs_plane": "XY",
#                             "position": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                                 ground_plane_position[1],
#                                                                 ground_plane_position[2]).split(","),
#                             "radius": "{}mm".format(cm2mm * ground_plane_hole_diameter_cm / 2),
#                             "numSides": 0,
#                             "matname": None}
# ground_plane_hole_geom = hfss.modeler.create_circle(**ground_plane_hole_params)
# ground_plane_geom.color = metal_color
#
# subtract_params = {
#     "blank_list": ["ground_plane"],
#     "tool_list": ["ground_plane_hole"],
#     "keep_originals": True
# }
# hfss.modeler.subtract(**subtract_params)
# ground_plane_hole_geom.delete()
if False:
    cavity_cylindrical_wall_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                                 cm2mm * antenna_coord_origin_xy_cm[1],
                                                 cm2mm * -cavity_height_cm])
    cavity_cylindrical_wall_params = {"name": "cavity_cylindrical_wall",
                                      "cs_axis": "Z",
                                      "position": "{}mm,{}mm,{}mm".format(cavity_cylindrical_wall_position[0],
                                                                          cavity_cylindrical_wall_position[1],
                                                                          cavity_cylindrical_wall_position[2]).split(
                                          ","),
                                      "radius": "{}mm".format(cm2mm * cavity_diameter_cm / 2),
                                      "height": "{}mm".format(cm2mm * cavity_height_cm),
                                      "numSides": 0,
                                      "matname": "aluminum"}
    cavity_cylindrical_wall_geom = hfss.modeler.create_cylinder(**cavity_cylindrical_wall_params)
    cavity_cylindrical_wall_geom.color = aluminum_color
    cavity_cylindrical_wall_geom.transparency = 0.2

    cavity_cylindrical_back_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                                 cm2mm * antenna_coord_origin_xy_cm[1],
                                                 cm2mm * -(cavity_height_cm + cavity_wall_thickness_cm)])
    cavity_cylindrical_back_params = {"name": "cavity_cylindrical_back",
                                      "cs_axis": "Z",
                                      "position": "{}mm,{}mm,{}mm".format(cavity_cylindrical_back_position[0],
                                                                          cavity_cylindrical_back_position[1],
                                                                          cavity_cylindrical_back_position[2]).split(
                                          ","),
                                      "radius": "{}mm".format(cm2mm * cavity_diameter_cm / 2),
                                      "height": "{}mm".format(cm2mm * cavity_wall_thickness_cm),
                                      "numSides": 0,
                                      "matname": "aluminum"}
    cavity_cylindrical_back_geom = hfss.modeler.create_cylinder(**cavity_cylindrical_back_params)
    cavity_cylindrical_back_geom.color = aluminum_color
    cavity_cylindrical_back_geom.transparency = 0.2

    unite_params = {
        "unite_list": ["cavity_cylindrical_wall", "cavity_cylindrical_back"],
        "purge": False,
        "keep_originals": False
    }
    hfss.modeler.unite(**unite_params)

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
    through_hole_contact_geom.delete()

    if PEC_ANTENNA:
        hfss.assign_perfecte_to_sheets(
            **{"sheet_list": arm_object_name,
               "sourcename": None,
               "is_infinite_gnd": False})
        sinuous_arm_polyline_list.append(sinuous_arm_polyline)
    else:
        sinuous_arm_object3d = hfss.modeler.thicken_sheet(objid=arm_object_name, thickness=copper_thickness)
        # Define the antenna material (e.g., copper)
        hfss.assign_material(sinuous_arm_object3d, "copper")
        sinuous_arm_polyline_list.append(sinuous_arm_object3d)

center_hole = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                        cm2mm * antenna_coord_origin_xy_cm[1],
                        cm2mm * -dielectric_height_cm])
center_hole_params = {"name": "center_hole",
                      "cs_axis": "Z",
                      "position": "{}mm,{}mm,{}mm".format(center_hole[0],
                                                          center_hole[1],
                                                          center_hole[2]).split(","),
                      "radius": "{}mm".format(cm2mm * 1.1 * ground_plane_hole_diameter_cm / 2),
                      "height": "{}mm".format(cm2mm * dielectric_height_cm),
                      "numSides": 20,
                      "matname": None}
# center_hole_geom = hfss.modeler.create_cylinder(**center_hole_params)

# detached_face_names = hfss.oeditor.DetachFaces(
#     ["NAME:Selections",
#      "Selections:=", "center_hole",
#      "NewPartsModelFlag:=", "Model"
#      ],
#     ["NAME:Parameters",
#      ["NAME:DetachFacesToParameters",
#       "FacesToDetach:=", [center_hole_geom.faces[0].id, center_hole_geom.faces[1].id]]])

# center_hole_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
#                                              cm2mm * antenna_coord_origin_xy_cm[1],
#                                              cm2mm * -dielectric_height_cm])
# center_hole_size = np.array([cm2mm * ground_plane_hole_diameter_cm,
#                                              cm2mm * ground_plane_hole_diameter_cm,
#                                              cm2mm * dielectric_height_cm])
#
# center_hole_params = {"name": "center_hole",
#                           "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
#                                                               dielectric_slab_position[1],
#                                                               dielectric_slab_position[2]).split(","),
#                           "dimensions_list": "{}mm,{}mm,{}mm".format(center_hole_size[0],
#                                                                      center_hole_size[1],
#                                                                      center_hole_size[2]).split(","),
#                           "matname": None}
# center_hole_geom = hfss.modeler.create_box(**center_hole_params)


# subtract_params = {
#     "blank_list": ["ground_plane", "dielectric_slab",
#                    sinuous_arm_polyline_list[0].name, sinuous_arm_polyline_list[1].name,
#                    sinuous_arm_polyline_list[2].name, sinuous_arm_polyline_list[3].name],
#     "tool_list": ["center_hole"],
#     "keep_originals": True
# }
# hfss.modeler.subtract(**subtract_params)
# center_hole_geom.delete()
# lumped_port_sheet_geom = hfss.modeler[detached_face_names[1]]
# # Integration line can be two points or one of the following:
# #           ``XNeg``, ``YNeg``, ``ZNeg``, ``XPos``, ``YPos``, and ``ZPos``
# lumped_port_params = {  # "signal": None,
#     "signal": detached_face_names[1],
#     "reference": "monopole_shield",
#     "create_port_sheet": False,
#     "port_on_plane": True,
#     # "integration_line": "XPos",
#     "integration_line": [lumped_port_sheet_geom.bottom_face_z.center,
#                          lumped_port_sheet_geom.bottom_face_z.top_edge_x.midpoint],
#     "impedance": 50,
#     "name": "source_port",
#     "renormalize": False,
#     "deembed": False,
#     "terminals_rename": False
# }
# hfss.lumped_port(**lumped_port_params)


# hfss.modeler.create_coordinate_system(**sma_connector_cs_params)
# sma_component_params = {"comp_file": "components/SMA_connector.a3dcomp",
#                         "geo_params": None,
#                         "sz_mat_params": "",
#                         "sz_design_params": "",
#                         "targetCS": "Global",
#                         "name": None,
#                         "password": "",
#                         "auxiliary_dict": False
#                         }

# hfss.create_open_region(open_region_params)
module = hfss.get_module("ModelSetup")
# module.CreateOpenRegion(
#     [
#         "NAME:Settings",
#         "OpFreq:=", "6GHz",
#         "Boundary:=", "Radiation",
#         "ApplyInfiniteGP:=", False
#     ])
# Set up radiation boundary
hfss.create_open_region(frequency=frequency_max)

# "definition": INFINITE_SPHERE_TYPE.ThetaPhi,
# far_field_infinite_sphere_params = {
#     "name": "Infinite_Sphere_Radiation",
#     "x_start": -180,
#     "x_stop": 180,
#     "x_step": 10,
#     "y_start": -180,
#     "y_stop": 180,
#     "y_step": 10,
#     "units": "deg",
#     "custom_radiation_faces": None,
#     "custom_coordinate_system": None,
#     "use_slant_polarization": False,
#     "polarization_angle": 45
# }
# infinite_sphere_far_field = hfss.insert_infinite_sphere(far_field_infinite_sphere_params)
# far_field_infinite_sphere_params_ext = {
#     "Polarization": "Linear",
#     "SlantAngle": '45deg',
#     "ElevationStart": '0deg',
#     "ElevationStop": '180deg',
#     "ElevationStep": '10deg',
#     "AzimuthStart": '0deg',
#     "AzimuthStop": '180deg',
#     "AzimuthStep": '10deg',
#     "UseLocalCS": False,
#     "CoordSystem": ''
# }
# infinite_sphere_far_field.props.update(far_field_infinite_sphere_params_ext)
# for field_setup in hfss.field_setups:
#     field_setup.azimuth_start = -180
#     field_setup.azimuth_stop = 180
#     field_setup.azimuth_step = 2
solver_setup = hfss.create_setup(setupname="Setup1", setuptype="HFSSDriven")
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

setup_ok = hfss.validate_full_design()
hfss.analyze_setup("Setup1")
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
# matlab_dict = {
#     # "num_cells": len(database),
#     "global_cell_size": cell_size,
#     # "database": database,
#     "compute_start_timestamp": current_time_str,
#     "compute_host": socket.gethostname(),
#     "compute_duration": time_difference_str
# }
#
# scipy.io.savemat(save_filename_matlab, matlab_dict)
###############################################################################
# Close Ansys Electronics Desktop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hfss.release_desktop(close_projects=True, close_desktop=True)
