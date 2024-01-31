"""
MTS: Unit Cell Simulation
Experiment 4

Automated construction of a transmission line metasurface antenna.
"""

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import scipy.signal
import socket
from datetime import datetime
import platform
import pyaedt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np

import uncc_mts_unit_cell as unit_cell
import uncc_mts_compute_config as compute_config

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
frequency = 10e9
wavelength = speed_of_light / frequency

height_mm = 1.57  # mm   <=== dielectric slab height in centimeters
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

path = "mts_databases/"
filename_prefix = "mts_dispersion_database"
datetime_str = "Dec16_07-57-12"
machine = "ece-emag1"
load_filename_matlab = path + filename_prefix + "_" + datetime_str + "_" + machine
# unit_cell_database = scipy.io.loadmat(load_filename_matlab)
# eigen_modes = unit_cell_database["database"][0][0]["mode_solutions"]
#

# DIELECTRIC MATERIALS
dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
# dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
ground_plane_material_name = "pec"
# ground_plane_material_name = "copper"
unit_cell_material_name = "pec"
# unit_cell_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
perfectly_matched_layer_color = [255, 128, 128]

d0 = 8  # mm
t0 = 3  # mm
L1 = 50  # mm
W = 22  # mm
t1 = 6  # mm
d1 = 4  # mm
wr = 13  # mm
t2 = 5  # mm
d2 = 2.5  # mm
wh = 13  # mm
L2 = 200  # mm
m = 9.4  # mm
p = 8  # mm
t3 = 7  # mm
d3 = 1.8  # mm
t4 = 5.7  # mm
d4 = 1.7  # mm
h = 0.787  # mm
# For the trapezoid unit, the geometric sizes are set as:
l3 = 2.9  # mm
l4 = 3.9  # mm
l5 = 4.9  # mm
s1 = 0.9  # mm
w3 = 0.2  # mm
l6 = 3  # mm
l7 = 4  # mm
l8 = 5  # mm
s2 = 1  # mm
antenna_length_mm = 2 * (d0 + d2) + L2
antenna_width_mm = W
# antenna_dimensions_xy_cm = np.array([40, 12.4])
# antenna_margin_xy_cm = np.array([1, 1])
# antenna_coord_origin_xy_cm = np.array([10, 6.2])
# antenna_dimensions_xy_cm = np.array([10, 5.4])
antenna_dimensions_xy_mm = np.array([antenna_length_mm, antenna_width_mm])
antenna_margin_xy_mm = np.array([0, 0])
# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
antenna_coord_origin_xy_mm = 0.5 * antenna_dimensions_xy_mm

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "MTS Antenna " + current_time_str
design_name = "MTS HFSS " + current_time_str

save_file_prefix = "mts_antenna"
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
# cm2mm = 10
hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_plane_position = -0.5 * np.array([antenna_dimensions_xy_mm[0],
                                         antenna_dimensions_xy_mm[1],
                                         height_mm])
ground_plane_size = np.array([antenna_dimensions_xy_mm[0],
                              antenna_dimensions_xy_mm[1]])

#  csPlane is either "XY", "YZ", or "XZ"

ground_plane_params = {"name": "cell_ground_plane",
                       "csPlane": "XY",
                       "position": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
                                                           ground_plane_position[1],
                                                           ground_plane_position[2]).split(","),
                       "dimension_list": "{}mm,{}mm".format(ground_plane_size[0],
                                                            ground_plane_size[1]).split(","),
                       "matname": ground_plane_material_name,
                       "is_covered": True}
ground_plane_geom = hfss.modeler.create_rectangle(**ground_plane_params)
ground_plane_geom.color = metal_color

# FIT ALL
hfss.modeler.fit_all()

dielectric_slab_position = ground_plane_position
dielectric_slab_size = np.array([ground_plane_size[0],
                                 ground_plane_size[1],
                                 height_mm])
dielectric_slab_params = {"name": "dielectric_slab",
                          "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
                                                              dielectric_slab_position[1],
                                                              dielectric_slab_position[2]).split(","),
                          "dimensions_list": "{}mm,{}mm,{}mm".format(dielectric_slab_size[0],
                                                                     dielectric_slab_size[1],
                                                                     dielectric_slab_size[2]).split(","),
                          "matname": dielectric_material_name}
dielectric_slab_geom = hfss.modeler.create_box(**dielectric_slab_params)
dielectric_slab_geom.color = dielectric_color

transmission_line_position = 0.5 * np.array([-L2,
                                             -antenna_dimensions_xy_mm[1],
                                             height_mm])
transmission_line_size = np.array([L2,
                                   antenna_dimensions_xy_mm[1]])
transmission_line_plane_params = {"name": "transmission_line",
                                  "csPlane": "XY",
                                  "position": "{}mm,{}mm,{}mm".format(transmission_line_position[0],
                                                                      transmission_line_position[1],
                                                                      transmission_line_position[2]).split(","),
                                  "dimension_list": "{}mm,{}mm".format(transmission_line_size[0],
                                                                       transmission_line_size[1]).split(","),
                                  "matname": ground_plane_material_name,
                                  "is_covered": True}
transmission_line_plane_geom = hfss.modeler.create_rectangle(**transmission_line_plane_params)
transmission_line_plane_geom.color = metal_color

port_index = 0
#  build a feed structure having feed_length_cm length along the Y axis (the propagation axis)
trapezoid_length_mm = d2
feed_rect_length_mm = 0.5 * (antenna_length_mm - L2 - 2 * trapezoid_length_mm)
feed_total_length_mm = feed_rect_length_mm + trapezoid_length_mm
feed_rect_width_mm = t0
trapezoid_top_width = feed_rect_width_mm
trapezoid_bottom_width = t2
feed_rect_position = np.array([-0.5 * antenna_length_mm,
                               -0.5 * feed_rect_width_mm,
                               0.5 * height_mm])
feed_rect_size = np.array([feed_rect_length_mm, feed_rect_width_mm])
feed_rect_params = {"name": "feed_rectanglar_portion_" + str(port_index),
                    "csPlane": "XY",
                    "position": "{}mm,{}mm,{}mm".format(feed_rect_position[0],
                                                        feed_rect_position[1],
                                                        feed_rect_position[2]).split(","),
                    "dimension_list": "{}mm,{}mm".format(feed_rect_size[0],
                                                         feed_rect_size[1]).split(","),
                    "matname": ground_plane_material_name,
                    "is_covered": True}
feed_rect_geom_0 = hfss.modeler.create_rectangle(**feed_rect_params)
feed_rect_geom_0.color = metal_color

trap_position_list = np.array([np.array([feed_rect_position[0] + feed_rect_size[0],
                                         feed_rect_position[1], feed_rect_position[2]]),
                               np.array([feed_rect_position[0] + feed_rect_size[0],
                                         feed_rect_position[1] + trapezoid_top_width, feed_rect_position[2]]),
                               np.array([feed_rect_position[0] + feed_rect_size[0] + trapezoid_length_mm,
                                         0.5 * trapezoid_bottom_width, feed_rect_position[2]]),
                               np.array([feed_rect_position[0] + feed_rect_size[0] + trapezoid_length_mm,
                                         -0.5 * trapezoid_bottom_width, feed_rect_position[2]])])
trap_position_list = [elem.tolist() for elem in trap_position_list]
trap_polyline_params = {
    "position_list": trap_position_list,
    "segment_type": None,
    "cover_surface": True,
    "close_surface": True,
    "name": "feed_trapezoid_portion_" + str(port_index),
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
feed_trap_geom_0 = hfss.modeler.create_polyline(**trap_polyline_params)
feed_trap_geom_0.color = metal_color

port_index = 1
#  build a feed structure having feed_length_cm length along the Y axis (the propagation axis)
feed_rect_position = np.array([0.5 * antenna_length_mm - feed_rect_length_mm,
                               0.5 * feed_rect_width_mm - 0.5 * feed_rect_width_mm,
                               0.5 * height_mm])
feed_rect_size = np.array([feed_rect_length_mm, feed_rect_width_mm])
feed_rect_params = {"name": "feed_rectanglar_portion_" + str(port_index),
                    "csPlane": "XY",
                    "position": "{}mm,{}mm,{}mm".format(feed_rect_position[0],
                                                        feed_rect_position[1],
                                                        feed_rect_position[2]).split(","),
                    "dimension_list": "{}mm,{}mm".format(feed_rect_size[0],
                                                         feed_rect_size[1]).split(","),
                    "matname": ground_plane_material_name,
                    "is_covered": True}
feed_rect_geom_1 = hfss.modeler.create_rectangle(**feed_rect_params)
feed_rect_geom_1.color = metal_color

trap_position_list = np.array([np.array([feed_rect_position[0],
                                         feed_rect_position[1], feed_rect_position[2]]),
                               np.array([feed_rect_position[0],
                                         feed_rect_position[1] + trapezoid_top_width, feed_rect_position[2]]),
                               np.array([feed_rect_position[0] - trapezoid_length_mm,
                                         0.5 * trapezoid_bottom_width + 0.5 * feed_rect_width_mm,
                                         feed_rect_position[2]]),
                               np.array([feed_rect_position[0] - trapezoid_length_mm,
                                         -0.5 * trapezoid_bottom_width + 0.5 * feed_rect_width_mm,
                                         feed_rect_position[2]])])
trap_position_list = [elem.tolist() for elem in trap_position_list]
trap_polyline_params = {
    "position_list": trap_position_list,
    "segment_type": None,
    "cover_surface": True,
    "close_surface": True,
    "name": "feed_trapezoid_portion_" + str(port_index),
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
feed_trap_geom_1 = hfss.modeler.create_polyline(**trap_polyline_params)
feed_trap_geom_1.color = metal_color

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": [ground_plane_geom.name,
                      transmission_line_plane_geom.name,
                      feed_rect_geom_0.name, feed_trap_geom_0.name,
                      feed_rect_geom_1.name, feed_trap_geom_1.name],
       "sourcename": None,
       "is_infinite_gnd": False})

l1 = 1.58
thickness = 0.2
gap_size = 0.16
num_vertices = 4  # THIS MUST BE EVEN
delta_angle_deg = 360 / num_vertices
angle_list = np.arange(0, 361, delta_angle_deg) - 180
outer_radius = l1
inner_radius = l1 - thickness
outer_position_list = []
inner_position_list = []
for angle in angle_list:
    x_outer = outer_radius * np.cos(angle * np.pi / 180)
    y_outer = outer_radius * np.sin(angle * np.pi / 180)
    x_inner = inner_radius * np.cos(angle * np.pi / 180)
    y_inner = inner_radius * np.sin(angle * np.pi / 180)
    outer_position_list.append(np.array([x_outer, y_outer, 0]))
    inner_position_list.append(np.array([x_inner, y_inner, 0]))

edge_vector_first = outer_position_list[1] - outer_position_list[0]
edge_vector_first *= 1 / np.linalg.norm(edge_vector_first)
outer_position_list[0] += np.abs(0.5 * gap_size / edge_vector_first[1]) * edge_vector_first
edge_vector_last = outer_position_list[-2] - outer_position_list[-1]
edge_vector_last *= 1 / np.linalg.norm(edge_vector_last)
outer_position_list[-1] += np.abs(0.5 * gap_size / edge_vector_last[1]) * edge_vector_last

edge_vector_first = inner_position_list[1] - inner_position_list[0]
edge_vector_first *= 1 / np.linalg.norm(edge_vector_first)
inner_position_list[0] += np.abs(0.5 * gap_size / edge_vector_first[1]) * edge_vector_first
edge_vector_last = inner_position_list[-2] - inner_position_list[-1]
edge_vector_last *= 1 / np.linalg.norm(edge_vector_last)
inner_position_list[-1] += np.abs(0.5 * gap_size / edge_vector_last[1]) * edge_vector_last

unit_cell_position_list = outer_position_list + inner_position_list[::-1]
for pt in unit_cell_position_list:
    pt[2] = 0.5 * height_mm

unit_cell_name_list = []
unit_cell_list = []
# ===LOOP==
index = 0
unit_cell_polyline_params = {
    "position_list": unit_cell_position_list,
    "segment_type": None,
    "cover_surface": True,
    "close_surface": True,
    "name": "unit_cell_" + str(index),
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
unit_cell_0 = hfss.modeler.create_polyline(**unit_cell_polyline_params)
unit_cell_0.color = metal_color
# hfss.assign_perfecte_to_sheets(
#     **{"sheet_list": [unit_cell_0.name],
#        "sourcename": None,
#        "is_infinite_gnd": False})
unit_cell_list.append(unit_cell_0)
unit_cell_name_list.append(unit_cell_0.name)
# ===LOOP==

subtract_params = {
    "blank_list": [transmission_line_plane_geom.name],
    "tool_list": unit_cell_name_list,
    "keep_originals": True
}
hfss.modeler.subtract(**subtract_params)

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
