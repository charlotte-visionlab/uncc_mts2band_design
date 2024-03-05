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

# height_mm = 1.57  # mm   <=== dielectric slab height in millimeters
height_mm = 1.0  # mm   <=== dielectric slab height in millimeters
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

frequency_GHz = 100.0
wavelength_mm = 1e3 * speed_of_light / (frequency_GHz * 1e9)

# DIELECTRIC MATERIALS
# dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "glass"
# dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
ground_plane_material_name = "pec"
# ground_plane_material_name = "copper"
unit_cell_material_name = "pec"
# unit_cell_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]  # green
dielectric_color = [255, 255, 128]  # yellow
radiation_box_color = [128, 255, 255]  # neon blue
subtract_tool_color = [64, 64, 64]  # dary gray
perfectly_matched_layer_color = [255, 128, 128]  # light red
port_color = [31, 81, 255]  # blue

num_patches = 10

strip_width_mm = 0.8  # mm
feed_trapezoid_start_width_mm = 0.25  # mm
feed_trapezoid_length_mm = 2.5  # mm
feed_rectangle_length_mm = 0.5  # mm
gap_length_mm = 1.5  # mm
patch_length_mm = 0.9  # mm
patch_width_mm = 0.8  # mm
feed_length_mm = feed_rectangle_length_mm + feed_trapezoid_length_mm
antenna_length_mm = num_patches * patch_length_mm + 1.5 * (num_patches + 1)  # mm
board_length_mm = (2 * feed_length_mm) + antenna_length_mm

board_margin_xy_mm = np.array([2, 0])
board_width_mm = patch_width_mm + 2 * board_margin_xy_mm[0]
antenna_dimensions_xy_mm = np.array([board_length_mm, board_width_mm])

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

port_index = 0
#  build a feed structure having feed_length_mm length along the X axis (the propagation axis)
feed_rect_length_mm = feed_rectangle_length_mm
feed_rect_width_mm = strip_width_mm
trapezoid_top_width = feed_trapezoid_start_width_mm
trapezoid_bottom_width = strip_width_mm
feed_rect_position = np.array([-0.5 * board_length_mm + feed_trapezoid_length_mm,
                               -0.5 * strip_width_mm,
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
board_edge = [-0.5 * board_length_mm, -0.5 * feed_trapezoid_start_width_mm]
trap_position_list = np.array([np.array([board_edge[0],
                                         board_edge[1], feed_rect_position[2]]),
                               np.array([board_edge[0],
                                         board_edge[1] + feed_trapezoid_start_width_mm, feed_rect_position[2]]),
                               np.array([board_edge[0] + feed_trapezoid_length_mm,
                                         0.5 * strip_width_mm, feed_rect_position[2]]),
                               np.array([board_edge[0] + feed_trapezoid_length_mm,
                                         -0.5 * strip_width_mm, feed_rect_position[2]])])
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
feed_trap_geom_0.transparency = 0

port_index = 1
#  build a feed structure having feed_length_cm length along the Y axis (the propagation axis)
feed_rect_position = np.array([0.5 * board_length_mm - feed_trapezoid_length_mm - feed_rect_length_mm,
                               -0.5 * strip_width_mm,
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
board_edge = [0.5 * board_length_mm, -0.5 * feed_trapezoid_start_width_mm]
trap_position_list = np.array([np.array([board_edge[0],
                                         board_edge[1], feed_rect_position[2]]),
                               np.array([board_edge[0],
                                         board_edge[1] + feed_trapezoid_start_width_mm, feed_rect_position[2]]),
                               np.array([board_edge[0] - feed_trapezoid_length_mm,
                                         0.5 * strip_width_mm, feed_rect_position[2]]),
                               np.array([board_edge[0] - feed_trapezoid_length_mm,
                                         -0.5 * strip_width_mm, feed_rect_position[2]])])
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
feed_trap_geom_1.transparency = 0

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": [ground_plane_geom.name,
                      # transmission_line_plane_geom.name,
                      feed_rect_geom_0.name, feed_trap_geom_0.name,
                      feed_rect_geom_1.name, feed_trap_geom_1.name],
       "sourcename": None,
       "is_infinite_gnd": False})

offset = -0.5 * board_length_mm + feed_length_mm + gap_length_mm
patch_geom_list = []
patch_name_list = []
for patch_index in np.arange(0, num_patches):
    patch_position = np.array([offset,
                               -0.5 * strip_width_mm,
                               0.5 * height_mm])
    patch_size = np.array([patch_length_mm, patch_width_mm])
    patch_params = {"name": "patch_" + str(patch_index),
                    "csPlane": "XY",
                    "position": "{}mm,{}mm,{}mm".format(patch_position[0],
                                                        patch_position[1],
                                                        patch_position[2]).split(","),
                    "dimension_list": "{}mm,{}mm".format(patch_size[0],
                                                         patch_size[1]).split(","),
                    "matname": ground_plane_material_name,
                    "is_covered": True}
    patch_geom = hfss.modeler.create_rectangle(**patch_params)
    patch_geom.color = metal_color
    offset += gap_length_mm + patch_length_mm
    patch_geom_list.append(patch_geom)
    patch_name_list.append(patch_geom.name)

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": patch_name_list,
       "sourcename": None,
       "is_infinite_gnd": False})

port_1_position = np.array([-0.5 * board_length_mm,
                            -0.5 * feed_trapezoid_start_width_mm,
                            -0.5 * height_mm])
port_1_size = np.array([feed_trapezoid_start_width_mm, height_mm])
port_1_params = {"name": "port_1",
                 "csPlane": "YZ",
                 "position": "{}mm,{}mm,{}mm".format(port_1_position[0],
                                                     port_1_position[1],
                                                     port_1_position[2]).split(","),
                 "dimension_list": "{}mm,{}mm".format(port_1_size[0],
                                                      port_1_size[1]).split(","),
                 "matname": None,
                 "is_covered": True}
port_1_geom = hfss.modeler.create_rectangle(**port_1_params)
port_1_geom.color = radiation_box_color

port_1_excitation_params = {"signal": port_1_geom,
                            "reference": ground_plane_geom,
                            "create_port_sheet": False,
                            "port_on_plane": True,
                            "integration_line": 0,
                            "impedance": 50,
                            "name": "port_1_excitation",
                            "renormalize": True,
                            "deembed": False,
                            "terminals_rename": True}
hfss.lumped_port(**port_1_excitation_params)

port_2_position = np.array([0.5 * board_length_mm,
                            -0.5 * feed_trapezoid_start_width_mm,
                            -0.5 * height_mm])
port_2_size = port_1_size
port_2_params = {"name": "port_2",
                 "csPlane": "YZ",
                 "position": "{}mm,{}mm,{}mm".format(port_2_position[0],
                                                     port_2_position[1],
                                                     port_2_position[2]).split(","),
                 "dimension_list": "{}mm,{}mm".format(port_2_size[0],
                                                      port_2_size[1]).split(","),
                 "matname": None,
                 "is_covered": True}
port_2_geom = hfss.modeler.create_rectangle(**port_2_params)
port_2_geom.color = radiation_box_color

port_2_excitation_params = {"signal": port_2_geom,
                            "reference": ground_plane_geom,
                            "create_port_sheet": False,
                            "port_on_plane": True,
                            "integration_line": 0,
                            "impedance": 50,
                            "name": "port_2_excitation",
                            "renormalize": True,
                            "deembed": False,
                            "terminals_rename": True}
hfss.lumped_port(**port_2_excitation_params)

open_region_params = {
    "Frequency": "{}GHz".format(frequency_GHz),
    "Boundary": "Radiation",
    "ApplyInfiniteGP": False,
    "GPAXis": "-z"}
success = hfss.create_open_region(**open_region_params)

analysis_plane_position = np.array([ground_plane_position[0], ground_plane_position[1], 0])
analysis_plane_size = ground_plane_size
analysis_plane_params = {"name": "plot_waveguide_mode",
                         "csPlane": "XY",
                         "position": "{}mm,{}mm,{}mm".format(analysis_plane_position[0],
                                                             analysis_plane_position[1],
                                                             analysis_plane_position[2]).split(","),
                         "dimension_list": "{}mm,{}mm".format(analysis_plane_size[0],
                                                              analysis_plane_size[1]).split(","),
                         "matname": None,
                         "is_covered": True}
analysis_plane_geom = hfss.modeler.create_rectangle(**analysis_plane_params)
analysis_plane_geom.color = radiation_box_color

nf_x_direction = [1, 0, 0]
nf_y_direction = [0, 1, 0]
nf_z_direction = [0, 0, 1]
top_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0, 0, 2 * wavelength_mm).split(","),
                             "reference_cs": "Global",
                             "name": "top_2lambda_CS",
                             "mode": "axis",
                             "view": "iso",
                             "x_pointing": nf_x_direction,
                             "y_pointing": nf_y_direction,
                             "psi": 0,
                             "theta": 0,
                             "phi": 0,
                             "u": None
                             }
top_plane_field_cs = hfss.modeler.create_coordinate_system(**top_plane_field_cs_params)

bottom_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0, 0, -2 * wavelength_mm).split(","),
                                "reference_cs": "Global",
                                "name": "bottom_2lambda_CS",
                                "mode": "axis",
                                "view": "iso",
                                "x_pointing": nf_x_direction,
                                "y_pointing": nf_y_direction,
                                "psi": 0,
                                "theta": 0,
                                "phi": 0,
                                "u": None
                                }
bottom_plane_field_cs = hfss.modeler.create_coordinate_system(**bottom_plane_field_cs_params)

left_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0,
                                                                -0.5 * board_width_mm - 2 * wavelength_mm,
                                                                0).split(","),
                              "reference_cs": "Global",
                              "name": "left_2lambda_CS",
                              "mode": "axis",
                              "view": "iso",
                              "x_pointing": nf_x_direction,
                              "y_pointing": nf_z_direction,
                              "psi": 0,
                              "theta": 0,
                              "phi": 0,
                              "u": None
                              }
left_plane_field_cs = hfss.modeler.create_coordinate_system(**left_plane_field_cs_params)
right_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0,
                                                                 0.5 * board_width_mm + 2 * wavelength_mm,
                                                                 0).split(","),
                               "reference_cs": "Global",
                               "name": "right_2lambda_CS",
                               "mode": "axis",
                               "view": "iso",
                               "x_pointing": nf_x_direction,
                               "y_pointing": nf_z_direction,
                               "psi": 0,
                               "theta": 0,
                               "phi": 0,
                               "u": None
                               }
right_plane_field_cs = hfss.modeler.create_coordinate_system(**right_plane_field_cs_params)

back_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(-0.5 * board_length_mm - 2 * wavelength_mm,
                                                                0,
                                                                0).split(","),
                              "reference_cs": "Global",
                              "name": "back_2lambda_CS",
                              "mode": "axis",
                              "view": "iso",
                              "x_pointing": nf_y_direction,
                              "y_pointing": nf_z_direction,
                              "psi": 0,
                              "theta": 0,
                              "phi": 0,
                              "u": None
                              }
back_plane_field_cs = hfss.modeler.create_coordinate_system(**back_plane_field_cs_params)
front_plane_field_cs_params = {"origin": "{}mm,{}mm,{}mm".format(0.5 * board_length_mm + 2 * wavelength_mm,
                                                                 0,
                                                                 0).split(","),
                               "reference_cs": "Global",
                               "name": "front_2lambda_CS",
                               "mode": "axis",
                               "view": "iso",
                               "x_pointing": nf_y_direction,
                               "y_pointing": nf_z_direction,
                               "psi": 0,
                               "theta": 0,
                               "phi": 0,
                               "u": None
                               }
front_plane_field_cs = hfss.modeler.create_coordinate_system(**front_plane_field_cs_params)

solver_setup = hfss.create_setup(setupname="MTS_SIWAntenna_Setup", setuptype="HFSSDriven")
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '{}GHz'.format(frequency_GHz),
                       "MaxDeltaS": 0.03,
                       "PortsOnly": False,
                       "UseMatrixConv": False,
                       "MaximumPasses": 30,
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

# frequency_sweep_params = {
#     "unit": "GHz",
#     "freqstart": frequency_GHz - 1.5,
#     "freqstop": frequency_GHz + 1.5,
#     "num_of_freq_points": 200,
#     "sweepname": "sweep",
#     "save_fields": True,
#     "save_rad_fields": False,
#     "sweep_type": "Discrete",
#     "interpolation_tol": 0.5,
#     "interpolation_max_solutions": 250
# }
# solver_setup.create_frequency_sweep(**frequency_sweep_params)


setup_ok = hfss.validate_full_design()

setup_solver_configuration_params = {
    "name": "MTS_SIWAntenna_Setup",
    "num_cores": solver_configuration["num_cores"],
    "num_tasks": 1,
    "num_gpu": solver_configuration["num_gpu"],
    "acf_file": None,
    "use_auto_settings": True,
    "num_variations_to_distribute": None,
    "allowed_distribution_types": None,
    "revert_to_initial_mesh": False,
    "blocking": True
}
hfss.analyze_setup(**setup_solver_configuration_params)

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
