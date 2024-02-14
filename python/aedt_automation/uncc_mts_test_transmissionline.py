"""
Standard Transmission Line Test
Experiment 5

Automated construction of a transmission line for simulation verification.
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

# height_mm = 1.57  # mm   <=== dielectric slab height in millimeters
height_mm = 0.787  # mm   <=== dielectric slab height in millimeters
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

frequency_GHz = 19.6

enclose_antenna_with_pec_boundary = True

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
metal_color = [143, 175, 143]  # green
dielectric_color = [255, 255, 128]  # yellow
radiation_box_color = [128, 255, 255]  # neon blue
subtract_tool_color = [64, 64, 64]  # dary gray
perfectly_matched_layer_color = [255, 128, 128]  # light red
port_color = [31, 81, 255]  # blue

margin_length = 4  # mm
design_size = 5  # mm
board_width_mm = 22  # mm
board_length_mm = 2 * (margin_length) + design_size

transmission_line_width_mm = 0.48  # mm
transmission_line_length_mm = board_length_mm

# antenna_dimensions_xy_cm = np.array([40, 12.4])
# antenna_margin_xy_cm = np.array([1, 1])
# antenna_coord_origin_xy_cm = np.array([10, 6.2])
# antenna_dimensions_xy_cm = np.array([10, 5.4])
board_dimensions_xy_mm = np.array([board_length_mm, board_width_mm])
antenna_margin_xy_mm = np.array([0, 0])
# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
antenna_coord_origin_xy_mm = 0.5 * board_dimensions_xy_mm

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "Microstrip " + current_time_str
design_name = "StripTest" + current_time_str

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
ground_plane_position = -0.5 * np.array([board_dimensions_xy_mm[0],
                                         board_dimensions_xy_mm[1],
                                         height_mm])
ground_plane_size = np.array([board_dimensions_xy_mm[0],
                              board_dimensions_xy_mm[1]])

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

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": [ground_plane_geom.name],
       "sourcename": None,
       "is_infinite_gnd": False})

# MAKE HOLES FOR THE RF 2.92 CONNECTOR BOLTS
hole_positions_y = np.array([-4.75, 4.75])
hole_positions_x = np.array([-0.5 * board_dimensions_xy_mm[0] + 2.5, 0.5 * board_dimensions_xy_mm[0] - 2.5])
hole_position_z = -0.5 * height_mm
hole_diameter_mm = 2.10  # mm
hole_index = 0
for hole_x_pos in hole_positions_x:
    for hole_y_pos in hole_positions_y:
        hole_params = {"name": "bolt_hole_" + str(hole_index),
                       "cs_axis": "Z",
                       "position": "{}mm,{}mm,{}mm".format(hole_x_pos,
                                                           hole_y_pos,
                                                           hole_position_z).split(","),
                       "radius": "{}mm".format(hole_diameter_mm / 2),
                       "height": "{}mm".format(dielectric_slab_size[2]),
                       "numSides": 0,
                       "matname": None}
        hole_geom = hfss.modeler.create_cylinder(**hole_params)
        hole_geom.color = radiation_box_color
        subtract_params = {
            "blank_list": [dielectric_slab_geom.name],
            "tool_list": [hole_geom.name],
            "keep_originals": False
        }
        hfss.modeler.subtract(**subtract_params)
        hole_index += 1

# RF292 Connector model has pin on +Y and up as +Z
rf292_input_position = 0.5 * np.array([-board_dimensions_xy_mm[0], 0, height_mm])
# add half the pin diameter to make sma pin lie on top of the dielectric slab surface
# rf292_pin_height_mm = 0.25
# rf292_input_position[2] += 0.5 * rf292_pin_height_mm
rf292_input_x_direction = [0, -1, 0]
rf292_input_y_direction = [1, 0, 0]
rf292_input_connector_cs_params = {"origin": "{}mm,{}mm,{}mm".format(rf292_input_position[0],
                                                                     rf292_input_position[1],
                                                                     rf292_input_position[2]).split(","),
                                   "reference_cs": "Global",
                                   "name": "rf292_port_1_CS",
                                   "mode": "axis",
                                   "view": "iso",
                                   "x_pointing": rf292_input_x_direction,
                                   "y_pointing": rf292_input_y_direction,
                                   "psi": 0,
                                   "theta": 0,
                                   "phi": 0,
                                   "u": None
                                   }
rf292_input_cs = hfss.modeler.create_coordinate_system(**rf292_input_connector_cs_params)
rf292_input_cs.set_as_working_cs()

rf292_component_list = []
port_index = 1
rf292_component_params = {"comp_file": "components/Rosenberger_292_Connector_v3.a3dcomp",
                          "geo_params": None,
                          "sz_mat_params": "",
                          "sz_design_params": "",
                          "targetCS": "rf292_port_" + str(port_index) + "_CS",
                          "name": "RF292_connector_" + str(port_index),
                          "password": "",
                          "auxiliary_dict": False
                          }
rf292_component = hfss.modeler.insert_3d_component(**rf292_component_params)
rf292_component.parameters['subH'] = str(height_mm) + "mm"
rf292_component.parameters['pinD'] = str(0.25) + "mm"
rf292_component_list.append(rf292_component)

# RF292 Connector model has pin on +Y and up as +Z
rf292_output_position = 0.5 * np.array([board_dimensions_xy_mm[0], 0, height_mm])
# add half the pin diameter to make sma pin lie on top of the dielectric slab surface
# rf292_pin_height_mm = 0.25
# rf292_output_position[2] += 0.5 * rf292_pin_height_mm
rf292_output_x_direction = [0, 1, 0]
rf292_output_y_direction = [-1, 0, 0]
rf292_output_connector_cs_params = {"origin": "{}mm,{}mm,{}mm".format(rf292_output_position[0],
                                                                      rf292_output_position[1],
                                                                      rf292_output_position[2]).split(","),
                                    "reference_cs": "Global",
                                    "name": "rf292_port_2_CS",
                                    "mode": "axis",
                                    "view": "iso",
                                    "x_pointing": rf292_output_x_direction,
                                    "y_pointing": rf292_output_y_direction,
                                    "psi": 0,
                                    "theta": 0,
                                    "phi": 0,
                                    "u": None
                                    }
rf292_output_cs = hfss.modeler.create_coordinate_system(**rf292_output_connector_cs_params)
rf292_output_cs.set_as_working_cs()

hfss.modeler.set_working_coordinate_system("Global")

port_index = 2
rf292_component_params = {"comp_file": "components/Rosenberger_292_Connector_v3.a3dcomp",
                          "geo_params": None,
                          "sz_mat_params": "",
                          "sz_design_params": "",
                          "targetCS": "rf292_port_" + str(port_index) + "_CS",
                          "name": "RF292_connector_" + str(port_index),
                          "password": "",
                          "auxiliary_dict": False
                          }
rf292_component = hfss.modeler.insert_3d_component(**rf292_component_params)
rf292_component.parameters['subH'] = str(height_mm) + "mm"
rf292_component.parameters['pinD'] = str(0.25) + "mm"
rf292_component_list.append(rf292_component)

hfss.modeler.set_working_coordinate_system("Global")

# port_1_position = np.array([-0.5 * antenna_length_mm,
#                             -0.5 * feed_rect_width_mm,
#                             -0.5 * height_mm])
# port_1_size = np.array([feed_rect_width_mm, height_mm])
# port_1_params = {"name": "port_1",
#                  "csPlane": "YZ",
#                  "position": "{}mm,{}mm,{}mm".format(port_1_position[0],
#                                                      port_1_position[1],
#                                                      port_1_position[2]).split(","),
#                  "dimension_list": "{}mm,{}mm".format(port_1_size[0],
#                                                       port_1_size[1]).split(","),
#                  "matname": None,
#                  "is_covered": True}
# port_1_geom = hfss.modeler.create_rectangle(**port_1_params)
# port_1_geom.color = radiation_box_color
#
# port_1_excitation_params = {"signal": port_1_geom,
#                             "reference": ground_plane_geom,
#                             "create_port_sheet": False,
#                             "port_on_plane": True,
#                             "integration_line": 0,
#                             "impedance": 50,
#                             "name": "port_1_excitation",
#                             "renormalize": True,
#                             "deembed": False,
#                             "terminals_rename": True}
# hfss.lumped_port(**port_1_excitation_params)
#
# port_2_position = np.array([0.5 * antenna_length_mm,
#                             -0.5 * feed_rect_width_mm,
#                             -0.5 * height_mm])
# port_2_size = port_1_size
# port_2_params = {"name": "port_2",
#                  "csPlane": "YZ",
#                  "position": "{}mm,{}mm,{}mm".format(port_2_position[0],
#                                                      port_2_position[1],
#                                                      port_2_position[2]).split(","),
#                  "dimension_list": "{}mm,{}mm".format(port_2_size[0],
#                                                       port_2_size[1]).split(","),
#                  "matname": None,
#                  "is_covered": True}
# port_2_geom = hfss.modeler.create_rectangle(**port_2_params)
# port_2_geom.color = radiation_box_color
#
# port_2_excitation_params = {"signal": port_2_geom,
#                             "reference": ground_plane_geom,
#                             "create_port_sheet": False,
#                             "port_on_plane": True,
#                             "integration_line": 0,
#                             "impedance": 50,
#                             "name": "port_2_excitation",
#                             "renormalize": True,
#                             "deembed": False,
#                             "terminals_rename": True}
# hfss.lumped_port(**port_2_excitation_params)

if enclose_antenna_with_pec_boundary:
    padding = 9.5 * 2  # mm The 2.92 connector extends 9.5 mm beyond board edge
    pec_box_inner_position = ground_plane_position - 0.5 * np.array([padding, padding, padding])
    pec_box_inner_size = padding + np.array([ground_plane_size[0],
                                             ground_plane_size[1],
                                             height_mm])
    pec_box_inner_params = {"name": "pec_box_inner",
                            "position": "{}mm,{}mm,{}mm".format(pec_box_inner_position[0],
                                                                pec_box_inner_position[1],
                                                                pec_box_inner_position[2]).split(","),
                            "dimensions_list": "{}mm,{}mm,{}mm".format(pec_box_inner_size[0],
                                                                       pec_box_inner_size[1],
                                                                       pec_box_inner_size[2]).split(","),
                            "matname": "pec"}
    pec_box_inner_geom = hfss.modeler.create_box(**pec_box_inner_params)
    pec_box_inner_geom.color = metal_color
    pec_box_inner_geom.transparency = 0.9

    padding = 10.5 * 2  # mm The 2.92 connector extends 9.5 mm beyond board edge
    pec_box_outer_position = ground_plane_position - 0.5 * np.array([padding, padding, padding])
    pec_box_outer_size = padding + np.array([ground_plane_size[0],
                                             ground_plane_size[1],
                                             height_mm])
    pec_box_outer_params = {"name": "pec_box_outer",
                            "position": "{}mm,{}mm,{}mm".format(pec_box_outer_position[0],
                                                                pec_box_outer_position[1],
                                                                pec_box_outer_position[2]).split(","),
                            "dimensions_list": "{}mm,{}mm,{}mm".format(pec_box_outer_size[0],
                                                                       pec_box_outer_size[1],
                                                                       pec_box_outer_size[2]).split(","),
                            "matname": "pec"}
    pec_box_outer_geom = hfss.modeler.create_box(**pec_box_outer_params)
    pec_box_outer_geom.color = metal_color
    pec_box_outer_geom.transparency = 0.9
    pec_box_subtract_params = {
        "blank_list": [pec_box_outer_geom.name],
        "tool_list": [pec_box_inner_geom.name],
        "keep_originals": False
    }
    hfss.modeler.subtract(**pec_box_subtract_params)

open_region_params = {
    "Frequency": "{}GHz".format(frequency_GHz),
    "Boundary": "Radiation",
    "ApplyInfiniteGP": False,
    "GPAXis": "-z"}
success = hfss.create_open_region(**open_region_params)

# oEditor = hfss.odesign.SetActiveEditor("3D Modeler")
# oEditor.ChangeProperty(
#     [
#         "NAME:AllTabs",
#         [
#             "NAME:Geometry3DCmdTab",
#             [
#                 "NAME:PropServers",
#                 "RadiatingSurface:CreateRegion:1"
#             ],
#             [
#                 "NAME:ChangedProps",
#                 [
#                     "NAME:+X Padding Data",
#                     "Value:=", "0mm"
#                 ],
#                 [
#                     "NAME:-X Padding Data",
#                     "Value:=", "0mm"
#                 ]
#             ]
#         ]
#     ])

analysis_plane_position = np.array([ground_plane_position[0], ground_plane_position[1], 0])
analysis_plane_size = ground_plane_size
analysis_plane_params = {"name": "plot_waveguide_fields",
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

solver_setup = hfss.create_setup(setupname="TLine_Setup", setuptype="HFSSDriven")
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

frequency_sweep_params = {
    "unit": "GHz",
    "freqstart": frequency_GHz - 1.5,
    "freqstop": frequency_GHz + 1.5,
    "num_of_freq_points": 200,
    "sweepname": "sweep",
    "save_fields": True,
    "save_rad_fields": False,
    "sweep_type": "Discrete",
    "interpolation_tol": 0.5,
    "interpolation_max_solutions": 250
}
solver_setup.create_frequency_sweep(**frequency_sweep_params)


##########################
# NON-LINEAR OPTIMIZER   #
##########################
from scipy.optimize import minimize

non_linear_feval_count = 0


def error_function(design_geometry_params):
    global non_linear_feval_count
    transmission_line_width_mm = design_geometry_params[0]
    print("Current parameters:", transmission_line_width_mm)
    design_components = []

    transmission_line_position = 0.5 * np.array([-transmission_line_length_mm,
                                                 -transmission_line_width_mm,
                                                 height_mm])
    transmission_line_size = np.array([transmission_line_length_mm,
                                       transmission_line_width_mm])
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

    design_components.append(transmission_line_plane_geom)

    hfss.assign_perfecte_to_sheets(
        **{"sheet_list": [transmission_line_plane_geom.name],
           "sourcename": None,
           "is_infinite_gnd": False})

    setup_ok = hfss.validate_full_design()

    setup_solver_configuration_params = {
        "name": "TLine_Setup",
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

    s11_solution_data = hfss.post.get_solution_data(expressions="dB(St(RF292_connector_1_excitation,RF292_connector_1_excitation))",
                                                    setup_sweep_name="TLine_Setup : sweep",
                                                    report_category="Terminal S Parameter")
    s21_solution_data = hfss.post.get_solution_data(expressions="dB(St(RF292_connector_2_excitation,RF292_connector_1_excitation))",
                                                    setup_sweep_name="TLine_Setup : sweep",
                                                    report_category="Terminal S Parameter")
    # 'Solution Convergence'
    # convergence_solution_data = hfss.post.get_solution_data(
    #       setup_sweep_name="MTS_EigenMode_Setup : LastAdaptive",
    #       report_category="Solution Convergence")

    vals_np_real = np.array(
        list(s11_solution_data.full_matrix_real_imag[0]['dB(St(RF292_connector_1_excitation,RF292_connector_1_excitation))'].values()))
    s11_vals = np.squeeze(vals_np_real)
    s11_freqs = np.array(s21_solution_data.primary_sweep_values)
    s11_freqs_units_str = s11_solution_data.units_sweeps['Freq']  # 'GHz'
    vals_np_real = np.array(
        list(s21_solution_data.full_matrix_real_imag[0]['dB(St(RF292_connector_2_excitation,RF292_connector_1_excitation))'].values()))
    s21_vals = np.squeeze(vals_np_real)
    s21_freqs = s21_solution_data.primary_sweep_values
    s21_freqs_units_str = s21_solution_data.units_sweeps['Freq']  # 'GHz'

    non_linear_feval_count += 1

    plt.close()
    plt.figure()
    plt.plot(s11_freqs, s11_vals)
    plt.plot(s21_freqs, s21_vals)
    plt.title("Non-Linear Optimization Evaluation " + str(non_linear_feval_count) +
              ": x = ({:.2f})".format( design_geometry_params[0]))
    plt.ylabel("dB")
    plt.xlabel(s21_freqs_units_str)
    plt.legend(["S_11", "S_21"])
    plt.savefig('feed_optimization_iteration_{:03d}.png'.format(non_linear_feval_count))

    freq_idxs_of_interest = np.argwhere((s11_freqs > frequency_GHz-1) & (s11_freqs < frequency_GHz+1))
    # error = np.sum(-s21_vals[freq_idxs_of_interest] < -2) + np.sum(s11_vals[freq_idxs_of_interest] > -9)
    error = np.sum(-4 * s21_vals[freq_idxs_of_interest]) + np.sum(s11_vals[freq_idxs_of_interest])
    print("Error = " + str(error))
    for component in design_components:
        component.delete()
    return error


# initial_feed_geometry = np.array([0.5, 0.1])
#  optimized result for 10 iterations
initial_feed_geometry = np.array([0.48])
# Use the minimize function to find the minimum of the objective function
parameter_bounds = [(0.15, 3.95)]
minimize_options = {"maxiter": 40, "disp": True}
# Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, and COBYLA methods
result = minimize(error_function, initial_feed_geometry,
                  bounds=parameter_bounds, method="L-BFGS-B", options=minimize_options)

print("Optimal parameters:", result.x)
print("Minimum value:", result.fun)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)
print("Optimization required {} time.".format(time_difference_str))
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
