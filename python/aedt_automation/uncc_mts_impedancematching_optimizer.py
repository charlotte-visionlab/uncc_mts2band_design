"""
Impedance matching optimization
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
frequency = 17e9
wavelength = speed_of_light / frequency
cm2mm = 10

height_cm = 100 * 1.57e-3  # <=== dielectric slab height in meters
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

# DIELECTRIC MATERIALS
# dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
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

# antenna_dimensions_xy_cm = np.array([40, 12.4])
# antenna_margin_xy_cm = np.array([1, 1])
# antenna_coord_origin_xy_cm = np.array([10, 6.2])
# antenna_dimensions_xy_cm = np.array([10, 5.4])
board_dimensions_xy_cm = np.array([5, 2.4])
board_margin_xy_cm = np.array([.25, .25])

antenna_pct_board_x = 0.75
antenna_size_cm = np.array([antenna_pct_board_x * board_dimensions_xy_cm[0], 0.7])
# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
# antenna_coord_origin_xy_cm = np.array([0.3 * board_dimensions_xy_cm[0],
#                                        0.5 * board_dimensions_xy_cm[1]])
# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "SMA Microstrip Feed " + current_time_str
design_name = "SMA Feed " + current_time_str

save_file_prefix = "sma_feed"
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
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_plane_position = cm2mm * np.array([-board_dimensions_xy_cm[0] / 2,
                                          -board_dimensions_xy_cm[1] / 2,
                                          -height_cm / 2])
ground_plane_size = cm2mm * np.array([board_dimensions_xy_cm[0],
                                      board_dimensions_xy_cm[1]])

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
                                 cm2mm * height_cm])
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

microstrip_position_cm = cm2mm * np.array([-antenna_size_cm[0] / 2,
                                           -antenna_size_cm[1] / 2,
                                           height_cm / 2])
microstrip_plane_params = {"name": "microstrip_plane",
                           "csPlane": "XY",
                           "position": "{}mm,{}mm,{}mm".format(microstrip_position_cm[0],
                                                               microstrip_position_cm[1],
                                                               microstrip_position_cm[2]).split(","),
                           "dimension_list": "{}mm,{}mm".format(cm2mm * antenna_size_cm[0],
                                                                cm2mm * antenna_size_cm[1]).split(","),
                           "matname": ground_plane_material_name,
                           "is_covered": True}
microstrip_plane_geom = hfss.modeler.create_rectangle(**microstrip_plane_params)
microstrip_plane_geom.color = metal_color

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "cell_ground_plane",
       "sourcename": None,
       "is_infinite_gnd": False})
hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "microstrip_plane",
       "sourcename": None,
       "is_infinite_gnd": False})

# SMA Connector model has pin on +Y and and up as +Z
sma_input_position = cm2mm * np.array([-board_dimensions_xy_cm[0] / 2, 0, height_cm / 2])
# add half the pin diameter to make sma pin lie on top of the dielectric slab surface
sma_pin_height_cm = 0.036
sma_input_position[2] += cm2mm * sma_pin_height_cm / 2
sma_input_x_direction = [0, -1, 0]
sma_input_y_direction = [1, 0, 0]
sma_input_connector_cs_params = {"origin": "{}mm,{}mm,{}mm".format(sma_input_position[0],
                                                                   sma_input_position[1],
                                                                   sma_input_position[2]).split(","),
                                 "reference_cs": "Global",
                                 "name": "sma_port_1_CS",
                                 "mode": "axis",
                                 "view": "iso",
                                 "x_pointing": sma_input_x_direction,
                                 "y_pointing": sma_input_y_direction,
                                 "psi": 0,
                                 "theta": 0,
                                 "phi": 0,
                                 "u": None
                                 }
sma_input_cs = hfss.modeler.create_coordinate_system(**sma_input_connector_cs_params)

sma_output_position = cm2mm * np.array([board_dimensions_xy_cm[0] / 2, 0, height_cm / 2])
# add half the pin diameter to make sma pin lie on top of the dielectric slab surface
sma_output_position[2] += cm2mm * sma_pin_height_cm / 2
sma_output_x_direction = [0, 1, 0]
sma_output_y_direction = [-1, 0, 0]
sma_output_connector_cs_params = {"origin": "{}mm,{}mm,{}mm".format(sma_output_position[0],
                                                                    sma_output_position[1],
                                                                    sma_output_position[2]).split(","),
                                  "reference_cs": "Global",
                                  "name": "sma_port_2_CS",
                                  "mode": "axis",
                                  "view": "iso",
                                  "x_pointing": sma_output_x_direction,
                                  "y_pointing": sma_output_y_direction,
                                  "psi": 0,
                                  "theta": 0,
                                  "phi": 0,
                                  "u": None
                                  }
sma_output_cs = hfss.modeler.create_coordinate_system(**sma_output_connector_cs_params)

sma_input_cs.set_as_working_cs()
feed_coordinate_systems = [sma_input_cs, sma_output_cs]

sma_components = []

for index, feed_cs in enumerate(feed_coordinate_systems):
    feed_cs.set_as_working_cs()
    port_index = index + 1
    sma_component_params = {"comp_file": "components/SMA_connector_v3.a3dcomp",
                            "geo_params": None,
                            "sz_mat_params": "",
                            "sz_design_params": "",
                            "targetCS": "sma_port_" + str(port_index) + "_CS",
                            "name": "SMA_connector_" + str(port_index),
                            "password": "",
                            "auxiliary_dict": False
                            }
    sma_component = hfss.modeler.insert_3d_component(**sma_component_params)
    sma_component.parameters['subH'] = str(cm2mm * height_cm) + "mm"
    sma_components.append(sma_component)

module = hfss.get_module("ModelSetup")
module.CreateOpenRegion(
    [
        "NAME:Settings",
        "OpFreq:=", "17GHz",
        "Boundary:=", "Radiation",
        "ApplyInfiniteGP:=", False
    ])
hfss.modeler.set_working_coordinate_system("Global")
oEditor = hfss.odesign.SetActiveEditor("3D Modeler")
oEditor.ChangeProperty(
    [
        "NAME:AllTabs",
        [
            "NAME:Geometry3DCmdTab",
            [
                "NAME:PropServers",
                "RadiatingSurface:CreateRegion:1"
            ],
            [
                "NAME:ChangedProps",
                [
                    "NAME:+Y Padding Data",
                    "Value:=", "0mm"
                ],
                [
                    "NAME:-Y Padding Data",
                    "Value:=", "0mm"
                ]
            ]
        ]
    ])

solver_setup = hfss.create_setup(setupname="AntennaFeed_Setup", setuptype="HFSSDriven")
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '17GHz',
                       "MaxDeltaS": 0.03,
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
    "freqstart": 5,
    "freqstop": 25,
    "num_of_freq_points": 200,
    "sweepname": "sweep",
    "save_fields": True,
    "save_rad_fields": False,
    "sweep_type": "Interpolating",
    "interpolation_tol": 0.5,
    "interpolation_max_solutions": 250
}
solver_setup.create_frequency_sweep(**frequency_sweep_params)

from scipy.optimize import minimize

non_linear_feval_count = 0


def error_function(feed_geometry_params):
    global non_linear_feval_count
    feed_rect_length_pct = feed_geometry_params[0]
    feed_rect_width_pct = feed_geometry_params[1]
    print("Current parameters:", feed_geometry_params)

    # feed_rect_length_pct = 0.5
    # feed_rect_width_pct = 0.1
    sma_board_edge_to_feed_gap_cm = 0.03
    feed_components = []

    for index, feed_cs in enumerate(feed_coordinate_systems):
        feed_cs.set_as_working_cs()
        port_index = index + 1
        #  build a feed structure having feed_length_cm length along the Y axis (the propagation axis)
        feed_total_length_cm = (board_dimensions_xy_cm[0] - antenna_size_cm[0]) / 2
        feed_rect_length_cm = (feed_rect_length_pct * feed_total_length_cm) - sma_board_edge_to_feed_gap_cm
        feed_rect_width_cm = feed_rect_width_pct * feed_total_length_cm
        feed_rect_position = cm2mm * np.array([-feed_rect_width_cm / 2,
                                               sma_board_edge_to_feed_gap_cm,
                                               -sma_pin_height_cm / 2])
        feed_rect_size = cm2mm * np.array([feed_rect_width_cm, feed_rect_length_cm])
        feed_rect_params = {"name": "feed_rectanglar_portion_" + str(port_index),
                            "csPlane": "XY",
                            "position": "{}mm,{}mm,{}mm".format(feed_rect_position[0],
                                                                feed_rect_position[1],
                                                                feed_rect_position[2]).split(","),
                            "dimension_list": "{}mm,{}mm".format(feed_rect_size[0],
                                                                 feed_rect_size[1]).split(","),
                            "matname": ground_plane_material_name,
                            "is_covered": True}
        feed_rect_geom = hfss.modeler.create_rectangle(**feed_rect_params)
        feed_rect_geom.color = metal_color
        feed_rect_geom.part_coordinate_system = feed_cs.name
        feed_components.append(feed_rect_geom)

        trap_position_list = cm2mm * np.array([np.array([-feed_rect_width_cm / 2,
                                                         sma_board_edge_to_feed_gap_cm + feed_rect_length_cm,
                                                         -sma_pin_height_cm / 2]),
                                               np.array([-antenna_size_cm[1] / 2,
                                                         feed_total_length_cm, -sma_pin_height_cm / 2]),
                                               np.array([antenna_size_cm[1] / 2,
                                                         feed_total_length_cm, -sma_pin_height_cm / 2]),
                                               np.array([feed_rect_width_cm / 2,
                                                         sma_board_edge_to_feed_gap_cm + feed_rect_length_cm,
                                                         -sma_pin_height_cm / 2])])
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
        feed_trap_geom = hfss.modeler.create_polyline(**trap_polyline_params)
        feed_trap_geom.color = metal_color
        feed_trap_geom.transparency = 0.2
        feed_trap_geom.part_coordinate_system = feed_cs.name
        hfss.assign_perfecte_to_sheets(
            **{"sheet_list": [feed_rect_geom.name, feed_trap_geom.name],
               "sourcename": None,
               "is_infinite_gnd": False})
        feed_components.append(feed_trap_geom)

    setup_ok = hfss.validate_full_design()

    setup_solver_configuration_params = {
        "name": "AntennaFeed_Setup",
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

    s11_solution_data = hfss.post.get_solution_data(expressions="dB(St(SMA_connector_1_1,SMA_connector_1_1))",
                                                    setup_sweep_name="AntennaFeed_Setup : sweep",
                                                    report_category="Terminal S Parameter")
    s21_solution_data = hfss.post.get_solution_data(expressions="dB(St(SMA_connector_2_1,SMA_connector_1_1))",
                                                    setup_sweep_name="AntennaFeed_Setup : sweep",
                                                    report_category="Terminal S Parameter")
    # 'Solution Convergence'
    # convergence_solution_data = hfss.post.get_solution_data(
    #       setup_sweep_name="MTS_EigenMode_Setup : LastAdaptive",
    #       report_category="Solution Convergence")

    vals_np_real = np.array(
        list(s11_solution_data.full_matrix_real_imag[0]['dB(St(SMA_connector_1_1,SMA_connector_1_1))'].values()))
    s11_vals = np.squeeze(vals_np_real)
    s11_freqs = np.array(s21_solution_data.primary_sweep_values)
    s11_freqs_units_str = s11_solution_data.units_sweeps['Freq']  # 'GHz'
    vals_np_real = np.array(
        list(s21_solution_data.full_matrix_real_imag[0]['dB(St(SMA_connector_2_1,SMA_connector_1_1))'].values()))
    s21_vals = np.squeeze(vals_np_real)
    s21_freqs = s21_solution_data.primary_sweep_values
    s21_freqs_units_str = s21_solution_data.units_sweeps['Freq']  # 'GHz'

    non_linear_feval_count += 1

    plt.close()
    plt.figure()
    plt.plot(s11_freqs, s11_vals)
    plt.plot(s21_freqs, s21_vals)
    plt.title("Non-Linear Optimization Evaluation " + str(non_linear_feval_count) +
              ": x = ({:.2f},{:.2f})".format(feed_geometry_params[0], feed_geometry_params[1]))
    plt.ylabel("dB")
    plt.xlabel(s21_freqs_units_str)
    plt.legend(["S_11", "S_21"])
    plt.savefig('feed_optimization_iteration_{:03d}.png'.format(non_linear_feval_count))

    freq_idxs_of_interest = np.argwhere((s11_freqs > 12) & (s11_freqs < 18))
    # error = np.sum(-s21_vals[freq_idxs_of_interest] < -2) + np.sum(s11_vals[freq_idxs_of_interest] > -9)
    error = np.sum(-4 * s21_vals[freq_idxs_of_interest]) + np.sum(s11_vals[freq_idxs_of_interest])
    print("Error = " + str(error))
    for component in feed_components:
        component.delete()
    return error


# feed_rect_length_pct = 0.5
# feed_rect_width_pct = 0.1
# initial_feed_geometry = np.array([0.5, 0.1])
#  optimized result for 10 iterations
initial_feed_geometry = np.array([0.229, 0.457])
# Use the minimize function to find the minimum of the objective function
parameter_bounds = [(0.05, 0.95), (0.05, 0.8)]
minimize_options = {"maxiter": 40, "disp": True}
# Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, and COBYLA methods
result = minimize(error_function, initial_feed_geometry,
                  bounds=parameter_bounds, method="L-BFGS-B", options=minimize_options)

print("Optimal parameters:", result.x)
print("Minimum value:", result.fun)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)

###############################################################################
# Close Ansys Electronics Desktop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hfss.release_desktop(close_projects=True, close_desktop=True)
