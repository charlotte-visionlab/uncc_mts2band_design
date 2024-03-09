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

from uncc_mts_transmissionline_metasurface_antenna import construct_mts_lw_antenna
from uncc_mts_transmissionline_metasurface_antenna_high import construct_hf_lw_antenna

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
height_mm = 0.787  # mm   <=== dielectric slab height in millimeters
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

frequency_GHz = 19.6
wavelength_mm = 1e3 * speed_of_light / (frequency_GHz * 1e9)

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

i_height1_mm = 0.787  # mm   <=== dielectric slab height in millimeters
# MATERIALS
dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
ground_plane_material_name = "pec"
I_OMIT_MTS_UNIT_CELLS = False
I_INSERT_HEXAGON_MTS_UNIT_CELLS = True
I_USE_CONNECTORS = True

ant1_geoms, ant1_attributes = construct_mts_lw_antenna(hfss=hfss,
                                                       height_mm=i_height1_mm,
                                                       ground_plane_material_name=ground_plane_material_name,
                                                       dielectric_material_name=dielectric_material_name,
                                                       OMIT_MTS_UNIT_CELLS=I_OMIT_MTS_UNIT_CELLS,
                                                       USE_CONNECTORS=I_USE_CONNECTORS,
                                                       INSERT_HEXAGON_MTS_UNIT_CELLS=I_INSERT_HEXAGON_MTS_UNIT_CELLS)

i_board_length_mm = 42
i_height2_mm = 0.09  # mm   <=== dielectric slab height in millimeters

i_num_patches = 16
i_feed_trapezoid_start_width_mm = 0.25
i_feed_trapezoid_length_mm = 0.9
i_patch_length_mm = 0.8  # mm
i_gap_length_mm = 1.5  # mm
i_strip_width_mm = 0.7  # mm
i_slot_width_pct = 0.8
i_phase_offset_mm = 0
dielectric_material_name = "glass"

antenna_parameters = [i_num_patches, i_feed_trapezoid_start_width_mm, i_feed_trapezoid_length_mm,
                      i_patch_length_mm, i_gap_length_mm, i_strip_width_mm, i_slot_width_pct,
                      i_phase_offset_mm]
args = hfss, i_board_length_mm, i_height2_mm, ground_plane_material_name, "transmission_line"

x_offset = 2.6
hfss.modeler.set_working_coordinate_system("Global")
for idx in np.arange(-11, 12):
    ant2_geoms, ant2_attributes = construct_hf_lw_antenna(antenna_parameters, args)

    for geom in ant2_geoms:
        hfss.modeler.rotate(geom, cs_axis="Z", angle=90.0, unit="deg")
        hfss.modeler.move(geom, vector=[x_offset + idx * 8, 0, 0.5 * (i_height1_mm + i_height2_mm)])

    i_board_width_mm = i_strip_width_mm + 0.5  # 2 * board_margin_xy_mm[0]
    antenna_dimensions_xy_mm = np.array([i_board_length_mm, i_board_width_mm])
    dielectric_slab_position = -0.5 * np.array([antenna_dimensions_xy_mm[0],
                                                antenna_dimensions_xy_mm[1],
                                                i_height2_mm])
    dielectric_slab_size = np.array([antenna_dimensions_xy_mm[0],
                                     antenna_dimensions_xy_mm[1],
                                     i_height2_mm])
    dielectric_slab_params = {"name": "dielectric_slab_hf_" + str(idx + 100),
                              "position": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
                                                                  dielectric_slab_position[1],
                                                                  dielectric_slab_position[2]).split(","),
                              "dimensions_list": "{}mm,{}mm,{}mm".format(dielectric_slab_size[0],
                                                                         dielectric_slab_size[1],
                                                                         dielectric_slab_size[2]).split(","),
                              "matname": dielectric_material_name}
    dielectric_slab_geom = hfss.modeler.create_box(**dielectric_slab_params)
    dielectric_slab_geom.color = dielectric_color
    hfss.modeler.rotate(dielectric_slab_geom, cs_axis="Z", angle=90.0, unit="deg")
    hfss.modeler.move(dielectric_slab_geom, vector=[x_offset + idx * 8, 0, 0.5 * (i_height1_mm + i_height2_mm)])

open_region_params = {
    "Frequency": "{}GHz".format(frequency_GHz),
    "Boundary": "Radiation",
    "ApplyInfiniteGP": False,
    "GPAXis": "-z"}
success = hfss.create_open_region(**open_region_params)

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
