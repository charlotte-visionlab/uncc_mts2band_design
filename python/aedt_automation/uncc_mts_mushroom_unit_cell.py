"""
MTS: Unit Cell Simulation
Experiment 0
-------------------
This example shows how you can use PyAEDT to create a multipart scenario in HFSS
and set up a metasurface unit cell dispersion analysis.

This is an HFSS implementation of dispersion calculation for a unit cell consisting of a small metal patch.

When the patch size is equal to the cell size the dispersion matches that of a parallel plate waveguide.
Results for this case match those shown on the web at: http://www.emtalk.com/tut_2.htm

The fill percentage, i.e., ratio of the metal patch dimensions in (X,Y) to the size to the unit cell, is controlled
via the fill_pct variable which takes on a value  (0,0) <= fill_pct <= (1,1) and determines the ratio of the square
metal patch size in (X,Y) to the unit cell size in (X,Y).

Note this simulation uses a Perfectly Matched Layer for the broadside radiation face of the unit cell (the top/up face).
When the unit cell size changes the parameters of the PML construction will also need to be adjusted manually.
"""

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import scipy.signal
import socket
from datetime import datetime

import pyaedt
import matplotlib.pyplot as plt
import numpy as np

os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
# os.environ["ANSYSEM_ROOT231"] = "C:\\Program Files\\AnsysEM\\v231\\Win64\\"
aedt_version = "2023.1"

###############################################################################
# Define program variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

speed_of_light = 2.99792458e8
frequency = 17e9
wavelength = speed_of_light / frequency
# sub_wavelength_factor = 10
sub_wavelength_factor = wavelength / 5.0e-3  # generates a 5mm x 5mm unit cell size
height = 1.27e-3  # m
mushroom_via_radius = 0.12e-3  # m
fill_pct = 0.96 * np.array([1.0, 1.0])  # fill_pct 0.96 ==> 4.8mm x 4.8mm square patch unit cell

# from Fong et. al. 2010 suggested as the primary/secondary phase difference across a cell w/ periodic BCs
# unit_cell_phase_difference = 72  # degrees

# Various values are suggested for the height of the solution volume
# 6*height of dielectric ==> from https://www.youtube.com/watch?v=AXOpFgJcI38
# 6*height of dielectric ==> from https://www.emtalk.com/tut_3.htm
# half the wavelength    ==> from B. H. Fong, J. S. Colburn, J. J. Ottusch, J. L. Visher and D. F. Sievenpiper,
#   "Scalar and Tensor Holographic Artificial Impedance Surfaces," in IEEE Transactions on Antennas and Propagation,
#   vol. 58, no. 10, pp. 3212-3221, Oct. 2010, doi: 10.1109/TAP.2010.2055812.
# radiation_volume_height = height + wavelength / 2.0
radiation_volume_height = height + 6 * height

# from Fong et. al. 2010 suggested as the position to integrate the mag(Ex)/mag(Hy) for impedance calculation
integration_surface_height = 0.5 * radiation_volume_height

time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")
save_file_prefix = "dispersion_data"
save_filename = save_file_prefix + "_" + current_time_str + "_" + socket.gethostname() + ".mat"

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
project_name = "Mushroom Unit Cell " + current_time_str
design_name = "MTS HFSS " + current_time_str

# Plot a course in the Brillouin zone 1
number_phase_steps_per_segment = 10
phase_step_size = 180 / (number_phase_steps_per_segment - 1)
x_phase_delays = np.arange(0, 180, phase_step_size)
y_phase_delays = [0] * len(x_phase_delays)
gamma_to_X_phase_pairs = {'path_name': 'Gamma_to_X', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
x_phase_delays = [180] * len(y_phase_delays)
y_phase_delays = np.arange(0, 180, phase_step_size)
X_to_M_phase_pairs = {'path_name': 'X_to_M', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
x_phase_delays = np.arange(180, 0, -phase_step_size)
y_phase_delays = np.arange(180, 0, -phase_step_size)
M_to_gamma_phase_pairs = {'path_name': 'M_to_Gamma', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
brillouin_zone_phase_list = [gamma_to_X_phase_pairs, X_to_M_phase_pairs, M_to_gamma_phase_pairs]

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
perfectly_matched_layer_color = [255, 128, 128]

# DIELECTRIC MATERIALS
# dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
ground_plane_material_name = "pec"
unit_cell_material_name = "pec"
radiation_box_material_name = "air"
# solution_object_material_name = "vacuum"


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


def get_ComplexMonostaticRCS_solution_data():
    solution_data = hfss.post.get_solution_data(expressions='ComplexMonostaticRCSPhi',
                                                setup_sweep_name=hfss.nominal_sweep,
                                                report_category='Monostatic RCS')
    # plot_units_intrinsic = solution_data.active_intrinsic
    plot_units_info = solution_data.primary_sweep_variations
    freq_vals = np.array(solution_data.primary_sweep_values)
    imag_vals = np.array(solution_data.data_imag())
    real_vals = np.array(solution_data.data_real())
    return real_vals + 1j * imag_vals, freq_vals, plot_units_info


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
    solution_type="Eigenmode",
    new_desktop_session=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical
)

hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(wavelength))
hfss.variable_manager.set_variable("height", expression="{}mm".format(height))
hfss.variable_manager.set_variable("sub_wavelength_factor", expression="{}".format(sub_wavelength_factor))
hfss.variable_manager.set_variable("x_fill_pct", expression="{}percent".format(fill_pct[0]))
hfss.variable_manager.set_variable("y_fill_pct", expression="{}percent".format(fill_pct[1]))
hfss.variable_manager.set_variable("radiation_volume_height", expression="{}mm".format(radiation_volume_height))
hfss.variable_manager.set_variable("integration_surface_height", expression="{}mm".format(integration_surface_height))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Construct the unit cell ground plane
ground_plane_position = [-wavelength / (2 * sub_wavelength_factor),
                         -wavelength / (2 * sub_wavelength_factor),
                         -height / 2]
ground_plane_size = [wavelength / sub_wavelength_factor,
                     wavelength / sub_wavelength_factor]

#  csPlane is either "XY", "YZ", or "XZ"
ground_plane_params = {"name": "cell_ground_plane",
                       "csPlane": "XY",
                       "position": "{}m,{}m,{}m".format(ground_plane_position[0],
                                                        ground_plane_position[1],
                                                        ground_plane_position[2]).split(","),
                       "dimension_list": "{}m,{}m".format(ground_plane_size[0],
                                                          ground_plane_size[1]).split(","),
                       "matname": ground_plane_material_name,
                       "is_covered": True}
ground_plane_geom = hfss.modeler.create_rectangle(**ground_plane_params)
ground_plane_geom.color = metal_color

# FIT ALL
hfss.modeler.fit_all()

# Construct the square metal patch unit cell surface
unit_cell_plane_position = [-fill_pct[0] * wavelength / (2 * sub_wavelength_factor),
                            -fill_pct[1] * wavelength / (2 * sub_wavelength_factor),
                            height / 2]
unit_cell_plane_size = fill_pct * [wavelength / sub_wavelength_factor,
                                   wavelength / sub_wavelength_factor]
unit_cell_plane_params = {"name": "unit_cell",
                          "csPlane": "XY",
                          "position": "{}m,{}m,{}m".format(unit_cell_plane_position[0],
                                                           unit_cell_plane_position[1],
                                                           unit_cell_plane_position[2]).split(","),
                          "dimension_list": "{}m,{}m".format(unit_cell_plane_size[0],
                                                             unit_cell_plane_size[1]).split(","),
                          "matname": unit_cell_material_name,
                          "is_covered": True}
unit_cell_plane_geom = hfss.modeler.create_rectangle(**unit_cell_plane_params)
unit_cell_plane_geom.color = metal_color

# Construct the via from the ground plane to the metal patch
# s_axis is "X", "Y" or "Z"
unit_cell_via_params = {"name": "unit_cell_via",
                        "cs_axis": "Z",
                        "position": "{}m,{}m,{}m".format(0,
                                                         0,
                                                         -height/2).split(","),
                        "radius": "{}m".format(mushroom_via_radius),
                        "height": "{}m".format(height),
                        "numSides": 0,
                        "matname": unit_cell_material_name}
unit_cell_via_geom = hfss.modeler.create_cylinder(**unit_cell_via_params)
unit_cell_via_geom.color = metal_color

# Construct the dielectric slab
dielectric_slab_position = ground_plane_position
dielectric_slab_size = [ground_plane_size[0], ground_plane_size[1], height]
dielectric_slab_params = {"name": "dielectric_slab",
                          "position": "{}m,{}m,{}m".format(dielectric_slab_position[0],
                                                           dielectric_slab_position[1],
                                                           dielectric_slab_position[2]).split(","),
                          "dimensions_list": "{}m,{}m,{}m".format(dielectric_slab_size[0],
                                                                  dielectric_slab_size[1],
                                                                  dielectric_slab_size[2]).split(","),
                          "matname": dielectric_material_name}
dielectric_slab_geom = hfss.modeler.create_box(**dielectric_slab_params)
dielectric_slab_geom.color = dielectric_color

# Construct a solution volume and typically it is assigned a vacuum of free space dielectric constant
radiation_volume_position = ground_plane_position
radiation_volume_size = [ground_plane_size[0], ground_plane_size[1], radiation_volume_height]
radiation_volume_params = {"name": "radiation_volume_1",
                          "position": "{}m,{}m,{}m".format(radiation_volume_position[0],
                                                           radiation_volume_position[1],
                                                           radiation_volume_position[2]).split(","),
                          "dimensions_list": "{}m,{}m,{}m".format(radiation_volume_size[0],
                                                                  radiation_volume_size[1],
                                                                  radiation_volume_size[2]).split(","),
                          "matname": radiation_box_material_name}
radiation_volume_geom = hfss.modeler.create_box(**radiation_volume_params)
radiation_volume_geom.color = radiation_box_color
radiation_volume_geom.display_wireframe = 1
radiation_volume_geom.transparency = 1

# Construct the unit cell ground plane
ground_plane_position = [-wavelength / (2 * sub_wavelength_factor),
                         -wavelength / (2 * sub_wavelength_factor),
                         -height / 2]
ground_plane_size = [wavelength / sub_wavelength_factor,
                     wavelength / sub_wavelength_factor]

# FIT ALL
hfss.modeler.fit_all()

# Construct a Perfectly Matched Layer (PML) box to absorb radiated energy at the top of the unit cell simulation volume
module = hfss.get_module("BoundarySetup")
module.CreatePML(
	[
		"NAME:PMLCreationSettings",
		"UserDrawnGroup:="	, False,
		"PMLFaces:="		, [radiation_volume_geom.faces[0].id],
		"Thickness:="		, "0.0025mm",
		"CreateJoiningObjs:="	, False,
		"PMLObj:="		, -1,
		"BaseObj:="		, radiation_volume_geom.id,
		"Orientation:="		, "Undefined",
		"UseFreq:="		, True,
		"MinFreq:="		, "1GHz",
		"MinBeta:="		, 20,
		"RadDist:="		, "0.00629666666666667mm"
	])

# FIT ALL
hfss.modeler.fit_all()

radiation_volume2_position = ground_plane_position
radiation_volume2_size = [ground_plane_size[0], ground_plane_size[1], radiation_volume_height + 0.0025]
radiation_volume2_params = {"name": "radiation_volume_2",
                          "position": "{}m,{}m,{}m".format(radiation_volume2_position[0],
                                                           radiation_volume2_position[1],
                                                           radiation_volume2_position[2]).split(","),
                          "dimensions_list": "{}m,{}m,{}m".format(radiation_volume2_size[0],
                                                                  radiation_volume2_size[1],
                                                                  radiation_volume2_size[2]).split(","),
                          "matname": radiation_box_material_name}
radiation_volume2_geom = hfss.modeler.create_box(**radiation_volume2_params)
radiation_volume2_geom.color = radiation_box_color
radiation_volume2_geom.display_wireframe = 1
radiation_volume2_geom.transparency = 1

# Construct a far field plane within the solution volume
solution_fields_surface_position = [ground_plane_position[0],
                                    ground_plane_position[1],
                                    integration_surface_height]
solution_fields_surface_size = ground_plane_size
solution_fields_surface_params = {"name": "solution_fields_plane",
                                  "csPlane": "XY",
                                  "position": "{}m,{}m,{}m".format(solution_fields_surface_position[0],
                                                                   solution_fields_surface_position[1],
                                                                   solution_fields_surface_position[2]).split(","),
                                  "dimension_list": "{}m,{}m".format(solution_fields_surface_size[0],
                                                                     solution_fields_surface_size[1]).split(","),
                                  "matname": radiation_box_material_name,
                                  "is_covered": True}
solution_fields_surface_geom = hfss.modeler.create_rectangle(**solution_fields_surface_params)
solution_fields_surface_geom.color = radiation_box_color
solution_fields_surface_geom.display_wireframe = 1
solution_fields_surface_geom.transparency = 1

# FIT ALL
hfss.modeler.fit_all()

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "cell_ground_plane",
       "sourcename": None,
       "is_infinite_gnd": False})
hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "unit_cell",
       "sourcename": None,
       "is_infinite_gnd": False})
hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "unit_cell_via",
       "sourcename": None,
       "is_infinite_gnd": False})

# Setup primary and secondary coupled, i.e., periodic boundary conditions
# BOX FACE INDICES ARE AS FOLLOWS
# faces = [ 0 = "+Z", "1" = "-Y", "2" = "-X", "3" = "+Y", "4" = "-Z", "5" = "+X"]
propagation_volume_geom = radiation_volume2_geom
propagation_volume_position = radiation_volume2_position
propagation_volume_size = radiation_volume2_size
# propagation_volume_geom = dielectric_slab_geom
# propagation_volume_position = dielectric_slab_position
# propagation_volume_size = dielectric_slab_size
primary_x_prop = {"face": propagation_volume_geom.faces[2],
                  "u_start": "{}m,{}m,{}m".format(propagation_volume_position[0],
                                                  propagation_volume_position[1],
                                                  propagation_volume_position[2]).split(","),
                  "u_end": "{}m,{}m,{}m".format(propagation_volume_position[0],
                                                propagation_volume_position[1] + propagation_volume_size[1],
                                                propagation_volume_position[2]).split(","),
                  "reverse_v": False,
                  "coord_name": "Global",
                  "primary_name": "x_lattice_prop_primary"}
x_primary_boundary = hfss.assign_primary(**primary_x_prop)

# Phase parameter can be "UseScanAngle", "UseScanUV",  and "InputPhaseDelay"
# Interpretation of (phase_delay_param1,phase_delay_param2) follows:
# "UseScanAngle"--(Phi,Theta) , "UseScanUV"--(U,V) "InputPhaseDelay"--(Phase, UNUSED)
secondary_x_prop = {"face": propagation_volume_geom.faces[5],
                    "primary_name": "x_lattice_prop_primary",
                    "u_start": "{}m,{}m,{}m".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                    propagation_volume_position[1],
                                                    propagation_volume_position[2]).split(","),
                    "u_end": "{}m,{}m,{}m".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                  propagation_volume_position[1] + propagation_volume_size[1],
                                                  propagation_volume_position[2]).split(","),
                    "reverse_v": True,
                    "phase_delay": "InputPhaseDelay",
                    "phase_delay_param1": "{}deg".format(90),
                    "phase_delay_param2": "0deg",
                    "coord_name": "Global",
                    "secondary_name": "x_lattice_prop_secondary"}
x_secondary_boundary = hfss.assign_secondary(**secondary_x_prop)

primary_y_prop = {"face": propagation_volume_geom.faces[1],
                  "u_start": "{}m,{}m,{}m".format(propagation_volume_position[0],
                                                  propagation_volume_position[1],
                                                  propagation_volume_position[2]).split(","),
                  "u_end": "{}m,{}m,{}m".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                propagation_volume_position[1],
                                                propagation_volume_position[2]).split(","),
                  "reverse_v": True,
                  "coord_name": "Global",
                  "primary_name": "y_lattice_prop_primary"}
y_primary_boundary = hfss.assign_primary(**primary_y_prop)
secondary_y_prop = {"face": propagation_volume_geom.faces[3],
                    "primary_name": "y_lattice_prop_primary",
                    "u_start": "{}m,{}m,{}m".format(propagation_volume_position[0],
                                                    propagation_volume_position[1] + propagation_volume_size[1],
                                                    propagation_volume_position[2]).split(","),
                    "u_end": "{}m,{}m,{}m".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                  propagation_volume_position[1] + propagation_volume_size[1],
                                                  propagation_volume_position[2]).split(","),
                    "reverse_v": False,
                    "phase_delay": "InputPhaseDelay",
                    "phase_delay_param1": "{}deg".format(0),
                    "phase_delay_param2": "0deg",
                    "coord_name": "Global",
                    "secondary_name": "y_lattice_prop_secondary"}
y_secondary_boundary = hfss.assign_secondary(**secondary_y_prop)

# radiation_faces_params = {"faces_id": [solution_volume_geom.faces[0],
#                                        solution_volume_geom.faces[4]],
#                           "boundary_name": "radiation_boundary"}
# hfss.assign_radiation_boundary_to_faces(**radiation_faces_params)

# setup_type = "HFSSDrivenAuto", "HFSSDrivenDefault", "HFSSEigen", "HFSSTransient", "HFSSSBR"
eigen_solver_params_default = {"MinimumFrequency": "1GHz",
                               "NumModes": 1,
                               "MaxDeltaFreq": 1,
                               "ConvergeOnRealFreq": True,
                               "MaximumPasses": 15,
                               "MinimumPasses": 3,
                               "MinimumConvergedPasses": 3,
                               "PercentRefinement": 30,
                               "IsEnabled": True,
                               # MeshLink', SetupProps([('ImportMesh', False)])),
                               "BasisOrder": 1,
                               "DoLambdaRefine": True,
                               "DoMaterialLambda": True,
                               "SetLambdaTarget": False,
                               "Target": 0.2,
                               "UseMaxTetIncrease": False}
eigen_solver_params_high_accuracy = {"MinimumFrequency": "1GHz",
                                     "NumModes": 4,
                                     "MaxDeltaFreq": 10,
                                     "ConvergeOnRealFreq": False,
                                     "MaximumPasses": 3,
                                     "MinimumPasses": 1,
                                     "MinimumConvergedPasses": 3,
                                     "PercentRefinement": 30,
                                     "IsEnabled": True,
                                     # MeshLink', SetupProps([('ImportMesh', False)])),
                                     "BasisOrder": 1,
                                     "DoLambdaRefine": True,
                                     "DoMaterialLambda": True,
                                     "SetLambdaTarget": False,
                                     "Target": 0.2,
                                     "UseMaxTetIncrease": False}
eigen_solver_setup = hfss.create_setup(**{"setupname": "MTS_EigenMode_Setup", "setuptype": "HFSSEigen"})
eigen_solver_setup.update(eigen_solver_params_default)
hfss.validate_simple()

mode_solutions = []
q_solutions = []
x_phase_vals = []
y_phase_vals = []
for brillouin_phase_list in brillouin_zone_phase_list:
    x_phase_delays = brillouin_phase_list['x_phase_deg']
    y_phase_delays = brillouin_phase_list['y_phase_deg']
    for phase_pair_index in range(len(x_phase_delays)):
        x_phase_delay = x_phase_delays[phase_pair_index]
        y_phase_delay = y_phase_delays[phase_pair_index]
        x_secondary_boundary.props['Phase'] = "{}deg".format(x_phase_delay)
        y_secondary_boundary.props['Phase'] = "{}deg".format(y_phase_delay)
        result_ok = hfss.analyze("MTS_EigenMode_Setup")

        mode_solution_data = hfss.post.get_solution_data(expressions='Mode(1)',
                                                         setup_sweep_name=hfss.nominal_sweep,
                                                         report_category='Eigen Modes')

        q_solution_data = hfss.post.get_solution_data(expressions='Q(1)',
                                                      setup_sweep_name=hfss.nominal_sweep,
                                                      report_category='Eigen Q')

        vals_np_real = np.array(list(mode_solution_data.full_matrix_real_imag[0]['Mode(1)'].values()))
        vals_np_imag = np.array(list(mode_solution_data.full_matrix_real_imag[1]['Mode(1)'].values()))
        mode_solution = vals_np_real + 1j * vals_np_imag

        vals_np_real = np.array(list(q_solution_data.full_matrix_real_imag[0]['Q(1)'].values()))
        vals_np_imag = np.array(list(q_solution_data.full_matrix_real_imag[1]['Q(1)'].values()))
        q_solution = vals_np_real + 1j * vals_np_imag

        mode_solutions.append(mode_solution)
        q_solutions.append(q_solution)
        x_phase_vals.append(x_phase_delay)
        y_phase_vals.append(y_phase_delay)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)
matlab_dict = {"mode_solutions": mode_solutions,
               "q_solutions": q_solutions,
               "phase_x": x_phase_vals,
               "phase_y": y_phase_vals,
               "freq": frequency,
               "height": height,
               "dielectric_material": dielectric_material_name,
               "sub_wavelength_factor": sub_wavelength_factor,
               "fill_pct": fill_pct,
               "cell_size": ground_plane_size,
               "compute_start_timestamp": current_time_str,
               "compute_host": socket.gethostname(),
               "compute_duration": time_difference_str
               }

scipy.io.savemat(save_filename, matlab_dict)
###############################################################################
# Close Ansys Electronics Desktop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.release_desktop(close_projects=True, close_desktop=True)
