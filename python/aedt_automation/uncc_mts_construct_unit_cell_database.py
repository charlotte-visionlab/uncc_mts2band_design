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
frequency = 15e9
wavelength = speed_of_light / frequency

# sub_wavelength_factor = 10
sub_wavelength_factor = wavelength / 3e-3
cell_size = wavelength / sub_wavelength_factor
height = 1.57e-3  # m
# height = 2.54e-3  # m
fill_pct = 0.5 * np.array([1.0, 1.0])

gap_size = 1.0e-3 * np.arange(0.2, 0.9, 0.6)
cell_gap_size = 0.5 * gap_size
pct_fill_values = 1.0 - cell_gap_size / cell_size
# database_fill_pct_values = np.arange(start=0.1, stop=1.0, step=0.4)
database_fill_pct_values = pct_fill_values

# cell_types can be from ["SQUARE PATCH", "SEIVENPIPER MUSHROOM", "ELLIPTICAL DISC" "CIRCULAR DISC"]
# cell_type = "CIRCULAR DISC"
# cell_type = "ELLIPTICAL DISC"
cell_type = "SQUARE PATCH"

# additional parameter for "SEIVENPIPER MUSHROOM"
mushroom_via_radius = 0.12e-3  # m
# additional parameter for "ELLIPTICAL DISC"
if cell_type == "ELLIPTICAL DISC":
    minor_to_major_axis_ratio = 0.75  # percent
    ellipse_XY_rotation_angle_deg = 45  # degrees
else:
    minor_to_major_axis_ratio = 1.0  # percent
    ellipse_XY_rotation_angle_deg = 0  # degrees

cell_base_parameters = {
    "cell_type": cell_type,
    "mushroom_via_radius": mushroom_via_radius,
    "minor_to_major_axis_ratio": minor_to_major_axis_ratio,
    "ellipse_XY_rotation_angle_deg": ellipse_XY_rotation_angle_deg
}

# from Fong et. al. 2010 suggested as the primary/secondary phase difference across a cell w/ periodic BCs
# unit_cell_phase_difference = 72  # degrees

# Various values are suggested for the height of the solution volume
# 6*height of dielectric ==> from https://www.youtube.com/watch?v=AXOpFgJcI38
# 6-8*height of dielectric ==> Ansoft Left-Handed Metamaterial Design Guide
# half the wavelength    ==> from B. H. Fong, J. S. Colburn, J. J. Ottusch, J. L. Visher and D. F. Sievenpiper,
#   "Scalar and Tensor Holographic Artificial Impedance Surfaces," in IEEE Transactions on Antennas and Propagation,
#   vol. 58, no. 10, pp. 3212-3221, Oct. 2010, doi: 10.1109/TAP.2010.2055812.
radiation_volume_height = height + 6 * height

RADIATION_VOLUME_CAP = "PEC_PLANE"
# RADIATION_VOLUME_CAP = "PML_BOX"

# Procedure to experimentally plot light line via computational electromagnetic simulation
# 1. Delete all structures until you only have the radiation boxes and the PML
# 2. Plot a new course in Brillouin zone 1 consisting of two short segments:
#       (a) Gamma_to_X ===> px = (0, 30, 5) py=0
#       (b) M_to_Gamma ===> px = (30, 0, -5), py = (30,0,-5)
# 3. Plot each dispersion curve (Gamma_to_X, M_to_Gamma) separately these are the light/free space lines
# 4. Deviation of the dispersion curve from the light line indicates dispersion is occurring.

# DIELECTRIC MATERIALS
# dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
ground_plane_material_name = "pec"
# ground_plane_material_name = "copper"
unit_cell_material_name = "pec"
# unit_cell_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"

# from Fong et. al. 2010 suggested as the position to integrate the mag(Ex)/mag(Hy) for impedance calculation
integration_surface_height = 0.5 * radiation_volume_height

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "Square Patch Unit Cell " + current_time_str
design_name = "MTS HFSS " + current_time_str

save_file_prefix = "mts_dispersion_database"
save_filename_no_extension = save_file_prefix + "_" + current_time_str + "_" + socket.gethostname()
save_filename_matlab = save_filename_no_extension + ".mat"
save_filename_numpy = save_filename_no_extension + ".npy"

# Plot a course in the Brillouin zone 1
# https://www.emtalk.com/tut_3.htm, https://www.emtalk.com/tut_3.htm both suggest the following path
# Ansoft Left-Handed Metamaterial Design Guide also suggests this path but there appears to be a typo in the
# second (X-M) parametric sweep described where px and py should be switched.
# step = 10
# Gamma_to_X ==> px=arange(step,180,step), py=0
# X_to_M ==> px=180, py=arange(step,180,step)
# M_to_Gamma ==> px=py=arange(180-step,0+step,-step)
number_phase_steps_per_segment = 19
phase_step_size = 180 / (number_phase_steps_per_segment - 1)
x_phase_delays = np.arange(phase_step_size, 180 + phase_step_size, phase_step_size)
y_phase_delays = [0] * len(x_phase_delays)
gamma_to_X_phase_pairs = {'path_name': 'Gamma_to_X', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
x_phase_delays = [180] * len(y_phase_delays)
y_phase_delays = np.arange(phase_step_size, 180 + phase_step_size, phase_step_size)
X_to_M_phase_pairs = {'path_name': 'X_to_M', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
x_phase_delays = np.arange(180 - phase_step_size, 0, -phase_step_size)
y_phase_delays = np.arange(180 - phase_step_size, 0, -phase_step_size)
M_to_gamma_phase_pairs = {'path_name': 'M_to_Gamma', 'x_phase_deg': x_phase_delays, 'y_phase_deg': y_phase_delays}
brillouin_zone_phase_list = [gamma_to_X_phase_pairs, X_to_M_phase_pairs, M_to_gamma_phase_pairs]

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
perfectly_matched_layer_color = [255, 128, 128]


def make_TransmissionLineSquarePatchUnitCell(fill_ratio):
    unit_cell_params_local = {
        "name": "cell_0",
        "position": meters2mm * np.array([0, 0, height / 2]),
        "size": meters2mm * np.array([fill_ratio[0] * cell_size, fill_ratio[1] * cell_size]),
        "scale_factor": 1.0,
        "material_name": unit_cell_material_name,
        "color": metal_color
    }
    cell_0_local = unit_cell.SquarePatchUnitCell(**unit_cell_params_local)
    cell_0_local.create_model(hfss)
    return cell_0_local

def make_SquarePatchUnitCell(fill_ratio):
    unit_cell_params_local = {
        "name": "cell_0",
        "position": meters2mm * np.array([0, 0, height / 2]),
        "size": meters2mm * np.array([fill_ratio[0] * cell_size, fill_ratio[1] * cell_size]),
        "scale_factor": 1.0,
        "material_name": unit_cell_material_name,
        "color": metal_color
    }
    cell_0_local = unit_cell.SquarePatchUnitCell(**unit_cell_params_local)
    cell_0_local.create_model(hfss)
    return cell_0_local


def make_SievenpiperMushroomUnitCell(fill_ratio):
    unit_cell_params_local = {
        "name": "cell_0",
        "position": meters2mm * np.array([0, 0, height / 2]),
        "size_patch": meters2mm * np.array([fill_ratio[0] * cell_size, fill_ratio[1] * cell_size]),
        "size_via": meters2mm * np.array([mushroom_via_radius, height]),
        "scale_factor": 1.0,
        "material_name": unit_cell_material_name,
        "color": metal_color
    }
    cell_0_local = unit_cell.SievenpiperMushroomUnitCell(**unit_cell_params_local)
    cell_0_local.create_model(hfss)
    return cell_0_local


def make_EllipticalDiscUnitCell(fill_ratio, rotation_xy_deg=0):
    fill_ratio[1] = minor_to_major_axis_ratio * fill_ratio[1]
    unit_cell_params_local = {
        "name": "cell_0",
        "position": meters2mm * np.array([0, 0, height / 2]),
        "size": meters2mm * np.array([fill_ratio[0] * cell_size,
                                      fill_ratio[1] * cell_size]),
        "rotation_xy_deg": rotation_xy_deg,
        "scale_factor": 1.0,
        "material_name": unit_cell_material_name,
        "color": metal_color
    }
    if fill_ratio[0] != fill_ratio[1]:
        cell_0_local = unit_cell.EllipticalDiscUnitCell(**unit_cell_params_local)
    else:
        cell_0_local = unit_cell.CircularDiscUnitCell(**unit_cell_params_local)
    cell_0_local.create_model(hfss)
    return cell_0_local


def make_CircularDiscUnitCell(fill_ratio):
    return make_EllipticalDiscUnitCell(np.array([fill_ratio[0], fill_ratio[0]]))


def make_unit_cell(cell_type_local, fill_ratio):
    if cell_type_local == "SQUARE PATCH":
        unit_cell_local = make_SquarePatchUnitCell(fill_ratio)
    elif cell_type_local == "SIEVENPIPER MUSHROOM":
        unit_cell_local = make_SievenpiperMushroomUnitCell(fill_ratio)
    elif cell_type_local == "ELLIPTICAL DISC":
        unit_cell_local = make_EllipticalDiscUnitCell(fill_ratio, ellipse_XY_rotation_angle_deg)
    elif cell_type_local == "CIRCULAR DISC":
        unit_cell_local = make_CircularDiscUnitCell(fill_ratio)
    return unit_cell_local


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
    solution_type="Eigenmode",
    new_desktop_session=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical
)

hfss.modeler.model_units = 'mm'
meters2mm = 1000
hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(meters2mm * wavelength))
hfss.variable_manager.set_variable("height", expression="{}mm".format(meters2mm * height))
hfss.variable_manager.set_variable("sub_wavelength_factor", expression="{}".format(sub_wavelength_factor))
hfss.variable_manager.set_variable("x_fill_pct", expression="{}percent".format(fill_pct[0]))
hfss.variable_manager.set_variable("y_fill_pct", expression="{}percent".format(fill_pct[1]))
hfss.variable_manager.set_variable("solution_volume_height",
                                   expression="{}mm".format(meters2mm * radiation_volume_height))
hfss.variable_manager.set_variable("integration_surface_height",
                                   expression="{}mm".format(meters2mm * integration_surface_height))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_plane_position = meters2mm * np.array([-wavelength / (2 * sub_wavelength_factor),
                                              -wavelength / (2 * sub_wavelength_factor),
                                              -height / 2])
ground_plane_size = meters2mm * np.array([wavelength / sub_wavelength_factor,
                                          wavelength / sub_wavelength_factor])

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

cell_0 = make_unit_cell(cell_type, fill_pct)

dielectric_slab_position = ground_plane_position
dielectric_slab_size = np.array([ground_plane_size[0],
                                 ground_plane_size[1],
                                 meters2mm * height])
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

radiation_volume_position = ground_plane_position
radiation_volume_size = np.array([ground_plane_size[0],
                                  ground_plane_size[1],
                                  meters2mm * radiation_volume_height])
radiation_volume_params = {"name": "radiation_volume_1",
                           "position": "{}mm,{}mm,{}mm".format(radiation_volume_position[0],
                                                               radiation_volume_position[1],
                                                               radiation_volume_position[2]).split(","),
                           "dimensions_list": "{}mm,{}mm,{}mm".format(radiation_volume_size[0],
                                                                      radiation_volume_size[1],
                                                                      radiation_volume_size[2]).split(","),
                           "matname": radiation_box_material_name}
radiation_volume_geom = hfss.modeler.create_box(**radiation_volume_params)
radiation_volume_geom.color = radiation_box_color
radiation_volume_geom.display_wireframe = 1
radiation_volume_geom.transparency = 1

radiation_volume_coupled_geom = radiation_volume_geom
radiation_volume_coupled_position = radiation_volume_position
radiation_volume_coupled_size = radiation_volume_size

solution_fields_surface_position = np.array([ground_plane_position[0],
                                             ground_plane_position[1],
                                             meters2mm * integration_surface_height])
solution_fields_surface_size = ground_plane_size
solution_fields_surface_params = {"name": "solution_fields_plane",
                                  "csPlane": "XY",
                                  "position": "{}mm,{}mm,{}mm".format(solution_fields_surface_position[0],
                                                                      solution_fields_surface_position[1],
                                                                      solution_fields_surface_position[2]).split(","),
                                  "dimension_list": "{}mm,{}mm".format(solution_fields_surface_size[0],
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
    **{"sheet_list": cell_0.get_model_names(),
       "sourcename": None,
       "is_infinite_gnd": False})

if RADIATION_VOLUME_CAP == "PEC_PLANE":
    #  csPlane is either "XY", "YZ", or "XZ"
    radiation_volume_cap_plane_position = np.array([ground_plane_position[0],
                                                    ground_plane_position[1],
                                                    meters2mm * (radiation_volume_height - height / 2)])
    radiation_volume_cap_plane_size = ground_plane_size
    radiation_volume_cap_plane_params = {"name": "radiation_volume_cap_plane",
                                         "csPlane": "XY",
                                         "position": "{}mm,{}mm,{}mm".format(radiation_volume_cap_plane_position[0],
                                                                             radiation_volume_cap_plane_position[1],
                                                                             radiation_volume_cap_plane_position[
                                                                                 2]).split(","),
                                         "dimension_list": "{}mm,{}mm".format(radiation_volume_cap_plane_size[0],
                                                                              radiation_volume_cap_plane_size[1]).split(
                                             ","),
                                         "matname": ground_plane_material_name,
                                         "is_covered": True}
    radiation_volume_cap_plane_geom = hfss.modeler.create_rectangle(**radiation_volume_cap_plane_params)
    radiation_volume_cap_plane_geom.color = metal_color

    hfss.assign_perfecte_to_sheets(
        **{"sheet_list": "radiation_volume_cap_plane",
           "sourcename": None,
           "is_infinite_gnd": False})

elif RADIATION_VOLUME_CAP == "PML_BOX":
    # Construct a Perfectly Matched Layer (PML) box to absorb radiated energy at the top of the unit cell simulation volume
    module = hfss.get_module("BoundarySetup")
    module.CreatePML(
        [
            "NAME:PMLCreationSettings",
            "UserDrawnGroup:=", False,
            "PMLFaces:=", [radiation_volume_geom.faces[0].id],
            "Thickness:=", "0.0025mm",
            "CreateJoiningObjs:=", False,
            "PMLObj:=", -1,
            "BaseObj:=", radiation_volume_geom.id,
            "Orientation:=", "Undefined",
            "UseFreq:=", True,
            "MinFreq:=", "1GHz",
            "MinBeta:=", 20,
            "RadDist:=", "0.00629666666666667mm"
        ])

    radiation_volume2_position = ground_plane_position
    radiation_volume2_size = [ground_plane_size[0], ground_plane_size[1],
                              meters2mm * (radiation_volume_height + 0.0025)]
    radiation_volume2_params = {"name": "radiation_volume_2",
                                "position": "{}mm,{}mm,{}mm".format(radiation_volume2_position[0],
                                                                    radiation_volume2_position[1],
                                                                    radiation_volume2_position[2]).split(","),
                                "dimensions_list": "{}mm,{}mm,{}mm".format(radiation_volume2_size[0],
                                                                           radiation_volume2_size[1],
                                                                           radiation_volume2_size[2]).split(","),
                                "matname": radiation_box_material_name}
    radiation_volume2_geom = hfss.modeler.create_box(**radiation_volume2_params)
    radiation_volume2_geom.color = radiation_box_color
    radiation_volume2_geom.display_wireframe = 1
    radiation_volume2_geom.transparency = 1
    radiation_volume_coupled_geom = radiation_volume2_geom
    radiation_volume_coupled_position = radiation_volume2_position
    radiation_volume_coupled_size = radiation_volume2_size

# Setup primary and secondary coupled, i.e., periodic boundary conditions
# BOX FACE INDICES ARE AS FOLLOWS
# faces = [ 0 = "+Z", "1" = "-Y", "2" = "-X", "3" = "+Y", "4" = "-Z", "5" = "+X"]
propagation_volume_geom = radiation_volume_coupled_geom
propagation_volume_position = radiation_volume_coupled_position
propagation_volume_size = radiation_volume_coupled_size
# propagation_volume_geom = dielectric_slab_geom
# propagation_volume_position = dielectric_slab_position
# propagation_volume_size = dielectric_slab_size
primary_x_prop = {"face": propagation_volume_geom.faces[2],
                  "u_start": "{}mm,{}mm,{}mm".format(propagation_volume_position[0],
                                                     propagation_volume_position[1],
                                                     propagation_volume_position[2]).split(","),
                  "u_end": "{}mm,{}mm,{}mm".format(propagation_volume_position[0],
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
                    "u_start": "{}mm,{}mm,{}mm".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                       propagation_volume_position[1],
                                                       propagation_volume_position[2]).split(","),
                    "u_end": "{}mm,{}mm,{}mm".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                     propagation_volume_position[1] + propagation_volume_size[1],
                                                     propagation_volume_position[2]).split(","),
                    "reverse_v": True,
                    "phase_delay": "InputPhaseDelay",
                    "phase_delay_param1": "{}deg".format(180),
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
                    "u_start": "{}mm,{}mm,{}mm".format(propagation_volume_position[0],
                                                       propagation_volume_position[1] + propagation_volume_size[1],
                                                       propagation_volume_position[2]).split(","),
                    "u_end": "{}mm,{}mm,{}mm".format(propagation_volume_position[0] + propagation_volume_size[0],
                                                     propagation_volume_position[1] + propagation_volume_size[1],
                                                     propagation_volume_position[2]).split(","),
                    "reverse_v": False,
                    "phase_delay": "InputPhaseDelay",
                    "phase_delay_param1": "{}deg".format(0),
                    "phase_delay_param2": "0deg",
                    "coord_name": "Global",
                    "secondary_name": "y_lattice_prop_secondary"}
y_secondary_boundary = hfss.assign_secondary(**secondary_y_prop)

# FIT ALL
hfss.modeler.fit_all()

module = hfss.get_module("MeshSetup")
module.AssignLengthOp(
    [
        "NAME:" + "dielectric_mesh_refinement",
        "RefineInside:=", False,
        "Enabled:=", True,
        "Objects:=", ["dielectric_slab"],
        "RestrictElem:=", False,
        "NumMaxElem:=", "1000",
        "RestrictLength:=", True,
        "MaxLength:=", "{}mm".format(meters2mm * wavelength / 25)
    ])
module.AssignTrueSurfOp(
    [
        "NAME:SurfApprox1",
        "Faces:=", [dielectric_slab_geom.faces[0].id],
        "CurvedSurfaceApproxChoice:=", "ManualSettings",
        "SurfDevChoice:=", 0,
        "NormalDevChoice:=", 1,
        "AspectRatioChoice:=", 2,
        "AspectRatio:=", "5"
    ])
# hfss.mesh.assign_length_mesh(**{"names": "dielectric_slab",
#                                 "isinside": True,
#                                 "maxlength": wavelength / 25.0,
#                                 "maxel": 1000,
#                                 "meshop_name": None})
eigen_mode_solver_setup = None
eigen_mode_solver_setup_linked = None

setup_solver_configuration_initial = {
    "setup_name": "MTS_EigenMode_Setup",
    "num_cores": solver_configuration["num_cores"],
    "num_tasks": solver_configuration["num_tasks"],
    "num_gpu": solver_configuration["num_gpu"],
    "acf_file": None,
    "use_auto_settings": False,
    "solve_in_batch": False,
    "machine": solver_configuration["machine"],
    "run_in_thread": False,
    "revert_to_initial_mesh": False,
    "blocking": True
}

setup_solver_configuration_linked = {
    "setup_name": "MTS_EigenMode_Setup1",
    "num_cores": solver_configuration["num_cores"],
    "num_tasks": solver_configuration["num_tasks"],
    "num_gpu": solver_configuration["num_gpu"],
    "acf_file": None,
    "use_auto_settings": False,
    "solve_in_batch": False,
    "machine": solver_configuration["machine"],
    "run_in_thread": False,
    "revert_to_initial_mesh": False,
    "blocking": True
}

# plotting dispersion curves
plt.ion()
fig = plt.figure(figsize=(6, 4))
plt.title("Dispersion Diagram" + " " + str(cell_type).lower() + " " +
          "cell size = {}mm x {}mm".format(meters2mm * cell_size, meters2mm * cell_size))
plt.ylabel("re(Mode(1)) [GHz]")
plt.xlabel("$\\beta$*p , $\\Delta\\phi$, Unit Cell Phase Change (rad)")
legend_strs = []

database = []
for fill_factor in database_fill_pct_values:
    fill_pct = fill_factor * np.array([1.0, 1.0])

    cell_0.delete_model()
    cell_0 = make_unit_cell(cell_type, fill_pct)

    hfss.assign_perfecte_to_sheets(
        **{"sheet_list": cell_0.get_model_names(),
           "sourcename": None,
           "is_infinite_gnd": False})

    if eigen_mode_solver_setup is not None:
        # eigen_mode_solver_setup_linked.delete()
        eigen_mode_solver_setup.delete()

    # setup_type = "HFSSDrivenAuto", "HFSSDrivenDefault", "HFSSEigen", "HFSSTransient", "HFSSSBR"
    eigen_solver_params_default = {"MinimumFrequency": "1GHz",
                                   "NumModes": 1,
                                   "MaxDeltaFreq": 1,
                                   "ConvergeOnRealFreq": True,
                                   "MaximumPasses": 25,
                                   "MinimumPasses": 3,
                                   "MinimumConvergedPasses": 2,
                                   "PercentRefinement": 30,
                                   "IsEnabled": True,
                                   # "MeshLink": SetupProps([("ImportMesh", False)])),
                                   "BasisOrder": 1,
                                   "DoLambdaRefine": True,
                                   "DoMaterialLambda": True,
                                   "SetLambdaTarget": False,
                                   "Target": 0.2,
                                   "UseMaxTetIncrease": False}
    eigen_mode_solver_setup = hfss.create_setup(**{"setupname": "MTS_EigenMode_Setup", "setuptype": "HFSSEigen"})
    eigen_mode_solver_setup.update(eigen_solver_params_default)

    # Converge for this phase pair is most demanding as this is typically where the field is changing most rapidly
    # x_secondary_boundary.props['Phase'] = "{}deg".format(0)
    # y_secondary_boundary.props['Phase'] = "{}deg".format(180)
    x_secondary_boundary.props['Phase'] = "{}deg".format(180)
    y_secondary_boundary.props['Phase'] = "{}deg".format(0)

    hfss.validate_simple()
    result_ok = hfss.analyze(**setup_solver_configuration_initial)

    module = hfss.get_module("AnalysisSetup")
    module.InsertMeshLinkedSetup("MTS_EigenMode_Setup")
    module.EditSetup("MTS_EigenMode_Setup1",
                     [
                         "NAME:" + "MTS_EigenMode_Setup1",
                         "MinimumFrequency:=", "1GHz",
                         "NumModes:=", 1,
                         "MaxDeltaFreq:=", 10,
                         "ConvergeOnRealFreq:=", True,
                         "MaximumPasses:=", 1,
                         "MinimumPasses:=", 1,
                         "MinimumConvergedPasses:=", 1,
                         "PercentRefinement:=", 30,
                         "IsEnabled:=", True,
                         [
                             "NAME:MeshLink",
                             "ImportMesh:=", True,
                             "Project:=", "This Project*",
                             "Product:=", "HFSS",
                             "Design:=", "This Design*",
                             "Soln:=", "MTS_EigenMode_Setup : LastAdaptive",
                             [
                                 "NAME:Params",
                                 "height:=", "height",
                                 "integration_surface_height:=", "integration_surface_height",
                                 "solution_volume_height:=", "solution_volume_height",
                                 "sub_wavelength_factor:=", "sub_wavelength_factor",
                                 "wavelength:=", "wavelength",
                                 "x_fill_pct:=", "x_fill_pct",
                                 "y_fill_pct:=", "y_fill_pct"
                             ],
                             "ForceSourceToSolve:=", False,
                             "PreservePartnerSoln:=", False,
                             "PathRelativeTo:=", "TargetProject",
                             "ApplyMeshOp:=", False
                         ],
                         "BasisOrder:=", 1,
                         "DoLambdaRefine:=", False,
                         "DoMaterialLambda:=", True,
                         "SetLambdaTarget:=", False,
                         "Target:=", 0.2,
                         "UseMaxTetIncrease:=", False
                     ])
    eigen_mode_solver_setup_linked = hfss.get_setup("MTS_EigenMode_Setup1")

    mode_solutions = [0]
    q_solutions = [0]
    x_phase_vals = [0]
    y_phase_vals = [0]

    # plotting dispersion curves
    current_trace, = plt.plot(x_phase_vals, mode_solutions)
    trace_str = "{:.2f} mm".format(cell_0.size[0])
    legend_strs.append(trace_str)
    plt.legend(legend_strs)

    for brillouin_phase_list in brillouin_zone_phase_list:
        x_phase_delays = brillouin_phase_list['x_phase_deg']
        y_phase_delays = brillouin_phase_list['y_phase_deg']
        for phase_pair_index in range(len(x_phase_delays)):
            x_phase_delay = x_phase_delays[phase_pair_index]
            y_phase_delay = y_phase_delays[phase_pair_index]
            x_secondary_boundary.props['Phase'] = "{}deg".format(x_phase_delay)
            y_secondary_boundary.props['Phase'] = "{}deg".format(y_phase_delay)

            result_ok = hfss.analyze(**setup_solver_configuration_linked)

            mode_solution_data = hfss.post.get_solution_data(expressions="Mode(1)",
                                                             setup_sweep_name="MTS_EigenMode_Setup1 : LastAdaptive",
                                                             report_category="Eigenmode Parameters")
            q_solution_data = hfss.post.get_solution_data(expressions='Q(1)',
                                                          setup_sweep_name="MTS_EigenMode_Setup1 : LastAdaptive",
                                                          report_category="Eigenmode Parameters")
            # 'Solution Convergence'
            # convergence_solution_data = hfss.post.get_solution_data(
            #       setup_sweep_name="MTS_EigenMode_Setup : LastAdaptive",
            #       report_category="Solution Convergence")

            vals_np_real = np.array(list(mode_solution_data.full_matrix_real_imag[0]['Mode(1)'].values()))
            vals_np_imag = np.array(list(mode_solution_data.full_matrix_real_imag[1]['Mode(1)'].values()))
            mode_solution = np.squeeze(vals_np_real + 1j * vals_np_imag)

            vals_np_real = np.array(list(q_solution_data.full_matrix_real_imag[0]['Q(1)'].values()))
            vals_np_imag = np.array(list(q_solution_data.full_matrix_real_imag[1]['Q(1)'].values()))
            q_solution = np.squeeze(vals_np_real + 1j * vals_np_imag)

            mode_solutions.append(mode_solution)
            q_solutions.append(q_solution)
            x_phase_vals.append(x_phase_delay)
            y_phase_vals.append(y_phase_delay)

            # plotting dispersion curves
            current_trace.set_xdata(np.append(current_trace.get_xdata(), phase_step_size * len(mode_solutions)))
            current_trace.set_ydata(np.append(current_trace.get_ydata(), np.real(mode_solution) / 1e9))
            fig.canvas.draw()
            fig.canvas.flush_events()
            ax = plt.gca()
            ax.relim()
            ax.autoscale_view()

            # end loop over current Brillioun segment
        # end loop over Brillioun segments
    # done traversing Brillioun segments

    # add a final value and replot the final dispersion curve
    mode_solutions.append(0)
    q_solutions.append(0)
    x_phase_vals.append(0)
    y_phase_vals.append(0)
    current_trace.set_xdata(np.append(current_trace.get_xdata(), phase_step_size * len(mode_solutions)))
    current_trace.set_ydata(np.append(current_trace.get_ydata(), 0 / 1e9))
    fig.canvas.draw()
    fig.canvas.flush_events()
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()

    time_end = datetime.now()
    time_difference = time_end - time_start
    time_difference_str = str(time_difference)
    database.append({"mode_solutions": mode_solutions,
                     "q_solutions": q_solutions,
                     "phase_x": x_phase_vals,
                     "phase_y": y_phase_vals,
                     "phase_step": phase_step_size,
                     "freq": frequency,
                     "height": height,
                     "dielectric_material": dielectric_material_name,
                     "sub_wavelength_factor": sub_wavelength_factor,
                     "fill_pct": fill_pct,
                     "cell_model_size": cell_0.size,
                     "cell_model_type": cell_type,
                     "cell_model_parameters": cell_base_parameters,
                     "compute_start_timestamp": current_time_str,
                     "compute_host": socket.gethostname(),
                     "compute_duration": time_difference_str
                     })
# end loop over unit cell geometry database modifications

plt.savefig(save_filename_no_extension + "_plots" + ".png")
# with open(save_filename_numpy, 'wb') as f:
#     np.save(f, database)

time_end = datetime.now()
time_difference = time_end - time_start
time_difference_str = str(time_difference)
matlab_dict = {"num_cells": len(database),
               "global_cell_size": cell_size,
               "database": database,
               "compute_start_timestamp": current_time_str,
               "compute_host": socket.gethostname(),
               "compute_duration": time_difference_str
               }

scipy.io.savemat(save_filename_matlab, matlab_dict)
###############################################################################
# Close Ansys Electronics Desktop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.release_desktop(close_projects=True, close_desktop=True)
