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

feed_geometry = "CLYLINDRICAL"
feed_size_cm = 100 * 1.0e-3  # mm

###############################################################################
# Define program variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed_of_light = 2.99792458e8
frequency = 17e9
wavelength = speed_of_light / frequency

height_cm = 100 * 1.57e-3  # <=== dielectric slab height in meters
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
# cell_type = str(np.squeeze(unit_cell_database["database"][0][0]["cell_type"].all()))
cell_type = "SQUARE_PATCH"

monopole_length_cm = 0.3  # cm

# additional parameter for "SEIVENPIPER MUSHROOM"
mushroom_via_radius = 0.12e-3  # m

if cell_type == "ELLIPTICAL DISC":
    minor_to_major_axis_ratio = 0.75  # percent
    ellipse_XY_rotation_angle_deg = 45  # degrees
else:
    minor_to_major_axis_ratio = 1.0  # percent
    ellipse_XY_rotation_angle_deg = 0  # degrees

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

radiation_volume_height_cm = height_cm + 6 * height_cm

Zvals = np.linspace(160, 250, num=600)
gap_list = np.linspace(0.2, 1, num=600)
impedance_curve = np.array([107 + 65.5 / gap - 12.7 / gap ** 2 + 0.94 / gap ** 3 for gap in gap_list])


def find_gap(impedance):
    x_axis_index = np.argmin(np.abs(impedance_curve - np.imag(impedance)))
    gap = gap_list[x_axis_index]
    return gap


modulation_constant_X = np.mean(impedance_curve)
modulation_constant_M = np.min(np.abs(np.array([np.min(impedance_curve) - modulation_constant_X,
                                                np.max(impedance_curve) - modulation_constant_X])))

Z0 = 376.73031366857
cell_size_cm = 3.0e-1
phi = 0
theta_L = 45 * np.pi / 180
dielectric_constant = 2.2  # Duroid 5880
# k0 = 2 * np.pi * frequency / speed_of_light
k0 = 2 * np.pi * frequency
# k0 = 2 * np.pi * frequency / speed_of_light
theta_L = 80 * np.pi / 180
phi = 30 * np.pi / 180

# antenna_dimensions_xy_cm = np.array([40, 12.4])
# antenna_margin_xy_cm = np.array([1, 1])
# antenna_coord_origin_xy_cm = np.array([10, 6.2])
# antenna_dimensions_xy_cm = np.array([10, 5.4])
antenna_dimensions_xy_cm = np.array([5, 2.4])
antenna_margin_xy_cm = np.array([.25, .25])
# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
antenna_coord_origin_xy_cm = np.array([0.3 * antenna_dimensions_xy_cm[0],
                                       0.5 * antenna_dimensions_xy_cm[1]])

num_cells_x = np.floor((antenna_dimensions_xy_cm[0] - 2 * antenna_margin_xy_cm[0]) / cell_size_cm)
num_cells_y = np.floor((antenna_dimensions_xy_cm[1] - 2 * antenna_margin_xy_cm[1]) / cell_size_cm)

if num_cells_x % 2 == 0:
    num_cells_x += 1
if num_cells_y % 2 == 0:
    num_cells_y += 1

lhp_cell_count = np.floor((num_cells_x - 1) * antenna_coord_origin_xy_cm[0] / antenna_dimensions_xy_cm[0])
rhp_cell_count = num_cells_x - 1 - lhp_cell_count
bhp_cell_count = np.floor(num_cells_y * antenna_coord_origin_xy_cm[1] / antenna_dimensions_xy_cm[1])
thp_cell_count = num_cells_y - 1 - bhp_cell_count

Z_surf_avg = modulation_constant_X  # value of X parameter
n_surf_avg = Z0 / Z_surf_avg  # Z0/Z_avg
k_surf_avg = k0 * n_surf_avg
# k_surf_avg = k0 * np.sqrt(dielectric_constant)
x_cell_centers_cm = np.arange(-cell_size_cm * (lhp_cell_count - 0.5),
                              cell_size_cm * (rhp_cell_count + 0.5 + 1.0e-4),
                              cell_size_cm)
y_cell_centers_cm = np.arange(-cell_size_cm * (bhp_cell_count - 0.5),
                              cell_size_cm * (thp_cell_count + 0.5 + 1.0e-4),
                              cell_size_cm)
[x_grid, y_grid] = np.meshgrid(x_cell_centers_cm, y_cell_centers_cm)
x_grid = x_grid * 1e-2  # convert to meters
y_grid = y_grid * 1e-2  # convert to meters
r = np.sqrt(x_grid ** 2 + y_grid ** 2)

phase_radiation_pointing_function = k0 * x_grid * np.sin(theta_L) + 1j * phi
phase_source_excitation = k_surf_avg * r

psi_rad = np.exp(1j * phase_radiation_pointing_function)
psi_surf = np.exp(-1j * phase_source_excitation)

# plt.plot(x, np.angle(psi_rad*np.conj(psi_surf)))
# plt.plot(x, real(psi_rad))
# radial_amplitude = np.cos(k * n * r)
# radiation_amplitude = np.cos(k * x_grid * np.sin(theta_L) + phi)
radial_amplitude = np.real(psi_surf)
radiation_amplitude = np.real(psi_rad)
# added_amplitude = np.cos(-k * n * r + k * n * x_grid * np.sin(theta_L))
added_amplitude = np.real(psi_rad * psi_surf)
# Z_xy = 1j * (modulation_constant_X + modulation_constant_M * np.cos(-k * n * r + k * n * x_grid * np.sin(theta_L)))
Z_xy_grid = 1j * (modulation_constant_X + modulation_constant_M * np.real(psi_rad * psi_surf))

plt.subplot(311)
plt.imshow(radial_amplitude, 'gray', origin='lower', interpolation='bicubic')
plt.subplot(312)
plt.imshow(radiation_amplitude, 'gray', origin='lower', interpolation='bicubic')
plt.subplot(313)
plt.imshow(added_amplitude, 'gray', origin='lower', interpolation='bicubic')
plt.show(block=False)
print("max = {}".format(np.max(Z_xy_grid.ravel())))
print("min = {}".format(np.min(Z_xy_grid.ravel())))

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


def make_SquarePatchUnitCell(fill_ratio):
    unit_cell_params_local = {
        "name": "cell_0",
        "position": cm2mm * np.array([0, 0, height_cm / 2]),
        "size": cm2mm * np.array([fill_ratio[0] * cell_size_cm, fill_ratio[1] * cell_size_cm]),
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
        "position": cm2mm * np.array([0, 0, height_cm / 2]),
        "size_patch": cm2mm * np.array([fill_ratio[0] * cell_size_cm, fill_ratio[1] * cell_size_cm]),
        "size_via": cm2mm * np.array([mushroom_via_radius, height_cm]),
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
        "position": cm2mm * np.array([0, 0, height_cm / 2]),
        "size": cm2mm * np.array([fill_ratio[0] * cell_size_cm,
                                  fill_ratio[1] * cell_size_cm]),
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
cm2mm = 10
hfss.autosave_disable()

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_plane_position = cm2mm * np.array([-antenna_coord_origin_xy_cm[0],
                                          -antenna_coord_origin_xy_cm[1],
                                          -height_cm / 2])
ground_plane_size = cm2mm * np.array([antenna_dimensions_xy_cm[0],
                                      antenna_dimensions_xy_cm[1]])

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

unit_cell_list = []
cell_idx = 0
x_grid = x_grid.ravel()
y_grid = y_grid.ravel()
for cell_idx, Z_xy in enumerate(Z_xy_grid.ravel()):
    coord_x_cm = 100 * x_grid[cell_idx]
    coord_y_cm = 100 * y_grid[cell_idx]
    # Z_xy = Z_xy_grid[cell_idx]
    gap_mm = find_gap(Z_xy)
    patch_side_length_mm = cm2mm * cell_size_cm - gap_mm
    fill_ratio = (patch_side_length_mm / (cm2mm * cell_size_cm)) * np.ones(2)
    cell_position = cm2mm * np.array([coord_x_cm - (cell_size_cm / 2),
                                      coord_y_cm - (cell_size_cm / 2),
                                      height_cm / 2])
    cell_size = cm2mm * np.array([fill_ratio[0] * cell_size_cm, fill_ratio[1] * cell_size_cm])
    unit_cell_params_local = {
        "name": "cell_{:05d}".format(cell_idx),
        "position": cell_position,
        "size": cell_size,
        "scale_factor": 1.0,
        "material_name": unit_cell_material_name,
        "color": metal_color
    }
    cell_idx += 1
    cell_0_local = unit_cell.SquarePatchUnitCell(**unit_cell_params_local)
    cell_0_local.create_model(hfss)
    hfss.assign_perfecte_to_sheets(
        **{"sheet_list": cell_0_local.get_model_names(),
           "sourcename": None,
           "is_infinite_gnd": False})
    unit_cell_list.append(unit_cell)

# MAKE A HOLE FOR THE CYLINDRICAL SOURCE
# Construct the via from the ground plane to the metal patch
# s_axis is "X", "Y" or "Z"
monopole_position = cm2mm * np.array([0, 0, height_cm / 2 - monopole_length_cm])
monopole_size_cm = 0.05  # mm
monopole_params = {"name": "monopole_antenna",
                   "cs_axis": "Z",
                   "position": "{}mm,{}mm,{}mm".format(monopole_position[0],
                                                       monopole_position[1],
                                                       monopole_position[2]).split(","),
                   "radius": "{}mm".format(cm2mm * monopole_size_cm / 2),
                   "height": "{}mm".format(cm2mm * monopole_length_cm),
                   "numSides": 0,
                   "matname": "pec"}
monopole_geom = hfss.modeler.create_cylinder(**monopole_params)
monopole_geom.color = metal_color

subtract_params = {
    "blank_list": ["dielectric_slab"],
    "tool_list": ["monopole_antenna"],
    "keep_originals": True
}
hfss.modeler.subtract(**subtract_params)

feed_hole_position = cm2mm * np.array([0, 0, -height_cm / 2])
feed_hole_size_cm = monopole_size_cm + 0.05  # mm
feed_hole_ellipse_params = {"name": "monopole_feed_hole",
                            "cs_plane": "XY",
                            "position": "{}mm,{}mm,{}mm".format(feed_hole_position[0],
                                                                feed_hole_position[1],
                                                                feed_hole_position[2]).split(","),
                            "major_radius": "{}mm".format(cm2mm * feed_hole_size_cm / 2),
                            "ratio": 1.0,
                            "matname": "pec",
                            "is_covered": True}
unit_cell_via_geom = hfss.modeler.create_ellipse(**feed_hole_ellipse_params)
unit_cell_via_geom.color = metal_color
subtract_params = {
    "blank_list": ["cell_ground_plane"],
    "tool_list": ["monopole_feed_hole"],
    "keep_originals": False
}
hfss.modeler.subtract(**subtract_params)

monopole_shield_position = np.array([0, 0, monopole_position[2]])
monopole_shield_length_mm = (cm2mm * -height_cm / 2) - monopole_position[2]
monopole_shield_params = {"name": "monopole_shield",
                          "cs_axis": "Z",
                          "position": "{}mm,{}mm,{}mm".format(monopole_shield_position[0],
                                                              monopole_shield_position[1],
                                                              monopole_shield_position[2]).split(","),
                          "radius": "{}mm".format(cm2mm * feed_hole_size_cm / 2),
                          "height": "{}mm".format(monopole_shield_length_mm),
                          "numSides": 0,
                          "matname": "pec"}
monopole_shield_geom = hfss.modeler.create_cylinder(**monopole_shield_params)
monopole_shield_geom.color = metal_color
monopole_shield_geom.transparency = 0.8

# hfss.oeditor.DetachFaces()
detached_face_names = hfss.oeditor.DetachFaces(
    ["NAME:Selections",
     "Selections:=", "monopole_shield",
     "NewPartsModelFlag:=", "Model"
     ],
    ["NAME:Parameters",
     ["NAME:DetachFacesToParameters",
      "FacesToDetach:=", [monopole_shield_geom.faces[0].id, monopole_shield_geom.faces[1].id]]])

hfss.modeler.delete(detached_face_names[0])

hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "cell_ground_plane",
       "sourcename": None,
       "is_infinite_gnd": False})
hfss.assign_perfecte_to_sheets(
    **{"sheet_list": "monopole_shield",
       "sourcename": None,
       "is_infinite_gnd": False})

lumped_port_sheet_geom = hfss.modeler[detached_face_names[1]]
# Integration line can be two points or one of the following:
#           ``XNeg``, ``YNeg``, ``ZNeg``, ``XPos``, ``YPos``, and ``ZPos``
lumped_port_params = {  # "signal": None,
    "signal": detached_face_names[1],
    "reference": "monopole_shield",
    "create_port_sheet": False,
    "port_on_plane": True,
    # "integration_line": "XPos",
    "integration_line": [lumped_port_sheet_geom.bottom_face_z.center,
                         lumped_port_sheet_geom.bottom_face_z.top_edge_x.midpoint],
    "impedance": 50,
    "name": "source_port",
    "renormalize": False,
    "deembed": False,
    "terminals_rename": False
}
hfss.lumped_port(**lumped_port_params)

# hfss.add_3d_component_array_from_json()
# sma_connector_cs_params = {
#     "origin": "{}mm,{}mm,{}mm".format(monopole_shield_position[0],
#                                       monopole_shield_position[1],
#                                       monopole_shield_position[2]).split(","),
#     "reference_cs": "Global",
#     "name": None,
#     "mode": "axis",
#     "view": "iso",
#     "x_pointing": [1, 0, 0],
#     "y_pointing": [0, 1, 0],
#     "psi": 0,
#     "theta": 0,
#     "phi": 0,
#     "u": None
# }
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
# hfss.modeler.insert_3d_component(**sma_component_params)
# open_region_params = {
#     "Frequency": "{}GHz".format(frequency/1e9), "Boundary": "Radiation", "ApplyInfiniteGP": False, "GPAXis": "-z"
# }
# hfss.create_open_region(open_region_params)
module = hfss.get_module("ModelSetup")
module.CreateOpenRegion(
    [
        "NAME:Settings",
        "OpFreq:=", "17GHz",
        "Boundary:=", "Radiation",
        "ApplyInfiniteGP:=", False
    ])
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
solver_setup = hfss.create_setup(setupname="MTS_Setup", setuptype="HFSSDriven")
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '17GHz',
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
    "freqstart": 15,
    "freqstop": 22,
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
hfss.analyze_setup("MTS_Setup")
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
