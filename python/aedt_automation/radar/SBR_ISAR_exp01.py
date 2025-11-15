"""
SBR+: doppler setup
-------------------
This example shows how you can use PyAEDT to create a multipart scenario in HFSS SBR+
and set up a doppler analysis.
"""

###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports.

import os
from collections import OrderedDict
import pyaedt
from pyaedt.generic.general_methods import remove_project_lock

import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.signal


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


# Launch AEDT
# ~~~~~~~~~~~

radar_parameters = {
    # 'Antenna Range': '10000m',
    'Nominal Operating Frequency': '10GHz',
    'Range Extent': '100m',  # X
    'Range Resolution': '1m',  # dX
    'Cross-Range Extent': '100m',  # Y
    'Cross-Range Resolution': '1m',  # dY
    'Polarization': 'Phi-Phi'
}
initial_azimuth_angle = 90

# Need to compute the phi variation of the experiment from the cross-range extent and resolution

os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
aedt_version = "2023.1"
projectname = "Inverse Synthetic Aperture Radar"
designname = "Range Profile Experiments"
source_project_path = 'projects'
source_project_name = os.path.join(source_project_path, 'SBR_ISAR_BaseProject.aedt')
scene_geom_output_img = os.path.join(source_project_path, 'SBR_ISAR_exp01_scene_top_down_image.jpg')

matplotlib_pyplot_setup()

# library_path = pyaedt.downloads.download_multiparts()
library_path = "radar_object_library"
speed_of_light = 2.99792458e8
parse_arg = lambda x: list(filter(None, re.split(r'([0-9.]+)(?!.*\d)', x)))
# antenna_range_val_and_units_str = parse_arg(radar_parameters['Antenna Range'])
freq_val_and_units_str = parse_arg(radar_parameters['Nominal Operating Frequency'])
x_range_val_and_units_str = parse_arg(radar_parameters['Range Extent'])
x_resolution_val_and_units_str = parse_arg(radar_parameters['Range Resolution'])
y_range_val_and_units_str = parse_arg(radar_parameters['Cross-Range Extent'])
y_resolution_val_and_units_str = parse_arg(radar_parameters['Cross-Range Resolution'])

freq_center_GHz = float(freq_val_and_units_str[0])
wavelength = speed_of_light / (freq_center_GHz * 1.0e9)

# Setup down range / short (or fast) time pulse characteristics
# ~~~~~~~~~~~~~~~~~~~~~~
bandwidth_GHz = speed_of_light / (2 * float(x_resolution_val_and_units_str[0]) * 1.0e9)
x_swath_extent = float(x_range_val_and_units_str[0])
y_swath_extent = float(y_range_val_and_units_str[0])
num_freq_samples = int(x_swath_extent / float(x_resolution_val_and_units_str[0]))
freq_start_GHz = freq_center_GHz - (bandwidth_GHz / 2)
freq_stop_GHz = freq_center_GHz + (bandwidth_GHz / 2)
N_x = np.round(x_swath_extent / float(x_resolution_val_and_units_str[0]))

# Setup cross range / long (or slow) time pulse characteristics
# ~~~~~~~~~~~~~~~~~~~~~~
# aperture_length = float(antenna_range_val_and_units_str[0])*wavelength/float(y_resolution_val_and_units_str[0])
# aperture is arc of circle having arc length  = delta_theta*radius
# delta_theta = aperture_length/float(antenna_range_val_and_units_str[0])
delta_phi = (wavelength / (2 * y_swath_extent)) * (180 / np.pi)
N_y = np.round(y_swath_extent / float(y_resolution_val_and_units_str[0]))
phi_start_deg = -delta_phi * (N_y / 2)
phi_stop_deg = delta_phi * (N_y / 2)
phi_start_deg += initial_azimuth_angle
phi_stop_deg += initial_azimuth_angle
N_phi = N_y

theta_start_deg = (np.pi / 2) * (180 / np.pi)
theta_stop_deg = (np.pi / 2) * (180 / np.pi)
delta_theta = 0
N_theta = 1

###############################################################################
# Set non-graphical mode
# ~~~~~~~~~~~~~~~~~~~~~~
# Set non-graphical mode. ``"PYAEDT_NON_GRAPHICAL"`` is needed to generate
# documentation only.
# You can set ``non_graphical`` either to ``True`` or ``False``.

non_graphical = False

###############################################################################
# Download and open project
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Download and open the project.

# project_name = pyaedt.generate_unique_project_name(project_name="doppler")
NewThread = True
desktop = pyaedt.launch_desktop(specified_version=aedt_version,
                                non_graphical=non_graphical,
                                new_desktop_session=NewThread
                                )
remove_project_lock(source_project_name)

hfss_geom = pyaedt.Hfss(
    specified_version=aedt_version,
    solution_type="SBR+",
    new_desktop_session=True,
    projectname=source_project_name,
    close_on_exit=True,
    non_graphical=non_graphical
)

hfss_geom.autosave_disable()

###############################################################################
# Save project and rename design
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save the project to the temporary folder and rename the design.

hfss_geom.rename_design(designname)

###############################################################################
# Set up library paths
# ~~~~~~~~~~~~~~~~~~~~
# Set up library paths to 3D components.

actor_lib = os.path.join(library_path, "actor_library")
env_lib = os.path.join(library_path, "environment_library")
radar_lib = os.path.join(library_path, "radar_modules")
env_folder = os.path.join(env_lib, "road1")
person_folder = os.path.join(actor_lib, "person3")
car_folder = os.path.join(actor_lib, "vehicle1")
bike_folder = os.path.join(actor_lib, "bike1")
bird_folder = os.path.join(actor_lib, "bird1")
reflector_folder = os.path.join(library_path, "reflectors")
missiles_folder = os.path.join(library_path, "missiles")
ships_folder = os.path.join(library_path, "ships")

###############################################################################
# Define environment
# ~~~~~~~~~~~~~~~~~~
# Define the background environment.

# road1 = app.modeler.add_environment(env_folder=env_folder, environment_name="Bari")
# trihedral_reflector1 = os.path.join(reflector_folder, "trihedral_reflector1")
# trihedral_reflector2 = os.path.join(reflector_folder, "trihedral_reflector2")
# agm88_1 = os.path.join(missiles_folder, "AGM88_1")
cablelaying_ship = os.path.join(ships_folder, "cablelaying_ship")

# reflector1 = hfss_geom.modeler.add_environment(env_folder=trihedral_reflector1,
#                                                environment_name="Reflector1",
#                                                global_offset=[0, 0, 0],
#                                                yaw=0,
#                                                pitch=0,
#                                                roll=0)
ship_model = hfss_geom.modeler.add_environment(env_folder=cablelaying_ship,
                                               environment_name="cablelaying_ship_1",
                                               global_offset=[0, 0, 0],
                                               yaw=0,
                                               pitch=0,
                                               roll=0)

# sheet_names = hfss_geom.modeler.sheet_names
# hfss_geom.assign_perfecte_to_sheets(sheet_list=sheet_names,sourcename=perfE_source)

plane_wave_source = hfss_geom.boundaries[0]
plane_wave_source_props = plane_wave_source.props
plane_wave_source.auto_update = False
plane_wave_source_props['PhiStart'] = '{0:.5f}deg'.format(phi_start_deg)
plane_wave_source_props['PhiStop'] = '{0:.5f}deg'.format(phi_stop_deg)
if N_phi > 0:
    plane_wave_source_props['PhiPoints'] = N_phi
plane_wave_source_props['ThetaStart'] = '{0:.5f}deg'.format(theta_start_deg)
plane_wave_source_props['ThetaStop'] = '{0:.5f}deg'.format(theta_stop_deg)
if N_theta > 0:
    plane_wave_source_props['ThetaPoints'] = N_theta
plane_wave_source.auto_update = True
plane_wave_source_props.update()
hfss_geom.boundaries[0].update()
plane_wave_source_props2 = {'ID': 0,
                            'BoundType': 'Plane Incident Wave',
                            'IsComponent': False,
                            'ParentBndID': -1,
                            'IsCartesian': False,
                            'EoX': '1',
                            'EoY': '0',
                            'EoZ': '0',
                            'kX': '0',
                            'kY': '0',
                            'kZ': '1',
                            'PhiStart': '{0:.5f}deg'.format(phi_start_deg),
                            'PhiStop': '{0:.5f}deg'.format(phi_stop_deg),
                            'PhiPoints': N_phi,
                            'ThetaStart': '{0:.5f}deg'.format(theta_start_deg),
                            'ThetaStop': '{0:.5f}deg'.format(theta_stop_deg),
                            'ThetaPoints': N_theta,
                            'EoPhi': '1', # This is 1 for Phi-X polarization (Horizontal)
                            'EoTheta': '0', # This is 1 for Theta-X polarization (Vertical)
                            'OriginX': '0mm',
                            'OriginY': '0mm',
                            'OriginZ': '0mm',
                            'IsPropagating': True,
                            'IsEvanescent': False,
                            'IsEllipticallyPolarized': False,
                            'PhiEnd': phi_stop_deg}
# hfss_geom.oboundary.EditIncidentWave(plane_wave_source._boundary_name, plane_wave_source._get_args())

# replace Boundary.py in pyaedt package
# import sys
# del sys.modules['pyaedt.modules.Boundary']
# sys.modules['pyaedt.modules.Boundary'] = __import__('Boundary_2')
# import pyaedt.modules.Boundary
hfss_geom.oboundary.EditIncidentWave(plane_wave_source._boundary_name, plane_wave_source._get_args())

# hfss_geom._create_boundary(name='IncPWave3', props=plane_wave_source_props2, boundary_type="Plane Incident Wave")

###############################################################################
# Place radar
# ~~~~~~~~~~~
# Place radar on the car. The radar is created relative to the car's coordinate
# system.
radar_coord_sys = hfss_geom.modeler.create_coordinate_system(origin=[70, 0, 0],
                                                             reference_cs="Global",
                                                             name="radar_cs",
                                                             mode="axis",
                                                             view="iso",
                                                             x_pointing=None,
                                                             y_pointing=None,
                                                             psi=0,
                                                             theta=0,
                                                             phi=0,
                                                             u=None)

# Create setups
# ~~~~~~~~~~~~
# Create setup and validate it. The ``create_sbr_pulse_doppler_setup`` method
# creates a setup and a parametric sweep on the time variable with a
# duration of two seconds. The step is computed automatically from CPI.

# hfss.create_perfecte_from_objects()
# hfss_geom.create_source_excitation()

# hfss_geom.get_excitations_name("IncPWave")
# hfss_geom.create_source_excitation()
# perfect_e_from_sheet = hfss_geom.assign_perfecte_to_sheets(sheet_list="Reflector1")


# setup_props1 = OrderedDict([('IsEnabled', True),
#                             ('MeshLink', OrderedDict([('ImportMesh', False)])),
#                             ('IsSbrRangeDoppler', False),
#                             ('RayDensityPerWavelength', 4),
#                             ('MaxNumberOfBounces', 5),
#                             ('RadiationSetup', ''),
#                             ('PTDUTDSimulationSettings', 'None'),
#                             ('Sweeps',
#                              OrderedDict([('Sweep',
#                                            OrderedDict([('RangeType',
#                                                          'LinearStep'),
#                                                         ('RangeStart',
#                                                          '1GHz'),
#                                                         ('RangeEnd', '10GHz'),
#                                                         ('RangeStep',
#                                                          '1GHz')]))])),
#                             ('ComputeFarFields', True)])

# rcs_setup = hfss_geom._create_setup(setupname="RCS_Setup_Custom", setuptype=4, props=setup_props1)
# rcs_setup=hfss_geom.get_setup("RCS_Setup")
# rcs_setup.props['ComputeFarFields'] = True
# rcs_setup.props['MaxNumberOfBounces'] = 5
# rcs_setup.props['RayDensityPerWavelength'] = 4
# rcs_setup.props['IsSbrRangeDoppler'] = False
# rcs_setup.props['IsMonostaticRCS'] = True
# rcs_setup.sweeps[0].props = {'ID': 0,
#                              'RangeType': 'LinearStep',
#                              'RangeStart': '9GHz',
#                              'RangeEnd': '11GHz',
#                              'RangeStep': '0.001GHz'}
# rcs_setup.sweeps[0].update()

rcs_setup = hfss_geom.get_setup("ISAR_Setup")
rcs_setup.props['ComputeFarFields'] = True
rcs_setup.props['MaxNumberOfBounces'] = 5
rcs_setup.props['RayDensityPerWavelength'] = 2
rcs_setup.props['IsSbrRangeDoppler'] = False
rcs_setup.props['IsMonostaticRCS'] = True
# rcs_setup.props["PTDUTDSimulationSettings"] = "None"
rcs_setup.props["PTDUTDSimulationSettings"] = "PTD Correction"
# rcs_setup.props["PTDUTDSimulationSettings"] = "PTD Correction + UTD Rays"
rcs_setup.sweeps[0].props = {'ID': 0,
                             'RangeType': 'LinearStep',
                             'RangeStart': '{0:.5f}GHz'.format(freq_start_GHz),
                             'RangeEnd': '{0:.5f}GHz'.format(freq_stop_GHz),
                             'RangeStep': '{0:.7f}GHz'.format(bandwidth_GHz / num_freq_samples)}
rcs_setup.sweeps[0].update()

# monostatic_rcs_report_props = OrderedDict([('report_category', 'Monostatic RCS'),
#                                            ('report_type', 'Rectangular Plot'),
#                                            ('context',
#                                             OrderedDict([('domain', 'Sweep'),
#                                                          ('primary_sweep', 'Freq'),
#                                                          ('primary_sweep_range', ['All']),
#                                                          ('secondary_sweep_range', ['All'])])),
#                                            ('expressions', 'im(ComplexMonostaticRCSPhi))'),
#                                            ('plot_name', 'Monostatic RCS Real Response1')])
# monorcs_real = hfss_geom.post.create_report("MonoRCS real vs. freq")
# monorcs_real.plot_name = "MonoRCS real vs. freq"
# monorcs_real.update_trace_in_report()

hfss_geom.validate_simple()
hfss_geom.analyze("ISAR_Setup")

vals = hfss_geom.post.get_solution_data(expressions='ComplexMonostaticRCSPhi',
                                        setup_sweep_name=hfss_geom.nominal_sweep,
                                        report_category='Monostatic RCS')
plot_units_intrinsic = vals.active_intrinsic
plot_units_info = vals.primary_sweep_variations
vals_np_xaxis = np.array(vals.primary_sweep_values)
vals_np_real = np.array(list(vals.full_matrix_real_imag[0]['ComplexMonostaticRCSPhi'].values()))
vals_np_imag = np.array(list(vals.full_matrix_real_imag[1]['ComplexMonostaticRCSPhi'].values()))
vals_np_real = vals_np_real.reshape((len(vals_np_xaxis), -1))
vals_np_imag = vals_np_imag.reshape((len(vals_np_xaxis), -1))

plt.subplot(311)
plt.title('Real Monostatic RCS Response')
plt.ylabel('re(ComplexMonostaticRCSPhi) [m2]')
plt.xlabel('Freq [GHz]')
plt.plot(vals_np_xaxis, vals_np_real)

plt.subplot(312)
plt.title('Image Monostatic RCS Response')
plt.ylabel('im(ComplexMonostaticRCSPhi) [m2]')
plt.xlabel('Freq [GHz]')
plt.plot(vals_np_xaxis, vals_np_imag)
# plt.show(block=True)
plt.show(block=False)

phase_history = vals_np_real + 1j * vals_np_imag

x_half_range = float(x_range_val_and_units_str[0]) / 2
y_half_range = float(y_range_val_and_units_str[0]) / 2

y_range_vals = np.linspace(-y_half_range, y_half_range, num=phase_history.shape[0])
x_range_vals = np.linspace(-x_half_range, x_half_range, num=phase_history.shape[1])
[x_coord_img, y_coord_img] = np.meshgrid(x_range_vals, y_range_vals)
z_coord_img = np.zeros((len(y_range_vals), len(x_range_vals)))
coords3D_img = np.stack((x_coord_img, y_coord_img, z_coord_img), axis=0)

phi_vals = np.linspace(phi_start_deg, phi_stop_deg, phase_history.shape[1])
theta_vals = np.array([90.0])
freq_vals = np.linspace(freq_start_GHz, freq_stop_GHz, phase_history.shape[0])

with open('phase_history.npy', 'wb') as f:
    np.save(f, phase_history)
    np.save(f, coords3D_img)
    np.save(f, phi_vals)
    np.save(f, theta_vals)
    np.save(f, freq_vals)

matlab_dict = {"phase_history": phase_history, "coords3D_img": coords3D_img,
               "phi_vals": phi_vals, "theta_vals": theta_vals, "freq_vals": freq_vals
               }
scipy.io.savemat("matlab_matrix.mat", matlab_dict)

plt.subplot(313)
isar_image = np.fft.fftshift(np.fft.ifft2(phase_history))
image_extent_xy = np.array([-x_swath_extent / 2, x_swath_extent / 2, -y_swath_extent / 2, y_swath_extent / 2])
plt.imshow(20 * np.log10(np.abs(isar_image) + 0.01), extent=image_extent_xy, cmap='jet')
plt.show(block=True)
# N = len(phase_history)
# Nr = 10
# if (N % 2 == 0):
#     if (Nr % 2 == 1):  # remove an even number of samples
#         Nr = Nr + 1
#     offset_start = (N - Nr) / 2
#     offset_end = N - offset_start - 1
# else:
#     if (Nr % 2 == 0):  # remove an odd number of samples
#         Nr = Nr + 1
#     offset_start = (N - 1 - Nr) / 2
#     offset_end = N - offset_start - 1
# phase_history[int(offset_start):int(offset_end)] = 0 + 1j * 0
N_pow2_x = 2 ** (int(np.log2(phase_history.shape[1])) + 1)
N_pow2_y = 2 ** (int(np.log2(phase_history.shape[0])) + 1)

N_pad_x = N_pow2_x - phase_history.shape[1]
# N_pad_x = 0
N_pad_y = N_pow2_y - phase_history.shape[0]

growth_factor_x = N_pow2_x / phase_history.shape[1]
growth_factor_y = N_pow2_y / phase_history.shape[0]

# sample_shift = np.exp(1j*)
if False:
    # Process the phase history range in the frequency domain
    # with a Taylor windowing filter nbar=4 sll=-35dB
    # see Armin Doerry, Anatomy of a SAR Impulse Response, SANDIA Tech. Report, 2007.
    range_window = scipy.signal.windows.taylor(phase_history.shape[1], nbar=4, sll=35, norm=True, sym=True)[
                       np.newaxis] / phase_history.shape[1]
    cross_range_window = scipy.signal.windows.taylor(phase_history.shape[0], nbar=4, sll=35, norm=True, sym=True)[
                             np.newaxis].T / phase_history.shape[0]
    # range_window = scipy.signal.windows.hamming(phase_history.shape[1], sym=True)[np.newaxis]/phase_history.shape[1]
    # cross_range_window = scipy.signal.windows.hamming(phase_history.shape[0], sym=True)[np.newaxis].T/phase_history.shape[0]
    # range_window = scipy.signal.windows.tukey(phase_history.shape[1], sym=True)[np.newaxis]/phase_history.shape[1]
    # cross_range_window = scipy.signal.windows.tukey(phase_history.shape[0], sym=True)[np.newaxis].T/phase_history.shape[0]
    sar_window_img_filter = cross_range_window @ range_window
    # range_window_img = np.tile(range_window, (phase_history.shape[0], 1))
    # cross_range_window_img = np.tile(cross_range_window, (1, phase_history.shape[1]))
    # sar_window_img_filter = range_window_img * cross_range_window_img
    # sar_window_img_fft = np.fft.fft2(sar_window_img_filter)
    # isar_image = np.fft.fftshift(np.fft.ifft2(phase_history))
    # phase_history = np.fft.fft2(np.fft.ifftshift(isar_image * sar_window_img_filter))
    phase_history = phase_history * sar_window_img_filter

phase_history_shifted = np.fft.fftshift(phase_history)

if phase_history_shifted.shape[0] % 2 == 1:
    phase_history_shifted = np.vstack(
        (np.zeros((np.ceil(N_pad_y / 2).astype('int'), phase_history_shifted.shape[1]), dtype=np.cdouble),
         phase_history_shifted,
         np.zeros((np.floor(N_pad_y / 2).astype('int'), phase_history_shifted.shape[1]), dtype=np.cdouble)))
else:
    phase_history_shifted = np.vstack(
        (np.zeros((np.ceil(N_pad_y / 2).astype('int'), phase_history_shifted.shape[1]), dtype=np.cdouble),
         phase_history_shifted[0, :] / 2,
         phase_history_shifted[1:, :],
         phase_history_shifted[0, :] / 2,
         np.zeros((np.floor(N_pad_y / 2).astype('int') - 1, phase_history_shifted.shape[1]), dtype=np.cdouble)))

if phase_history_shifted.shape[1] % 2 == 1:
    phase_history_shifted = np.hstack(
        (np.zeros((phase_history_shifted.shape[0], np.ceil(N_pad_x / 2).astype('int')), dtype=np.cdouble),
         phase_history_shifted,
         np.zeros((phase_history_shifted.shape[0], np.floor(N_pad_x / 2).astype('int')), dtype=np.cdouble)))
else:
    phase_history_shifted = np.hstack(
        (np.zeros((phase_history_shifted.shape[0], np.ceil(N_pad_x / 2).astype('int')), dtype=np.cdouble),
         phase_history_shifted[:, 0][np.newaxis].T / 2,
         phase_history_shifted[:, 1:],
         phase_history_shifted[:, 0][np.newaxis].T / 2,
         np.zeros((phase_history_shifted.shape[0], np.floor(N_pad_x / 2).astype('int') - 1), dtype=np.cdouble)))

phase_history_padded = np.fft.ifftshift(phase_history_shifted)

if False:
    # Process the phase history range in the frequency domain
    # with a Taylor windowing filter nbar=4 sll=-35dB
    # see Armin Doerry, Anatomy of a SAR Impulse Response, SANDIA Tech. Report, 2007.
    # range_window = scipy.signal.windows.taylor(phase_history_padded.shape[1], nbar=4, sll=35, norm=True, sym=True)
    # cross_range_window = scipy.signal.windows.taylor(phase_history_padded.shape[0], nbar=4, sll=35, norm=True, sym=True)[np.newaxis].T
    range_window = scipy.signal.windows.hamming(phase_history_padded.shape[1], sym=True) / phase_history_padded.shape[1]
    cross_range_window = scipy.signal.windows.hamming(phase_history_padded.shape[0], sym=True)[np.newaxis].T / \
                         phase_history_padded.shape[0]
    range_window_img = np.tile(range_window, (phase_history_padded.shape[0], 1))
    cross_range_window_img = np.tile(cross_range_window, (1, phase_history_padded.shape[1]))
    sar_window_img_filter = range_window_img * cross_range_window_img
    # sar_window_img_fft = np.fft.fft2(sar_window_img_filter)
    phase_history_padded = phase_history_padded * sar_window_img_filter

isar_image_complex = np.fft.fftshift(np.fft.ifft2(phase_history_padded))

x_range_vals = np.linspace(-x_half_range, x_half_range, num=isar_image_complex.shape[1])
y_range_vals = np.linspace(-y_half_range, y_half_range, num=isar_image_complex.shape[0])

[x_coord_img, y_coord_img] = np.meshgrid(x_range_vals, y_range_vals)
z_coord_img = np.zeros((len(y_range_vals), len(x_range_vals)))
coords3D_img = np.stack((x_coord_img, y_coord_img, z_coord_img), axis=0)
# r0 = np.array([[0], [0], [0]])
# dr_i = np.linalg.norm(r0) - np.linalg.norm(coord3D_img - r0, axis=0)

# isar_image_complex = np.roll(isar_image_complex, int(isar_image_complex.shape[0] / 2), axis=0)
# isar_image_complex = np.roll(isar_image_complex, int(isar_image_complex.shape[1] / 2), axis=1)
plt.figure
# plt.subplot(313)
extent = [min(x_range_vals), max(x_range_vals), min(y_range_vals), max(y_range_vals)]
plt.xlabel('Range [m]')
plt.ylabel('Range [m]')
plt.imshow(20 * np.log10(np.abs(isar_image_complex)), extent=extent, cmap='jet')
plt.show(block=True)

# steup = hfss.create_linear_count_sweep(        setupname,
#         unit,
#         freqstart,
#         freqstop,
#         num_of_freq_points=None,
#         sweepname=None,
#         save_fields=True,
#         save_rad_fields=False,
#         sweep_type="Discrete",
#         interpolation_tol=0.5,
#         interpolation_max_solutions=250,)
# AA=hfss_geom.create_source_excitation()
# radar_freq_sweep = hfss_geom.create_linear_count_sweep(setupname="Setup1",
#                                                        unit="GHz",
#                                                        freqstart="9",
#                                                        freqstop="11",
#                                                        num_of_freq_points="50",
#                                                        sweepname="RadarSweep",
#                                                        save_fields=True,
#                                                        save_rad_fields=False,
#                                                        sweep_type="Discrete",
#                                                        interpolation_tol=0.5,
#                                                        interpolation_max_solutions=250)
# hfss_geom.create_sbr_pulse_doppler_setup()

# props={}
# props["RadiationSetup"] = "ATK_3D"
# props["ComputeFarFields"] = True
# props["RayDensityPerWavelength"] = 2
# props["MaxNumberOfBounces"] = 3
# setup1 = hfss_geom.create_setup(setupname="Setup2", setuptype=4)
# setup1.auto_update = True
# {
#                     "RayDensityPerWavelength": 4,
#                     "MaxNumberOfBounces": 5,
#                     "EnableCWRays": False,
#                     "EnableSBRSelfCoupling": False,
#                     "UseSBRAdvOptionsGOBlockage": False,
#                     "UseSBRAdvOptionsWedges": False,
#                     "PTDUTDSimulationSettings": "None",
#                     "SkipSBRSolveDuringAdaptivePasses": True,
#                     "UseSBREnhancedRadiatedPowerCalculation": False,
#                     "AdaptFEBIWithRadiation": False,
#                 }
# hfss_geom.post.create_report()

###############################################################################
# Place radar
# ~~~~~~~~~~~
# Place radar on the car. The radar is created relative to the car's coordinate
# system.

# radar1 = hfss.create_sbr_radar_from_json(
#     radar_file=radar_lib,
#     radar_name="Example_1Tx_1Rx",
#     offset=[2.57, 0, 0.54],
#     use_relative_cs=True,
#     relative_cs_name=car1.cs_name,
# )

###############################################################################
# Create setup
# ~~~~~~~~~~~~
# Create setup and validate it. The ``create_sbr_pulse_doppler_setup`` method
# creates a setup and a parametric sweep on the time variable with a
# duration of two seconds. The step is computed automatically from CPI.

# setup, sweep = hfss.create_sbr_pulse_doppler_setup(sweep_time_duration=2)
# hfss.set_sbr_current_sources_options()

###############################################################################
# Plot model
# ~~~~~~~~~~
# Plot the model.

plot_obj = hfss_geom.plot(show=False, plot_air_objects=True, view="xy")
plot_obj.background_color = [153, 203, 255]
plot_obj.zoom = 0.5
# plot_obj.show_grid = False
# plot_obj.show_axes = False
# plot_obj.bounding_box = False
plot_obj.plot(export_image_path=scene_geom_output_img)

###############################################################################
# Solve and release AEDT
# ~~~~~~~~~~~~~~~~~~~~~~
# Solve and release AEDT. To solve, uncomment the ``app.analyze_setup`` command
# to activate the simulation.

# hfss.save_project()
hfss_geom.release_desktop(close_projects=True, close_desktop=True)

# setup_rcs = hfss_geom.create_setup("RCS_calculator", 4)
# "EnableCWRays": False,
# "MaxNumberOfBounces": 5,
# setup_rcs_props = {
#     "ComputeFarFields": True,
#     "RayDensityPerWavelength": 4,
#     "EnableSBRSelfCoupling": False,
#     "UseSBRAdvOptionsGOBlockage": False,
#     "UseSBRAdvOptionsWedges": False,
#     "PTDUTDSimulationSettings": "None",
#     "SkipSBRSolveDuringAdaptivePasses": True,
#     "UseSBREnhancedRadiatedPowerCalculation": False,
#     "AdaptFEBIWithRadiation": False
# }

# setup_rcs.update(update_dictionary=setup_rcs_props)
#     ["ComputeFarFields"] = True
# setup_rcs["RayDensityPerWavelength"] = 4
# setup_rcs["MaxNumberOfBounces"] = 5
# setup_rcs["EnableCWRays"] = False
# setup_rcs["EnableSBRSelfCoupling"] = False
# setup_rcs["UseSBRAdvOptionsGOBlockage"] = False
# setup_rcs["UseSBRAdvOptionsWedges"] = False
# setup_rcs["PTDUTDSimulationSettings"] = "None"
# setup_rcs["SkipSBRSolveDuringAdaptivePasses"] = True,
# setup_rcs["UseSBREnhancedRadiatedPowerCalculation"] = False,
# setup_rcs["AdaptFEBIWithRadiation"] = False
