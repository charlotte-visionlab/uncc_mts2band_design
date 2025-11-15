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

speed_of_light = 2.99792458e8
os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
aedt_version = "2023.1"
projectname = "Pulse Compression"
designname = "Range Profile Experiments"
source_project_path = 'projects'
source_project_name = os.path.join(source_project_path, 'SBR_RCS_BaseProject.aedt')
scene_geom_output_img = os.path.join(source_project_path, 'SBR_RCS_scene_top_down_image.jpg')

# library_path = pyaedt.downloads.download_multiparts()
library_path = "radar_object_library"

def read_ansys_directivity_file(filename):
    import re
    with open(filename, 'r') as file:
        csv_string = file.read()
    data = csv_string.split('\n')
    data_values = []
    for row in range(7,len(data)):
        data[row] = data[row].strip()
        number_strings = re.split('\s+', data[row])
        if len(number_strings)==2:
            row_values = [float(x) for x in number_strings]
        data_values.append(row_values)
    data_values = np.array(data_values)
    plt.xlabel('Angle (deg.)')
    plt.ylabel('Gain (dB)')
    plt.plot(data_values[:,0], data_values[:,1])
    return data_values

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


def configure_sbr(name, ray_density=1, max_number_of_bounces=5, freq_start_GHz=1, freq_stop_GHz=10, freq_step_GHz=1):
    # rcs_setup = hfss_geom.get_setup(name)
    rcs_setup = hfss_geom.setups[0]
    rcs_setup.props['ComputeFarFields'] = True
    rcs_setup.props['MaxNumberOfBounces'] = max_number_of_bounces
    rcs_setup.props['RayDensityPerWavelength'] = ray_density
    rcs_setup.props['IsSbrRangeDoppler'] = False
    rcs_setup.props['IsMonostaticRCS'] = True
    rcs_setup.sweeps[0].props = {'ID': 0,
                                 'RangeType': 'LinearStep',
                                 'RangeStart': '{0:.5f}GHz'.format(freq_start_GHz),
                                 'RangeEnd': '{0:.5f}GHz'.format(freq_stop_GHz),
                                 'RangeStep': '{0:.7f}GHz'.format(freq_step_GHz)}
    rcs_setup.sweeps[0].update()


def get_sbr_configuration(radar_parameters):
    parse_arg = lambda x: list(filter(None, re.split(r'([0-9.]+)(?!.*\d)', x)))
    freq_val_and_units_str = parse_arg(radar_parameters['Nominal Operating Frequency'])
    range_val_and_units_str = parse_arg(radar_parameters['Range Extent'])
    resolution_val_and_units_str = parse_arg(radar_parameters['Range Resolution'])

    freq_center_GHz = float(freq_val_and_units_str[0])
    bandwidth_GHz = speed_of_light / (2 * float(resolution_val_and_units_str[0]) * 1e9)
    num_freq_samples = int(float(range_val_and_units_str[0]) / float(resolution_val_and_units_str[0]))
    freq_start_GHz = freq_center_GHz - (bandwidth_GHz / 2)
    freq_stop_GHz = freq_center_GHz + (bandwidth_GHz / 2)
    freq_step_GHz = bandwidth_GHz / num_freq_samples
    range_val = float(range_val_and_units_str[0])
    range_resolution = float(resolution_val_and_units_str[0])
    return freq_start_GHz, freq_stop_GHz, freq_step_GHz, range_val, range_resolution


def get_ComplexMonostaticRCS_solution_data():
    solution_data = hfss_geom.post.get_solution_data(expressions='ComplexMonostaticRCSPhi',
                                                     setup_sweep_name=hfss_geom.nominal_sweep,
                                                     report_category='Monostatic RCS')
    # plot_units_intrinsic = solution_data.active_intrinsic
    plot_units_info = solution_data.primary_sweep_variations
    freq_vals = np.array(solution_data.primary_sweep_values)
    imag_vals = np.array(solution_data.data_imag())
    real_vals = np.array(solution_data.data_real())
    return real_vals + 1j * imag_vals, freq_vals, plot_units_info


matplotlib_pyplot_setup()

radar_parameters = {
    'Nominal Operating Frequency': '10GHz',
    'Range Extent': '30m',  # X
    'Range Resolution': '0.05m',  # dX
    'Polarization': 'Phi-Phi'
}

antenna_directivity = read_ansys_directivity_file('9THz_lamo5_ipdip_hd5um_N25_theta_n2_Xs_275_m0p15.txt')
# Launch AEDT
# ~~~~~~~~~~~

# parse_arg = lambda x: list(filter(None, re.split(r'([0-9.]+)(?!.*\d)', x)))
# freq_val_and_units_str = parse_arg(radar_parameters['Nominal Operating Frequency'])
# range_val_and_units_str = parse_arg(radar_parameters['Range Extent'])
# resolution_val_and_units_str = parse_arg(radar_parameters['Range Resolution'])
#
# freq_center_GHz = float(freq_val_and_units_str[0])
# bandwidth_GHz = speed_of_light / (2 * float(resolution_val_and_units_str[0]) * 1e9)
# num_freq_samples = int(float(range_val_and_units_str[0]) / float(resolution_val_and_units_str[0]))
# freq_start_GHz = freq_center_GHz - (bandwidth_GHz / 2)
# freq_stop_GHz = freq_center_GHz + (bandwidth_GHz / 2)

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
###############################################################################
# Define environment
# ~~~~~~~~~~~~~~~~~~
# Define the background environment.

# road1 = app.modeler.add_environment(env_folder=env_folder, environment_name="Bari")
trihedral_reflector1 = os.path.join(reflector_folder, "trihedral_reflector1")
trihedral_reflector2 = os.path.join(reflector_folder, "trihedral_reflector2")
# reflector1 = hfss_geom.modeler.add_environment(env_folder=trihedral_reflector1,
#                                                environment_name="Reflector1",
#                                                global_offset=[0, 0, 0],
#                                                yaw=0,
#                                                pitch=0,
#                                                roll=0)
sphere_object_list = []

sphere_position = [0, 0, 0]
sphere_radius = 0.3
sphere_name = 'Big_Sphere'
sphere_object = hfss_geom.modeler.create_sphere(sphere_position, sphere_radius, name=None)
sphere_object_list.append(sphere_object)
hfss_geom.assign_perfecte_to_sheets(sphere_object)

for sphere_num in range(0, 25):
    sphere_position = [sphere_num - 12 - 1, sphere_num - 12, 0]
    sphere_radius = 0.005
    sphere_name = 'Sphere_{0:d}'.format(sphere_num)
    sphere_object = hfss_geom.modeler.create_sphere(sphere_position, sphere_radius, name=None)
    sphere_object_list.append(sphere_object)
    hfss_geom.assign_perfecte_to_sheets(sphere_object)
# sheet_names = hfss_geom.modeler.sheet_names
# hfss_geom.assign_perfecte_to_sheets(sheet_list=sheet_names,sourcename=perfE_source)

plane_wave_source = hfss_geom.boundaries[0]
plane_wave_source_props = plane_wave_source.props
plane_wave_source.auto_update = False
plane_wave_source_props['PhiStart'] = '0deg'
plane_wave_source_props['PhiStop'] = '0deg'
plane_wave_source_props['ThetaStart'] = '90deg'
plane_wave_source_props['ThetaStop'] = '90deg'
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
                            'PhiStart': '0deg',
                            'PhiStop': '0deg',
                            'PhiPoints': 1,
                            'ThetaStart': '90deg',
                            'ThetaStop': '90deg',
                            'ThetaPoints': 1,
                            'EoPhi': '1',
                            'EoTheta': '0',
                            'OriginX': '0mm',
                            'OriginY': '0mm',
                            'OriginZ': '0mm',
                            'IsPropagating': True,
                            'IsEvanescent': False,
                            'IsEllipticallyPolarized': False,
                            'PhiEnd': 0}
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
radar_coord_sys = hfss_geom.modeler.create_coordinate_system(origin=[5, 0, 0],
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

freq_start_GHz, freq_stop_GHz, freq_step_GHz, range_val, range_resolution = get_sbr_configuration(radar_parameters)

configure_sbr('RCS_Setup', ray_density=1, max_number_of_bounces=5, freq_start_GHz=freq_start_GHz,
              freq_stop_GHz=freq_stop_GHz, freq_step_GHz=freq_step_GHz)

# rcs_setup = hfss_geom.get_setup("RCS_Setup")
# rcs_setup.props['ComputeFarFields'] = True
# rcs_setup.props['MaxNumberOfBounces'] = 5
# rcs_setup.props['RayDensityPerWavelength'] = 1
# rcs_setup.props['IsSbrRangeDoppler'] = False
# rcs_setup.props['IsMonostaticRCS'] = True
# rcs_setup.sweeps[0].props = {'ID': 0,
#                              'RangeType': 'LinearStep',
#                              'RangeStart': '{0:.5f}GHz'.format(freq_start_GHz),
#                              'RangeEnd': '{0:.5f}GHz'.format(freq_stop_GHz),
#                              'RangeStep': '{0:.7f}GHz'.format(bandwidth_GHz / num_freq_samples)}
# rcs_setup.sweeps[0].update()

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
hfss_geom.analyze("RCS_Setup")
phase_history, freq_vals, plot_units = get_ComplexMonostaticRCS_solution_data()

radar_parameters['Nominal Operating Frequency'] = '100GHz'
freq_start_GHz, freq_stop_GHz, freq_step_GHz, range_val, range_resolution = get_sbr_configuration(radar_parameters)
configure_sbr('RCS_Setup', ray_density=1, max_number_of_bounces=5, freq_start_GHz=freq_start_GHz,
              freq_stop_GHz=freq_stop_GHz, freq_step_GHz=freq_step_GHz)
hfss_geom.validate_simple()
hfss_geom.analyze("RCS_Setup")
phase_history_100GHz, freq_vals_100GHz, plot_units_100GHz = get_ComplexMonostaticRCS_solution_data()

# vals = hfss_geom.post.get_solution_data(expressions='ComplexMonostaticRCSPhi',
#                                         setup_sweep_name=hfss_geom.nominal_sweep,
#                                         report_category='Monostatic RCS')
# plot_units_intrinsic = vals.active_intrinsic
# plot_units_info = vals.primary_sweep_variations
# vals_np_xaxis = np.array(vals.primary_sweep_values)
# vals_np_imag = np.array(vals.data_imag())
# vals_np_real = np.array(vals.data_real())

plt.figure
plt.subplot(311)
plt.title('Real Monostatic RCS Response')
plt.ylabel('re(ComplexMonostaticRCSPhi) [m2]')
plt.xlabel('Freq [GHz]')
plt.plot(freq_vals, phase_history.real)
plt.plot(freq_vals, phase_history_100GHz.real)
plt.subplot(312)
plt.title('Imag Monostatic RCS Response')
plt.ylabel('im(ComplexMonostaticRCSPhi) [m2]')
plt.xlabel('Freq [GHz]')
plt.plot(freq_vals, phase_history.imag)
plt.plot(freq_vals, phase_history_100GHz.imag)
# plt.show(block=True)
plt.show(block=False)

# phase_history = vals_np_real + 1j * vals_np_imag
# Calculate the next power of 2 to pad the signal
N_pow2 = 2 ** (int(np.log2(len(phase_history))) + 1)
N_pad = N_pow2 - len(phase_history)
# sample_shift = np.exp(1j*)
phase_history = np.pad(phase_history, (0, N_pad), 'constant')
phase_history_100GHz = np.pad(phase_history_100GHz, (0, N_pad), 'constant')

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

range_profile = np.fft.ifft(phase_history)
range_profile_100GHz = np.fft.ifft(phase_history_100GHz)
# range_profile = np.fft.ifft(phase_history)
half_range = range_val / 2
range_vals = np.linspace(-half_range, half_range, num=len(range_profile))

plt.subplot(313)
plt.title('Range Profile')
plt.ylabel('RCS Power Received [dB]')
plt.xlabel('Range [m]')
# range_vals = np.roll(range_vals, int(num_freq_samples / 2)).copy()
plt.plot(range_vals, 20 * np.log10(np.abs(np.roll(range_profile, int(len(range_profile) / 2)))), label='10 GHz')
plt.plot(range_vals, 20 * np.log10(np.abs(np.roll(range_profile_100GHz, int(len(range_profile_100GHz) / 2)))), label='100 GHz')
# plt.plot(range_vals, 20 * np.log10(np.abs(range_profile)))
plt.subplots_adjust(top=1.00, bottom=0.1, hspace=0.4, wspace=0.4)
plt.legend(loc="upper right")
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
# radar_freq_sweep = hfss_geom.create_linear_count_sweep(setupname="RCS_Setup",
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
