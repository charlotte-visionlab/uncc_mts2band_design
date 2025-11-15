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

# Launch AEDT
# ~~~~~~~~~~~
# Launch AEDT.
os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
aedt_version = "2023.1"
projectname = "DB-Doppler exp 00"
designname = "Dual Band Doppler exp 00"

source_project_path = 'projects'
source_project_name = os.path.join(source_project_path, 'SBR_ISAR_BaseProject.aedt')
scene_geom_output_img = os.path.join(source_project_path, 'SBR_ISAR_scene_top_down_image.jpg')

# library_path = pyaedt.downloads.download_multiparts()
library_path = "radar_object_library"

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
trihedral_reflector = os.path.join(reflector_folder, "trihedral_reflector1")
reflector1 = hfss_geom.modeler.add_environment(env_folder=trihedral_reflector,
                                               environment_name="Reflector1",
                                               global_offset=[0, 0, 0],
                                               yaw=0,
                                               pitch=0,
                                               roll=0)
# prim = hfss_geom.modeler
# sheet_names = hfss_geom.modeler.sheet_names
# hfss_geom.assign_perfecte_to_sheets(sheet_list=sheet_names,sourcename=perfE_source)

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
radar1 = hfss_geom.create_sbr_radar_from_json(
    radar_file=radar_lib,
    radar_name="Example_1Tx_1Rx_exp00",
    offset=[2.57, 0, 0.54],
    use_relative_cs=True,
    relative_cs_name="radar_cs"
)
# hfss_geom.post.create_report(setup_sweep_name="re(ComplexMonostaticRCSPhi)", primary_sweep_variable="Freq", )

# Create setups
# ~~~~~~~~~~~~
# Create setup and validate it. The ``create_sbr_pulse_doppler_setup`` method
# creates a setup and a parametric sweep on the time variable with a
# duration of two seconds. The step is computed automatically from CPI.

setup, sweep = hfss_geom.create_sbr_pulse_doppler_setup(sweep_time_duration=2)
hfss_geom.set_sbr_current_sources_options()
hfss_geom.validate_simple()
hfss_geom.analyze(setup.name)
###############################################################################
# Plot model
# ~~~~~~~~~~
# Plot the model.
# app.plot(show=False, export_path=os.path.join(app.working_directory, "Image.jpg"), plot_air_objects=True)
# hfss.plot(show=False, export_path=os.path.join('.', "Image.jpg"), plot_air_objects=True)

# hfss.create_perfecte_from_objects()

# Get Data
# ~~~~~~~~~~~~
# vals = hfss_geom.post.get_solution_data(expressions='ComplexMonostaticRCSPhi', setup_sweep_name=hfss_geom.nominal_sweep,
#                                         report_category='Monostatic RCS')
# plot_units_intrinsic = vals.active_intrinsic
# plot_units_info = vals.primary_sweep_variations
# vals_np_xaxis = np.array(vals.primary_sweep_values)
# vals_np_imag = np.array(vals.data_imag())
# vals_np_real = np.array(vals.data_real())
# plt.plot(vals_np_xaxis, vals_np_imag)
# plt.plot(vals_np_xaxis, vals_np_real)
# plt.show(block=True)

###############################################################################
# Solve and release AEDT
# ~~~~~~~~~~~~~~~~~~~~~~
# Solve and release AEDT. To solve, uncomment the ``app.analyze_setup`` command
# to activate the simulation.

# hfss.save_project()
hfss_geom.release_desktop(close_projects=True, close_desktop=True)

