"""
SBR+: Import Geometry from Maps
-------------------------------
This example shows how you can use PyAEDT to create an HFSS SBR+ project from an
OpenStreeMaps.
"""
###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports and set up the local path to the PyAEDT
# directory path.

import os
import pyaedt
# from pyaedt import Hfss
os.environ["ANSYSEM_ROOT231"] = "/opt/AnsysEM/v231/Linux64/"
aedt_version = "2023.1"
working_directory = 'projects'
project_file = 'projects/EPIC_Map.aedt'
# project_file = 'EPIC_Map_2000'
###############################################################################
# Define Location to import
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Define latitude and longitude to import.
# ansys_home = [40.273726, -80.168269]
# uncc_practice_field = [35.312504, -80.739100]
uncc_epic = [35.309925, -80.740539]
terrain_radius = 500

###############################################################################
# Set non-graphical mode
# ~~~~~~~~~~~~~~~~~~~~~~
# Set non-graphical mode. ``"PYAEDT_NON_GRAPHICAL"`` is needed to generate
# documentation only.
# You can set ``non_graphical`` either to ``True`` or ``False``.

non_graphical = False
NewThread = True
desktop = pyaedt.launch_desktop(specified_version=aedt_version,
                                non_graphical=non_graphical,
                                new_desktop_session=NewThread
                                )

###############################################################################
# Define designs
# ~~~~~~~~~~~~~~
# Define two designs, one source and one target.
# Each design is connected to a different object.
# project_name = pyaedt.generate_unique_project_name(project_name="doppler")
app = pyaedt.Hfss(
    designname="Ansys",
    solution_type="SBR+",
    specified_version=aedt_version,
    new_desktop_session=True,
    projectname=project_file,
    non_graphical=non_graphical
)

###############################################################################
# Generate map and import
# ~~~~~~~~~~~~~~~~~~~~~~~
# Assign boundaries.
app.modeler.import_from_openstreet_map(uncc_epic,
                                       terrain_radius=terrain_radius,
                                       road_step=3,
                                       plot_before_importing=False,
                                       import_in_aedt=True)

###############################################################################
# Plot model
# ~~~~~~~~~~
# Plot the model.

plot_obj = app.plot(show=False, plot_air_objects=True)
plot_obj.background_color = [153, 203, 255]
plot_obj.zoom = 1.5
plot_obj.show_grid = False
plot_obj.show_axes = False
plot_obj.bounding_box = False
plot_obj.plot(os.path.join(app.working_directory, "Source.jpg"))

###############################################################################
# Release AEDT
# ~~~~~~~~~~~~
# Release AEDT and close the example.
app.save_project()
app.release_desktop()