from datetime import datetime
import pyaedt
import platform
import os
import numpy as np

if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT252"] = "/opt/Ansys/v252/AnsysEM/"
else:
    os.environ["ANSYSEM_ROOT252"] = "C:\\Program Files\\AnsysEM\\v252\\Win64\\"

aedt_version = "2025.2"

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
perfectly_matched_layer_color = [255, 128, 128]

time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "Demo 1 " + current_time_str
design_name = "Demo 1 " + current_time_str
non_graphical = False
# Launch HFSS Student (auto-creates project and design)
# hfss = Hfss(student_version=True, non_graphical=False)
hfss = pyaedt.Hfss(
    version=aedt_version,
    solution_type="Terminal",
    new_desktop=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical
)
design = hfss  # The Hfss object handles the design directly

# --- Patch Array Parameters ---
freq = 2.4e9             # Frequency (Hz)
patch_w = 0.038          # Patch width (m)
patch_l = 0.029          # Patch length (m)
substrate_thickness = 0.003
gap = 0.01               # Gap between patches
num_elements = 5

# --- Create substrate ---
substrate_length = num_elements * (patch_w + gap)
substrate = design.modeler.create_box([0, 0, 0],
                                      [substrate_length, patch_w, substrate_thickness],
                                      name="substrate",
                                      matname="FR4_epoxy")

# --- Create patches in a linear array ---
patches = []
for i in range(num_elements):
    x_pos = i * (patch_w + gap)
    patch = design.modeler.create_box([x_pos, 0, substrate_thickness],
                                      [patch_w, patch_l, 0.001],  # patch thickness
                                      name=f"patch_{i+1}",
                                      matname="copper")
    patches.append(patch)

# --- Create ground plane ---
ground = design.modeler.create_box([0, 0, 0],
                                   [substrate_length, patch_l, 0.001],
                                   name="ground",
                                   matname="copper")

# --- Assign lumped ports to each patch ---
# for i, patch in enumerate(patches):
#     face = patch.faces[0]  # select one face for port
#     design.create_lumped_port(face=face,
#                               name=f"port_{i+1}",
#                               use_object_coordinate_system=True)
# MAKE A HOLE FOR THE CYLINDRICAL SOURCE
# Construct the via from the ground plane to the metal patch
# s_axis is "X", "Y" or "Z"
cm2mm = 10
monopole_size_cm = 0.05
monopole_length_cm = 0.3
monopole_position = np.array([0, 0, -substrate_thickness])
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


# --- Save project ---
hfss.save_project("PatchArrayProject.aedt")

print(f"{num_elements}-element patch array created successfully!")

# --- Release AEDT ---
hfss.release_desktop()

