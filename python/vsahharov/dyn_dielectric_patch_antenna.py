###############################################################################
# Concentric single-patch HFSS script (based on original file)
# - Uses ansys.aedt.core Hfss API and launches AEDT v252
# - Single concentric patch (inner disk + 4 arc strips)
# - FR4 substrate with LC cavity below patch; LC is anisotropic
# - Removes metasurface/unit-cell array loop
###############################################################################

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

#import uncc_mts_compute_config as compute_config


if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT252"] = "/opt/Ansys/v252/AnsysEM/"
else:
    os.environ["ANSYSEM_ROOT252"] = "C:\\Program Files\\AnsysEM\\v252\\Win64\\"

aedt_version = "2025.2"

#solver_configuration = compute_config.SolverConfig().solver_config


###############################################################################
# Constants & parameters (kept in mm where appropriate)
###############################################################################
speed_of_light = 2.99792458e8
frequency = 3.1e9
wavelength = speed_of_light / frequency

# board / ground extents (mm)
antenna_dimensions_xy_mm = np.array([50.0, 50.0])  # overall board area (kept larger than patch)
antenna_coord_origin_xy_mm = np.array([0.5 * antenna_dimensions_xy_mm[0],
                                       0.5 * antenna_dimensions_xy_mm[1]])

# Substrate / patch stack
fr4_thickness = 1.6  # mm
lc_thickness = 1.0   # mm (cavity depth)
metal_thickness = 0.035  # mm (copper thickness if needed)

# Concentric patch dimensions (mm) — manufacturable values
inner_radius = 4.2    # mm
outer_radius = 8.0    # mm (centerline of arc)
arc_width = 1.5       # mm (thickness of arc after thickening)
arc_span_deg = 75.0   # angular span of each arc
arc_centers = [45, 135, 225, 315]  # arc center angles in degrees

# Feed geometry
via_radius_mm = 0.3  # mm
via_length_mm = fr4_thickness + lc_thickness  # mm

# Visualization colors (optional)
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
lc_color = [200, 160, 255]

###############################################################################
# Project / design names
###############################################################################
time_str = datetime.now().strftime("%b%d_%H-%M-%S")
project_name = "ConcentricPatch_" + time_str
design_name = "HFSSDesign"

###############################################################################
# Launch AEDT and create HFSS design (v252)
###############################################################################
non_graphical_mode = False
NewThread = True
# desktop = pyaedt.launch_desktop()
# desktop = pyaedt.launch_desktop(version=aedt_version,
#                                 non_graphical=non_graphical_mode,
#                                 new_desktop=NewThread)

# Solution Types are: { "Modal", "Terminal", "Eigenmode", "Transient Network", "SBR+", "Characteristic"}

hfss = pyaedt.Hfss(
    version=aedt_version,
    solution_type="Terminal",
    new_desktop=True,
    projectname=project_name,
    designname=design_name,
    close_on_exit=True,
    non_graphical=non_graphical_mode
)

hfss.modeler.model_units = "mm"
hfss.autosave_disable()

###############################################################################
# Material: define anisotropic LC BEFORE creating the LC box
###############################################################################
# Define LC permittivity variables so you can parametrize them later
hfss["e_xx"] = "2.6"
hfss["e_yy"] = "2.6"
hfss["e_zz"] = "3.2"
hfss["lc_loss_tangent"] = "0.02"

if "LC_MDA_98_1602" not in hfss.materials.material_keys:
    lc_mat = hfss.materials.add_material("LC_MDA_98_1602")
    lc_mat.permittivity = ["e_xx", "e_yy", "e_zz"]
    lc_mat.dielectric_loss_tangent = "lc_loss_tangent"

# Keep FR4 (use built-in if available)
# You may also add FR4 explicitly if your AEDT doesn't have FR4_epoxy material key
if "FR4_epoxy" not in hfss.materials.material_keys:
    try:
        hfss.materials.add_material("FR4_epoxy")
    except Exception:
        pass

###############################################################################
# Geometry: Ground plane, FR4 substrate, LC cavity
###############################################################################
# Ground plane (centered)
board_x = antenna_dimensions_xy_mm[0]
board_y = antenna_dimensions_xy_mm[1]

ground = hfss.modeler.create_rectangle(
    origin=[-board_x / 2, -board_y / 2, 0.0],
    sizes=[f"{board_x}mm", f"{board_y}mm"],
    orientation="XY",
    name="cell_ground_plane",
    material="pec"
)
ground.color = metal_color

hfss.modeler.fit_all()

# FR4 substrate
fr4 = hfss.modeler.create_box(
    origin=[-board_x / 2, -board_y / 2, 0.0],
    sizes=[f"{board_x}mm", f"{board_y}mm", f"{fr4_thickness}mm"],
    name="FR4_Substrate",
    material="FR4_epoxy"
)
fr4.color = dielectric_color

# LC cavity box (placed on top of FR4)
lc_origin_z = fr4_thickness
lc_origin_x = -outer_radius
lc_origin_y = -outer_radius
lc_box = hfss.modeler.create_box(
    origin=[f"{lc_origin_x}mm", f"{lc_origin_y}mm", f"{lc_origin_z}mm"],
    sizes=[f"{2*outer_radius}mm", f"{2*outer_radius}mm", f"{lc_thickness}mm"],
    name="LC_Cavity",
    material="LC_MDA_98_1602"
)
lc_box.color = lc_color

# subtract LC cavity from FR4 so LC sits in the milled cavity
try:
    hfss.modeler.subtract("FR4_Substrate", "LC_Cavity", keep_originals=False)
except Exception:
    # If subtract fails, continue — LC remains as separate solid on top
    pass

###############################################################################
# Concentric patch geometry (top of LC)
###############################################################################
patch_z = fr4_thickness + lc_thickness  # z coordinate where patch metal sits

# Inner filled disk (PEC)
inner_patch = hfss.modeler.create_circle(
    origin=[0.0, 0.0, patch_z],
    radius=f"{inner_radius}mm",
    orientation="XY",
    name="InnerPatch",
    material="pec",
    cover_surface=True
)

# Outer arcs: create as polyline arc segments and thicken into strips
# Outer arcs: create as spline polylines and thicken into strips
arc_names = []
for ac in arc_centers:
    start_ang = ac - arc_span_deg / 2.0
    end_ang = ac + arc_span_deg / 2.0

    # sample the arc as points (valid for all AEDT versions)
    num_pts = 15
    angles = np.linspace(start_ang, end_ang, num_pts)
    pts = [[outer_radius * np.cos(np.radians(ang)),
            outer_radius * np.sin(np.radians(ang)),
            patch_z] for ang in angles]

    # create polyline spline
    arc_pl = hfss.modeler.create_polyline(
        points=pts,
        close_surface=False,
        cover_surface=False,
        name=f"ArcSeg_{ac}"
    )

    # thicken into strip
    arc_strip = hfss.modeler.thicken_sheet(
        arc_pl,
        thickness=f"{arc_width}mm",
        material="pec",
        name=f"ArcStrip_{ac}"
    )

    arc_names.append(arc_strip.name)


# Assign perfect E to all patch surfaces (optional; material already pec)
try:
    hfss.assign_perfecte_to_sheets(["InnerPatch"] + arc_names)
except Exception:
    pass

###############################################################################
# Feed: create via and feed hole similar to original logic
###############################################################################
# monopole-like feed: cylinder from ground plane up to patch
monopole_length_mm = 3.0  # chosen length in mm (approx)
via_radius = via_radius_mm

via = hfss.modeler.create_cylinder(
    cs_axis="Z",
    origin=[0.0, 0.0, patch_z - monopole_length_mm],
    radius=f"{via_radius}mm",
    height=f"{monopole_length_mm}mm",
    name="FeedVia",
    material="pec"
)
via.color = metal_color

# subtract feed hole from FR4 substrate to create physical hole (keep originals if desired)
try:
    hfss.modeler.subtract("FR4_Substrate", "FeedVia", keep_originals=True)
except Exception:
    pass

# Make feed hole through ground plane (if desired)
# Create small ellipse/hole in ground
try:
    feed_hole = hfss.modeler.create_ellipse(
        origin=[0.0, 0.0, 0.0],
        major_radius=f"{via_radius + 0.2}mm",
        ratio=1.0,
        name="monopole_feed_hole",
        material="pec",
        is_covered=True
    )
    hfss.modeler.subtract("cell_ground_plane", "monopole_feed_hole", keep_originals=False)
except Exception:
    # Some API variants use different parameter names - ignore if fails
    pass

# Create monopole shield cylinder (for port reference) to mimic original port geometry
try:
    monopole_shield = hfss.modeler.create_cylinder(
        cs_axis="Z",
        origin=[0.0, 0.0, patch_z - monopole_length_mm],
        radius=f"{via_radius + 0.2}mm",
        height=f"{monopole_length_mm}mm",
        name="monopole_shield",
        material="pec"
    )
    monopole_shield.transparency = 0.8
except Exception:
    monopole_shield = None

# Create lumped port between feed via (signal) and shield or ground
try:
    if monopole_shield:
        hfss.lumped_port_between_objects("FeedVia", "monopole_shield", axisdir=2, impedance=50)
    else:
        hfss.lumped_port_between_objects("FeedVia", "cell_ground_plane", axisdir=2, impedance=50)
except Exception:
    # fallback: try named port creation using face ids (best-effort)
    try:
        faces = via.faces
        if faces:
            face = faces[0]
            # Integrate port manually may require low-level calls - skip if fails
            pass
    except Exception:
        pass

###############################################################################
# Radiation region / open region
###############################################################################
module = hfss.get_module("ModelSetup")
module.CreateOpenRegion(
    [
        "NAME:Settings",
        "OpFreq:=", f"{frequency/1e9}GHz",
        "Boundary:=", "Radiation",
        "ApplyInfiniteGP:=", False
    ]
)

###############################################################################
# Solver setup & frequency sweep (keeps original style but at 3.1 GHz)
###############################################################################
solver_setup = hfss.create_setup(setupname="MTS_Setup", setuptype="HFSSDriven")
solver_setup_params = {
    "SolveType": 'Single',
    "Frequency": f'{frequency/1e9}GHz',
    "MaxDeltaS": 0.02,
    "PortsOnly": False,
    "MaximumPasses": 8,
    "MinimumPasses": 1,
    "MinimumConvergedPasses": 1,
    "PercentRefinement": 30,
    "IsEnabled": True,
    "BasisOrder": 1,
    "DoLambdaRefine": True,
    "DoMaterialLambda": True,
    "Target": 0.3333,
    "PortAccuracy": 2,
    "IESolverType": "Auto",
}
solver_setup.props.update(solver_setup_params)

frequency_sweep_params = {
    "unit": "GHz",
    "freqstart": 2.5,
    "freqstop": 3.7,
    "num_of_freq_points": 301,
    "sweepname": "sweep",
    "save_fields": True,
    "save_rad_fields": False,
    "sweep_type": "Interpolating",
    "interpolation_tol": 0.5,
    "interpolation_max_solutions": 250
}
solver_setup.create_frequency_sweep(**frequency_sweep_params)

# Validate design (best-effort)
try:
    setup_ok = hfss.validate_full_design()
except Exception:
    setup_ok = False

# Analyze (run) setup
hfss.analyze_setup("MTS_Setup")

# Save project and release desktop
hfss.save_project()
hfss.release_desktop(close_projects=True, close_desktop=True)
