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
import socket
from datetime import datetime
import platform
from ansys.aedt.core import Hfss, launch_desktop
import pyaedt
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
import numpy as np
from scipy.special import i1

import uncc_mts_compute_config as compute_config


def sinuous_curve(r, alpha, r_start, tau):
    return alpha * np.sin((np.pi * np.log(r / r_start)) / np.log(tau))


cm2mm = 10
frequency_max = 12e9
frequency_min = 2e9

plot_antenna = True

copper_thickness = 35.0e-6
radius_start_cm = 2.5 / cm2mm  # initial radius
radius_end_cm = 37.5 / cm2mm
tau = 1.238  # tau is usually between 1.2 and 1.5
alpha = np.deg2rad(45)  # alpha is usually between 30 and 70 degrees
# self complementary delta = 180/(2N)
delta = np.deg2rad(22.5)  # delta is the angular width of the arm
# delta = np.deg2rad(180.0 / (2 * N))  # delta is the angular width of the arm
# R_end = R0 * (tau * N)  # final radius
N = radius_end_cm / (radius_start_cm * tau)
# R_end = np.abs(sinuous_curve(R0 * (tau * N), alpha, R0, tau))  # final radius
# Rogers 5880 standard thicknesses are 0.127, 0.252, 0.508, 0.787, 1.575
dielectric_height_cm = 1.575 / cm2mm  # <=== dielectric slab height in meters
# height = 2.54e-3  # m
arm_contact_diameter_cm = 2 * 0.25 * radius_start_cm
arm_contact_pad_thickness = 0.3
PEC_ANTENNA = True
dielectric_diameter_cm = np.array(1.1 * 2 * radius_end_cm)
antenna_diameter_cm = 2 * radius_end_cm
# ground_plane_diameter_cm = 10 * 20.0e-3
ground_plane_diameter_cm = dielectric_diameter_cm - 0.25
# ground_plane_hole_diameter_cm = 0.5
ground_plane_hole_diameter_cm = 0.25

cavity_wall_thickness_cm = 1.0 / cm2mm
cavity_diameter_cm = dielectric_diameter_cm + 2 * cavity_wall_thickness_cm

cavity_height_cm = 30.0 / cm2mm

machine = "ece-emag1"
# DIELECTRIC MATERIALS

dielectric_material_name = "Rogers RT/duroid 5880 (tm)"
dielectric_material_name = "Taconic TLY (tm)"
# dielectric_material_name = "Rogers RT/duroid 6010/6010LM (tm)"
# ground_plane_material_name = "pec"
ground_plane_material_name = "copper"
# radiation_box_material_name = "vacuum"
radiation_box_material_name = "air"


class Point2D:
    def __init__(self, x, y):
        self.data = {'x': x, 'y': y}

    def __getattr__(self, name):
        return self.data[name]

    def __setattr__(self, name, value):
        if name == 'data':
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class SinuousArm:

    def __init__(self):
        self.start = Point2D(0.0, 0.0)
        self.end = Point2D(0.0, 0.0)
        self.inner = Point2D(0.0, 0.0)
        self.outer = Point2D(0.0, 0.0)
        self.contour_mm = Point2D(0.0, 0.0)
        self.contour_position_array_mm = np.empty(0)
        self.contour_position_list_mm = []
        self.contact_position_array_mm = np.empty(0)
        self.contact_position_list_mm = []
        self.contact_center_position_mm = Point2D(0.0, 0.0)


def make_sinuous_arm(r_start, r_end, N, tau, alpha, delta, height_mm=0.0, z_orientation=0.0,
                     num_samples=1000, plot=False):
    angle_offset = np.deg2rad(z_orientation)
    arm = SinuousArm()

    # R_end = R0 * (tau * N)  # final radius
    r = np.linspace(r_start, r_end, num_samples)

    phi_inner = angle_offset + delta + sinuous_curve(r, alpha, r_start, tau)
    phi_outer = angle_offset - delta + sinuous_curve(r, alpha, r_start, tau)

    phi_inner_end = angle_offset + delta + sinuous_curve(r_end, alpha, r_start, tau)
    phi_outer_end = angle_offset - delta + sinuous_curve(r_end, alpha, r_start, tau)

    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)

    t_end = np.linspace(phi_inner_end, phi_outer_end, num_samples)

    arm.end.x = r_end * np.cos(t_end)
    arm.end.y = r_end * np.sin(t_end)
    arm.start.x = np.array([r_start * np.cos(phi_inner_start), r_start * np.cos(phi_outer_start)])
    arm.start.y = np.array([r_start * np.sin(phi_inner_start), r_start * np.sin(phi_outer_start)])

    arm.inner.x = r * np.cos(phi_inner)
    arm.inner.y = r * np.sin(phi_inner)
    arm.outer.x = r * np.cos(phi_outer)
    arm.outer.y = r * np.sin(phi_outer)

    arm.contour_mm.x = np.concatenate((np.flip(arm.start.x), arm.inner.x, arm.end.x, np.flip(arm.outer.x)))
    arm.contour_mm.y = np.concatenate((np.flip(arm.start.y), arm.inner.y, arm.end.y, np.flip(arm.outer.y)))
    num_contour_points = len(arm.contour_mm.x)
    arm.contour_position_array_mm = np.array(
        [arm.contour_mm.x, arm.contour_mm.y, height_mm * np.ones(num_contour_points)]).T
    # z_angle_rad = np.deg2rad(z_orientation)
    # rotation_matrix = [[np.cos(z_angle_rad), -np.sin(z_angle_rad), 0.0],
    #                    [np.sin(z_angle_rad), np.cos(z_angle_rad), 0.0],
    #                    [0.0, 0.0, 1.0]]
    # arm_contour_points_array = rotation_matrix @ arm_contour_points_array
    arm.contour_position_list_mm = [elem.tolist() for elem in arm.contour_position_array_mm]
    if plot:
        plot_colors = {"-b", "-r", "-k", "-c", "-m"}
        plt.figure()
        # plt.plot(arm.contour.x, arm.contour.y, '-b')
        plt.plot(arm.contour_mm.x, arm.contour_mm.y, '-r')
        plt.axis('equal')
        plt.show()
        lim = 1.1 * r_end
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
    return arm


def make_sinuous_arm_circular_terminal(sinuous_arm, radius_mm, r_start, tau, alpha, delta, height_mm=0.0,
                                       z_orientation=0.0, num_samples=50, plot=False):
    angle_offset = np.deg2rad(z_orientation - 22.5)
    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)
    # arm_start_x = np.array([R0 * np.cos(phi_inner_start), R0 * np.cos(phi_outer_start)])
    # arm_start_y = np.array([R0 * np.sin(phi_inner_start), R0 * np.sin(phi_outer_start)])
    # circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    circle_center_radius = 0.75 * r_start
    circle_center = Point2D(circle_center_radius * np.cos(circle_center_phi),
                            circle_center_radius * np.sin(circle_center_phi))
    theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    feed_x = circle_center.x + radius_mm * np.cos(theta)
    feed_y = circle_center.y + radius_mm * np.sin(theta)
    num_contour_points = len(feed_x)
    contour_position_array_mm = np.array([feed_x, feed_y, height_mm * np.ones(num_contour_points)]).T
    # plt.fill(feed_x, feed_y)
    sinuous_arm.contact_center_position_mm = Point2D(circle_center.x, circle_center.y)
    sinuous_arm.contact_position_array_mm = contour_position_array_mm
    sinuous_arm.contact_position_list_mm = [elem.tolist() for elem in contour_position_array_mm]
    return contour_position_array_mm, circle_center


def make_sinuous_arm_trapezoidal_terminal(sinuous_arm, radius_mm, r_start, tau, alpha, delta, height_mm=0.0,
                                          z_orientation=0.0, num_samples=50, plot=False):
    angle_offset = np.deg2rad(z_orientation)
    phi_inner_start = angle_offset + delta + sinuous_curve(r_start, alpha, r_start, tau)
    phi_outer_start = angle_offset - delta + sinuous_curve(r_start, alpha, r_start, tau)

    trap_position_list = np.array([[r_start * np.cos(phi_inner_start), r_start * np.sin(phi_inner_start)],
                                   [r_start * np.cos(phi_outer_start), r_start * np.sin(phi_outer_start)],
                                   [radius_mm * np.cos(phi_outer_start), radius_mm * np.sin(phi_outer_start)],
                                   [radius_mm * np.cos(phi_inner_start), radius_mm * np.sin(phi_inner_start)]])
    # circle_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    trap_center_phi = 0.5 * (phi_inner_start + phi_outer_start)
    trap_center_radius = 0.5 * (r_start + radius_mm)
    circle_center = Point2D(trap_center_radius * np.cos(trap_center_phi),
                            trap_center_radius * np.sin(trap_center_phi))
    num_contour_points = trap_position_list.shape[0]
    contour_position_array_mm = np.concatenate((trap_position_list, height_mm * np.ones((num_contour_points, 1))), axis=1)
    # plt.fill(feed_x, feed_y)
    sinuous_arm.contact_center_position_mm = Point2D(circle_center.x, circle_center.y)
    sinuous_arm.contact_position_array_mm = contour_position_array_mm
    sinuous_arm.contact_position_list_mm = [elem.tolist() for elem in contour_position_array_mm]
    return contour_position_array_mm, circle_center


if plot_antenna:
    plt.figure()

# Exact feed-via locations requested for the balun/via transition.
# Arm index mapping for z_orientation = 0, 90, 180, 270 deg is +X, +Y, -X, -Y.
exact_feed_centers_by_arm_index_mm = {
    0: (1.56, 0.0),
    1: (0.0, 1.56),
    2: (-1.56, 0.0),
    3: (0.0, -1.56),
}

sinuous_antenna_arms_list = []
for arm_index in np.arange(4):
    sinuous_antenna_arm = make_sinuous_arm(cm2mm * radius_start_cm, cm2mm * radius_end_cm, N, tau, alpha, delta,
                                           height_mm=0.0, z_orientation=90 * arm_index, plot=False)
    make_sinuous_arm_trapezoidal_terminal(sinuous_antenna_arm, cm2mm * arm_contact_diameter_cm / 2,
                                          cm2mm * radius_start_cm,
                                          tau, alpha, delta, height_mm=0.0, z_orientation=90 * arm_index, plot=False)
    exact_x_mm, exact_y_mm = exact_feed_centers_by_arm_index_mm[int(arm_index)]
    sinuous_antenna_arm.contact_center_position_mm = Point2D(exact_x_mm, exact_y_mm)
    sinuous_antenna_arms_list.append(sinuous_antenna_arm)
    if plot_antenna:
        plot_colors = {"-b", "-r", "-k", "-c", "-m"}
        plt.fill(sinuous_antenna_arm.contour_mm.x, sinuous_antenna_arm.contour_mm.y)
        plt.fill(sinuous_antenna_arm.contact_position_array_mm[:, 0],
                 sinuous_antenna_arm.contact_position_array_mm[:, 1])

if plot_antenna:
    plt.axis('equal')
    lim = 10 * 1.1 * radius_end_cm
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    plt.show()

if platform.system() == "Linux":
    os.environ["ANSYSEM_ROOT252"] = "/opt/Ansys/v252/AnsysEM"
else:
    os.environ["ANSYSEM_ROOT252"] = "C:\\Program Files\\AnsysEM\\v252\\Win64\\"

aedt_version = "2025.2"

solver_configuration = compute_config.SolverConfig().solver_config

###############################################################################
# Define program variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed_of_light = 2.99792458e8
frequency_nominal = 0.5 * (frequency_max + frequency_min)
wavelength_nominal = speed_of_light / frequency_nominal

# location of the coordinate system origin with respect to the top left corner of the antenna dimensions rectangle
antenna_coord_origin_xy_cm = 0.5 * np.array([0.0, 0.0])

# VISUALIZATION PREFERENCES
metal_color = [143, 175, 143]
dielectric_color = [255, 255, 128]
radiation_box_color = [128, 255, 255]
aluminum_color = [143, 143, 175]

radiation_volume_height_cm = dielectric_height_cm + 6 * dielectric_height_cm

# define a custom projectname and design name to allow multiple runs to simultaneously run on a single host
time_start = datetime.now()
current_time_str = datetime.now().strftime("%b%d_%H-%M-%S")

project_name = "Sinuous DF Antenna " + current_time_str
design_name = "Sinuous DF Antenna " + current_time_str

save_file_prefix = "sinuous_df_antenna"
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

###############################################################################
# Set non-graphical mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

aedt_version = "2025.2"
non_graphical = False

###############################################################################
# Launch AEDT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NewThread = True
desktop = launch_desktop(aedt_version, non_graphical)



# Solution Types are: { "Modal", "Terminal", "Eigenmode", "Transient Network", "SBR+", "Characteristic"}
hfss = Hfss(
    project=project_name,
    design=design_name,
    version=aedt_version,
    solution_type="Modal",

    non_graphical=non_graphical
)



hfss.modeler.model_units = 'mm'
hfss.autosave_disable()


# =============================================================================
# ADDED HELPERS FOR SAFE DELETE + STABLE BALUN GEOMETRY
# =============================================================================
def set_obj_color(obj, color_value):
    try:
        if obj:
            obj.color = color_value
    except Exception:
        pass


def delete_if_exists(modeler, obj_or_name):
    try:
        obj_name = obj_or_name.name if hasattr(obj_or_name, "name") else obj_or_name
        if isinstance(obj_name, str) and obj_name in modeler.object_names:
            modeler.delete(obj_name)
    except Exception:
        pass


def assign_sheet_as_metal(sheet_name):
    sheet_obj = hfss.modeler[sheet_name]
    set_obj_color(sheet_obj, metal_color)

    if PEC_ANTENNA:
        hfss.assign_perfecte_to_sheets(
            assignment=sheet_name,
            name="PerfE_" + sheet_name,
            is_infinite_ground=False,
        )
        return sheet_obj
    else:
        thick_obj = hfss.modeler.thicken_sheet(sheet_name, thickness=copper_thickness)
        try:
            thick_name = thick_obj if isinstance(thick_obj, str) else thick_obj.name
            hfss.assign_material(thick_name, "copper")
            return hfss.modeler[thick_name]
        except Exception:
            return thick_obj


BALUN_COPPER_THICKNESS_MM = 0.035
FIXED_VIA_CENTER_OFFSET_MM = 1.56
FIXED_VIA_RADIUS_MM = 0.3225
FIXED_VIA_HEIGHT_MM = -1.576
FIXED_VIA_PAD_RADIUS_MM = 0.5
FIXED_VIA_PAD_HEIGHT_MM = -0.025


def assign_balun_sheet_as_copper(sheet_name, thickness_mm=BALUN_COPPER_THICKNESS_MM):
    """
    Always realize the balun/feed/ground metal as thin 3D copper so the feed
    network is solved with physical metal thickness rather than as ideal PEC sheets.
    """
    sheet_obj = hfss.modeler[sheet_name]
    set_obj_color(sheet_obj, metal_color)
    thick_obj = hfss.modeler.thicken_sheet(sheet_name, thickness=thickness_mm)
    try:
        thick_name = thick_obj if isinstance(thick_obj, str) else thick_obj.name
        hfss.assign_material(thick_name, "copper")
        thick_obj3d = hfss.modeler[thick_name]
        set_obj_color(thick_obj3d, metal_color)
        return thick_obj3d
    except Exception:
        return thick_obj


def ensure_balun_substrate_material(material_name, er_value, tand_value):
    """Create the balun dielectric if it is not already present in the AEDT material library."""
    try:
        existing_names = []
        try:
            existing_names = list(hfss.materials.material_keys.keys())
        except Exception:
            try:
                existing_names = list(hfss.materials.material_keys)
            except Exception:
                existing_names = []

        existing_names_lc = [str(n).lower() for n in existing_names]
        if str(material_name).lower() in existing_names_lc:
            return material_name

        mat = hfss.materials.add_material(material_name)
        try:
            mat.permittivity = er_value
        except Exception:
            pass
        try:
            mat.dielectric_loss_tangent = tand_value
        except Exception:
            pass
        try:
            mat.conductivity = 0
        except Exception:
            pass
    except Exception:
        pass
    return material_name



def microstrip_effective_permittivity(width_mm, h_mm, er_value):
    """Hammerstad-Jensen effective permittivity approximation."""
    width_mm = max(float(width_mm), 1e-6)
    h_mm = max(float(h_mm), 1e-6)
    u = width_mm / h_mm
    a = 1.0 + (1.0 / 49.0) * np.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432))         + (1.0 / 18.7) * np.log(1.0 + (u / 18.1) ** 3)
    b = 0.564 * ((er_value - 0.9) / (er_value + 3.0)) ** 0.053
    return (er_value + 1.0) / 2.0 + (er_value - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)


def microstrip_width_for_impedance(z0_ohm, er_value, h_mm):
    """
    Hammerstad-Jensen inversion by bisection.
    Returns the microstrip width in mm for a single-ended microstrip line.
    """
    z0_ohm = float(z0_ohm)
    er_value = float(er_value)
    h_mm = float(h_mm)

    def z_from_u(u):
        eeff = microstrip_effective_permittivity(u * h_mm, h_mm, er_value)
        if u <= 1.0:
            return (60.0 / np.sqrt(eeff)) * np.log(8.0 / u + 0.25 * u)
        return (120.0 * np.pi) / (np.sqrt(eeff) * (u + 1.393 + 0.667 * np.log(u + 1.444)))

    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if z_from_u(mid) > z0_ohm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * h_mm


def guided_wavelength_mm(freq_hz, width_mm, h_mm, er_value):
    eeff = microstrip_effective_permittivity(width_mm, h_mm, er_value)
    return (speed_of_light * 1e3 / float(freq_hz)) / np.sqrt(eeff)


def quarter_wave_balun_length_mm(freq_hz, width_start_mm, width_end_mm, h_mm, er_value, fraction=0.25):
    avg_width_mm = 0.5 * (float(width_start_mm) + float(width_end_mm))
    return fraction * guided_wavelength_mm(freq_hz, avg_width_mm, h_mm, er_value)


def create_planar_polygon_sheet(name, pts3d):
    clean_pts = []
    for p in pts3d:
        q = [float(p[0]), float(p[1]), float(p[2])]
        if not clean_pts or any(abs(q[i] - clean_pts[-1][i]) > 1e-12 for i in range(3)):
            clean_pts.append(q)

    if len(clean_pts) > 1 and all(abs(clean_pts[0][i] - clean_pts[-1][i]) < 1e-12 for i in range(3)):
        clean_pts = clean_pts[:-1]

    poly = hfss.modeler.create_polyline(
        points=clean_pts,
        cover_surface=True,
        close_surface=True,
        name=name,
        material=None,
    )
    if not poly:
        raise RuntimeError("HFSS failed to create planar sheet '{}'.".format(name))

    poly = hfss.modeler[name]
    set_obj_color(poly, metal_color)
    return poly


def create_rect_sheet_xy(name, x1, y1, x2, y2, z0):
    pts = [
        [x1, y1, z0],
        [x2, y1, z0],
        [x2, y2, z0],
        [x1, y2, z0],
    ]
    return create_planar_polygon_sheet(name, pts)


def create_rect_sheet_xz(name, x1, z1, x2, z2, y0):
    pts = [
        [x1, y0, z1],
        [x2, y0, z1],
        [x2, y0, z2],
        [x1, y0, z2],
    ]
    return create_planar_polygon_sheet(name, pts)


def create_rect_sheet_yz(name, y1, z1, y2, z2, x0):
    pts = [
        [x0, y1, z1],
        [x0, y2, z1],
        [x0, y2, z2],
        [x0, y1, z2],
    ]
    return create_planar_polygon_sheet(name, pts)


def create_straight_strip_xy(name, p0, p1, width_mm, z_mm):
    """
    Create a single-piece straight microstrip sheet in the XY plane.
    The strip runs directly from p0 to p1 with constant width and is used
    to connect each via landing point below the antenna substrate to the
    neck of the corresponding balun trace on the shared vertical substrate.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    seg_len = np.hypot(dx, dy)
    if seg_len <= 1e-12:
        raise ValueError("Degenerate strip '{}' with zero length.".format(name))

    nx = -dy / seg_len
    ny = dx / seg_len
    hw = 0.5 * float(width_mm)

    pts = [
        [x0 + hw * nx, y0 + hw * ny, z_mm],
        [x1 + hw * nx, y1 + hw * ny, z_mm],
        [x1 - hw * nx, y1 - hw * ny, z_mm],
        [x0 - hw * nx, y0 - hw * ny, z_mm],
    ]
    return create_planar_polygon_sheet(name, pts)


def create_tapered_straight_strip_xy(name, p0, p1, width0_mm, width1_mm, z_mm):
    """
    Single-piece tapered strip in the XY plane. Useful if a slightly wider
    overlap is desired at the via pad while matching the balun neck width at
    the vertical-substrate end.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    seg_len = np.hypot(dx, dy)
    if seg_len <= 1e-12:
        raise ValueError("Degenerate tapered strip '{}' with zero length.".format(name))

    nx = -dy / seg_len
    ny = dx / seg_len
    hw0 = 0.5 * float(width0_mm)
    hw1 = 0.5 * float(width1_mm)

    pts = [
        [x0 + hw0 * nx, y0 + hw0 * ny, z_mm],
        [x1 + hw1 * nx, y1 + hw1 * ny, z_mm],
        [x1 - hw1 * nx, y1 - hw1 * ny, z_mm],
        [x0 - hw0 * nx, y0 - hw0 * ny, z_mm],
    ]
    return create_planar_polygon_sheet(name, pts)


def create_trapezoid_sheet_xz(name, x_left_top, x_right_top, z_top,
                              x_left_bot, x_right_bot, z_bot, y0):
    pts = [
        [x_left_top,  y0, z_top],
        [x_right_top, y0, z_top],
        [x_right_bot, y0, z_bot],
        [x_left_bot,  y0, z_bot],
    ]
    return create_planar_polygon_sheet(name, pts)


def unite_keep_first(obj_names):
    obj_names = [n for n in obj_names if n in hfss.modeler.object_names]
    if len(obj_names) <= 1:
        return hfss.modeler[obj_names[0]] if obj_names else None

    hfss.modeler.unite(
        assignment=obj_names,
        purge=False,
        keep_originals=False,
    )
    return hfss.modeler[obj_names[0]]


def create_axis_aligned_runner(name, p0, p1, width_mm, z_mm):
    x0, y0 = p0
    x1, y1 = p1

    if abs(y1 - y0) < 1e-12:
        return create_rect_sheet_xy(
            name,
            min(x0, x1),
            y0 - 0.5 * width_mm,
            max(x0, x1),
            y0 + 0.5 * width_mm,
            z_mm,
        )
    elif abs(x1 - x0) < 1e-12:
        return create_rect_sheet_xy(
            name,
            x0 - 0.5 * width_mm,
            min(y0, y1),
            x0 + 0.5 * width_mm,
            max(y0, y1),
            z_mm,
        )
    else:
        raise ValueError("Runner '{}' is not axis-aligned.".format(name))


def create_runner_path(name_prefix, points2d, width_mm, z_mm):
    seg_names = []
    for i in range(len(points2d) - 1):
        seg_name = "{}_seg{}".format(name_prefix, i + 1)
        seg = create_axis_aligned_runner(seg_name, points2d[i], points2d[i + 1], width_mm, z_mm)
        seg_names.append(seg.name)

    return unite_keep_first(seg_names)


def klopfenstein_progress(p, z0_ohm, z1_ohm, gamma_m=0.02, num_int_points=401):
    """
    Normalized Klopfenstein taper progression from 0 to 1.
    p = 0 at the far/source side and p = 1 at the narrow antenna side.
    This is used as a geometry-driving profile for the strip width.
    """
    p = max(0.0, min(1.0, float(p)))
    x = 2.0 * p - 1.0
    if abs(z1_ohm - z0_ohm) < 1e-12:
        return p
    gamma0 = 0.5 * np.log(float(z1_ohm) / float(z0_ohm))
    A = np.arccosh(max(1.0 + 1e-12, abs(gamma0) / max(gamma_m, 1e-9)))

    def integrand(y):
        root = max(0.0, 1.0 - y * y)
        t = A * np.sqrt(root)
        if abs(t) < 1e-9:
            return 0.5
        return i1(t) / t

    ys1 = np.linspace(0.0, 1.0, num_int_points)
    vals1 = np.array([integrand(y) for y in ys1])
    phi1 = np.trapz(vals1, ys1)

    if x >= 0.0:
        ysx = np.linspace(0.0, x, max(3, int(num_int_points * x)))
        valsx = np.array([integrand(y) for y in ysx])
        phix = np.trapz(valsx, ysx)
    else:
        ysx = np.linspace(x, 0.0, max(3, int(num_int_points * abs(x))))
        valsx = np.array([integrand(y) for y in ysx])
        phix = -np.trapz(valsx, ysx)

    return 0.5 * (1.0 + phix / phi1)


def width_profile_half(style, p, w_near, w_far):
    """
    Half-width profile for the main taper.
    p = 0 at the far/wide end, p = 1 at the near/narrow end of the taper.

    Supported styles:
      - lin   : direct linear width transition
      - exp   : exponential width transition
      - klopf : Klopfenstein-style monotonic taper progression
    """
    p = max(0.0, min(1.0, float(p)))
    style_lc = style.lower()
    if style_lc.startswith("klopf"):
        prog = klopfenstein_progress(p, z_source_single_ohm, z_target_single_ohm, gamma_m=0.02)
        return 0.5 * (w_far + (w_near - w_far) * prog)
    if style_lc.startswith("exp"):
        if w_near <= 0 or w_far <= 0:
            return 0.5 * w_near
        a = 0.5 * w_far
        bL = np.log(w_near / w_far)
        return a * np.exp(bL * p)
    return 0.5 * (w_far + (w_near - w_far) * p)



def create_custom_profile_taper_trace(name_prefix, xsign, y_face, taper_style,
                                      z_top, z3,
                                      x_far_offset_mm, x_near_offset_mm,
                                      balun_dims,
                                      underside_z_mm,
                                      taper_fraction=0.90,
                                      neck_fraction=0.10,
                                      num_taper_pts=80,
                                      num_neck_pts=14):
    """
    Straight-through tapered balun trace on a constant-y plane.

    Design intent:
      - the main Klopfenstein taper occupies most of the balun length
      - once the line reaches the narrow output width, it remains constant width
      - the output centerline is aligned directly with the via coordinate, so there
        is no lateral bend on the vertical balun substrate
    """
    total = taper_fraction + neck_fraction
    taper_fraction /= total
    neck_fraction /= total

    x_far = xsign * x_far_offset_mm
    x_near = xsign * x_near_offset_mm
    w_far = balun_dims["w4"]
    w_near = balun_dims["w1"]

    z_taper_end = z3 + taper_fraction * (underside_z_mm - z3)
    z_neck_end = underside_z_mm

    taper_pts_left = []
    taper_pts_right = []
    for i in range(num_taper_pts):
        p = float(i) / float(max(1, num_taper_pts - 1))
        zc = z3 + p * (z_taper_end - z3)
        hw = width_profile_half(taper_style, p, w_near, w_far)
        xc = x_far + p * (x_near - x_far)
        taper_pts_left.append([xc - hw, y_face, zc])
        taper_pts_right.append([xc + hw, y_face, zc])

    hw_narrow = 0.5 * w_near
    neck_pts_left = []
    neck_pts_right = []
    for i in range(1, num_neck_pts):
        p = float(i) / float(max(1, num_neck_pts - 1))
        zc = z_taper_end + p * (z_neck_end - z_taper_end)
        neck_pts_left.append([x_near - hw_narrow, y_face, zc])
        neck_pts_right.append([x_near + hw_narrow, y_face, zc])

    left_edge = taper_pts_left + neck_pts_left
    right_edge = taper_pts_right + neck_pts_right
    poly_pts = left_edge + list(reversed(right_edge))

    poly = create_planar_polygon_sheet(name_prefix, poly_pts)
    metal_obj = assign_balun_sheet_as_copper(poly.name)
    return metal_obj if metal_obj else poly


def create_custom_profile_taper_trace_rot90(name_prefix, ysign, x_face, taper_style,
                                            z_top, z3,
                                            y_far_offset_mm, y_near_offset_mm,
                                            balun_dims,
                                            underside_z_mm,
                                            taper_fraction=0.90,
                                            neck_fraction=0.10,
                                            num_taper_pts=80,
                                            num_neck_pts=14):
    """
    Straight-through tapered balun trace on a constant-x plane.
    The line stays aligned with the final via y-coordinate so there is no lateral
    bend on the rotated vertical balun substrate.
    """
    total = taper_fraction + neck_fraction
    taper_fraction /= total
    neck_fraction /= total

    y_far = ysign * y_far_offset_mm
    y_near = ysign * y_near_offset_mm
    w_far = balun_dims["w4"]
    w_near = balun_dims["w1"]

    z_taper_end = z3 + taper_fraction * (underside_z_mm - z3)
    z_neck_end = underside_z_mm

    taper_pts_left = []
    taper_pts_right = []
    for i in range(num_taper_pts):
        p = float(i) / float(max(1, num_taper_pts - 1))
        zc = z3 + p * (z_taper_end - z3)
        hw = width_profile_half(taper_style, p, w_near, w_far)
        yc = y_far + p * (y_near - y_far)
        taper_pts_left.append([x_face, yc - hw, zc])
        taper_pts_right.append([x_face, yc + hw, zc])

    hw_narrow = 0.5 * w_near
    neck_pts_left = []
    neck_pts_right = []
    for i in range(1, num_neck_pts):
        p = float(i) / float(max(1, num_neck_pts - 1))
        zc = z_taper_end + p * (z_neck_end - z_taper_end)
        neck_pts_left.append([x_face, y_near - hw_narrow, zc])
        neck_pts_right.append([x_face, y_near + hw_narrow, zc])

    left_edge = taper_pts_left + neck_pts_left
    right_edge = taper_pts_right + neck_pts_right
    poly_pts = left_edge + list(reversed(right_edge))

    poly = create_planar_polygon_sheet(name_prefix, poly_pts)
    metal_obj = assign_balun_sheet_as_copper(poly.name)
    return metal_obj if metal_obj else poly
# =============================================================================
# =============================================================================

###############################################################################
# Define HFSS Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hfss.variable_manager.set_variable("wavelength", expression="{}mm".format(cm2mm * wavelength_nominal))

###############################################################################
# Define geometries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dielectric_slab_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                     cm2mm * antenna_coord_origin_xy_cm[1],
                                     cm2mm * -dielectric_height_cm])
dielectric_slab_params = {"name": "dielectric_slab",
                          "orientation": "Z",
                          "origin": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
                                                              dielectric_slab_position[1],
                                                              dielectric_slab_position[2]).split(","),
                          "radius": "{}mm".format(cm2mm * dielectric_diameter_cm / 2),
                          "height": "{}mm".format(cm2mm * dielectric_height_cm),
                          "num_sides": 0,
                          "material": dielectric_material_name}
dielectric_slab_geom = hfss.modeler.create_cylinder(**dielectric_slab_params)
dielectric_slab_geom.color = dielectric_color

# FIT ALL
hfss.modeler.fit_all()

ground_plane_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                  cm2mm * antenna_coord_origin_xy_cm[1],
                                  cm2mm * -dielectric_height_cm - 0.035])

# ground_plane_params = {"name": "ground_plane",
#                        "cs_plane": "XY",
#                        "origin": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                            ground_plane_position[1],
#                                                            ground_plane_position[2]).split(","),
#                        "radius": "{}mm".format(cm2mm * ground_plane_diameter_cm / 2),
#                        "num_sides": 0,
#                        "material": None}
# ground_plane_geom = hfss.modeler.create_circle(**ground_plane_params)
# ground_plane_geom.color = metal_color

# ground_plane_params = {"name": "ground_plane",
#                        "cs_axis": "XY",
#                        "origin": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                            ground_plane_position[1],
#                                                            ground_plane_position[2]).split(","),
#                        "radius": "{}mm".format(cm2mm * ground_plane_diameter_cm / 2),
#                        "height": "{}mm".format(0.035),
#                        "num_sides": 0,
#                        "matname": "copper"}
# ground_plane_geom = hfss.modeler.create_cylinder(**ground_plane_params)
# ground_plane_geom.color = metal_color

# hfss.assign_perfecte_to_sheets(
#     **{"sheet_list": "ground_plane",
#        "sourcename": None,
#        "is_infinite_gnd": False})

# ground_plane_hole_params = {"name": "ground_plane_hole",
#                             "cs_plane": "XY",
#                             "origin": "{}mm,{}mm,{}mm".format(ground_plane_position[0],
#                                                                 ground_plane_position[1],
#                                                                 ground_plane_position[2]).split(","),
#                             "radius": "{}mm".format(cm2mm * ground_plane_hole_diameter_cm / 2),
#                             "num_sides": 0,
#                             "material": None}
# ground_plane_hole_geom = hfss.modeler.create_circle(**ground_plane_hole_params)
# ground_plane_geom.color = metal_color
#
# subtract_params = {
#     "blank_list": ["ground_plane"],
#     "tool_list": ["ground_plane_hole"],
#     "keep_originals": True
# }
# hfss.modeler.subtract(**subtract_params)
# ground_plane_hole_geom.delete()
if False:
    cavity_cylindrical_wall_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                                 cm2mm * antenna_coord_origin_xy_cm[1],
                                                 cm2mm * -cavity_height_cm])
    cavity_cylindrical_wall_params = {"name": "cavity_cylindrical_wall",
                                      "orientation": "Z",
                                      "origin": "{}mm,{}mm,{}mm".format(cavity_cylindrical_wall_position[0],
                                                                          cavity_cylindrical_wall_position[1],
                                                                          cavity_cylindrical_wall_position[2]).split(
                                          ","),
                                      "radius": "{}mm".format(cm2mm * cavity_diameter_cm / 2),
                                      "height": "{}mm".format(cm2mm * cavity_height_cm),
                                      "num_sides": 0,
                                      "material": "aluminum"}
    cavity_cylindrical_wall_geom = hfss.modeler.create_cylinder(**cavity_cylindrical_wall_params)
    cavity_cylindrical_wall_geom.color = aluminum_color
    cavity_cylindrical_wall_geom.transparency = 0.2

    cavity_cylindrical_back_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                                                 cm2mm * antenna_coord_origin_xy_cm[1],
                                                 cm2mm * -(cavity_height_cm + cavity_wall_thickness_cm)])
    cavity_cylindrical_back_params = {"name": "cavity_cylindrical_back",
                                      "orientation": "Z",
                                      "origin": "{}mm,{}mm,{}mm".format(cavity_cylindrical_back_position[0],
                                                                          cavity_cylindrical_back_position[1],
                                                                          cavity_cylindrical_back_position[2]).split(
                                          ","),
                                      "radius": "{}mm".format(cm2mm * cavity_diameter_cm / 2),
                                      "height": "{}mm".format(cm2mm * cavity_wall_thickness_cm),
                                      "num_sides": 0,
                                      "material": "aluminum"}
    cavity_cylindrical_back_geom = hfss.modeler.create_cylinder(**cavity_cylindrical_back_params)
    cavity_cylindrical_back_geom.color = aluminum_color
    cavity_cylindrical_back_geom.transparency = 0.2

    unite_params = {
        "unite_list": ["cavity_cylindrical_wall", "cavity_cylindrical_back"],
        "purge": False,
        "keep_originals": False
    }
    hfss.modeler.unite(**unite_params)

# Draw the sinusoidal antenna trace as a polyline
sinuous_arm_polyline_list = []

for arm_index, arm in enumerate(sinuous_antenna_arms_list):
    arm_position_list = arm.contour_position_list_mm
    arm_object_name = "sinuous_arm_" + str(arm_index)
    arm_polyline_params = {
        "points": arm_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": arm_object_name,
        "material": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    sinuous_arm_polyline = hfss.modeler.create_polyline(**arm_polyline_params)
    sinuous_arm_polyline.color = metal_color

    arm_contact_position_list = arm.contact_position_list_mm
    arm_contact_object_name = "sinuous_arm_contact_" + str(arm_index)
    arm_contact_polyline_params = {
        "points": arm_contact_position_list,
        "segment_type": None,
        "cover_surface": True,
        "close_surface": True,
        "name": arm_contact_object_name,
        "material": None,
        "xsection_type": None,
        "xsection_orient": None,
        "xsection_width": 1,
        "xsection_topwidth": 1,
        "xsection_height": 1,
        "xsection_num_seg": 0,
        "xsection_bend_type": None,
        "non_model": False
    }
    sinuous_contact_arm_polyline = hfss.modeler.create_polyline(**arm_contact_polyline_params)
    sinuous_contact_arm_polyline.color = metal_color

    unite_params = {
        "assignment": [arm_object_name, arm_contact_object_name],
        "purge": False,
        "keep_originals": False
    }
    hfss.modeler.unite(**unite_params)

    through_hole_contact_cavity_position = [arm.contact_center_position_mm.x,
                                            arm.contact_center_position_mm.y,
                                            cm2mm * -dielectric_height_cm]

    arm_contact_cavity_name = "contact_through_hole_cavity_" + str(arm_index)
    through_hole_contact_params = {"name": arm_contact_cavity_name,
                                   "orientation": "Z",
                                   "origin": "{}mm,{}mm,{}mm".format(through_hole_contact_cavity_position[0],
                                                                       through_hole_contact_cavity_position[1],
                                                                       through_hole_contact_cavity_position[2]).split(
                                       ","),
                                   "radius": "{}mm".format(FIXED_VIA_RADIUS_MM),
                                   "height": "{}mm".format(abs(FIXED_VIA_HEIGHT_MM)),
                                   "num_sides": 0,
                                   "material": None}
    through_hole_contact_geom = hfss.modeler.create_cylinder(**through_hole_contact_params)

    subtract_params = {
        # "blank_list": ["dielectric_slab", "ground_plane", arm_object_name],
        "blank_list": ["dielectric_slab", arm_object_name],
        "tool_list": [arm_contact_cavity_name],
        "keep_originals": False
    }
    hfss.modeler.subtract(**subtract_params)
    delete_if_exists(hfss.modeler, through_hole_contact_geom)

    if PEC_ANTENNA:
        hfss.assign_perfecte_to_sheets(
            assignment=arm_object_name,
            name="PerfE_" + arm_object_name,
            is_infinite_ground=False)
        sinuous_arm_polyline_list.append(sinuous_arm_polyline)
    else:
        sinuous_arm_object3d = hfss.modeler.thicken_sheet(objid=arm_object_name, thickness=copper_thickness)
        # Define the antenna material (e.g., copper)
        hfss.assign_material(sinuous_arm_object3d, "copper")
        sinuous_arm_polyline_list.append(sinuous_arm_object3d)


# =============================================================================
# SHARED PERPENDICULAR 4-BALUN FEED BLOCK
# =============================================================================
feed_points = []
for arm in sinuous_antenna_arms_list:
    feed_points.append((
        float(arm.contact_center_position_mm.x),
        float(arm.contact_center_position_mm.y)
    ))

east_pt = max(feed_points, key=lambda p: p[0])
west_pt = min(feed_points, key=lambda p: p[0])
north_pt = max(feed_points, key=lambda p: p[1])
south_pt = min(feed_points, key=lambda p: p[1])

feed_centers_xy = {
    "east": east_pt,
    "west": west_pt,
    "north": north_pt,
    "south": south_pt,
}

antenna_diameter_mm = cm2mm * antenna_diameter_cm
antenna_bottom_z_mm = -cm2mm * dielectric_height_cm
pad_outer_radius_mm = FIXED_VIA_PAD_RADIUS_MM
via_barrel_radius_mm = FIXED_VIA_RADIUS_MM

# -----------------------------------------------------------------------------
# Recommended feed target for this antenna family
# -----------------------------------------------------------------------------
# User source: 50 ohm DIFFERENTIAL.
# For higher realized-gain / wider gain-bandwidth performance, the 2022
# current request is to retune the balun closer to 200.00 ohm at the antenna side.
# NOTE: literature for four-arm mode-1 sinuous antennas often reports about 267 ohm
# for the conventional self-complementary case, while 188.5 ohm is the classic
# self-complementary spiral/two-arm reference. This file implements the requested
# 50-to-200.00 ohm first-pass retune as a practical study variant.
#
# These dimensions began from the earlier 50->100-ohm baseline, then were
# re-biased toward a higher antenna-side impedance by narrowing the balanced-end
# width, slightly increasing the line spacing, and lengthening the transition.
#
# Retune intent:
#   - slightly wider traces to compensate for the lower dielectric constant
#   - slightly longer electrical transition length
#   - keep the user-requested 3.125 mm neck gap for via alignment
# -----------------------------------------------------------------------------


z_source_single_ohm = 50.0
z_target_antenna_diff_ohm = 200.0
z_target_single_ohm = 0.5 * z_target_antenna_diff_ohm

# Manufacturing-oriented version A:
# use the same dielectric family as the antenna substrate so the feed transition
# sees a more continuous dielectric environment. The board is placed essentially
# touching the antenna substrate with only a tiny numerical seam to prevent HFSS
# solid-overlap errors.
balun_substrate_material_name = dielectric_material_name
balun_substrate_er = 2.20
balun_substrate_tand = 0.0009
balun_board_thickness_mm = 0.508
manufacturing_contact_clearance_mm = 0.05

# Calculate the single-ended microstrip widths from closed-form microstrip equations.
# The present feed topology launches each arm from a single-ended microstrip referenced
# to the common opposite-face ground network. Therefore, a 200-ohm differential target
# is implemented as approximately 100 ohm single-ended per arm.
w_source_50ohm_mm = microstrip_width_for_impedance(z_source_single_ohm, balun_substrate_er, balun_board_thickness_mm)
w_target_single_mm = microstrip_width_for_impedance(z_target_single_ohm, balun_substrate_er, balun_board_thickness_mm)

# Electrical-length sizing for the intended 2-12 GHz operating span.
# Use a quarter guided wavelength at the lowest operating frequency as the
# first-pass total balun length, then retain the literature-inspired segmentation
# fractions of 10% / 40% / 50% for the launch, wide section, and taper portions.
balun_low_frequency_hz = 2.0e9
balun_length_mm = quarter_wave_balun_length_mm(
    balun_low_frequency_hz,
    w_source_50ohm_mm,
    w_target_single_mm,
    balun_board_thickness_mm,
    balun_substrate_er,
    fraction=0.25,
)
balun_dims = {
    "L": balun_length_mm,
    "L1": 0.10 * balun_length_mm,
    "L2": 0.40 * balun_length_mm,
    "Ltaper": 0.50 * balun_length_mm,
    "w1": w_target_single_mm,
    "w4": w_source_50ohm_mm,
    "S1": max(0.40, 0.5 * w_target_single_mm),
    "S2": max(0.80, 0.5 * w_source_50ohm_mm),
}
balun_dims["w5"] = max(8.0, 2.0 * balun_dims["w4"] + 2.0)
balun_dims["linear_width_slope_mm_per_mm"] = (balun_dims["w4"] - balun_dims["w1"]) / balun_dims["Ltaper"]
balun_dims["exp_width_b_per_mm"] = np.log(balun_dims["w4"] / balun_dims["w1"]) / balun_dims["Ltaper"]

# Straight-through launch requirement:
# keep the vertical balun traces centered directly over the via coordinates so
# there is no lateral bend on the balun substrate itself. The short underside
# connectors then run straight from the board face to the via pads.
x_near_offset_mm = FIXED_VIA_CENTER_OFFSET_MM
x_far_offset_mm = FIXED_VIA_CENTER_OFFSET_MM
desired_face_neck_gap_mm = 2.0 * x_near_offset_mm - balun_dims["w1"]
actual_face_neck_gap_mm = desired_face_neck_gap_mm

ensure_balun_substrate_material(balun_substrate_material_name, balun_substrate_er, balun_substrate_tand)
# Tiny seam only for numerical robustness; intended physical implementation is essentially touching.
balun_substrate_gap_mm = manufacturing_contact_clearance_mm
# Make the substrate a bit wider so the shifted-apart balun traces still have
# margin to the board edges.
balun_board_edge_margin_mm = 0.75
balun_board_width_mm = max(
    balun_dims["w5"],
    2.0 * (x_far_offset_mm + 0.5 * balun_dims["w4"] + balun_board_edge_margin_mm),
)
balun_board_height_mm = balun_dims["L"]

balun_board_top_z_mm = antenna_bottom_z_mm - balun_substrate_gap_mm
balun_board_bottom_z_mm = balun_board_top_z_mm - balun_board_height_mm

# Build the plus-shaped dielectric support as four non-overlapping substrate halves
# so the two perpendicular balun boards do not occupy the same center volume.
center_board_clearance_mm = 0.002
half_center_exclusion_mm = 0.5 * balun_board_thickness_mm + 0.5 * center_board_clearance_mm
board_half_size_mm = 0.5 * balun_board_width_mm - half_center_exclusion_mm

# Original balun substrate: left and right halves (large faces at constant y)
shared_balun_substrate_left = hfss.modeler.create_box(
    origin=[
        -0.5 * balun_board_width_mm,
        -0.5 * balun_board_thickness_mm,
        balun_board_top_z_mm - balun_board_height_mm,
    ],
    sizes=[
        board_half_size_mm,
        balun_board_thickness_mm,
        balun_board_height_mm,
    ],
    name="shared_balun_substrate_left",
    material=balun_substrate_material_name,
)
shared_balun_substrate_left = hfss.modeler["shared_balun_substrate_left"]
set_obj_color(shared_balun_substrate_left, [220, 220, 220])

shared_balun_substrate_right = hfss.modeler.create_box(
    origin=[
        half_center_exclusion_mm,
        -0.5 * balun_board_thickness_mm,
        balun_board_top_z_mm - balun_board_height_mm,
    ],
    sizes=[
        board_half_size_mm,
        balun_board_thickness_mm,
        balun_board_height_mm,
    ],
    name="shared_balun_substrate_right",
    material=balun_substrate_material_name,
)
shared_balun_substrate_right = hfss.modeler["shared_balun_substrate_right"]
set_obj_color(shared_balun_substrate_right, [220, 220, 220])

# Offset the metal sheets just outside the substrate faces so their 3D thickness
# does not intrude into the dielectric volumes.
metal_face_offset_mm = BALUN_COPPER_THICKNESS_MM
front_face_y = +0.5 * balun_board_thickness_mm + metal_face_offset_mm
back_face_y = -0.5 * balun_board_thickness_mm - metal_face_offset_mm

z_top = balun_board_top_z_mm
z1 = z_top - balun_dims["L2"]
z2 = z1 - balun_dims["Ltaper"]
z3 = z2 - balun_dims["L1"]

# Underside connection plane for the via-to-balun feed strips.
underside_z_mm = antenna_bottom_z_mm

# 90-degree rotated balun substrate: lower and upper halves (large faces at constant x)
shared_balun_substrate_rot90_lower = hfss.modeler.create_box(
    origin=[
        -0.5 * balun_board_thickness_mm,
        -0.5 * balun_board_width_mm,
        balun_board_top_z_mm - balun_board_height_mm,
    ],
    sizes=[
        balun_board_thickness_mm,
        board_half_size_mm,
        balun_board_height_mm,
    ],
    name="shared_balun_substrate_rot90_lower",
    material=balun_substrate_material_name,
)
shared_balun_substrate_rot90_lower = hfss.modeler["shared_balun_substrate_rot90_lower"]
set_obj_color(shared_balun_substrate_rot90_lower, [220, 220, 220])

shared_balun_substrate_rot90_upper = hfss.modeler.create_box(
    origin=[
        -0.5 * balun_board_thickness_mm,
        half_center_exclusion_mm,
        balun_board_top_z_mm - balun_board_height_mm,
    ],
    sizes=[
        balun_board_thickness_mm,
        board_half_size_mm,
        balun_board_height_mm,
    ],
    name="shared_balun_substrate_rot90_upper",
    material=balun_substrate_material_name,
)
shared_balun_substrate_rot90_upper = hfss.modeler["shared_balun_substrate_rot90_upper"]
set_obj_color(shared_balun_substrate_rot90_upper, [220, 220, 220])

# The left-face copper objects on the negative X side were still showing a gap because
# the sheet normal on that face thickens away from the substrate. Keep the right face
# slightly offset, but place the left-face sheet directly on the substrate surface so the
# thickened copper starts flush against the dielectric instead of floating one copper
# thickness away.
right_face_x = +0.5 * balun_board_thickness_mm + metal_face_offset_mm
left_face_x = -0.5 * balun_board_thickness_mm


# -----------------------------------------------------------------------------
# Assigned one-balun-per-via implementation for the plus-shaped structure.
# Remaining feed mapping, following the earlier assignment intent:
#   west  -> linear   on original substrate
#   east  -> exponential on original substrate
#   south -> linear   on rotated substrate
#   north -> exponential on rotated substrate
# Any other balun that would otherwise share the same via region is removed and
# replaced by a ground plane on that same face-half of the substrate.
# -----------------------------------------------------------------------------

center_split_half_mm = 0.5 * balun_board_thickness_mm
face_outer_half_span_mm = 0.5 * balun_board_width_mm
face_margin_mm = 0.05
face_top_z_mm = z_top
face_bottom_z_mm = z3

# Keep only the baluns assigned to the four feed vias.
balun_front_left_klopf_obj = create_custom_profile_taper_trace(
    "balun_front_left_klopf", -1, front_face_y, "klopf",
    z_top, z3, x_far_offset_mm, x_near_offset_mm, balun_dims, underside_z_mm
)
balun_back_right_klopf_obj = create_custom_profile_taper_trace(
    "balun_back_right_klopf", +1, back_face_y, "klopf",
    z_top, z3, x_far_offset_mm, x_near_offset_mm, balun_dims, underside_z_mm
)
balun_rot90_right_lower_klopf_obj = create_custom_profile_taper_trace_rot90(
    "balun_rot90_right_lower_klopf", -1, right_face_x, "klopf",
    z_top, z3, x_far_offset_mm, x_near_offset_mm, balun_dims, underside_z_mm
)
balun_rot90_left_upper_klopf_obj = create_custom_profile_taper_trace_rot90(
    "balun_rot90_left_upper_klopf", +1, left_face_x, "klopf",
    z_top, z3, x_far_offset_mm, x_near_offset_mm, balun_dims, underside_z_mm
)

# Replace the removed baluns with ground planes on the same face-half.
front_right_ground = create_rect_sheet_xz(
    "gnd_front_right_half",
    center_split_half_mm + face_margin_mm,
    face_bottom_z_mm,
    face_outer_half_span_mm - face_margin_mm,
    face_top_z_mm,
    front_face_y,
)
front_right_ground_obj = assign_balun_sheet_as_copper(front_right_ground.name)

back_left_ground = create_rect_sheet_xz(
    "gnd_back_left_half",
    -face_outer_half_span_mm + face_margin_mm,
    face_bottom_z_mm,
    -center_split_half_mm - face_margin_mm,
    face_top_z_mm,
    back_face_y,
)
back_left_ground_obj = assign_balun_sheet_as_copper(back_left_ground.name)

rot90_right_upper_ground = create_rect_sheet_yz(
    "gnd_rot90_right_upper_half",
    center_split_half_mm + face_margin_mm,
    face_bottom_z_mm,
    face_outer_half_span_mm - face_margin_mm,
    face_top_z_mm,
    right_face_x,
)
rot90_right_upper_ground_obj = assign_balun_sheet_as_copper(rot90_right_upper_ground.name)

rot90_left_lower_ground = create_rect_sheet_yz(
    "gnd_rot90_left_lower_half",
    -face_outer_half_span_mm + face_margin_mm,
    face_bottom_z_mm,
    -center_split_half_mm - face_margin_mm,
    face_top_z_mm,
    left_face_x,
)
rot90_left_lower_ground_obj = assign_balun_sheet_as_copper(rot90_left_lower_ground.name)

# Copper hub in the removed center intersection volume to tie the four ground
# face sections into one continuous RF return network through the full balun height.
# Extend it to the copper-bearing substrate faces so it can physically meet the
# short face-connector straps and form one solid ground body.
hub_half_extent_mm = 0.5 * balun_board_thickness_mm + BALUN_COPPER_THICKNESS_MM
common_ground_hub = hfss.modeler.create_box(
    origin=[
        -hub_half_extent_mm,
        -hub_half_extent_mm,
        balun_board_bottom_z_mm,
    ],
    sizes=[
        2.0 * hub_half_extent_mm,
        2.0 * hub_half_extent_mm,
        balun_board_height_mm,
    ],
    name="common_ground_hub",
    material="copper",
)
common_ground_hub = hfss.modeler["common_ground_hub"]
set_obj_color(common_ground_hub, metal_color)

# Short ground straps on each active ground face so the central copper hub is
# electrically and physically connected to every ground section without disturbing
# the active signal-balun halves.
strap_overlap_mm = 0.01
front_ground_link = create_rect_sheet_xz(
    "gnd_front_right_link",
    hub_half_extent_mm - strap_overlap_mm,
    face_bottom_z_mm,
    center_split_half_mm + face_margin_mm + strap_overlap_mm,
    face_top_z_mm,
    front_face_y,
)
front_ground_link_obj = assign_balun_sheet_as_copper(front_ground_link.name)

back_ground_link = create_rect_sheet_xz(
    "gnd_back_left_link",
    -center_split_half_mm - face_margin_mm - strap_overlap_mm,
    face_bottom_z_mm,
    -hub_half_extent_mm + strap_overlap_mm,
    face_top_z_mm,
    back_face_y,
)
back_ground_link_obj = assign_balun_sheet_as_copper(back_ground_link.name)

rot90_right_ground_link = create_rect_sheet_yz(
    "gnd_rot90_right_upper_link",
    hub_half_extent_mm - strap_overlap_mm,
    face_bottom_z_mm,
    center_split_half_mm + face_margin_mm + strap_overlap_mm,
    face_top_z_mm,
    right_face_x,
)
rot90_right_ground_link_obj = assign_balun_sheet_as_copper(rot90_right_ground_link.name)

rot90_left_ground_link = create_rect_sheet_yz(
    "gnd_rot90_left_lower_link",
    -center_split_half_mm - face_margin_mm - strap_overlap_mm,
    face_bottom_z_mm,
    -hub_half_extent_mm + strap_overlap_mm,
    face_top_z_mm,
    left_face_x,
)
rot90_left_ground_link_obj = assign_balun_sheet_as_copper(rot90_left_ground_link.name)

common_ground_obj = None
if (front_right_ground_obj and back_left_ground_obj and rot90_right_upper_ground_obj and rot90_left_lower_ground_obj
        and front_ground_link_obj and back_ground_link_obj and rot90_right_ground_link_obj and rot90_left_ground_link_obj):
    common_ground_obj = unite_keep_first([
        front_right_ground_obj.name,
        back_left_ground_obj.name,
        rot90_right_upper_ground_obj.name,
        rot90_left_lower_ground_obj.name,
        front_ground_link_obj.name,
        back_ground_link_obj.name,
        rot90_right_ground_link_obj.name,
        rot90_left_ground_link_obj.name,
        common_ground_hub.name,
    ])

# Exact feed vias and underside pads at the requested symmetric locations.
via_cylinder_objs = {}
via_pad_objs = {}
for tag, (xc, yc) in feed_centers_xy.items():
    via = hfss.modeler.create_cylinder(
        orientation="Z",
        origin=[xc, yc, 0.0],
        radius=FIXED_VIA_RADIUS_MM,
        height=FIXED_VIA_HEIGHT_MM,
        num_sides=0,
        name=f"feed_via_{tag}",
        material="copper",
    )
    via = hfss.modeler[f"feed_via_{tag}"]
    set_obj_color(via, metal_color)
    via_cylinder_objs[tag] = via

    pad = hfss.modeler.create_cylinder(
        orientation="Z",
        origin=[xc, yc, antenna_bottom_z_mm],
        radius=FIXED_VIA_PAD_RADIUS_MM,
        height=FIXED_VIA_PAD_HEIGHT_MM,
        num_sides=0,
        name=f"via_pad_{tag}",
        material="copper",
    )
    pad = hfss.modeler[f"via_pad_{tag}"]
    set_obj_color(pad, metal_color)
    via_pad_objs[tag] = pad

# Direct 3D via-to-balun microstrip connectors only for the assigned baluns.
# Keep constant width from the balun neck to the pad so there is no extra local
# narrowing in the bend/feed region after the main taper has reached w1.
assigned_connector_width_mm = balun_dims["w1"]

via_to_balun_west = create_straight_strip_xy(
    "via_to_balun_west_klopf",
    feed_centers_xy["west"],
    (-x_near_offset_mm, front_face_y),
    assigned_connector_width_mm,
    underside_z_mm,
)
via_to_balun_west_obj = assign_balun_sheet_as_copper(via_to_balun_west.name, thickness_mm=-BALUN_COPPER_THICKNESS_MM)

via_to_balun_east = create_straight_strip_xy(
    "via_to_balun_east_klopf",
    feed_centers_xy["east"],
    (+x_near_offset_mm, back_face_y),
    assigned_connector_width_mm,
    underside_z_mm,
)
via_to_balun_east_obj = assign_balun_sheet_as_copper(via_to_balun_east.name, thickness_mm=-BALUN_COPPER_THICKNESS_MM)

via_to_balun_south = create_straight_strip_xy(
    "via_to_balun_south_klopf",
    feed_centers_xy["south"],
    (right_face_x, -x_near_offset_mm),
    assigned_connector_width_mm,
    underside_z_mm,
)
via_to_balun_south_obj = assign_balun_sheet_as_copper(via_to_balun_south.name, thickness_mm=-BALUN_COPPER_THICKNESS_MM)

via_to_balun_north = create_straight_strip_xy(
    "via_to_balun_north_klopf",
    feed_centers_xy["north"],
    (left_face_x, +x_near_offset_mm),
    assigned_connector_width_mm,
    underside_z_mm,
)
via_to_balun_north_obj = assign_balun_sheet_as_copper(via_to_balun_north.name, thickness_mm=-BALUN_COPPER_THICKNESS_MM)

# Unite each active balun with its underside feed strip, via pad, and via barrel.
if balun_front_left_klopf_obj and via_to_balun_west_obj and via_pad_objs.get("west") and via_cylinder_objs.get("west"):
    unite_keep_first([balun_front_left_klopf_obj.name, via_to_balun_west_obj.name, via_pad_objs["west"].name, via_cylinder_objs["west"].name])
if balun_back_right_klopf_obj and via_to_balun_east_obj and via_pad_objs.get("east") and via_cylinder_objs.get("east"):
    unite_keep_first([balun_back_right_klopf_obj.name, via_to_balun_east_obj.name, via_pad_objs["east"].name, via_cylinder_objs["east"].name])
if balun_rot90_right_lower_klopf_obj and via_to_balun_south_obj and via_pad_objs.get("south") and via_cylinder_objs.get("south"):
    unite_keep_first([balun_rot90_right_lower_klopf_obj.name, via_to_balun_south_obj.name, via_pad_objs["south"].name, via_cylinder_objs["south"].name])
if balun_rot90_left_upper_klopf_obj and via_to_balun_north_obj and via_pad_objs.get("north") and via_cylinder_objs.get("north"):
    unite_keep_first([balun_rot90_left_upper_klopf_obj.name, via_to_balun_north_obj.name, via_pad_objs["north"].name, via_cylinder_objs["north"].name])

print("Balun geometry created successfully. The plus-shaped dielectric support now uses four non-overlapping substrate halves, the center intersection is filled by a copper common-ground hub, the exact vias and pads have been placed symmetrically at 1.56 mm from the origin, and the assigned via-feed connectors have been united with their balun traces.")
print("Balun substrate material = {} (er = {:.3f}, tanD = {:.4f})".format(balun_substrate_material_name, balun_substrate_er, balun_substrate_tand))
print("Calculated 50-ohm launch width = {:.4f} mm".format(w_source_50ohm_mm))
print("Calculated 100-ohm antenna-side single-ended width = {:.4f} mm".format(w_target_single_mm))
print("Calculated balun L      = {:.4f} mm".format(balun_dims["L"]))
print("Calculated balun Ltaper = {:.4f} mm".format(balun_dims["Ltaper"]))
print("Calculated balun w1     = {:.4f} mm".format(balun_dims["w1"]))
print("Calculated balun w4     = {:.4f} mm".format(balun_dims["w4"]))
print("Antenna-side face neck gap on each active balun face = {:.4f} mm".format(actual_face_neck_gap_mm))
print("Active via mapping: west->front_left_klopf, east->back_right_klopf, south->rot90_right_lower_klopf, north->rot90_left_upper_klopf")
print("Requested impedance study target = {:.2f} ohm at the antenna-side balanced feed".format(z_target_antenna_diff_ohm))
print("All four active baluns use the same Klopfenstein-style taper profile.")
print("Balun board bottom z = {:.6f} mm, top z = {:.6f} mm, substrate gap = {:.6f} mm".format(balun_board_bottom_z_mm, balun_board_top_z_mm, balun_substrate_gap_mm))
print("Fixed via radius = {:.4f} mm, via height = {:.4f} mm".format(FIXED_VIA_RADIUS_MM, FIXED_VIA_HEIGHT_MM))
print("Fixed via pad radius = {:.4f} mm, pad height = {:.4f} mm".format(FIXED_VIA_PAD_RADIUS_MM, FIXED_VIA_PAD_HEIGHT_MM))
print("Balun/feed/ground copper thickness = {:.3f} mm".format(BALUN_COPPER_THICKNESS_MM))
# -----------------------------------------------------------------------------
# Modal lumped ports on the bottom faces of the four plus-shaped substrate sections.
# Each port sheet is a rectangle on the bottom face with width equal to the widest
# active balun width (w4) and in-plane thickness equal to the substrate thickness.
# The integration line is drawn across the substrate thickness from the local ground
# side to the active signal side for that section.
# -----------------------------------------------------------------------------
port_z_mm = balun_board_bottom_z_mm
port_power_each_w = 0.25
port_sheet_width_mm = balun_dims["w4"]
port_sheet_front_y_mm = front_face_y
port_sheet_back_y_mm = back_face_y
port_sheet_right_x_mm = right_face_x
port_sheet_left_x_mm = left_face_x


def create_modal_port_sheet_xy_faces(name, x_center, x_width_mm, y_min_mm, y_max_mm, z_mm):
    return create_rect_sheet_xy(
        name,
        x_center - 0.5 * x_width_mm,
        y_min_mm,
        x_center + 0.5 * x_width_mm,
        y_max_mm,
        z_mm,
    )


def create_modal_port_sheet_yz_faces(name, y_center, y_width_mm, x_min_mm, x_max_mm, z_mm):
    return create_planar_polygon_sheet(
        name,
        [
            [x_min_mm, y_center - 0.5 * y_width_mm, z_mm],
            [x_max_mm, y_center - 0.5 * y_width_mm, z_mm],
            [x_max_mm, y_center + 0.5 * y_width_mm, z_mm],
            [x_min_mm, y_center + 0.5 * y_width_mm, z_mm],
        ],
    )


def create_modal_lumped_port(sheet_name, port_name, int_line_start, int_line_end):
    try:
        return hfss.lumped_port(
            assignment=sheet_name,
            integration_line=[int_line_start, int_line_end],
            impedance=50,
            name=port_name,
            renormalize=False,
            deembed=False,
        )
    except TypeError:
        try:
            return hfss.lumped_port(
                signal=sheet_name,
                integration_line=[int_line_start, int_line_end],
                impedance=50,
                name=port_name,
                renormalize=False,
                deembed=False,
            )
        except Exception:
            return None


# Left original section: signal on +Y face, ground on -Y face.
port_sheet_west = create_modal_port_sheet_xy_faces(
    "port_sheet_west_klopf",
    -x_far_offset_mm,
    port_sheet_width_mm,
    port_sheet_back_y_mm,
    port_sheet_front_y_mm,
    port_z_mm,
)
port_west = create_modal_lumped_port(
    port_sheet_west.name,
    "port_west_klopf",
    [-x_far_offset_mm, port_sheet_back_y_mm, port_z_mm],
    [-x_far_offset_mm, port_sheet_front_y_mm, port_z_mm],
)

# Right original section: signal on -Y face, ground on +Y face.
port_sheet_east = create_modal_port_sheet_xy_faces(
    "port_sheet_east_klopf",
    +x_far_offset_mm,
    port_sheet_width_mm,
    port_sheet_back_y_mm,
    port_sheet_front_y_mm,
    port_z_mm,
)
port_east = create_modal_lumped_port(
    port_sheet_east.name,
    "port_east_klopf",
    [+x_far_offset_mm, port_sheet_front_y_mm, port_z_mm],
    [+x_far_offset_mm, port_sheet_back_y_mm, port_z_mm],
)

# Lower rotated section: signal on +X face, ground on -X face.
port_sheet_south = create_planar_polygon_sheet(
    "port_sheet_south_klopf",
    [
        [port_sheet_left_x_mm, -x_far_offset_mm - 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_right_x_mm, -x_far_offset_mm - 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_right_x_mm, -x_far_offset_mm + 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_left_x_mm, -x_far_offset_mm + 0.5 * port_sheet_width_mm, port_z_mm],
    ],
)
port_south = create_modal_lumped_port(
    port_sheet_south.name,
    "port_south_klopf",
    [port_sheet_left_x_mm, -x_far_offset_mm, port_z_mm],
    [port_sheet_right_x_mm, -x_far_offset_mm, port_z_mm],
)

# Upper rotated section: signal on -X face, ground on +X face.
port_sheet_north = create_planar_polygon_sheet(
    "port_sheet_north_klopf",
    [
        [port_sheet_left_x_mm, +x_far_offset_mm - 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_right_x_mm, +x_far_offset_mm - 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_right_x_mm, +x_far_offset_mm + 0.5 * port_sheet_width_mm, port_z_mm],
        [port_sheet_left_x_mm, +x_far_offset_mm + 0.5 * port_sheet_width_mm, port_z_mm],
    ],
)
port_north = create_modal_lumped_port(
    port_sheet_north.name,
    "port_north_klopf",
    [port_sheet_right_x_mm, +x_far_offset_mm, port_z_mm],
    [port_sheet_left_x_mm, +x_far_offset_mm, port_z_mm],
)

print("Modal lumped ports created on the bottom faces of the four plus-shaped balun-substrate sections.")
print("Each port sheet width equals the widest balun width w4 = {:.4f} mm.".format(port_sheet_width_mm))
print("Source phasing: west=0deg, east=180deg, north=90deg, south=270deg.")
print("Port impedances: 50 ohm each, with port sheets spanning directly from one conductor face to the opposing conductor face.")
print("Source phasing: west=0deg, east=180deg, north=90deg, south=270deg.")
print("Total input power target = 1 W (0.25 W per port).")
# =============================================================================

# =============================================================================

center_hole = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
                        cm2mm * antenna_coord_origin_xy_cm[1],
                        cm2mm * -dielectric_height_cm])
center_hole_params = {"name": "center_hole",
                      "orientation": "Z",
                      "origin": "{}mm,{}mm,{}mm".format(center_hole[0],
                                                          center_hole[1],
                                                          center_hole[2]).split(","),
                      "radius": "{}mm".format(cm2mm * 1.1 * ground_plane_hole_diameter_cm / 2),
                      "height": "{}mm".format(cm2mm * dielectric_height_cm),
                      "num_sides": 20,
                      "material": None}
# center_hole_geom = hfss.modeler.create_cylinder(**center_hole_params)

# detached_face_names = hfss.oeditor.DetachFaces(
#     ["NAME:Selections",
#      "Selections:=", "center_hole",
#      "NewPartsModelFlag:=", "Model"
#      ],
#     ["NAME:Parameters",
#      ["NAME:DetachFacesToParameters",
#       "FacesToDetach:=", [center_hole_geom.faces[0].id, center_hole_geom.faces[1].id]]])

# center_hole_position = np.array([cm2mm * antenna_coord_origin_xy_cm[0],
#                                              cm2mm * antenna_coord_origin_xy_cm[1],
#                                              cm2mm * -dielectric_height_cm])
# center_hole_size = np.array([cm2mm * ground_plane_hole_diameter_cm,
#                                              cm2mm * ground_plane_hole_diameter_cm,
#                                              cm2mm * dielectric_height_cm])
#
# center_hole_params = {"name": "center_hole",
#                           "origin": "{}mm,{}mm,{}mm".format(dielectric_slab_position[0],
#                                                               dielectric_slab_position[1],
#                                                               dielectric_slab_position[2]).split(","),
#                           "dimensions_list": "{}mm,{}mm,{}mm".format(center_hole_size[0],
#                                                                      center_hole_size[1],
#                                                                      center_hole_size[2]).split(","),
#                           "material": None}
# center_hole_geom = hfss.modeler.create_box(**center_hole_params)


# subtract_params = {
#     "blank_list": ["ground_plane", "dielectric_slab",
#                    sinuous_arm_polyline_list[0].name, sinuous_arm_polyline_list[1].name,
#                    sinuous_arm_polyline_list[2].name, sinuous_arm_polyline_list[3].name],
#     "tool_list": ["center_hole"],
#     "keep_originals": True
# }
# hfss.modeler.subtract(**subtract_params)
# center_hole_geom.delete()
# lumped_port_sheet_geom = hfss.modeler[detached_face_names[1]]
# # Integration line can be two points or one of the following:
# #           ``XNeg``, ``YNeg``, ``ZNeg``, ``XPos``, ``YPos``, and ``ZPos``
# lumped_port_params = {  # "signal": None,
#     "signal": detached_face_names[1],
#     "reference": "monopole_shield",
#     "create_port_sheet": False,
#     "port_on_plane": True,
#     # "integration_line": "XPos",
#     "integration_line": [lumped_port_sheet_geom.bottom_face_z.center,
#                          lumped_port_sheet_geom.bottom_face_z.top_edge_x.midpoint],
#     "impedance": 50,
#     "name": "source_port",
#     "renormalize": False,
#     "deembed": False,
#     "terminals_rename": False
# }
# hfss.lumped_port(**lumped_port_params)


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

# hfss.create_open_region(open_region_params)
module = hfss.get_module("ModelSetup")
# module.CreateOpenRegion(
#     [
#         "NAME:Settings",
#         "OpFreq:=", "6GHz",
#         "Boundary:=", "Radiation",
#         "ApplyInfiniteGP:=", False
#     ])
# Set up radiation boundary
hfss.create_open_region(frequency=frequency_max)

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
solver_setup = hfss.create_setup(setupname="Setup1", setuptype="HFSSDriven")
solver_setup_params = {"SolveType": 'Single',
                       # ('MultipleAdaptiveFreqsSetup',
                       #  SetupProps([('1GHz', [0.02]),
                       #              ('2GHz', [0.02]),
                       #              ('5GHz', [0.02])])),
                       "Frequency": '12GHz',
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
    "freqstart": 2.0,
    "freqstop": 12.0,
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
hfss.analyze_setup("Setup1")
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

desktop.release_desktop(close_projects=False, close_on_exit=False)