import numpy as np


class UnitCell:
    def __init__(self, name=None, position=[], size=[], scale_factor=1.0, material_name=None, color=None):
        if color is None:
            color = [143, 175, 143]
        self.name = name
        self.position = position
        self.size = size
        self.scale_factor = scale_factor
        self.material_name = material_name
        self.color = color
        self.geom_list = []
        self.geom_name_list = []

    def create_model(self, hffs_design):
        print("error no geometry in the UnitCell base class.")
        return self.geom_list

    def delete_model(self):
        for geom in self.geom_list:
            geom.delete()

    def get_model_names(self):
        return self.geom_name_list

    def __str__(self):
        return "default unit cell"



class TransmissionLineSquarePatchUnitCell(UnitCell):
    def __init__(self, name=None, position=None, size=None, scale_factor=None,
                 material_name=None, color=None):
        super().__init__(name, position, size, scale_factor, material_name, color)

    def create_model(self, hfss_design):

        unit_cell_plane_position = self.scale_factor * np.array([self.position[0] - (self.size[0] / 2),
                                                                 self.position[1] - (self.size[1] / 2),
                                                                 self.position[2]])
        unit_cell_plane_size = self.scale_factor * np.array([self.size[0],
                                                             self.size[0]])
        unit_cell_plane_params = {"name": self.name + "_patch",
                                  "csPlane": "XY",
                                  "position": "{}mm,{}mm,{}mm".format(unit_cell_plane_position[0],
                                                                      unit_cell_plane_position[1],
                                                                      unit_cell_plane_position[2]).split(","),
                                  "dimension_list": "{}mm,{}mm".format(unit_cell_plane_size[0],
                                                                       unit_cell_plane_size[1]).split(","),
                                  "matname": self.material_name,
                                  "is_covered": True}
        unit_cell_plane_geom = hfss_design.modeler.create_rectangle(**unit_cell_plane_params)
        unit_cell_plane_geom.color = self.color
        self.geom_list.append(unit_cell_plane_geom)
        self.geom_name_list.append(self.name + "_patch")
        return self.geom_list

class SquarePatchUnitCell(UnitCell):
    def __init__(self, name=None, position=None, size=None, scale_factor=None,
                 material_name=None, color=None):
        super().__init__(name, position, size, scale_factor, material_name, color)

    def create_model(self, hfss_design):
        unit_cell_plane_position = self.scale_factor * np.array([self.position[0] - (self.size[0] / 2),
                                                                 self.position[1] - (self.size[1] / 2),
                                                                 self.position[2]])
        unit_cell_plane_size = self.scale_factor * np.array([self.size[0],
                                                             self.size[0]])
        unit_cell_plane_params = {"name": self.name + "_patch",
                                  "csPlane": "XY",
                                  "position": "{}mm,{}mm,{}mm".format(unit_cell_plane_position[0],
                                                                      unit_cell_plane_position[1],
                                                                      unit_cell_plane_position[2]).split(","),
                                  "dimension_list": "{}mm,{}mm".format(unit_cell_plane_size[0],
                                                                       unit_cell_plane_size[1]).split(","),
                                  "matname": self.material_name,
                                  "is_covered": True}
        unit_cell_plane_geom = hfss_design.modeler.create_rectangle(**unit_cell_plane_params)
        unit_cell_plane_geom.color = self.color
        self.geom_list.append(unit_cell_plane_geom)
        self.geom_name_list.append(self.name + "_patch")
        return self.geom_list

    def __str__(self):
        return "square patch unit cell"


# Sievenpiper Mushroom
class SievenpiperMushroomUnitCell(SquarePatchUnitCell):
    def __init__(self, name=None, position=None, size_patch=None,
                 size_via=None, scale_factor=None,
                 material_name=None, color=None):
        super().__init__(name, position, size_patch, scale_factor, material_name, color)
        #  via stored in [radius, height] order
        self.size_via = size_via

    def create_model(self, hfss_design):
        # add the square planar patch
        unit_cell_plane_geom = super().create_model(hfss_design)
        # Construct the via from the ground plane to the metal patch
        # s_axis is "X", "Y" or "Z"
        unit_cell_via_position = self.scale_factor * np.array([self.position[0],
                                                               self.position[1],
                                                               self.position[2] - self.size_via[1]])
        unit_cell_via_params = {"name": self.name + "_via",
                                "cs_axis": "Z",
                                "position": "{}mm,{}mm,{}mm".format(unit_cell_via_position[0],
                                                                    unit_cell_via_position[1],
                                                                    unit_cell_via_position[2]).split(","),
                                "radius": "{}mm".format(self.scale_factor * self.size_via[0]),
                                "height": "{}mm".format(self.size_via[1]),
                                "numSides": 0,
                                "matname": self.material_name}
        unit_cell_via_geom = hfss_design.modeler.create_cylinder(**unit_cell_via_params)
        unit_cell_via_geom.color = self.color
        self.geom_list.append(unit_cell_via_geom)
        self.geom_name_list.append(self.name + "_via")
        return self.geom_list

    def __str__(self):
        return "Sievenpiper Mushroom unit cell"


class EllipticalDiscUnitCell(UnitCell):
    def __init__(self, name=None, position=None, size=None, scale_factor=None,
                 rotation_xy_deg=0, material_name=None, color=None):
        super().__init__(name, position, size, scale_factor, material_name, color)
        self.rotation_xy_deg = rotation_xy_deg

    def create_model(self, hfss_design):
        unit_cell_disc_position = self.scale_factor * np.array([self.position[0],
                                                                self.position[1],
                                                                self.position[2]])
        #  convert diameter to radius
        unit_cell_disc_size = self.scale_factor * np.array([self.size[0] / 2,
                                                            self.size[1] / 2])
        unit_cell_ellipse_params = {"name": self.name + "_patch",
                                    "cs_plane": "XY",
                                    "position": "{}mm,{}mm,{}mm".format(unit_cell_disc_position[0],
                                                                        unit_cell_disc_position[1],
                                                                        unit_cell_disc_position[2]).split(","),
                                    "major_radius": unit_cell_disc_size[0],
                                    "ratio": unit_cell_disc_size[1] / unit_cell_disc_size[0],
                                    "matname": self.material_name,
                                    "is_covered": True}
        unit_cell_disc_geom = hfss_design.modeler.create_ellipse(**unit_cell_ellipse_params)
        unit_cell_disc_geom.color = self.color
        if self.rotation_xy_deg != 0:
            hfss_design.modeler.rotate(**{"objid": unit_cell_disc_geom, "cs_axis": "Z",
                                          "angle": self.rotation_xy_deg, "unit": "deg"})
        self.geom_list.append(unit_cell_disc_geom)
        self.geom_name_list.append(self.name + "_patch")
        return self.geom_list

    def __str__(self):
        return "elliptical unit cell"


class CircularDiscUnitCell(EllipticalDiscUnitCell):
    def __init__(self, name=None, position=None, size=None, scale_factor=None,
                 material_name=None, color=None, **kwargs):
        ellipse_size = np.array([size[0], size[0]])
        super().__init__(name, position, ellipse_size, scale_factor, material_name, color)

    def __str__(self):
        return "disc unit cell"
