import cv2
import numpy as np
import trimesh
from matplotlib.tri import Triangulation
import yaml
import argparse
import os
import sys

class MapConverter():
    def __init__(self, map_dir, export_dir, threshold=105, height=2.0):
        
        self.threshold = threshold
        self.height = height
        self.export_dir = export_dir
        self.map_dir = map_dir

    def map_callback(self):
        
        map_array = cv2.imread(self.map_dir)
        map_array = cv2.flip(map_array, 0)
        print(f'loading map file: {self.map_dir}')
        try:
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
        except cv2.error as err:
            print(err, "Conversion failed: Invalid image input, please check your file path")    
            sys.exit()
        info_dir = self.map_dir.replace('pgm','yaml')

        with open(info_dir, 'r') as stream:
            map_info = yaml.load(stream, Loader=yaml.FullLoader) 
        
        # set all -1 (unknown) values to 0 (unoccupied)
        map_array[map_array < 0] = 0
        contours = self.get_occupied_regions(map_array)
        print('Processing...')
        meshes = [self.contour_to_mesh(c, map_info) for c in contours]

        corners = list(np.vstack(contours))
        corners = [c[0] for c in corners]
        mesh = trimesh.util.concatenate(meshes)
        if not self.export_dir.endswith('/'):
            self.export_dir = self.export_dir + '/'
        file_dir = self.export_dir + map_info['image'].replace('pgm','stl')
        print(f'export STL file: {file_dir}')
        
        with open(file_dir, 'wb') as f:
            mesh.export(f, "stl")
        
        self.create_sdf_file(map_info, file_dir)

    def create_sdf_file(self, map_info, stl_file_path):
        """
        Create a Gazebo SDF file that references the generated STL mesh
        """
        map_name = os.path.splitext(map_info['image'])[0]
        
        sdf_file_path = stl_file_path.replace('.stl', '.sdf')
        
        stl_absolute_path = os.path.abspath(stl_file_path)
        
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.9" xmlns:xacro="http://ros.org/wiki/xacro">
  <xacro:arg name="headless" default="false" />
  
  <world name="{map_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics">
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster">
    </plugin>

    <!-- Conditional plugins for GUI mode -->
    <xacro:unless value="$(arg headless)">
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      </plugin>
    </xacro:unless>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <direction>0 0 -1</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Map STL model -->
    <model name="{map_name}_model">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://{stl_absolute_path}</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://{stl_absolute_path}</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>"""

        with open(sdf_file_path, 'w') as f:
            f.write(sdf_content)
        
        print(f'export SDF file: {sdf_file_path}')
        print(f'Map name: {map_name}')
        print(f'STL path in SDF: file://{stl_absolute_path}')

    def get_occupied_regions(self, map_array):
        """
        Get occupied regions of map
        """
        map_array = map_array.astype(np.uint8)
        _, thresh_map = cv2.threshold(
                map_array, self.threshold, 100, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
                thresh_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]
        output_contours = []
        for idx, contour in enumerate(contours):
            output_contours.append(contour) if 0 not in contour else print('Remove image boundary')
            
        return output_contours

    def contour_to_mesh(self, contour, metadata):
        height = np.array([0, 0, self.height])
        meshes = []
        for point in contour:
            x, y = point[0]
            vertices = []
            new_vertices = [
                    self.coords_to_loc((x, y), metadata),
                    self.coords_to_loc((x, y+1), metadata),
                    self.coords_to_loc((x+1, y), metadata),
                    self.coords_to_loc((x+1, y+1), metadata)]
            vertices.extend(new_vertices)
            vertices.extend([v + height for v in new_vertices])
            faces = [[0, 2, 4],
                     [4, 2, 6],
                     [1, 2, 0],
                     [3, 2, 1],
                     [5, 0, 4],
                     [1, 0, 5],
                     [3, 7, 2],
                     [7, 6, 2],
                     [7, 4, 6],
                     [5, 4, 7],
                     [1, 5, 3],
                     [7, 3, 5]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not mesh.is_volume:
                mesh.fix_normals()
            meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        mesh.update_faces(mesh.unique_faces())
        return mesh

    def coords_to_loc(self,coords, metadata):
        x, y = coords
        loc_x = x * metadata['resolution'] + metadata['origin'][0]
        loc_y = y * metadata['resolution'] + metadata['origin'][1]
        return np.array([loc_x, loc_y, 0.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--map_dir', type=str, required=True,
        help='File name of the map to convert'
    )

    parser.add_argument(
        '--export_dir', type=str, default=os.path.abspath('.'),
        help='Mesh output directory'
    )

    option = parser.parse_args()

    Converter = MapConverter(option.map_dir, option.export_dir)
    Converter.map_callback()
    print('Conversion Done')
