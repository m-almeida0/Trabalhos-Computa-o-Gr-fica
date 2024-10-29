from OpenGL.GL import *
import numpy as np
import math

from PIL import Image

def gen_rot_matrix_x(angle):
    rotation_matrix = np.identity(4, dtype=np.float32)
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    rotation_matrix[1][1] = cos_t
    rotation_matrix[1][2] = -sin_t
    rotation_matrix[2][1] = sin_t
    rotation_matrix[2][2] = cos_t
    return rotation_matrix

def gen_rot_matrix_y(angle):
    rotation_matrix = np.identity(4, dtype=np.float32)
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    rotation_matrix[0][0] = cos_t
    rotation_matrix[2][0] = -sin_t
    rotation_matrix[0][2] = sin_t
    rotation_matrix[2][2] = cos_t
    return rotation_matrix


def gen_rot_matrix_z(angle):
    rotation_matrix = np.identity(4, dtype=np.float32)
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    rotation_matrix[0][0] = cos_t
    rotation_matrix[0][1] = -sin_t
    rotation_matrix[1][0] = sin_t
    rotation_matrix[1][1] = cos_t
    return rotation_matrix

def gen_translation_matrix(x, y, z):
    translation_matrix = np.identity(4, dtype=np.float32)
    translation_matrix[0][3] = x
    translation_matrix[1][3] = y
    translation_matrix[2][3] = z
    return translation_matrix

def gen_scale_matrix(x, y, z):
    scale_matrix = np.identity(4, dtype=np.float32)
    scale_matrix[0][0] = x
    scale_matrix[1][1] = y
    scale_matrix[2][2] = z
    return scale_matrix

class RenderableObject:
    def __init__(self, tag, center, obj_filepath, texture_filepath):
        self.tag = tag
        self.center = center
        self.scale = [1.0, 1.0, 1.0]

        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.texture_index = []
        self.normal_index = []

        self.model = []
        self.tex_img_data = None
        self.texture = None
        self.img_width = 0
        self.img_height = 0
        self.texture_offset = 0

        self.vertices_VBO = glGenBuffers(1)        
        self.textures_EBO = glGenBuffers(1)

        self.translation_matrix = np.identity(4, dtype=np.float32)
        self.translation_matrix[0][3] = center[0]
        self.translation_matrix[1][3] = center[1]
        self.translation_matrix[2][3] = center[2]

        self.scale_matrix = np.identity(4, dtype=np.float32)

        self.rotation_matrix_x = np.identity(4, dtype=np.float32)
        self.rotation_matrix_y = np.identity(4, dtype=np.float32)
        self.rotation_matrix_z = np.identity(4, dtype=np.float32)

        self.rotation_matrix = np.identity(4, dtype=np.float32)

        self.load_model(obj_filepath)
        self.load_texture(texture_filepath)

        self.texture = glGenTextures(1)

    def load_model(self, file):
        count = 0
        for line in open(file, 'r'):
            count = count + 1
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vt':
                self.text_coords.append(values[1:3])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])

            if values[0] == 'f':
                #print(values)
                len_values = len(values)
                self.vertex_by_face = len_values-1
                #print(len_values)
                face_i = []
                text_i = []
                norm_i = []

                for v in values[1:len_values]:
                    w = v.split('/')
                    face_i.append(int(w[0])-1)
                    if(w[1] != ''):
                        text_i.append(int(w[1])-1)
                    else:
                        text_i.append(0)
                    norm_i.append(int(w[2])-1)
                
                if len_values == 4:  # Triangle face_i
                    self.vertex_index.append(face_i)
                    self.texture_index.append(text_i)
                    self.normal_index.append(norm_i)
                elif len_values == 5:  # Quad face_i, split into two triangles
                    t1 = [face_i[0], face_i[1], face_i[2]]
                    t2 = [face_i[0], face_i[2], face_i[3]]
                    self.vertex_index.append(t1)  # First triangle
                    self.vertex_index.append(t2)  # Second triangle
                    tex1 = [text_i[0], text_i[1], text_i[2]]
                    tex2 = [text_i[0], text_i[2], text_i[3]]
                    self.texture_index.append(tex1)
                    self.texture_index.append(tex2)
                    norm1 = [norm_i[0], norm_i[1], norm_i[2]]
                    norm2 = [norm_i[0], norm_i[2], norm_i[3]]
                    self.normal_index.append(norm1)
                    self.normal_index.append(norm2)
        #print(self.normal_index)
        #print(count)
        self.vertex_index = [y for x in self.vertex_index for y in x]
        self.texture_index = [y for x in self.texture_index for y in x]
        self.normal_index = [y for x in self.normal_index for y in x]

        temp = []

        for i in self.vertex_index:
            self.model.extend(self.vert_coords[i])

        for i in self.texture_index:
            self.model.extend(self.text_coords[i])
            temp.extend(self.text_coords[i])

        for i in self.normal_index:
            self.model.extend(self.norm_coords[i])

        self.vert_coords = np.array(self.vert_coords, dtype=np.float32)
        self.text_coords = np.array(temp, dtype=np.float32)
        self.norm_coords = np.array(self.norm_coords, dtype=np.float32)

        self.vertex_index = np.array(self.vertex_index, dtype=np.int32)
        self.texture_index = np.array(self.texture_index, dtype=np.float32)
        self.normal_index = np.array(self.normal_index, dtype=np.float32)

        self.model = np.array(self.model, dtype='float32')
        self.texture_offset = len(self.vertex_index)*12

    def load_texture(self, file):
        image = Image.open(file)
        #img_data = np.array(image, np.uint8)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.width, self.height = image.size
        rgb_im = image.convert('RGB')
        self.tex_img_data = np.array(rgb_im, np.uint8)

    def bind_buffers(self, position_location, TexCoord_location):
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*len(self.model), self.model, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.model.itemsize * 3, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position_location)

        self.textures_EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.textures_EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(self.text_coords), self.text_coords, GL_STATIC_DRAW)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, self.model.itemsize * 2, ctypes.c_void_p(self.texture_offset))
        glEnableVertexAttribArray(TexCoord_location)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, self.tex_img_data)

    def update_matrices(self):
        self.translation_matrix = gen_translation_matrix(self.center[0], self.center[1], self.center[2])
        #self.rotation_matrix = self.rotation_matrix_x @ self.rotation_matrix_y @ self.rotation_matrix_z
        self.rotation_matrix = np.dot(self.rotation_matrix_x, np.dot(self.rotation_matrix_y, self.rotation_matrix_z))
        self.scale_matrix = gen_scale_matrix(self.scale[0], self.scale[1], self.scale[2])
    
    def set_scale(self, scale_x, scale_y, scale_z):
        self.scale[0] = scale_x
        self.scale[1] = scale_y
        self.scale[2] = scale_z

    def set_rotation(self, x:float, y:float, z:float):
        self.rotation_matrix_x = gen_rot_matrix_x(x)
        self.rotation_matrix_y = gen_rot_matrix_y(y)
        self.rotation_matrix_z = gen_rot_matrix_z(z)
    
    def set_pos(self, x:float, y:float, z:float):
        self.center[0] = x
        self.center[1] = y
        self.center[2] = z

import numpy as np

class Camera:
    def __init__(self, position=None, target_mod=None, theta_z=None, up=None):
        self.position = np.array(position if position is not None else [0.0, 0.0, -1.0], dtype=np.float32)
        #self.target = np.array(target if target is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        if(target_mod is None and theta_z is None):
            self.target_mod = 1.0
            self.theta_z = 0.0
            self.target = np.array([self.position[0], self.position[1], self.position[2]+1], dtype=np.float32)
        else:
            self.calculate_target(target_mod, theta_z)
        self.up = np.array(up if up is not None else [0.0, 1.0, 0.0], dtype=np.float32)

        self.angular_speed = 0.1
        self.linear_speed = 0.1
        
        self.view_matrix = self.calculate_view_matrix()

    def calculate_target(self, target_mod = None, theta_z = None):
        if target_mod is not None:
            self.target_mod = target_mod
        if theta_z is not None:
            self.theta_z = theta_z
        self.target = np.array([self.position[0]+self.target_mod*math.sin(self.theta_z), self.position[1], self.position[2]+self.target_mod*math.cos(self.theta_z)], dtype=np.float32)

    def calculate_view_matrix(self):
        self.calculate_target()
        # Calculate forward, right, and up vectors
        f = self.target - self.position
        f = f / np.linalg.norm(f)  # Normalize forward vector

        r = np.cross(f, self.up)
        r = r / np.linalg.norm(r)  # Normalize right vector

        u = np.cross(r, f)  # Calculate actual up vector

        # Create view matrix
        view = np.identity(4, dtype=np.float32)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[0, 3] = -np.dot(r, self.position)
        view[1, 3] = -np.dot(u, self.position)
        view[2, 3] = np.dot(f, self.position)

        return view

    def move_camera(self, x = 0.0, y = 0.0, z = 0.0, d_z = 0.0):
        self.theta_z= self.theta_z+d_z
        aux_sin = math.sin(self.theta_z)
        aux_cos = math.cos(self.theta_z)
        self.position = np.array([self.position[0] + x*aux_cos + z*aux_sin, self.position[1] + y, self.position[2] - x*aux_sin + z*aux_cos], dtype=np.float32)
        
        #self.target = np.array([self.position[0] + math.sin(d_z), self.position[1], self.position[2]+math.cos(d_z)], dtype=np.float32)
        self.view_matrix = self.calculate_view_matrix()
        

    def set_position(self, x, y, z):
        self.position = np.array([x, y, z], dtype=np.float32)
        self.view_matrix = self.calculate_view_matrix()

    def set_target(self, x, y, z):
        self.target = np.array([x, y, z], dtype=np.float32)
        self.view_matrix = self.calculate_view_matrix()

    def set_up(self, x, y, z):
        self.up = np.array([x, y, z], dtype=np.float32)
        self.view_matrix = self.calculate_view_matrix()

    def get_position(self):
        return self.position

    def get_view_matrix(self):
        return self.view_matrix


def create_3d_star(color, center_side, point_len):
    d = center_side/2
    p = point_len
    vertices = np.array([
        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
          0.0,   0.0,   d+p, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],
          0.0,   0.0,  -d-p, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],
          d+p,   0.0,   0.0, color[0], color[1], color[2],

        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],
         -d-p,   0.0,   0.0, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],

        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2]
    ], dtype=np.float32)

    indices = np.array([
         0,  1,  4,  1,  2,  4,  2,  3,  4,  3,  0,  4,
         5,  6,  9,  6,  7,  9,  7,  8,  9,  8,  9,  5,
        10, 11, 14, 11, 12, 14, 12, 13, 14, 13, 10, 14,
        15, 16, 19, 16, 17, 19, 17, 18, 19, 18, 19, 15,
        20, 21, 22, 20, 22, 23,
        24, 25, 26, 24, 26, 27
    ], dtype=np.uint32)

    return vertices, indices

def create_cube(color, side):
    d = side/2
    vertices = np.array([
        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2],

        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2],

        0.0+d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0+d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0+d, 0.0-d, color[0], color[1], color[2],

        0.0+d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0+d, color[0], color[1], color[2],
        0.0+d, 0.0-d, 0.0-d, color[0], color[1], color[2],
        0.0-d, 0.0-d, 0.0-d, color[0], color[1], color[2]
    ], dtype=np.float32)

    indices = np.array([
        0,  1,  2,  1,  2,  3,
        4,  5,  6,  5,  6,  7,
        8,  9,  10, 9,  10, 11,
        12, 13, 14, 13, 14, 15,
        16, 17, 18, 17, 18, 19,
        20, 21, 22, 21, 22, 23
    ], dtype=np.uint32)

    return vertices, indices

"""def create_catavento(l):
    l = 0.4
    vertices = np.array([
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        l, 0.0, 0.0, 0.0, 1.0, 0.0,
        l,   l, 0.0, 0.0, 1.0, 0.0,
        0.0,   l, 0.0, 0.0, 1.0, 0.0,
        0.0, 2*l, 0.0, 0.0, 1.0, 0.0,

        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        -l, 0.0, 0.0, 0.0, 0.0, 1.0,
        -l,   l, 0.0, 0.0, 0.0, 1.0,
        0.0,   l, 0.0, 0.0, 0.0, 1.0,
        -2*l, 0.0, 0.0, 0.0, 0.0, 1.0,
        
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        -l, 0.0, 0.0, 1.0, 0.0, 0.0,
        -l,  -l, 0.0, 1.0, 0.0, 0.0,
        0.0,  -l, 0.0, 1.0, 0.0, 0.0,
        0.0,-2*l, 0.0, 1.0, 0.0, 0.0,

        0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        l, 0.0, 0.0, 1.0, 1.0, 0.0,
        l,  -l, 0.0, 1.0, 1.0, 0.0,
        0.0,  -l, 0.0, 1.0, 1.0, 0.0,
        2*l, 0.0, 0.0, 1.0, 1.0, 0.0
    ], dtype=np.float32)

    indices = np.array([
        0,  1,  2,  0,  2,  3,  2,  3,  4,
        5,  6,  7,  5,  7,  8,  6,  7,  9,
        10, 11, 12, 10, 12, 13, 12, 13, 14,
        15, 16, 17, 15, 17, 18, 16, 17, 19
    ], dtype=np.uint32)
    return vertices, indices
"""

def create_catavento(l, cor_q1, cor_q2, cor_q3, cor_q4):
    #l = 0.4
    vertices = np.array([
        0.0, 0.0, 0.0, cor_q1[0], cor_q1[1], cor_q1[2],
          l, 0.0, 0.0, cor_q1[0], cor_q1[1], cor_q1[2],
          l,   l, 0.0, cor_q1[0], cor_q1[1], cor_q1[2],
        0.0,   l, 0.0, cor_q1[0], cor_q1[1], cor_q1[2],
        0.0, 2*l, 0.0, cor_q1[0], cor_q1[1], cor_q1[2],

        0.0, 0.0, 0.0, cor_q2[0], cor_q2[1], cor_q2[2],
         -l, 0.0, 0.0, cor_q2[0], cor_q2[1], cor_q2[2],
         -l,   l, 0.0, cor_q2[0], cor_q2[1], cor_q2[2],
        0.0,   l, 0.0, cor_q2[0], cor_q2[1], cor_q2[2],
       -2*l, 0.0, 0.0, cor_q2[0], cor_q2[1], cor_q2[2],
        
        0.0, 0.0, 0.0, cor_q3[0], cor_q3[1], cor_q3[2],
         -l, 0.0, 0.0, cor_q3[0], cor_q3[1], cor_q3[2],
         -l,  -l, 0.0, cor_q3[0], cor_q3[1], cor_q3[2],
        0.0,  -l, 0.0, cor_q3[0], cor_q3[1], cor_q3[2],
        0.0,-2*l, 0.0, cor_q3[0], cor_q3[1], cor_q3[2],

        0.0, 0.0, 0.0, cor_q4[0], cor_q4[1], cor_q4[2],
          l, 0.0, 0.0, cor_q4[0], cor_q4[1], cor_q4[2],
          l,  -l, 0.0, cor_q4[0], cor_q4[1], cor_q4[2],
        0.0,  -l, 0.0, cor_q4[0], cor_q4[1], cor_q4[2],
        2*l, 0.0, 0.0, cor_q4[0], cor_q4[1], cor_q4[2]
    ], dtype=np.float32)

    indices = np.array([
        0,  1,  2,  0,  2,  3,  2,  3,  4,
        5,  6,  7,  5,  7,  8,  6,  7,  9,
        10, 11, 12, 10, 12, 13, 12, 13, 14,
        15, 16, 17, 15, 17, 18, 16, 17, 19
    ], dtype=np.uint32)
    return vertices, indices

def create_horn(base_side, length, height):
    a = base_side
    b = length
    c = 1.5*base_side
    if length < 0:
        c = -1*c
    d = height
    horn_vertices = np.array([
        0.0,  a,  a, 0.0, 1.0, 0.0,#0
        0.0,  a, -a, 0.0, 1.0, 0.0,#1
        0.0, -a, -a, 0.0, 1.0, 0.0,#2
        0.0, -a,  a, 0.0, 1.0, 0.0,#3

        b-c,  a,  a, 0.0, 1.0, 0.0,#4
        b-c,  a, -a, 0.0, 1.0, 0.0,#5
        b, -a, -a, 0.0, 1.0, 0.0,#6
        b, -a,  a, 0.0, 1.0, 0.0,#7

        b-c/2, d, 0.0, 0.0, 1.0, 0.0#8
    ], dtype=np.float32)

    horn_indices = np.array([
        #0, 1, 2, 4, 5, 6,
        0, 1, 4, 1, 4, 5,
        1, 2, 5, 2, 5, 6,
        2, 3, 6, 3, 6, 7,
        3, 7, 0, 0, 7, 4,

        4, 5, 8, 5, 6, 8, 6, 7, 8, 4, 7, 8
    ], dtype=np.int32)

    return horn_vertices, horn_indices