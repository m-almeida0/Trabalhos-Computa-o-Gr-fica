import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

import numpy as np
from PIL import Image

import random
import math

#import glm

import utils

#cameraPos   = np.array([0.0,  0.0,  1.0], dtype=np.float32)
#cameraFront = np.array([0.0,  0.0, -1.0], dtype=np.float32)
#cameraUp    = np.array([0.0,  1.0,  0.0], dtype=np.float32)
#
#def view():
#    global cameraPos, cameraFront, cameraUp
#    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
#    mat_view = np.array(mat_view)
#    return mat_view

polygon = False
spinning = False
tree_scale = 1.0

def perspective_projection(fov, aspect_ratio, near, far):
    # Convert field of view from degrees to radians
    fov_rad = np.radians(fov)
    
    # Calculate the scale factors
    f = 1 / np.tan(fov_rad / 2)
    
    # Initialize the 4x4 projection matrix with zeros
    projection_matrix = np.zeros((4, 4))
    
    # Set the perspective projection matrix elements
    projection_matrix[0, 0] = f / aspect_ratio
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = -(far + near) / (far - near)
    projection_matrix[2, 3] = -(2 * far * near) / (far - near)
    projection_matrix[3, 2] = -1
    projection_matrix[3, 3] = 0
    
    return projection_matrix

obj_array = {}

def main():
    if not glfw.init():
        return
    
    window = glfw.create_window(800, 600, "My window", None, None)
    
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    vertex_shader = """
    #version 330 core
    layout (location=0) in vec3 position;
    layout (location=1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 rotation;
    uniform mat4 translation;
    uniform mat4 scale;
    uniform mat4 camera;
    uniform mat4 projection;

    void main()
    {
        mat4 final = projection*camera*translation*rotation*scale;
        gl_Position = final*vec4(position, 1.0);
        TexCoord = aTexCoord;
    }
    """

    fragment_shader = """
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;

    uniform sampler2D texture1;

    void main()
    {
        FragColor = texture(texture1, TexCoord);
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    position = glGetAttribLocation(shader, "position")
    TexCoord = glGetAttribLocation(shader, "aTexCoord")
    rotation_loc = glGetUniformLocation(shader, "rotation")
    translation_loc = glGetUniformLocation(shader, "translation")
    scale_loc = glGetUniformLocation(shader, "scale")
    camera_loc = glGetUniformLocation(shader, "camera")
    projection_loc = glGetUniformLocation(shader, "projection")

    #att: adicionando os objetos ao array
    #obj_array = {}
    global obj
    obj_array["cottage"] = utils.RenderableObject("cottage", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/cottage/cottage_obj.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/cottage/textures/cottage_diffuse.png")
    obj_array["nightstand"] = utils.RenderableObject("nightstand", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/Nightstand_obj/Nightstand.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/Nightstand_obj/Wood1_Albedo.png")
    obj_array["little_ghost"] = utils.RenderableObject("little_ghost", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/ghost/little_gost.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/ghost/ghost.png")
    obj_array['skybox'] = utils.RenderableObject("skybox", [0.0, 0.6, -1.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/skybox/skybox.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/skybox/skybox.png")
    obj_array['wood_floor'] = utils.RenderableObject("wood_floor", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wood_floor/wood_floor.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wood_floor/wood_floor.jpg")
    obj_array['grass'] = utils.RenderableObject("grass", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/grass/grass.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/grass/SM_Prop_Grass_1001_BaseColor.png")
    obj_array['wind_rooster'] = utils.RenderableObject("woord_rooster", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wind_rooster/wind_rooster.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wind_rooster/wind_rooster.png")
    obj_array['wolf'] = utils.RenderableObject("wolf", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wolf/wolf.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/wolf/wolf.jpg")
    obj_array['bed'] = utils.RenderableObject("bed", [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/bed/bed.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/bed/bed.png")

    obj_array['tree'] = utils.RenderableObject(f'tree_i', [0.0, 0.0, 0.0], "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/tree/tree.obj", "C:/Users/olavo/Desktop/Trabalhos-Computa-o-Gr-fica/models/tree/tree.png")
    obj_array['tree'].set_scale(0.01,0.01,0.01)
    obj_array['tree'].update_matrices()
    
    random.seed(42)
    tree_pos = []
    for i in range(1,21):
        X = random.choice([random.randint(20, 100), random.randint(-100, -20)])
        Z = random.choice([random.randint(20, 100), random.randint(-100, -20)])
        S = float(random.randint(5, 20))/1000.0
        tree_pos.append([float(X), float(Z), S])
    
    camera = utils.Camera(position=[0.0, 2.0, -1.0])

    fov = 60  # Field of view in degrees
    aspect_ratio = 16 / 9  # Aspect ratio of the viewport
    near = 0.1  # Near clipping plane
    far = 10000.0  # Far clipping plane

    projection_matrix = perspective_projection(fov, aspect_ratio, near, far)

    # att: modificando as propriedades de um objeto
    obj_array["nightstand"].set_pos(2.0, 0.11, -7)
    obj_array["nightstand"].set_scale(6,6,6)
    obj_array["nightstand"].update_matrices()

    obj_array['skybox'].set_scale(10,10,10)
    obj_array['skybox'].update_matrices()

    obj_array['grass'].set_pos(0,0.11,0)
    obj_array['grass'].set_scale(500,1,500)
    obj_array['grass'].update_matrices()

    obj_array['wind_rooster'].set_scale(0.05,0.05,0.05)
    obj_array['wind_rooster'].set_pos(0,15.8,0)
    obj_array['wind_rooster'].update_matrices()

    # obj_array['tree'].set_scale(0.01,0.01,0.01)
    # obj_array['tree'].set_pos(17,0,0)
    # obj_array['tree'].update_matrices()

    obj_array['wood_floor'].set_pos(0,0.2,0)
    obj_array['wood_floor'].set_scale(13,1,8)
    obj_array['wood_floor'].update_matrices()

    obj_array['wolf'].set_rotation(-90,0,math.pi)
    obj_array['wolf'].update_matrices()
    obj_array['wolf'].set_pos(25,0.11,45)
    obj_array['wolf'].update_matrices()

    obj_array['bed'].set_scale(0.04,0.04,0.04)
    obj_array['bed'].set_pos(0,0,5)
    obj_array['bed'].update_matrices()

    glUseProgram(shader)
    texture1_loc = glGetUniformLocation(shader, "texture1")
    glUniform1i(texture1_loc, 0)  # 0 corresponds to GL_TEXTURE0

    glClearColor(1.0, 1.0, 1.0, 0.5)
    glEnable(GL_DEPTH_TEST)

    identity = np.array([1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def key_event(window, key, scancode, action, mods):
        global polygon, spinning, tree_scale
        w, a, s, d, q, e = 87, 65, 83, 68, 81, 69
        translation_keys = [w, a, s, d, q, e]
        j, k, r, f, p = 74, 75, 82, 70, 80
        m, n, x = 77, 78, 88
        speed_inc = 2
        max_speed = 300000
        min_speed = 0.05
        max_squared_radius = 50000.0
        ground_level = 0.6
        max_scale = 2.0
        min_scale = 0.5
        scale_inc = 0.1
        #print(key)
        if (action == 1 or action == 2):
            if key in translation_keys:
                old_camera_pos = camera.get_position()
                if key == a:
                    camera.move_camera(camera.linear_speed)
                if key == d:
                    camera.move_camera(-camera.linear_speed)
                if key == w:
                    camera.move_camera(z=camera.linear_speed)
                if key == s:
                    camera.move_camera(z=-camera.linear_speed)
                if key == q:
                    if (old_camera_pos[1] - camera.linear_speed) > ground_level:
                        camera.move_camera(y = -camera.linear_speed)
                if key == e:
                    camera.move_camera(y = camera.linear_speed)
                camera_pos = camera.get_position()
                sq_dist_from_origin = camera_pos[0]**2+camera_pos[1]**2+camera_pos[2]**2
                #print(camera_pos)
                #print(sq_dist_from_origin)
                if sq_dist_from_origin > max_squared_radius:
                    camera.set_position(old_camera_pos[0], old_camera_pos[1], old_camera_pos[2])
                    camera_pos = old_camera_pos
                obj_array['skybox'].set_pos(camera_pos[0], camera_pos[1], camera_pos[2])
                obj_array['skybox'].update_matrices()
                return

            if key == x:
                spinning = not spinning
                return
            if key == m:
                tree_scale += scale_inc
                if tree_scale > max_scale:
                    tree_scale = max_scale
                return
            if key == n:
                tree_scale -= scale_inc
                if tree_scale < min_scale:
                    tree_scale = min_scale
                return

            if key == j:
                camera.move_camera(d_z=(camera.angular_speed))
                return
            if key == k:
                camera.move_camera(d_z=(-camera.angular_speed))
                return
            if key == r:
                camera.linear_speed += speed_inc
                if camera.linear_speed > max_speed:
                    camera.linear_speed = max_speed
                return
            if key == f:
                camera.linear_speed -= speed_inc
                if camera.linear_speed < min_speed:
                    camera.linear_speed = min_speed
                return
            if key == p:
                polygon = not polygon
                return
            #print(camera.view_matrix)

    glfw.set_key_callback(window,key_event)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if polygon == True:
            glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)

        for key in obj_array.keys():
            obj = obj_array[key]
            obj.bind_buffers(position, TexCoord)

            glUniformMatrix4fv(projection_loc, 1, GL_TRUE, projection_matrix)
            glUniformMatrix4fv(camera_loc, 1, GL_TRUE, camera.view_matrix)
            glUniformMatrix4fv(rotation_loc, 1, GL_TRUE, obj.rotation_matrix)
            glUniformMatrix4fv(translation_loc, 1, GL_TRUE, obj.translation_matrix)
            glUniformMatrix4fv(scale_loc, 1, GL_TRUE, obj.scale_matrix)

            if key == 'tree':
                #print(tree_pos)
                for pos in tree_pos:
                    glUniformMatrix4fv(translation_loc, 1, GL_TRUE, utils.gen_translation_matrix(pos[0], 0.0, pos[1]))
                    glUniformMatrix4fv(scale_loc, 1, GL_TRUE, utils.gen_scale_matrix(pos[2]*tree_scale, pos[2]*tree_scale, pos[2]*tree_scale))
                    glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))
            else:
                if key == 'wind_rooster' and spinning:
                    obj_array['wind_rooster'].set_rotation(0.0, glfw.get_time(), 0.0)
                    obj_array['wind_rooster'].update_matrices()
                glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))

        glfw.swap_buffers(window)
        #print(camera.get_position())

if __name__ == "__main__":
    main()