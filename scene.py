import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

import numpy as np
from PIL import Image

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
    obj_array = {}
    obj_array["cottage"] = utils.RenderableObject("cottage", [0.0, 0.0, 0.0], "models/cottage/cottage_obj.obj", "models/cottage/textures/cottage_diffuse.png")
    obj_array["nightstand"] = utils.RenderableObject("nightstand", [0.0, 0.0, 0.0], "models/Nightstand_obj/Nightstand.obj", "models/Nightstand_obj/Wood1_Albedo.png")
    obj_array["little_ghost"] = utils.RenderableObject("little_ghost", [0.0, 0.0, 0.0], "models/little_gost.obj", "models/ghost.png")
    #obj = utils.RenderableObject("skybox", [0.0, 0.0, 0.0], "models/skybox.obj", "models/Nightstand_obj/Wood1_Albedo.png")
    camera = utils.Camera(position=[0.0, 0.4, -1.0])

    fov = 60  # Field of view in degrees
    aspect_ratio = 16 / 9  # Aspect ratio of the viewport
    near = 0.1  # Near clipping plane
    far = 100.0  # Far clipping plane

    projection_matrix = perspective_projection(fov, aspect_ratio, near, far)

    #att: modificando as propriedades de um objeto
    obj_array["nightstand"].set_pos(2.0, 0.5, 0.0)
    obj_array["nightstand"].set_scale(1.5)
    obj_array["nightstand"].update_matrices()

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
        global polygon
        w, a, s, d, j, k, q, e, r, f, p = 87, 65, 83, 68, 74, 75, 81, 69, 82, 70, 80
        speed_inc = 0.02
        max_speed = 0.3
        #print(key)
        if (action == 1 or action == 2):
            if key == a:
                camera.move_camera(camera.linear_speed)
            if key == d:
                camera.move_camera(-camera.linear_speed)
            if key == w:
                camera.move_camera(z=camera.linear_speed)
            if key == s:
                camera.move_camera(z=-camera.linear_speed)
            if key == j:
                camera.move_camera(d_z=(camera.angular_speed))
            if key == k:
                camera.move_camera(d_z=(-camera.angular_speed))
            if key == q:
                camera.move_camera(y = -camera.linear_speed)
            if key == e:
                camera.move_camera(y = camera.linear_speed)
            if key == r:
                camera.linear_speed += speed_inc
            if key == f:
                camera.linear_speed -= speed_inc
            if key == p:
                polygon = not polygon
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

            #glDrawElements(GL_TRIANGLES, len(loader.vertex_index), GL_UNSIGNED_INT, None)
            glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))

        glfw.swap_buffers(window)

if __name__ == "__main__":
    main()