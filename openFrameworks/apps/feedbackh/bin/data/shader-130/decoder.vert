#version 130

void main() {
    // get the homogeneous 2d position
    gl_Position = ftransform();
}
