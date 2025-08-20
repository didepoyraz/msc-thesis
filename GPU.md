## OpenGL 

### libs
```bash
sudo apt install libglew-dev
sudo apt install libgl1-mesa-dev
sudo apt install libglfw3-dev
```

### compile
```bash
g++ opengl.cpp -o opengl -lGL -lGLEW -lglfw
```
**Note:** Shader files `vertex_shader.glsl` and `MatrixMultiplicationShader.glsl` are in the same directory as your program.

### run
```bash
./opengl N
```
- **N**: matrix size
- Currently, **Maximum N** is 1024. For larger matrix sizes, we will need to adjust the window width/height

- **Results (ms):** <br>
N = [16    32    64   128   256  512    1024]<br>
T = [0.66  0.61  0.7  1.7  9.5  71.7  570]

## Vulkan

### Vulkan libs and driver
```bash
sudo apt install mesa-vulkan-drivers libvulkan-dev
```
Try to install directly, or refer to [https://www.ravbug.com/tutorials/rpi-vulkan/](https://www.ravbug.com/tutorials/rpi-vulkan/) (incluidng examples)


### compile
```bash
sudo apt install ninja-build cmake -y
cd vulkan && mkdir build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
(ninja clean)
```
- **NOTE**: Before running the matmul in `bin`, we need to compile the shader

### Compile shader
- go to the folders `shaders/glsl`
- `matmul.comp`: a headless compute shader
- use `compileshaders.py` to compile `matmul.comp` -> `matmul.comp.spv`

```bash
sudo apt install glslang-tools
python compileshaders.py --glslang /usr/bin/glslangValidator
```

### run
```bash
./matmul N TILE
```
- **N**: matrix size
- **TILE**: tile size

### metric gathering
```bash
python3 metric_gathering.py
```