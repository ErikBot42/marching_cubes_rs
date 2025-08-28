# Marching Cubes in Rust + WGSL
![image](https://github.com/user-attachments/assets/31db73f9-d02b-46ce-b2f1-7d3c04d15f36)

Marching Cubes implementation and rendering where all compute and culling is done on the GPU.

The space is divided into a grid of chunks, containing 32x32x32 cubes. In each vertex of all the cubes, we need some metric for how much inside the volume a given point is. Here, the mandelbox fractal as an sdf is used. Then, for each cube, the number of triangles that will be generated is computed. A parallel prefix sum over that per cube number gives both the total number of triangles that will be generated and what their relative memory locations are. Then triangles are written to the computed locations.

Typically in computer graphics, triangles are represented using 9 32-bit floating point numbers plus material information. Instead this information is compressed into a single 32-bit number, because we only care about the position within a chunk. This reduces the per-triangle memory bandwidth significantly.

Culling is done by another kernel that compares the bounding box of each chunk with the view frustum, discarding any chunks that are too far away or not visible in a given direction. To avoid having any per-chunk data, the position of the chunk is encoded in the vertex number in the draw call.
With this setup, a single indirect draw call can render all chunks.
