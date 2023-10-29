# Building dependencies
1. create a python environment to run scripts.
    ```
    conda env create -f conf/env.yaml
    ```
2. openMVG and openMVS: Follow the building tutorial of [openMVG](https://github.com/openMVG/openMVG/blob/develop/BUILD.md) and [openMVS](https://github.com/cdcseacave/openMVS/blob/master/BUILD.md) to build these two libraries using our provided packages in this [link]().
3. pcl and vtk: Follow the building tutorial of [pcl](https://github.com/PointCloudLibrary/pcl) and VTK to build the libraries.
4. Update the paths in the file `scripts/path.py`. 
