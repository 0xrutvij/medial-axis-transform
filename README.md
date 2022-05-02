# Medial Axis Transform

Medial axis transform using Delaunay triangulations


## Environment Setup

- Windows

  Note: Replace `<PythonPath>` with path to the Python>=3.8 interpreter in your
  system.

  ```powershell

  pip install virtualenv

  virtualenv --python <PythonPath> computational_geometry

  .\computational_geometry\Scripts\activate

  pip install -r requirements.txt

  ```

- *nix

  ```sh

  pip3 install virtualenv

  virtualenv --python $(which python3) computational_geometry

  ./computational_geometry/bin/activate

  pip install -r requirements.txt

  ```

### Running The Program

```
python3 mat_example.py --file shapes/horse-1.txt
```
