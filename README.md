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

```sh
usage: mat_example.py [-h] [-f FILE] (-s | -m)

A script to extract the MAT from a shapes point set.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  file containing points as x-y pairs (format same as
                        that for files in shapes folder)
  -s, --skeleton        Display the Medial Axis Skeleton of the shape in given
                        input file.
  -m, --shape-matching  Find the best match shape for the shape in given input
                        file.
```

```
python3 mat_example.py --file shapes/horse-1.txt
```
