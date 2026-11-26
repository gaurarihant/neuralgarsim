# Third-party notices

Parts of this repository are derived from the projects below. Each code project is released under the MIT License, whose text is at the end of this file.

- **NeuralClothSim** (https://github.com/navamikairanda/neuralclothsim). Copyright (c) 2024 Navami Kairanda. The simulator structure, configuration and material handling build on it, and `experiments/belytschko/run_neuralclothsim.py` is the NeuralClothSim baseline.
- **DiffUDF** (https://github.com/LIA-DiTella/DiffUDF). Copyright (c) 2024 Laboratorio de Inteligencia Artificial - UTDT. The reference-geometry network in `src/garment_tdc/models.py`.
- **SIREN** (https://github.com/vsitzmann/siren). Copyright (c) 2020 Vincent Sitzmann. The sine layers in `src/garment_tdc/models.py` and the differential operators in `src/garment_tdc/differential_operators.py`.

The nonlinear material parameters in `material/` (cotton, silk, canvas) come from Clyde et al., "Modeling and Data-Driven Parameter Estimation for Woven Fabrics", SCA 2017.

The sample garment `data/garments/tee.obj` is from the Dataset of 3D Garments with Sewing Patterns by Maria Korosteleva and Sung-Hee Lee (https://doi.org/10.5281/zenodo.5267549), licensed under Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/). Changes: centred, scaled to a unit maximum absolute coordinate, and rotated to Z-up. Vertex order is unchanged.

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
