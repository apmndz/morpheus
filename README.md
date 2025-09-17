# ✈️ Morpheus

Morpheus is an **aero design automation tool** that transforms a single multi-element airfoil into **thousands of unique, valid geometries**.  
With only a starting **STL file**, Morpheus parameterizes and exports designs, making it easy to explore vast aerodynamic design spaces.

<p align="center">
  <img src="https://github.com/user-attachments/assets/56f02c4e-f1c2-4c44-bea5-abb7ef59555e" width="350"/>
  <img src="https://github.com/user-attachments/assets/2f66afee-a0da-4ed0-a742-0c54dc2b7581" width="350"/>
  <img src="https://github.com/user-attachments/assets/06a09dda-62ba-41a6-95c2-95e82548eb65" width="350"/>
  <img src="https://github.com/user-attachments/assets/bf342105-dd8a-4c11-b028-af300ce1a35a" width="350"/>
</p>

---

## 🔧 Features
- **Airfoil parametrization**: Rotate, translate, and reshape slats and flaps with flexible bounds.  
- **Automated geometry generation**: Create hundreds or thousands of STLs with a single command.  
- **Customizable constraints**: Define chord lengths, element gaps, and transformation ranges.  
- **Visualization on the fly**: Preview selected geometries during batch generation.  

---

## 📐 Airfoil Parametrization

Each aerodynamic element can be transformed according to well-defined rules:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c8a59cc4-1158-41b3-818c-b087c70549e1" width="800"/>
</p>

- **Slat rotation**: Rotate about its trailing edge; angle defined between the horizontal axis and slat chord line.  
- **Slat translation**: Move along both **x** and **y** axes.  
- **Slat overhang**: Minimum horizontal distance between the slat and the main body’s leading edge (positive or negative).  
- **Flap rotation**: Rotate about its leading edge; angle defined between the horizontal axis and flap chord line.  
- **Flap translation**: Move along both **x** and **y** axes.  
- **Flap overhang**: Minimum horizontal distance between the flap and the trailing edge of the main body.  
- **Gap width**: Minimum distance allowed between adjacent elements.  
- All ranges are fully configurable in [`config.yaml`](./config.yaml).  

---

## 📦 Requirements
Morpheus depends on the following Python libraries:
- `numpy`  
- `matplotlib`  
- `pygeo`  
- `pyspline`  
- `trimesh`  
- `pyyaml`  

To install them all at once, run:
```bash
pip install -r requirements.txt
