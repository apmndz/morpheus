<h1>Morpheus</h1>

<p>An aero design tool that can turn one multi-element airfoil into thousands of unique designs. <strong>All you need</strong> is a starting <strong>STL</strong> file, and Morpheus will do the rest!</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/56f02c4e-f1c2-4c44-bea5-abb7ef59555e" width="350"/>
  <img src="https://github.com/user-attachments/assets/2f66afee-a0da-4ed0-a742-0c54dc2b7581" width="350"/>
  <img src="https://github.com/user-attachments/assets/06a09dda-62ba-41a6-95c2-95e82548eb65" width="350"/>
  <img src="https://github.com/user-attachments/assets/bf342105-dd8a-4c11-b028-af300ce1a35a" width="350"/>
</p>

<h2>Airfoil Parametrization</h2>

This diagram illustrates the transformations applied to each aerodynamic element:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c8a59cc4-1158-41b3-818c-b087c70549e1" width="800"/>
</p>

- The **slat can be rotated** about its **trailing edge**; the angle is defined as being between the horizontal and the slat’s chord line.
- The **slat can be translated** in the **x** and **y** directions.
- The **slat overhang** is the minimum **horizontal** distance between the slat and the **leading edge** of the main body. Overhang can be **positive or negative**.
- The **flap can be rotated** about its **leading edge**; the angle is defined as being between the horizontal and the flap’s chord line.
- The **flap can be translated** in the **x** and **y** directions.
- The **flap overhang** is the minimum **horizontal** distance between the flap and the **trailing edge** of the main body.
- The **gap width** is the minimum distance that can exist between the main body and the other elements.
- A **desired range** of each of these parameters can be set by the user in input/config.py.
