#ifndef _TESTEXAMPLE_H_
#define _TESTEXAMPLE_H_

#include "VectorFunction.h"
#include "VelocityField.h"

class TestIT {
  using Point = Vec<Real, 2>;

  template <class T>
  using Vector = std::vector<T>;

public:
  Point center;                 // center of initial circle
  Real radio;                   // radio of initial circle
  Real dt;                      // time step
  Real rtiny;                   // rtiny of MARS
  Real T;                       // total time
  VectorFunction<2> *velocity;  // velocity fiel
  int n;  // cut the initial circle to n equal parts to generate the initial
          // YinSet
  int opstride;      // out put stride, for plot
  std::string name;  // file name of the out put target, for plot

  TestIT(Point _center,
         Real _radio,
         int _n,
         Real _dt,
         Real _T,
         VectorFunction<2> *_v,
         std::string _name,
         int _opstride,
         Real _rtiny = 0.1) :
      center(_center),
      radio(_radio),
      dt(_dt),
      rtiny(_rtiny),
      T(_T),
      velocity(_v),
      n(_n),
      opstride(_opstride),
      name(_name){};
};

// two velocity fields
VectorFunction<2> *vortex = new Vortex(8);
VectorFunction<2> *deformation = new Deformation(2);
VectorFunction<2> *translation = new Translation(1, 1);
VectorFunction<2> *rotation = new Rotation(0, 0, M_PI);
VectorFunction<2> *revrotation = new RevRotation(0, 0, M_PI, 2);
VectorFunction<2> *circleshrink = new CircleShrink(1);
VectorFunction<2> *squareshrink = new SquareShrink();

TestIT getTest(int i) {
  Vector<TestIT> test;

  // test 0
  // Vortex(8)
  // n = 64; dt = 0.04; rtiny = 0.01
  // ERK
  // 4.80e-5  4.25  2.52e-6  4.96  8.08e-8  7.41  4.77e-10  4.09  2.80e-11(exact)
  // time_IML  2.83e-01  1.92  1.07e+00  1.95  4.14e+00  1.99  1.64e+01  2.02  6.65e+01
  // time_vect 2.91e-01  1.87  1.06e+00  1.95  4.11e+00  1.99  1.64e+01  2.02  6.63e+01
  // time_list 3.13e-01  1.87  1.15e+00  1.96  4.45e+00  1.98  1.76e+01  2.01  7.11e+01

  // ESDIRK4
  // 4.64e-5  4.23  2.47e-6  4.96  7.92e-8  7.53  4.29e-10  4.01  2.67e-11
  // time_IML: 2.00e+00  1.76e+00  6.74e+00  1.83e+00  2.39e+01  1.73e+00  7.94e+01
  // 1.92e+00  3.01e+02
  test.push_back(TestIT(
      Point{0.5, 0.75}, 0.15, 64, 0.04, 8, vortex, "Vortex", 100, 0.01));

  // test 1
  // Deformation(2, 4)
  // n = 128; dt = 0.01; rtiny = 0.01
  // 2.44e-5 3.77 1.79e-6 5.23 4.76e-8(richardson)
  // 2.74e-5 3.81 1.95e-6 5.23 5.22e-8(exact)
  // time_list 1.73e+1 2.36e+2 3.58e+3
  // time_vect 1.75e+1 2.43e+2 3.77e+3
  test.push_back(TestIT(Point{0.5, 0.5},
                        0.15,
                        128,
                        0.01,
                        2,
                        deformation,
                        "Deformation",
                        40,
                        0.01));

  // test 2
  // Deformation(2, 4)
  // n = 128; dt = 0.01; rtiny = 0.1
  // 3.60e-5 4.61 1.47e-6 5.40 3.49e-8(exact)
  // time_list 1.43e+1 1.93e+2 3.99e+3
  // time_vect 1.43e+1 2.01e+2 4.05e+3
  test.push_back(TestIT(
      Point{0.5, 0.5}, 0.15, 128, 0.01, 2, deformation, "Deformation", 40));

  // test 3
  // Deformation(2, 4) plot use
  test.push_back(TestIT(
      Point{0.5, 0.5}, 0.15, 256, 0.005, 2, deformation, "Deformation", 50));

  // test 4
  // Vortex(8) plot use
  test.push_back(TestIT(
      Point{0.5, 0.75}, 0.15, 128, 0.01, 8, vortex, "Vortex", 100, 0.01));

  // test 5
  // Deformation(2, 4)
  // n = 128; dt = 0.04; rtiny = 0.01
  // 4.93e-5  4.41  2.31e-6  5.10  6.77e-8  6.21  9.12e-10
  // 4.55e-5  4.66  1.80e-6  5.16  5.06e-8  5.85  8.74e-10 (coarse to fine)
  // time_list 3.7478e+00  5.4055e+01  7.8974e+02  1.2461e+04
  // time_vect 3.9465e+00  5.6405e+01  8.6824e+02  1.2688e+04
  test.push_back(TestIT(Point{0.5, 0.5},
                        0.15,
                        128,
                        0.04,
                        2,
                        deformation,
                        "Deformation",
                        40,
                        0.01));

  // test 6
  // Deformation(2, 4)
  // n = 128; dt = 0.02; rtiny = 0.01
  // ERK4
  // 2.43e-5  3.67  1.91e-6  5.20  5.19e-8  6.56  5.50e-10  4.17  3.07e-11
  // time_IML  3.04e-1  1.97  1.20e+0  2.01  4.81e+0  2.00  1.92e+1  1.84  6.90e+1
  // time_vect 3.01e-1  1.99  1.19e+0  2.01  4.81e+0  2.00  1.93e+1  1.84  6.92e+1
  // time_list 3.26e-1  1.97  1.28e+0  2.01  5.13e+0  1.99  2.04e+1  1.84  7.29e+1

  // ESDIRK4
  // 2.31e-05  3.60e+00  1.91e-06
  // time_IML(LAPACKE) 1.84e+03  3.80e+00  2.57e+04
  // time_IML(EigSpLU) 1.50e+04  2.76e+00  1.02e+05
  test.push_back(TestIT(Point{0.5, 0.5},
                        0.15,
                        128,
                        0.02,
                        2,
                        deformation,
                        "Deformation",
                        40,
                        0.01));

  // test 7
  // Deformation(2, 4)
  // n = 128; dt = 0.02; rtiny = 0.4
  // 2.42e-05  3.66  1.91e-06  5.20  5.19e-08  6.56  5.49e-10
  // time_IML  4.29e-01  1.94  1.65e+00  1.99  6.54e+00  1.88  2.41e+01
  // time_vect 4.15e-01  1.97  1.63e+00  1.98  6.46e+00  1.90  2.42e+01
  // time_list 4.55e-01  1.95  1.76e+00  1.98  6.93e+00  1.88  2.55e+01
  test.push_back(TestIT(Point{0.5, 0.5},
                        0.15,
                        128,
                        0.02,
                        2,
                        deformation,
                        "Deformation",
                        40,
                        0.4));

  // test 8
  // Rotaion
  // n = 32; dt = 0.1; rtiny = 0.1
  test.push_back(
      TestIT(Point{0.5, 0}, 0.5, 32, 0.1, 2, rotation, "Rotation", 40, 0.1));

  // test 9
  // RevRotation
  // n = 16; dt = 0.1; rtiny = 0.1
  // 5.54e-05  4.06e+00  3.32e-06  4.02e+00  2.05e-07  4.01e+00  1.27e-08
  // time_IMV(EigSpLU) 1.55e+00  2.55e+00  9.06e+00  2.24e+00  4.27e+01  2.78e+00
  // 2.92e+02
  test.push_back(TestIT(
      Point{0.5, 0}, 0.5, 16, 0.1, 2, revrotation, "RevRotation", 40, 0.1));

  // test 10
  // CircleShrink
  // n = 32; dt = 0.05; rtiny = 0.3
  test.push_back(TestIT(
      Point{0, 0}, 1, 32, 0.05, 0.5, circleshrink, "CircleShrink", 40, 0.3));

  return test[i];
}

#endif