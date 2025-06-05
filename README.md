<p align="center">
<h1 align="center"><strong>🤖 SLAM and Path Planning</strong></h1>
  <p align="center">
    <a href='mailto:tingde.liu@gmail.com' target='_blank'>Tingde Liu</a><br>
    Supervised by Prof. Dr.-Ing. Claus Brenner<br>
    Leibniz Universität Hannover
  </p>
</p>

---

## 📌 Overview

This repository contains code and conceptual exercises developed as part of the **SLAM and Path Planning** course at Leibniz University Hannover. The exercises cover the full SLAM pipeline, from motion modeling and probabilistic filtering to localization, mapping, and path planning.

All implementations are written in Python and are structured to follow the weekly course topics.

---

## 📚 Course Units

### 🅰️ Unit A – Motion Model

- Motor ticks measurement
- LIDAR measurements and observation models

### 🅱️ Unit B – Landmark-Based Localization

- Feature-based position correction
- Similarity transform
- Limitations and weaknesses of naive correction

### 🅲 Unit C – Filtering Basics

- Systematic and random errors
- Discrete probability distributions
- Error propagation problems
- Convolution of distributions
- Measurement uncertainty and sensor noise
- Discrete Bayes filter / Histogram filter
- Gaussian bell curve & normal distribution
- Kalman Filter fundamentals

### 🅳 Unit D – Extended Kalman Filter

- EKF derivation and implementation
- Linearization of non-linear systems

### 🅴 Unit E – Localization

- Resampling techniques
- Density estimation
- Probabilistic robot positioning

### 🅵 Unit F – SLAM (Simultaneous Localization and Mapping)

- Difference between Full SLAM and Online SLAM
- EKF-SLAM approach

### 🅶 Unit G – Fast SLAM

- Particle Filter SLAM algorithm
- Efficiency improvements and drawbacks

### 🅷 Unit H – Path Planning

- Dijkstra’s Algorithm and optimization
- Path generation and smoothing
- Limitations of Dijkstra
- Greedy Algorithm and A\* (A-star) search
- Modifications: potential fields
- Practical considerations: curvature and turning radius

---

## 🛠️ Requirements

- Python 3.8+
- numpy
- matplotlib
- (optional) jupyter, scipy

Install via:

```bash
pip install numpy matplotlib

