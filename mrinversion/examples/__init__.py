# coding: utf-8
import os

path = os.path.split(__file__)[0]

sideband01 = os.path.join(path, "1/PASS_spectrum.csdf.temp")
MAF01 = os.path.join(path, "1/MAF_spectrum.csdf.temp")
true_distribution01 = os.path.join(path, "1/pdf_simulation.csdf.temp")

sideband02 = os.path.join(path, "2/PASS_spectrum.csdf.temp")
MAF02 = os.path.join(path, "2/MAF_spectrum.csdf.temp")
true_distribution02 = os.path.join(path, "2/pdf_simulation.csdf.temp")

sideband03 = os.path.join(path, "bimodal/PASS_spectrum.csdf.temp")
MAF03 = os.path.join(path, "bimodal/MAF_spectrum.csdf.temp")
true_distribution03 = os.path.join(path, "bimodal/pdf_simulation.csdf.temp")


exp1 = os.path.join(path, "exp/Na2O.4.7SiO2.csdf")
exp2 = os.path.join(path, "exp/Rb2O.2.25SiO2.csdf")
exp3 = os.path.join(path, "exp/Cs2O.4.72SiO2.csdf")
exp4 = os.path.join(path, "exp/mat_790Hz_KMg0.5.4SiO2.csdf")
exp5 = os.path.join(path, "exp/2Na2O.3SiO2.csdf")
exp6 = os.path.join(path, "exp/pass_K1.5Mg0.25O.4SiO2.csdf")
exp7 = os.path.join(path, "exp/MAF_K1.5M0.25_4SiO2.csdf")
