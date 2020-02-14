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


exp1 = os.path.join(path, "Na2O4p7SiO2/Na2O.4.7SiO2.csdf.temp")
