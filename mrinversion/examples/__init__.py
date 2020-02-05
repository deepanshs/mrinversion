# coding: utf-8
import os

path = os.path.split(__file__)[0]

sideband01 = os.path.join(path, "1/PASS_spectrum.csdf.temp")
MAF01 = os.path.join(path, "1/MAF_spectrum.csdf.temp")
true_distribution01 = os.path.join(path, "1/pdf_simulation.csdf.temp")

sideband02 = os.path.join(path, "2/PASS_spectrum.csdf.temp")
MAF02 = os.path.join(path, "2/MAF_spectrum.csdf.temp")
true_distribution02 = os.path.join(path, "2/pdf_simulation.csdf.temp")
