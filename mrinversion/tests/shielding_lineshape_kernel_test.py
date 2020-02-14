import csdmpy as cp
import numpy as np
from mrsimulator import Dimension
from mrsimulator import Isotopomer
from mrsimulator import Simulator
from mrsimulator import Site
from mrsimulator.methods import one_d_spectrum

from mrinversion.kernel import MAF
from mrinversion.kernel import NuclearShieldingTensor
from mrinversion.kernel import SpinningSidebands
from mrinversion.kernel.lineshape import zeta_eta_to_x_y
from mrinversion.linear_model import TSVDCompression

anisotropic_dimension = [
    cp.Dimension(type="linear", count=96, increment="208.33 Hz", complex_fft=True)
]

inverse_dimension = [
    cp.Dimension(type="linear", count=4, increment="3 kHz"),
    cp.Dimension(type="linear", count=4, increment="3 kHz"),
]


def generate_shielding_kernel(zeta_, eta_, angle, freq, n_sidebands):
    mrsim_dim = Dimension(
        isotope="29Si",
        magnetic_flux_density=9.4,
        number_of_points=96,
        spectral_width=208.33 * 96,
        reference_offset=0,
        rotor_angle=angle,
        rotor_frequency=freq,
    )
    isotopomers = []
    for z, e in zip(zeta_, eta_):
        z_ = z / (mrsim_dim.larmor_frequency / 1e6)
        isotopomers.append(
            Isotopomer(
                sites=[
                    Site(
                        isotope="29Si",
                        isotropic_chemical_shift=0,
                        shielding_symmetric={"zeta": z_, "eta": e},
                    )
                ]
            )
        )
    sim = Simulator()
    sim.isotopomers = isotopomers
    sim.dimensions = [mrsim_dim]
    sim.config.decompose = True
    sim.config.number_of_sidebands = n_sidebands
    _, sim_lineshape = sim.run(method=one_d_spectrum)
    sim_lineshape = np.asarray(sim_lineshape).reshape(4, 4, 96)
    sim_lineshape = sim_lineshape / sim_lineshape[0, 0].sum()
    sim_lineshape[0, :, :] /= 2.0
    sim_lineshape[:, 0, :] /= 2.0
    sim_lineshape.shape = (16, 96)
    return sim_lineshape


def test_zeta_eta_from_x_y():
    ns_obj = NuclearShieldingTensor(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimension,
        isotope="29Si",
        magnetic_flux_density="9.4 T",
        rotor_angle="90 deg",
        rotor_frequency="14 kHz",
        number_of_sidebands=1,
    )

    x = np.arange(4) * 3000
    y = np.arange(4) * 3000
    factor_ = 4 / np.pi
    zeta_ = []
    eta_ = []
    for y_ in y:
        for x_ in x:
            z = np.sqrt(x_ ** 2 + y_ ** 2)
            if x_ <= y_:
                eta_.append(factor_ * np.arctan(x_ / y_))
                zeta_.append(z)
            else:
                eta_.append(factor_ * np.arctan(y_ / x_))
                zeta_.append(-z)
            if x_ == y_ == 0:
                eta_[-1] = 1
    zeta, eta = ns_obj._get_zeta_eta(supersampling=1)

    assert np.allclose(zeta, np.asarray(zeta_))
    assert np.allclose(eta, np.asarray(eta_))


def test_x_y_from_zeta_eta():
    x, y = zeta_eta_to_x_y([10, np.sqrt(2) * 25], [0, 1])
    assert np.allclose(x, [0, 25])
    assert np.allclose(y, [10, 25])


def test_MAF_lineshape_kernel():
    ns_obj = NuclearShieldingTensor(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimension,
        isotope="29Si",
        magnetic_flux_density="9.4 T",
        rotor_angle="90 deg",
        rotor_frequency="14 kHz",
        number_of_sidebands=1,
    )
    zeta, eta = ns_obj._get_zeta_eta(supersampling=1)
    K = ns_obj.kernel(supersampling=1)
    sim_lineshape = generate_shielding_kernel(zeta, eta, np.pi / 2, 14000, 1).T
    assert np.allclose(K, sim_lineshape, rtol=1.0e-3, atol=1e-3)

    ns_obj = MAF(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimension,
        isotope="29Si",
        magnetic_flux_density="9.4 T",
    )
    zeta, eta = ns_obj._get_zeta_eta(supersampling=1)
    K = ns_obj.kernel(supersampling=1)
    sim_lineshape = generate_shielding_kernel(zeta, eta, np.pi / 2, 14000, 1).T

    assert np.allclose(K, sim_lineshape, rtol=1.0e-3, atol=1e-3)

    _ = TSVDCompression(K, s=np.arange(96))
    assert _.truncation_index == 15


def test_spinning_sidebands_kernel():
    ns_obj = NuclearShieldingTensor(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimension,
        isotope="29Si",
        magnetic_flux_density="9.4 T",
        rotor_angle="54.735 deg",
        rotor_frequency="100 Hz",
        number_of_sidebands=96,
    )
    zeta, eta = ns_obj._get_zeta_eta(supersampling=1)
    K = ns_obj.kernel(supersampling=1)
    sim_lineshape = generate_shielding_kernel(zeta, eta, 0.9553059660790962, 100, 96).T

    assert np.allclose(K, sim_lineshape, rtol=1.0e-3, atol=1e-3)

    ns_obj = SpinningSidebands(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimension,
        isotope="29Si",
        magnetic_flux_density="9.4 T",
    )
    zeta, eta = ns_obj._get_zeta_eta(supersampling=1)
    K = ns_obj.kernel(supersampling=1)
    sim_lineshape = generate_shielding_kernel(
        zeta, eta, 0.9553059660790962, 208.33, 96
    ).T

    assert np.allclose(K, sim_lineshape, rtol=1.0e-3, atol=1e-3)

    _ = TSVDCompression(K, s=np.arange(96))
    assert _.truncation_index == 15
