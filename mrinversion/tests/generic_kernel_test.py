import csdmpy as cp
import pytest

from mrinversion.kernel import NuclearShieldingTensor


def test_number_of_dimensions_for_lineshape_kernel():
    kernel_dimensions = [
        cp.Dimension(type="linear", count=96, increment="208.33 Hz", complex_fft=True)
    ]

    inverse_dimension = [
        cp.Dimension(type="linear", count=25, increment="370 Hz"),
        cp.Dimension(type="linear", count=25, increment="370 Hz"),
    ]

    error = r"Exactly 2 inverse dimension\(s\) is/are required for the"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=inverse_dimension[0],
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=[inverse_dimension[0]],
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    error = r"Exactly 1 direct dimension\(s\) is/are required for the"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=inverse_dimension,
            inverse_dimension=inverse_dimension,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    kernel_dimension__ = {}
    error = r"The value of the `kernel_dimension` attribute must be one of "
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimension__,
            inverse_dimension=inverse_dimension,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    inverse_dimension = ["", ""]
    error = "The element at index 0 of the `inverse_dimension` list must be an"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=inverse_dimension,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    inverse_kernel_dimension__ = [
        {"type": "linear", "count": 10, "increment": "1 Hz"},
        "string",
    ]
    error = "The element at index 0 of the `inverse_dimension` list must be an"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=inverse_kernel_dimension__,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )


def test_dimensionality_of_lineshape_kernel():
    kernel_dimensions = [
        cp.Dimension(type="linear", count=96, increment="208.33 Hz", complex_fft=True)
    ]

    inverse_dimension = [
        cp.Dimension(type="linear", count=25, increment="370 Hz"),
        cp.Dimension(type="linear", count=25, increment="370 m"),
    ]

    error = r"dimension with quantity name `\['frequency'\]` is required for the"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=inverse_dimension,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )

    kernel_dimensions = cp.Dimension(
        type="linear", count=96, increment="208.33 ms", complex_fft=True
    )
    error = r"dimension with quantity name `\['frequency'\]` is required for the"
    with pytest.raises(ValueError, match=".*{0}.*".format(error)):
        NuclearShieldingTensor(
            anisotropic_dimension=kernel_dimensions,
            inverse_dimension=inverse_dimension,
            isotope="29Si",
            magnetic_flux_density="9.4 T",
            rotor_angle="90 deg",
            rotor_frequency="14 kHz",
            number_of_sidebands=1,
        )
