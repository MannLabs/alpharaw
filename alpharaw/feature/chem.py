from __future__ import annotations

import numpy as np
from numba import float32, float64, int32, njit, types
from numba.experimental import jitclass
from numba.typed import Dict

# Some legacy chemistry functions
# Maybe replacd with alphabase?


averagine_aa = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

averagine_aa["C"] = 4.9384
averagine_aa["H"] = 7.7583
averagine_aa["N"] = 1.3577
averagine_aa["O"] = 1.4773
averagine_aa["S"] = 0.0417

averagine_avg = 111.1254

DELTA_M = 1.00286864
DELTA_S = 0.0109135

M_PROTON = 1.00727646687

maximum_offset = DELTA_M + DELTA_S


spec = [
    ("m0", float32),
    ("dm", int32),
    ("intensities", float32[:]),
]


@jitclass(spec)
class Isotope:
    """
    Jit-compatible class to store isotopes

    Attributes:
        m0 (int): Mass of pattern
        dm0 (int): dm of pattern (number of isotopes)
        int0 (np.float32[:]): Intensities of pattern
    """

    def __init__(self, m0: int, dm: int, intensities: np.ndarray):
        self.m0 = m0
        self.dm = dm
        self.intensities = intensities


isotopes = Dict.empty(
    key_type=types.unicode_type, value_type=Isotope.class_type.instance_type
)

isotopes["C"] = Isotope(12, 3, np.array([0.9893, 0.0107, 0.0], dtype=np.float32))
isotopes["H"] = Isotope(
    1.007940, 3, np.array([0.999885, 0.000115, 0.0], dtype=np.float32)
)
isotopes["O"] = Isotope(
    15.9949146221, 3, np.array([0.99757, 0.00038, 0.00205], dtype=np.float32)
)
isotopes["N"] = Isotope(
    14.0030740052, 2, np.array([0.99636, 0.00364], dtype=np.float32)
)
isotopes["S"] = Isotope(
    31.97207069, 4, np.array([0.9499, 0.0075, 0.0425, 0.0001], dtype=np.float32)
)

isotopes["I"] = Isotope(126.904473, 1, np.array([1], dtype=np.float32))
isotopes["K"] = Isotope(
    38.9637069, 3, np.array([0.932581, 0.000117, 0.067302], dtype=np.float32)
)


def test_isotope():
    assert isotopes["C"].m0 == 12
    assert isotopes["C"].dm == 3
    assert np.allclose(isotopes["C"].intensities[0], 0.9893)
    assert np.allclose(isotopes["C"].intensities[1], 0.0107)
    assert np.allclose(isotopes["C"].intensities[2], 0)


# test_isotope()


spec = [
    ("m0", float32),
    ("dm", int32),
    ("intensities", float64[:]),
]


# This class and the methods fast_add, numba_bin are tested in the wrapper function dict_to_dist
@jitclass(spec)
class IsotopeDistribution:
    """Class to represent isotope distributions.

    Attributes:
        m0 (int): the mono-isotopic mass.
        dm (int): number of isotopes.
        intensities (np.ndarray): isotope intensities.

    Methods:
        add: add another isotope distribution.
        copy: create a copy of the current isotope distribution.
        mult: multiply the current isotope distribution.

    """

    def __init__(self):
        self.m0 = 0
        self.dm = 1
        self.intensities = np.array([1.0])

    def add(self, x: IsotopeDistribution):
        """Add another isotope distribution.

        Args:
            x (IsotopeDistribution): IsotopeDistribution.
        """
        self.m0, self.dm, self.intensities = fast_add(
            self.m0, self.dm, self.intensities, x.m0, x.dm, x.intensities
        )

    def copy(self) -> IsotopeDistribution:
        """Copies the current isotope distribution.

        Returns:
            IsotopeDistribution: A copy of the current isotope distribution.
        """
        i = IsotopeDistribution()
        i.m0 = self.m0
        i.dm = self.dm
        i.intensities = self.intensities

        return i

    def mult(self, n: int) -> IsotopeDistribution:
        """Multiplies the current isotope distribution.

        Args:
            n (int): Multiplication factor.

        Returns:
            IsotopeDistribution: A multiplied version of the current isotope distribution.
        """
        binary = numba_bin(n)

        if n == 1:
            return self.copy()
        else:
            i = IsotopeDistribution()

            multiples = self.copy()

            for count in binary[::-1]:
                if count == 1:
                    i.add(multiples)
                multiples.add(multiples)

            return i


@njit
def fast_add(
    m0: int,
    dm0: int,
    int0: np.ndarray,
    m1: int,
    dm1: int,
    int1: np.ndarray,
    prune_level: float = 0.000001,
) -> (int, int, np.ndarray):
    """Helper function to quickly add isotope patterns.

    Args:
        m0 (float): the mono-isotopic mass of the first isotope distribution.
        dm0 (int): the number of isotope intensities in the first isotope distribution.
        int0 (np.ndarray:float): the intensity pattern of the first isotope distribution.
        m1 (float): the mono-isotopic mass of the second isotope distribution.
        dm1 (int): the number of isotope intensities in the second isotope distribution.
        int1 (np.ndarray:float): the intensity pattern of the second isotope distribution.
        prune_level (float): Precision threshold. Defaults to 0.000001.


    Returns:
        int: Mass of new pattern.
        int: Number of isotopes in new pattern.
        np.ndarray: Intensity of new pattern.

    """
    m0 += m1

    ni = np.zeros(dm0 + dm1 - 1)
    for i in range(dm0):
        for j in range(dm1):
            ni[i + j] += int0[i] * int1[j]

    dm0 += dm1 - 1

    int0 = ni / np.max(ni)

    while ni[dm0 - 1] < prune_level:
        dm0 -= 1

    return m0, dm0, int0


@njit
def numba_bin(decimal: int) -> list:
    """Numba compatible function to convert a decimal number to a binary (list).

    Args:
        decimal (int): Decimal number.

    Returns:
        list: Number in binary.
    """

    binary = []

    while decimal != 0:
        bit = int(decimal % 2)
        binary.insert(0, bit)
        decimal = int(decimal / 2)

    return binary


@njit
def dict_to_dist(counted_AA: Dict, isotopes: Dict) -> IsotopeDistribution:
    """Function to convert a dictionary with counts of atoms to an isotope distribution.

    Args:
        counted_AA (Dict): Numba-typed dict with counts of atoms.
        isotopes (Dict): Numba-typed lookup dict with isotopes.

    Returns:
        IsotopeDistribution: The calculated isotope distribution for the chemical compound.
    """

    dist = IsotopeDistribution()
    for AA in counted_AA:
        x = IsotopeDistribution()
        x.add(isotopes[AA])
        x = x.mult(counted_AA[AA])

        dist.add(x)

    return dist


def test_dict_to_dist():
    abundances = (
        np.array(
            [
                58.83,
                0.18,
                100,
                0.29,
                81.29,
                0.22,
                42.05,
                0.11,
                15.54,
                0.04,
                4.36,
                0.01,
                0.97,
                0.17,
                0.03,
            ]
        )
        / 100
    )

    counted_AA = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    counted_AA["K"] = 23
    counted_AA["I"] = 22
    dist = dict_to_dist(counted_AA, isotopes)

    rmse = np.mean(np.sqrt((dist.intensities[: len(abundances)] - abundances) ** 2))

    assert rmse < 0.05


# test_dict_to_dist()


@njit
def get_average_formula(
    molecule_mass: float, averagine_aa: Dict, isotopes: Dict, sulphur: bool = True
) -> Dict:
    """Function to calculate the averagine formula for a molecule mass.

    Args:
        molecule_mass (float): Input molecule mass for which the averagine model should be calculated.
        averagine_aa (Dict): Numba-typed dictionary with averagine masses. See alphapept.constants.averagine_aa.
        isotopes (Dict): Numba-typed lookup dict with isotopes.
        sulphur (bool, optional): Flag to consider sulphur. Defaults to True.

    Raises:
        NotImplementedError: If mode w/o sulphur is selected.

    Returns:
        Dict: Numba-typed dict with averagine composition.
    """

    if sulphur:
        averagine_units = molecule_mass / averagine_avg
    else:
        raise NotImplementedError("Mode w/o sulphur is not implemented yet")

    counted_AA = Dict.empty(key_type=types.unicode_type, value_type=types.int64)

    final_mass = 0

    # Calculate integral mnumbers of atoms
    for AA in averagine_aa:
        counted_AA[AA] = int(np.round(averagine_units * averagine_aa[AA]))
        final_mass += counted_AA[AA] * isotopes[AA].m0

    # Correct with H atoms
    h_correction = int(np.round((molecule_mass - final_mass) / isotopes["H"].m0))
    counted_AA["H"] += h_correction

    return counted_AA


def test_get_average_formula():
    for molecule_mass in [100, 200, 300, 400, 500]:
        average_formula = get_average_formula(
            molecule_mass, averagine_aa, isotopes, sulphur=True
        )
        mass = np.sum([average_formula[AA] * isotopes[AA].m0 for AA in average_formula])
        assert np.abs(mass - molecule_mass) < isotopes["H"].m0


# test_get_average_formula()

ISOTOPE_MASS = DELTA_M


@njit
def mass_to_dist(
    molecule_mass: float, averagine_aa: Dict, isotopes: Dict
) -> (np.ndarray, np.ndarray):
    """Function to calculate an isotope distribution from a molecule mass using the averagine model.

    Args:
        molecule_mass (float, averagine_aa): input molecule mass.
        averagine_aa (Dict): Numba-typed dictionary with averagine masses.
        isotopes (Dict): Numba-typed lookup dict with isotopes.

    Returns:
        np.ndarray: isotope masses.
        np.ndarray: isotope intensity.
    """
    counted_AA = get_average_formula(molecule_mass, averagine_aa, isotopes)

    dist = dict_to_dist(counted_AA, isotopes)

    masses = np.array(
        [dist.m0 + i * ISOTOPE_MASS for i in range(len(dist.intensities))]
    )
    ints = dist.intensities

    return masses, ints


# print(mass_to_dist(300, averagine_aa, isotopes))
