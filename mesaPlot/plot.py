# Copyright (c) 2017, Robert Farmer r.j.farmer@uva.nl

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import bisect
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import matplotlib.patheffects as path_effects
import os
import random
from io import BytesIO
from cycler import cycler
from scipy.interpolate import interp1d
from distutils.spawn import find_executable
import numpy.ma as ma


class plot(object):
    def __init__(self, rcparams_fixed=True):
        self.colors = {
            "clr_Black": [0.0, 0.0, 0.0],
            "clr_Blue": [0.0, 0.0, 1.0],
            "clr_BrightBlue": [0.0, 0.4, 1.0],
            "clr_LightSkyBlue": [0.53, 0.808, 0.98],
            "clr_LightSkyGreen": [0.125, 0.698, 0.668],
            "clr_MediumSpringGreen": [0.0, 0.98, 0.604],
            "clr_Goldenrod": [0.855, 0.648, 0.125],
            "clr_Lilac": [0.8, 0.6, 1.0],
            "clr_Coral": [1.0, 0.498, 0.312],
            "clr_FireBrick": [0.698, 0.132, 0.132],
            "clr_RoyalPurple": [0.4, 0.0, 0.6],
            "clr_Gold": [1.0, 0.844, 0.0],
            "clr_Crimson": [0.8, 0.0, 0.2],
            "clr_SlateGray": [0.44, 0.5, 0.565],
            "clr_SeaGreen": [0.18, 0.545, 0.34],
            "clr_Teal": [0.0, 0.5, 0.5],
            "clr_LightSteelBlue": [0.69, 0.77, 0.87],
            "clr_MediumSlateBlue": [0.484, 0.408, 0.932],
            "clr_MediumBlue": [0.0, 0.0, 0.804],
            "clr_RoyalBlue": [0.255, 0.41, 0.884],
            "clr_LightGray": [0.828, 0.828, 0.828],
            "clr_Silver": [0.752, 0.752, 0.752],
            "clr_DarkGray": [0.664, 0.664, 0.664],
            "clr_Gray": [0.5, 0.5, 0.5],
            "clr_IndianRed": [0.804, 0.36, 0.36],
            "clr_Tan": [0.824, 0.705, 0.55],
            "clr_LightOliveGreen": [0.6, 0.8, 0.6],
            "clr_CadetBlue": [0.372, 0.62, 0.628],
            "clr_Beige": [0.96, 0.96, 0.864],
        }

        self.mix_names = [
            "None",
            "Conv",
            "Soften",
            "Over",
            "Semi",
            "Thermo",
            "Rot",
            "Mini",
            "Anon",
        ]
        self.mix_col = [
            self.colors["clr_SeaGreen"],  # None
            self.colors["clr_LightSkyBlue"],  # Convection
            self.colors["clr_LightSteelBlue"],  # Softened convection
            self.colors["clr_SlateGray"],  # Overshoot
            self.colors["clr_Lilac"],  # Semi convection
            self.colors["clr_LightSkyGreen"],  # Thermohaline
            self.colors["clr_BrightBlue"],  # Rotation
            self.colors["clr_Beige"],  # Minimum
            self.colors["clr_Tan"],  # Anonymous
        ]

        # Conviently the index of this list is the proton number
        self.elementsPretty = [
            "neut",
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Uub",
            "Uut",
            "Uuq",
            "Uup",
            "Uuh",
            "Uus",
            "Uuo",
        ]
        self.elements = [x.lower() for x in self.elementsPretty]

        self._getMESAPath()

        self.msun = 1.9892 * 10**33
        self.secyear = 60.0 * 60.0 * 24.0 * 365.25

        self._kip_cbar_label = r"$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$"

        # ..names of the stable isotopes
        self.stable_isos = [
            "h1",
            "h2",
            "he3",
            "he4",
            "li6",
            "li7",
            "be9",
            "b10",
            "b11",
            "c12",
            "c13",
            "n14",
            "n15",
            "o16",
            "o17",
            "o18",
            "f19",
            "ne20",
            "ne21",
            "ne22",
            "na23",
            "mg24",
            "mg25",
            "mg26",
            "al27",
            "si28",
            "si29",
            "si30",
            "p31",
            "s32",
            "s33",
            "s34",
            "s36",
            "cl35",
            "cl37",
            "ar36",
            "ar38",
            "ar40",
            "k39",
            "k40",
            "k41",
            "ca40",
            "ca42",
            "ca43",
            "ca44",
            "ca46",
            "ca48",
            "sc45",
            "ti46",
            "ti47",
            "ti48",
            "ti49",
            "ti50",
            "v50",
            "v51",
            "cr50",
            "cr52",
            "cr53",
            "cr54",
            "mn55",
            "fe54",
            "fe56",
            "fe57",
            "fe58",
            "co59",
            "ni58",
            "ni60",
            "ni61",
            "ni62",
            "ni64",
            "cu63",
            "cu65",
            "zn64",
            "zn66",
            "zn67",
            "zn68",
            "zn70",
            "ga69",
            "ga71",
            "ge70",
            "ge72",
            "ge73",
            "ge74",
            "ge76",
            "as75",
            "se74",
            "se76",
            "se77",
            "se78",
            "se80",
            "se82",
            "br79",
            "br81",
            "kr78",
            "kr80",
            "kr82",
            "kr83",
            "kr84",
            "kr86",
            "rb85",
            "rb87",
            "sr84",
            "sr86",
            "sr87",
            "sr88",
            "y89",
            "zr90",
            "zr91",
            "zr92",
            "zr94",
            "zr96",
            "nb93",
            "mo92",
            "mo94",
            "mo95",
            "mo96",
            "mo97",
            "mo98",
            "mo100",
            "ru96",
            "ru98",
            "ru99",
            "ru100",
            "ru101",
            "ru102",
            "ru104",
            "rh103",
            "pd102",
            "pd104",
            "pd105",
            "pd106",
            "pd108",
            "pd110",
            "ag107",
            "ag109",
            "cd106",
            "cd108",
            "cd110",
            "cd111",
            "cd112",
            "cd113",
            "cd114",
            "cd116",
            "in113",
            "in115",
            "sn112",
            "sn114",
            "sn115",
            "sn116",
            "sn117",
            "sn118",
            "sn119",
            "sn120",
            "sn122",
            "sn124",
            "sb121",
            "sb123",
            "te120",
            "te122",
            "te123",
            "te124",
            "te125",
            "te126",
            "te128",
            "te130",
            "i127",
            "xe124",
            "xe126",
            "xe128",
            "xe129",
            "xe130",
            "xe131",
            "xe132",
            "xe134",
            "xe136",
            "cs133",
            "ba130",
            "ba132",
            "ba134",
            "ba135",
            "ba136",
            "ba137",
            "ba138",
            "la138",
            "la139",
            "ce136",
            "ce138",
            "ce140",
            "ce142",
            "pr141",
            "nd142",
            "nd143",
            "nd144",
            "nd145",
            "nd146",
            "nd148",
            "nd150",
            "sm144",
            "sm147",
            "sm148",
            "sm149",
            "sm150",
            "sm152",
            "sm154",
            "eu151",
            "eu153",
            "gd152",
            "gd154",
            "gd155",
            "gd156",
            "gd157",
            "gd158",
            "gd160",
            "tb159",
            "dy156",
            "dy158",
            "dy160",
            "dy161",
            "dy162",
            "dy163",
            "dy164",
            "ho165",
            "er162",
            "er164",
            "er166",
            "er167",
            "er168",
            "er170",
            "tm169",
            "yb168",
            "yb170",
            "yb171",
            "yb172",
            "yb173",
            "yb174",
            "yb176",
            "lu175",
            "lu176",
            "hf174",
            "hf176",
            "hf177",
            "hf178",
            "hf179",
            "hf180",
            "ta180",
            "ta181",
            "w180",
            "w182",
            "w183",
            "w184",
            "w186",
            "re185",
            "re187",
            "os184",
            "os186",
            "os187",
            "os188",
            "os189",
            "os190",
            "os192",
            "ir191",
            "ir193",
            "pt190",
            "pt192",
            "pt194",
            "pt195",
            "pt196",
            "pt198",
            "au197",
            "hg196",
            "hg198",
            "hg199",
            "hg200",
            "hg201",
            "hg202",
            "hg204",
            "tl203",
            "tl205",
            "pb204",
            "pb206",
            "pb207",
            "pb208",
            "bi209",
            "th232",
            "u235",
            "u238",
        ]

        self.solar_is_set = False

        # ..anders & grevesse 1989 solar mass fractions
        self._sol_comp_ag89 = [
            7.0573e-01,
            4.8010e-05,
            2.9291e-05,
            2.7521e-01,
            6.4957e-10,
            9.3490e-09,
            1.6619e-10,
            1.0674e-09,
            4.7301e-09,
            3.0324e-03,
            3.6501e-05,
            1.1049e-03,
            4.3634e-06,
            9.5918e-03,
            3.8873e-06,
            2.1673e-05,
            4.0515e-07,
            1.6189e-03,
            4.1274e-06,
            1.3022e-04,
            3.3394e-05,
            5.1480e-04,
            6.7664e-05,
            7.7605e-05,
            5.8052e-05,
            6.5301e-04,
            3.4257e-05,
            2.3524e-05,
            8.1551e-06,
            3.9581e-04,
            3.2221e-06,
            1.8663e-05,
            9.3793e-08,
            2.5320e-06,
            8.5449e-07,
            7.7402e-05,
            1.5379e-05,
            2.6307e-08,
            3.4725e-06,
            4.4519e-10,
            2.6342e-07,
            5.9898e-05,
            4.1964e-07,
            8.9734e-07,
            1.4135e-06,
            2.7926e-09,
            1.3841e-07,
            3.8929e-08,
            2.2340e-07,
            2.0805e-07,
            2.1491e-06,
            1.6361e-07,
            1.6442e-07,
            9.2579e-10,
            3.7669e-07,
            7.4240e-07,
            1.4863e-05,
            1.7160e-06,
            4.3573e-07,
            1.3286e-05,
            7.1301e-05,
            1.1686e-03,
            2.8548e-05,
            3.6971e-06,
            3.3579e-06,
            4.9441e-05,
            1.9578e-05,
            8.5944e-07,
            2.7759e-06,
            7.2687e-07,
            5.7528e-07,
            2.6471e-07,
            9.9237e-07,
            5.8765e-07,
            8.7619e-08,
            4.0593e-07,
            1.3811e-08,
            3.9619e-08,
            2.7119e-08,
            4.3204e-08,
            5.9372e-08,
            1.7136e-08,
            8.1237e-08,
            1.7840e-08,
            1.2445e-08,
            1.0295e-09,
            1.0766e-08,
            9.1542e-09,
            2.9003e-08,
            6.2529e-08,
            1.1823e-08,
            1.1950e-08,
            1.2006e-08,
            3.0187e-10,
            2.0216e-09,
            1.0682e-08,
            1.0833e-08,
            5.4607e-08,
            1.7055e-08,
            1.1008e-08,
            4.3353e-09,
            2.8047e-10,
            5.0468e-09,
            3.6091e-09,
            4.3183e-08,
            1.0446e-08,
            1.3363e-08,
            2.9463e-09,
            4.5612e-09,
            4.7079e-09,
            7.7706e-10,
            1.6420e-09,
            8.7966e-10,
            5.6114e-10,
            9.7562e-10,
            1.0320e-09,
            5.9868e-10,
            1.5245e-09,
            6.2225e-10,
            2.5012e-10,
            8.6761e-11,
            5.9099e-10,
            5.9190e-10,
            8.0731e-10,
            1.5171e-09,
            9.1547e-10,
            8.9625e-10,
            3.6637e-11,
            4.0775e-10,
            8.2335e-10,
            1.0189e-09,
            1.0053e-09,
            4.5354e-10,
            6.8205e-10,
            6.4517e-10,
            5.3893e-11,
            3.9065e-11,
            5.5927e-10,
            5.7839e-10,
            1.0992e-09,
            5.6309e-10,
            1.3351e-09,
            3.5504e-10,
            2.2581e-11,
            5.1197e-10,
            1.0539e-10,
            7.1802e-11,
            3.9852e-11,
            1.6285e-09,
            8.6713e-10,
            2.7609e-09,
            9.8731e-10,
            3.7639e-09,
            5.4622e-10,
            6.9318e-10,
            5.4174e-10,
            4.1069e-10,
            1.3052e-11,
            3.8266e-10,
            1.3316e-10,
            7.1827e-10,
            1.0814e-09,
            3.1553e-09,
            4.9538e-09,
            5.3600e-09,
            2.8912e-09,
            1.7910e-11,
            1.6223e-11,
            3.3349e-10,
            4.1767e-09,
            6.7411e-10,
            3.3799e-09,
            4.1403e-09,
            1.5558e-09,
            1.2832e-09,
            1.2515e-09,
            1.5652e-11,
            1.5125e-11,
            3.6946e-10,
            1.0108e-09,
            1.2144e-09,
            1.7466e-09,
            1.1240e-08,
            1.3858e-12,
            1.5681e-09,
            7.4306e-12,
            9.9136e-12,
            3.5767e-09,
            4.5258e-10,
            5.9562e-10,
            8.0817e-10,
            3.6533e-10,
            7.1757e-10,
            2.5198e-10,
            5.2441e-10,
            1.7857e-10,
            1.7719e-10,
            2.9140e-11,
            1.4390e-10,
            1.0931e-10,
            1.3417e-10,
            7.2470e-11,
            2.6491e-10,
            2.2827e-10,
            1.7761e-10,
            1.9660e-10,
            2.5376e-12,
            2.8008e-11,
            1.9133e-10,
            2.6675e-10,
            2.0492e-10,
            3.2772e-10,
            2.9180e-10,
            2.8274e-10,
            8.6812e-13,
            1.4787e-12,
            3.7315e-11,
            3.0340e-10,
            4.1387e-10,
            4.0489e-10,
            4.6047e-10,
            3.7104e-10,
            1.4342e-12,
            1.6759e-11,
            3.5397e-10,
            2.4332e-10,
            2.8557e-10,
            1.6082e-10,
            1.6159e-10,
            1.3599e-12,
            3.2509e-11,
            1.5312e-10,
            2.3624e-10,
            1.7504e-10,
            3.4682e-10,
            1.4023e-10,
            1.5803e-10,
            4.2293e-12,
            1.0783e-12,
            3.4992e-11,
            1.2581e-10,
            1.8550e-10,
            9.3272e-11,
            2.4131e-10,
            1.1292e-14,
            9.4772e-11,
            7.8768e-13,
            1.6113e-10,
            8.7950e-11,
            1.8989e-10,
            1.7878e-10,
            9.0315e-11,
            1.5326e-10,
            5.6782e-13,
            5.0342e-11,
            5.1086e-11,
            4.2704e-10,
            5.2110e-10,
            8.5547e-10,
            1.3453e-09,
            1.1933e-09,
            2.0211e-09,
            8.1702e-13,
            5.0994e-11,
            2.1641e-09,
            2.2344e-09,
            1.6757e-09,
            4.8231e-10,
            9.3184e-10,
            2.3797e-12,
            1.7079e-10,
            2.8843e-10,
            3.9764e-10,
            2.2828e-10,
            5.1607e-10,
            1.2023e-10,
            2.7882e-10,
            6.7411e-10,
            3.1529e-10,
            3.1369e-09,
            3.4034e-09,
            9.6809e-09,
            7.6127e-10,
            1.9659e-10,
            3.8519e-13,
            5.3760e-11,
        ]

        # ..charge of the stable isotopes

        self._stable_charge = [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            8,
            8,
            8,
            9,
            10,
            10,
            10,
            11,
            12,
            12,
            12,
            13,
            14,
            14,
            14,
            15,
            16,
            16,
            16,
            16,
            17,
            17,
            18,
            18,
            18,
            19,
            19,
            19,
            20,
            20,
            20,
            20,
            20,
            20,
            21,
            22,
            22,
            22,
            22,
            22,
            23,
            23,
            24,
            24,
            24,
            24,
            25,
            26,
            26,
            26,
            26,
            27,
            28,
            28,
            28,
            28,
            28,
            29,
            29,
            30,
            30,
            30,
            30,
            30,
            31,
            31,
            32,
            32,
            32,
            32,
            32,
            33,
            34,
            34,
            34,
            34,
            34,
            34,
            35,
            35,
            36,
            36,
            36,
            36,
            36,
            36,
            37,
            37,
            38,
            38,
            38,
            38,
            39,
            40,
            40,
            40,
            40,
            40,
            41,
            42,
            42,
            42,
            42,
            42,
            42,
            42,
            44,
            44,
            44,
            44,
            44,
            44,
            44,
            45,
            46,
            46,
            46,
            46,
            46,
            46,
            47,
            47,
            48,
            48,
            48,
            48,
            48,
            48,
            48,
            48,
            49,
            49,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            51,
            51,
            52,
            52,
            52,
            52,
            52,
            52,
            52,
            52,
            53,
            54,
            54,
            54,
            54,
            54,
            54,
            54,
            54,
            54,
            55,
            56,
            56,
            56,
            56,
            56,
            56,
            56,
            57,
            57,
            58,
            58,
            58,
            58,
            59,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
            62,
            62,
            62,
            62,
            62,
            62,
            62,
            63,
            63,
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            65,
            66,
            66,
            66,
            66,
            66,
            66,
            66,
            67,
            68,
            68,
            68,
            68,
            68,
            68,
            69,
            70,
            70,
            70,
            70,
            70,
            70,
            70,
            71,
            71,
            72,
            72,
            72,
            72,
            72,
            72,
            73,
            73,
            74,
            74,
            74,
            74,
            74,
            75,
            75,
            76,
            76,
            76,
            76,
            76,
            76,
            76,
            77,
            77,
            78,
            78,
            78,
            78,
            78,
            78,
            79,
            80,
            80,
            80,
            80,
            80,
            80,
            80,
            81,
            81,
            82,
            82,
            82,
            82,
            83,
            90,
            92,
            92,
        ]

        # ..number of nucleons (protons and neutrons) in the stable isotopes

        self._stable_a = [
            1,
            2,
            3,
            4,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            35,
            37,
            36,
            38,
            40,
            39,
            40,
            41,
            40,
            42,
            43,
            44,
            46,
            48,
            45,
            46,
            47,
            48,
            49,
            50,
            50,
            51,
            50,
            52,
            53,
            54,
            55,
            54,
            56,
            57,
            58,
            59,
            58,
            60,
            61,
            62,
            64,
            63,
            65,
            64,
            66,
            67,
            68,
            70,
            69,
            71,
            70,
            72,
            73,
            74,
            76,
            75,
            74,
            76,
            77,
            78,
            80,
            82,
            79,
            81,
            78,
            80,
            82,
            83,
            84,
            86,
            85,
            87,
            84,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            94,
            96,
            93,
            92,
            94,
            95,
            96,
            97,
            98,
            100,
            96,
            98,
            99,
            100,
            101,
            102,
            104,
            103,
            102,
            104,
            105,
            106,
            108,
            110,
            107,
            109,
            106,
            108,
            110,
            111,
            112,
            113,
            114,
            116,
            113,
            115,
            112,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            122,
            124,
            121,
            123,
            120,
            122,
            123,
            124,
            125,
            126,
            128,
            130,
            127,
            124,
            126,
            128,
            129,
            130,
            131,
            132,
            134,
            136,
            133,
            130,
            132,
            134,
            135,
            136,
            137,
            138,
            138,
            139,
            136,
            138,
            140,
            142,
            141,
            142,
            143,
            144,
            145,
            146,
            148,
            150,
            144,
            147,
            148,
            149,
            150,
            152,
            154,
            151,
            153,
            152,
            154,
            155,
            156,
            157,
            158,
            160,
            159,
            156,
            158,
            160,
            161,
            162,
            163,
            164,
            165,
            162,
            164,
            166,
            167,
            168,
            170,
            169,
            168,
            170,
            171,
            172,
            173,
            174,
            176,
            175,
            176,
            174,
            176,
            177,
            178,
            179,
            180,
            180,
            181,
            180,
            182,
            183,
            184,
            186,
            185,
            187,
            184,
            186,
            187,
            188,
            189,
            190,
            192,
            191,
            193,
            190,
            192,
            194,
            195,
            196,
            198,
            197,
            196,
            198,
            199,
            200,
            201,
            202,
            204,
            203,
            205,
            204,
            206,
            207,
            208,
            209,
            232,
            235,
            238,
        ]

        # jcode tells the type progenitors each stable species can have.
        # jcode = 0 if the species is the only stable one of that a
        #       = 1 if the species can have proton-rich progenitors
        #       = 2 if the species can have neutron-rich progenitors
        #       = 3 if the species can only be made as itself (eg k40)

        self._jcode = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            1,
            0,
            0,
            0,
            2,
            2,
            0,
            1,
            0,
            1,
            0,
            2,
            3,
            0,
            1,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            2,
            2,
            0,
            1,
            1,
            0,
            2,
            2,
            2,
            0,
            0,
            1,
            1,
            1,
            0,
            2,
            2,
            0,
            2,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            2,
            2,
            2,
            0,
            1,
            1,
            0,
            3,
            0,
            2,
            2,
            1,
            1,
            0,
            1,
            0,
            2,
            2,
            0,
            1,
            1,
            0,
            2,
            2,
            2,
            0,
            0,
            1,
            1,
            1,
            0,
            2,
            2,
            2,
            2,
            1,
            2,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            2,
            2,
            2,
            0,
            2,
            1,
            1,
            1,
            3,
            0,
            2,
            2,
            2,
            0,
            1,
            1,
            1,
            0,
            3,
            0,
            2,
            2,
            2,
            0,
            1,
            1,
            1,
            0,
            3,
            0,
            2,
            3,
            0,
            1,
            1,
            0,
            2,
            0,
            1,
            0,
            2,
            0,
            0,
            2,
            2,
            1,
            0,
            1,
            0,
            1,
            2,
            2,
            0,
            0,
            1,
            1,
            0,
            2,
            0,
            2,
            2,
            0,
            1,
            1,
            1,
            0,
            2,
            0,
            2,
            0,
            1,
            1,
            0,
            0,
            2,
            2,
            0,
            1,
            1,
            0,
            0,
            0,
            2,
            2,
            0,
            3,
            1,
            1,
            0,
            0,
            0,
            2,
            3,
            0,
            1,
            0,
            0,
            2,
            2,
            0,
            2,
            1,
            1,
            1,
            0,
            0,
            2,
            2,
            0,
            0,
            1,
            1,
            0,
            0,
            2,
            2,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        if rcparams_fixed:
            # Set default matplotlib rcParameters (True by default)
            if find_executable("latex"):
                mpl.rc("text", usetex=True)
            else:
                mpl.rc("text", usetex=False)

            mpl.rc("font", size=30)
            mpl.rc("xtick", labelsize=28)
            mpl.rc("ytick", labelsize=28)
            mpl.rcParams["axes.linewidth"] = 2.0
            mpl.rcParams["xtick.major.size"] = 18  # major tick size in points
            mpl.rcParams["xtick.minor.size"] = 9  # minor tick size in points
            mpl.rcParams["ytick.major.size"] = 18  # major tick size in points
            mpl.rcParams["ytick.minor.size"] = 9  # minor tick size in points

            mpl.rcParams["xtick.major.width"] = 0.8  # major tick size in points
            mpl.rcParams["xtick.minor.width"] = 0.6  # minor tick size in points
            mpl.rcParams["ytick.major.width"] = 0.8  # major tick size in points
            mpl.rcParams["ytick.minor.width"] = 0.6  # minor tick size in points

    _core_masses = {
        "He": {"names": ["he_core_mass"], "color": "clr_Teal"},
        "C": {"names": ["c_core_mass", "co_core_mass"], "color": "clr_LightOliveGreen"},
        "O": {"names": ["o_core_mass", "one_core_mass"], "color": "clr_SeaGreen"},
        "Si": {"names": ["si_core_mass"], "color": "clr_Lilac"},
        "Fe": {"names": ["fe_core_mass"], "color": "clr_Crimson"},
    }

    def set_solar(self, solar="ag89"):
        if str(solar) == "ag89":
            self.sol_comp = self._sol_comp_ag89
        else:
            raise ValueError("Must pass ag89")

        self.solar_is_set = True

    def is_solar_set(self):
        if not self.solar_is_set:
            raise ValueError("Must call set_solar first")

    def _getMESAPath(self):
        self.mesa_dir = os.getenv("MESA_DIR")
        # if self.mesa_dir==None:
        # raise ValueError("Must set $MESA_DIR in terminal or call setMESAPath(mesa_dir)")

    def setMESAPath(self, mesa_dir):
        self.mesa_dir = mesa_dir

    def _loadBurnData(self):
        try:
            dataDir = self.mesa_dir + "/data/star_data/plot_info/"
        except TypeError:
            raise ValueError("Must set $MESA_DIR or call setMESAPath(MESA_DIR)")

        self._hburn = np.genfromtxt(
            dataDir + "hydrogen_burn.data", names=["logRho", "logT"]
        )
        self._heburn = np.genfromtxt(
            dataDir + "helium_burn.data", names=["logRho", "logT"]
        )
        self._cburn = np.genfromtxt(
            dataDir + "carbon_burn.data", names=["logRho", "logT"]
        )
        self._oburn = np.genfromtxt(
            dataDir + "oxygen_burn.data", names=["logRho", "logT"]
        )

        self._psi4 = np.genfromtxt(dataDir + "psi4.data", names=["logRho", "logT"])
        self._elect = np.genfromtxt(dataDir + "elect.data", names=["logRho", "logT"])
        self._gamma4 = np.genfromtxt(
            dataDir + "gamma_4_thirds.data", names=["logRho", "logT"]
        )
        self._kap = np.genfromtxt(
            dataDir + "kap_rad_cond_eq.data", names=["logRho", "logT"]
        )
        self._opal = np.genfromtxt(dataDir + "opal_clip.data", names=["logRho", "logT"])
        self._scvh = np.genfromtxt(dataDir + "scvh_clip.data", names=["logRho", "logT"])

    def labels(self, label, log=False, center=False):
        l = ""
        ls = ""
        if type(label) is bytes:
            label = label.decode()

        if "$" in label:
            return label

        if log or "log_" in label:
            ls = r"$\log_{10}\;$"
            label = label.replace("log_", "")

        if "D_" in label:
            ls = ""
            l = r"$D_{" + label.split("_")[-1] + "}$"
        elif "am_D_" in label:
            l = r"$D_{" + label.split("_")[-1] + "}$"
        elif label == "xq":
            # ls=''
            l = r"$\left(1-q\right)$"
        elif label == "mass":
            l = l + r"$\rm{Mass}\; [M_{\odot}]$"
        elif label == "model":
            l = l + r"$\rm{Model\; number}$"
        elif "teff" in label or label == "logT" or "_T" in label:
            if label == "logT":
                ls = r"$\log_{10}\;$"
            if center:
                l = l + r"$T_{\rm{eff},c}\; [K]$"
            else:
                l = l + r"$T_{\rm{eff}}\; [K]$"
            label = label.replace("teff", "")
        elif "rho" in label or "Rho" in label:
            if center:
                l = l + r"$\rho_{c}\; [\rm{g\;cm^{-3}}]$"
            else:
                l = l + r"$\rho\; [\rm{g\;cm^{-3}}]$"
        elif label == "column_depth":
            l = ls + r"$y\; [\rm{g}\; \rm{cm}^{-2}]$"
        elif ("lum" in label) and ("column" not in label):
            l = l + r"$L\; [L_{\odot}]$"
        if "star_age" in label:
            l = l + r"T$\;$"
            if "sec" in label:
                l = l + "[s]"
            if "hr" in label:
                l = l + "[hr]"
            if "day" in label:
                l = l + "[day]"
            if "yr" in label:
                l = l + "[yr]"
        elif "burn" in label:
            l = l + r"$\epsilon_{" + label.split("_")[1].capitalize() + r"}$"
        elif label == "pp":
            l = l + r"$\epsilon_{pp}$"
        elif label == "tri_alfa":
            l = l + r"$\epsilon_{3\alpha}$"
        elif label == "c12_c12":
            l = l + r"$\epsilon_{c12,c12}$"
        elif label == "c12_o16":
            l = l + r"$\epsilon_{c12,o16}$"
        elif label == "cno":
            l = l + r"$\epsilon_{cno}$"
        elif label == "o16_o16":
            l = l + r"$\epsilon_{o16,o16}$"
        elif label == "pnhe4":
            l = l + r"$\epsilon_{pnhe4}$"
        elif label == "photo":
            l = l + r"$\epsilon_{\gamma}$"
        elif label == "other":
            l = l + r"$\epsilon_{other}$"
        elif "abundance" in label:
            l = l + r"$\chi\; [M_{\odot}]$"
        elif label == "non_nuc_neu":
            l = l + r"$\epsilon_{\nu,total}$"
        elif label == "nonnucneu_brem":
            l = l + r"$\epsilon_{\nu,brem}$"
        elif label == "nonnucneu_pair":
            l = l + r"$\epsilon_{\nu,pair}$"
        elif label == "nonnucneu_phot":
            l = l + r"$\epsilon_{\nu,photo}$"
        elif label == "nonnucneu_plas":
            l = l + r"$\epsilon_{\nu,plasma}$"
        elif label == "nonnucneu_reco":
            l = l + r"$\epsilon_{\nu,reco}$"

        if len(l) == 0:
            if "$" not in label:
                l = label.replace("_", " ")
            else:
                l = label

        l = ls + l

        return l

    def safeLabel(self, label, axis, strip=None):
        outLabel = ""
        if label is not None:
            outLabel = label
        else:
            if isinstance(axis, str) or isinstance(axis, bytes):
                outLabel = self.labels(axis)

        if strip is not None:
            outLabel = outLabel.replace(strip, "")

        if type(outLabel) is bytes:
            outLabel = outLabel.decode()

        return outLabel

    def _listRates(self, data, prefix=" "):
        res = []
        for i in data.data.dtype.names:
            if i.startswith(prefix):
                res.append(i)
        return res

    def _listAbun(self, data, prefix=""):
        abun_list = []
        for j in data.data.dtype.names:
            if prefix in j:
                i = j[len(prefix) :]
                if len(i) <= 5 and len(i) >= 2 and "burn_" not in j:
                    if (
                        i[0].isalpha()
                        and (i[1].isalpha() or i[1].isdigit())
                        and any(char.isdigit() for char in i)
                        and i[-1].isdigit()
                    ):
                        if (len(i) == 5 and i[-1].isdigit() and i[-2].isdigit()) or len(
                            i
                        ) < 5:
                            abun_list.append(j)
                if i == "neut" or i == "prot":
                    abun_list.append(j)

        for idx, i in enumerate(abun_list):
            if type(i) is bytes:
                abun_list[idx] = i.decode()

        return abun_list

    def _splitIso(self, iso, prefix=""):
        name = ""
        mass = ""
        if prefix in iso:
            iso = iso[len(prefix) :]
        for i in iso:
            if i.isdigit():
                mass += i
            else:
                name += i

        if "neut" in name or "prot" in name:
            mass = 1
        return name, int(mass)

    def _getIso(self, iso, prefix=""):
        name, mass = self._splitIso(iso, prefix)
        if "prot" in name:
            p = 1
            n = 0
        else:
            p = self.elements.index(name)
            n = mass - p
        return name, p, n

    def _listBurn(self, data):
        burnList = []
        ignore = ["qtop", "type"]
        extraBurn = [
            "pp",
            "cno",
            "tri_alfa",
            "tri_alpha",
            "c12_c12",
            "c12_O16",
            "o16_o16",
            "pnhe4",
            "photo",
            "other",
        ]
        for i in data:
            if ("burn_" in i or i in extraBurn) and not any(j in i for j in ignore):
                burnList.append(str(i))
        return burnList

    def _listMix(self, data):
        mixList = [
            "log_D_conv",
            "log_D_semi",
            "log_D_ovr",
            "log_D_th",
            "log_D_thrm",
            "log_D_minimum",
            "log_D_anon",
            "log_D_rayleigh_taylor",
            "log_D_soft",
        ]
        mixListOut = []
        for i in data:
            if i in mixList:
                mixListOut.append(str(i))
        return mixListOut

    def _abunSum(self, m, iso, mass_min=0.0, mass_max=9999.0):
        ind = (m.prof.mass >= mass_min) & (m.prof.mass <= mass_max)
        return (
            np.sum(m.prof.data[iso][ind] * 10 ** m.prof.logdq[ind])
            * m.prof.star_mass
            / np.minimum(m.prof.star_mass, mass_max - mass_min)
        )

    def _eleSum(self, m, ele, mass_min=0.0, mass_max=9999.0):
        ind = (m.prof.mass >= mass_min) & (m.prof.mass <= mass_max)

        la = m.prof.listAbun()
        x = 0.0
        for i in la:
            if ele == i[0 : len(ele)]:
                x = x + np.sum(
                    m.prof.data[i][ind] * 10 ** m.prof.logdq[ind]
                ) * m.prof.star_mass / np.minimum(m.prof.star_mass, mass_max - mass_min)
        return x

    def _setMixRegionsCol(self, kip=True, mix=False):
        cmap = mpl.colors.ListedColormap(self.mix_col)

        cmap.set_over((1.0, 1.0, 1.0))
        cmap.set_under((0.0, 0.0, 0.0))
        bounds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def _setTicks(self, ax, log=False):
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if not log:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="y", pad=10)

    def _plotBurnRegions(self, m, ax, x, y, show_x, show_line, yrng=None, ind=None):
        # non 0.0, yellow 1, orange 10**4, red 10**7
        ylim = ax.get_ylim()

        if show_x:
            yy = np.zeros(np.size(x))
            if yrng is not None:
                yy[:] = yrng[0]
            else:
                yy[:] = ylim[0]
            size = 350
        else:
            yy = y
            size = 180

        if ind is not None:
            netEng = m.prof.data["net_nuclear_energy"][ind]
        else:
            netEng = m.prof.data["net_nuclear_energy"]

        ind2 = (netEng >= 1.0) & (netEng <= 4.0)
        ax.scatter(x[ind2], yy[ind2], c="yellow", s=size, linewidths=0, alpha=1.0)
        ind2 = (netEng >= 4.0) & (netEng <= 7.0)
        ax.scatter(x[ind2], yy[ind2], c="orange", s=size, linewidths=0, alpha=1.0)
        ind2 = netEng >= 7.0
        ax.scatter(x[ind2], yy[ind2], c="red", s=size, edgecolor="none", alpha=1.0)
        ax.set_ylim(ylim)

    def _plotMixRegions(self, m, ax, x, y, show_x, show_line, yrng=None, ind=None):

        if ind is not None:
            if np.count_nonzero(ind) == 0:
                return

        ylim = ax.get_ylim()

        if show_x:
            yy = np.zeros(np.size(x))
            if yrng is not None:
                yy[:] = yrng[0]
            else:
                yy[:] = ylim[0]
            size = 150
        else:
            yy = y
            size = 60

        cmap, norm = self._setMixRegionsCol()

        isSet = False
        col = np.zeros(np.size(x))
        for mixLabel in ["mixing_type", "conv_mixing_type"]:
            try:
                col = m.prof.data[mixLabel]
                isSet = True
                break
            except:
                pass

        if isSet is None:
            raise ValueError(
                "Need mixing type in profile file for showing mix regions, either its mixing_type or conv_mixing_type"
            )

        if ind is not None:
            col = col[ind]
            x = x[ind]
            yy = yy[ind]

        ax.scatter(x, yy, c=col, s=size, cmap=cmap, norm=norm, linewidths=0)

        ax.set_ylim(ylim)

    def _annotateLine(
        self,
        ax,
        x,
        y,
        num_labels,
        xmin,
        xmax,
        text,
        xlog=False,
        line=None,
        color=None,
        fontsize=None,
        offset=None,
    ):

        if fontsize is None:
            fontsize = int(mpl.rcParams["font.size"] // 2)

        if xmin < np.nanmin(x):
            xmin = np.nanmin(x)
        if xmax > np.nanmax(x):
            xmax = np.nanmax(x)

        if xlog:
            xmin = np.log10(xmin)
            xmax = np.log10(xmax)

        ind = np.argsort(x)
        x = x[ind]
        y = y[ind]

        # Dont add points at the edge of the plot
        xx = np.linspace(xmin, xmax, num_labels + 2)[1:-1]

        if xlog:
            xx = 10**xx

        yy = y[np.searchsorted(x, xx)]

        if type(text) is bytes:
            text = text.decode()

        for xp1, yp1 in zip(xx, yy):
            if line is None:
                col = color
            else:
                col = line.get_color()

            x = xp1
            y = yp1
            if offset is not None:
                x = x + offset[0]
                y = y + offset[1]

            # print(text,type(text))
            ax.annotate(
                text,
                xy=(xp1, yp1),
                xytext=(x, y),
                color=col,
                fontsize=fontsize,
                clip_on=True,
            )

    def _setYLim(self, ax, yrngIn, yrngOut, rev=False, log=False):
        yrng = []
        if yrngOut is not None:
            yrng = yrngOut
        else:
            yrng = yrngIn

        if rev:
            yrng = yrng[::-1]
        # if (log==True or log=='log') and log!='linear':
        # yrng=np.log10(yrng)
        ax.set_ylim(yrng)

    def _setXAxis(self, xx, xmin, xmax, fx):
        x = xx
        if fx is not None:
            x = fx(x)

        xrngL = [0, 0]
        if xmin is not None:
            xrngL[0] = xmin
        else:
            xrngL[0] = np.min(x)

        if xmax is not None:
            xrngL[1] = xmax
        else:
            xrngL[1] = np.max(x)

        ind = (x >= xrngL[0]) & (x <= xrngL[1])

        return x, xrngL, ind

    def _cycleColors(self, ax, colors=None, cmap="", num_plots=0, random_col=False):
        if colors is None:
            c = [cmap(i) for i in np.linspace(0.0, 1.0, num_plots)]
        else:
            c = colors
        if random_col:
            random.shuffle(c)
        try:
            ax.set_prop_cycle(cycler("color", c))
        except AttributeError:
            ax.set_color_cylce(c)  # Fallback for older matplotlibs

    def _showBurnData(self, ax):
        self._loadBurnData()
        ax.plot(
            self._hburn["logRho"], self._hburn["logT"], color=self.colors["clr_Gray"]
        )
        ax.annotate(
            "H burn",
            xy=(self._hburn["logRho"][-1], self._hburn["logT"][-1]),
            xytext=(self._hburn["logRho"][-1], self._hburn["logT"][-1]),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

        ax.plot(
            self._heburn["logRho"], self._heburn["logT"], color=self.colors["clr_Gray"]
        )
        ax.annotate(
            "He burn",
            xy=(self._heburn["logRho"][-1], self._heburn["logT"][-1]),
            xytext=(self._heburn["logRho"][-1], self._heburn["logT"][-1]),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

        ax.plot(
            self._cburn["logRho"], self._cburn["logT"], color=self.colors["clr_Gray"]
        )
        ax.annotate(
            "C burn",
            xy=(self._cburn["logRho"][-1], self._cburn["logT"][-1]),
            xytext=(self._cburn["logRho"][-1], self._cburn["logT"][-1]),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

        ax.plot(
            self._oburn["logRho"], self._oburn["logT"], color=self.colors["clr_Gray"]
        )
        ax.annotate(
            "O burn",
            xy=(self._oburn["logRho"][-1], self._oburn["logT"][-1]),
            xytext=(self._oburn["logRho"][-1], self._oburn["logT"][-1]),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

    def _showPgas(self, ax):
        lr1 = -8
        lr2 = 5
        lt1 = np.log10(3.2 * 10**7) + (lr1 - np.log10(0.7)) / 3.0
        lt2 = np.log10(3.2 * 10**7) + (lr2 - np.log10(0.7)) / 3.0
        ax.plot([lr1, lr2], [lt1, lt2], color=self.colors["clr_Gray"])
        ax.annotate(
            r"$P_{rad}\approx P_{gas}$",
            xy=(-4.0, 6.5),
            xytext=(-4.0, 6.5),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

    def _showDegeneracy(self, ax):
        ax.plot(self._psi4["logRho"], self._psi4["logT"], color=self.colors["clr_Gray"])
        ax.annotate(
            r"$\epsilon_F/KT\approx 4$",
            xy=(2.0, 6.0),
            xytext=(2.0, 6.0),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

    def _showGamma4(self, ax):
        ax.plot(
            self._gamma4["logRho"],
            self._gamma4["logT"],
            color=self.colors["clr_Crimson"],
        )
        ax.annotate(
            r"$\Gamma_{1} <4/3$",
            xy=(3.8, 9.2),
            xytext=(3.8, 9.2),
            color=self.colors["clr_Crimson"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

    def _showEOS(self, ax):
        logRho1 = 2.7
        logRho2 = 2.5
        logRho3 = -1.71
        logRho4 = -2.21
        logRho5 = -9.0
        logRho6 = -9.99
        logRho7 = -12
        logT1 = 7.7
        logT2 = 7.6
        logT3 = 4.65
        logT4 = 4.75
        logT5 = 3.60
        logT6 = 3.50
        logT7 = 2.3
        logT8 = 2.2

        ax.plot(
            [logRho2, logRho7], [logT1, logT1], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho2, logRho7], [logT2, logT2], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho2, logRho1], [logT1, logT2], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho1, logRho1], [logT2, logT3], color=self.colors["clr_LightSkyGreen"]
        )

        ax.plot(
            [logRho4, logRho5], [logT7, logT7], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho4, logRho5], [logT8, logT8], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho4, logRho3], [logT8, logT7], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho6, logRho5], [logT7, logT8], color=self.colors["clr_LightSkyGreen"]
        )

        ax.plot(
            [logRho2, logRho2], [logT2, logT4], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho3, logRho1], [logT7, logT3], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho4, logRho2], [logT7, logT4], color=self.colors["clr_LightSkyGreen"]
        )

        ax.plot(
            [logRho5, logRho5], [logT7, logT6], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho6, logRho6], [logT7, logT6], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho5, logRho6], [logT6, logT5], color=self.colors["clr_LightSkyGreen"]
        )

        ax.plot(
            [logRho6, logRho7], [logT5, logT5], color=self.colors["clr_LightSkyGreen"]
        )
        ax.plot(
            [logRho6, logRho7], [logT6, logT6], color=self.colors["clr_LightSkyGreen"]
        )

        logRho0 = logRho1
        logRho1 = 2.2
        logRho2 = 1.2
        logRho3 = -2.0
        logRho4 = -3.8
        logRho5 = -5.8
        logRho6 = -6.8
        logRho7 = -10
        logT1 = 6.6
        logT2 = 6.5
        logT3 = 4.0
        logT4 = 3.4
        logT5 = 3.3

        ax.plot(
            [logRho0, logRho2], [logT1, logT1], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho2, logRho4], [logT1, logT3], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho4, logRho5], [logT3, logT4], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho5, logRho7], [logT4, logT4], color=self.colors["clr_LightSkyBlue"]
        )

        ax.plot(
            [logRho0, logRho1], [logT2, logT2], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho1, logRho3], [logT2, logT3], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho3, logRho5], [logT3, logT5], color=self.colors["clr_LightSkyBlue"]
        )
        ax.plot(
            [logRho5, logRho7], [logT5, logT5], color=self.colors["clr_LightSkyBlue"]
        )

        ax.annotate(
            "HELM",
            xy=(8.6, 8.6),
            xytext=(8.6, 8.6),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )
        ax.annotate(
            "OPAL",
            xy=(-7.2, 5.8),
            xytext=(-7.2, 5.8),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )
        ax.annotate(
            "SCVH",
            xy=(-0.8, 3.7),
            xytext=(-0.8, 3.7),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )
        ax.annotate(
            "PC",
            xy=(7.1, 5.1),
            xytext=(7.1, 5.1),
            color=self.colors["clr_Gray"],
            fontsize=mpl.rcParams["font.size"] - 12,
        )

    def _showBurnMixLegend(self, ax, mix=True, burn=True):

        label = []
        color = []

        if burn:
            label.append(r"$>1\; \rm{erg}^{-1}\;s^{-1}$")
            color.append(self.colors["clr_Gold"])
            label.append(r"$>1000\; \rm{erg}^{-1}\;s^{-1}$")
            color.append(self.colors["clr_Coral"])
            label.append(r"$>10^7\; \rm{erg}^{-1}\;s^{-1}$")
            color.append(self.colors["clr_Crimson"])

        if mix:
            cmap, norm = self._setMixRegionsCol(mix=True)
            for i, j in zip(self.mix_names, self.mix_col):
                label.append(i)
                color.append(j)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for (
            i,
            j,
        ) in zip(label, color):
            ax.plot([0, 0], [0, 0], color="w", label=i, alpha=0.0)

        leg = ax.legend(
            framealpha=0, labelspacing=0.0, numpoints=1, loc=4, handlelength=1
        )
        for text, i, j in zip(leg.get_texts(), label, color):
            plt.setp(text, color=j, fontsize=16)
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ]
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _plotCoreLoc(self, prof, ax, xaxis, x, ymin, ymax, linecol="k"):

        for name, value in self._core_masses.items():
            cn = None
            # Find cell where we have core mass and use that to index the actual x axis
            for n in value.names:
                if n in prof.head.dtype.names:
                    cm = n

            pos = bisect.bisect_right(prof.data["mass"][::-1], prof.head[cm])
            if pos < np.size(prof.data[xaxis]) and pos > 0 and prof.head[cm] > 0.0:
                pos = np.size(prof.data["mass"]) - pos
                ax.plot(
                    [prof.data[xaxis][pos], prof.data[xaxis][pos]],
                    [ymin, ymax],
                    "--",
                    color=linecol,
                )
                xp1 = prof.data[xaxis][pos]
                yp1 = 0.95 * (ymax - ymin) + ymin
                ax.annotate(
                    name,
                    xy=(xp1, yp1),
                    xytext=(xp1, yp1),
                    color=linecol,
                    fontsize=mpl.rcParams["font.size"] - 12,
                )

    def _showMassLoc(self, m, fig, ax, x, modInd):

        l = []
        c = []
        for name, val in self._core_masses.items():
            y = None
            for n in val["names"]:
                if n in m.hist.data.dtype.names:
                    y = m.hist.data[n][modInd]
            if y is None:
                continue

            if np.any(y):
                ax.plot(x, y, color=self.colors[val["color"]], linewidth=5)
                l.append(name)
                c.append(self.colors[val["color"]])

        self._addExtraLabelsToAxis(
            fig,
            l,
            c,
            num_left=0,
            num_right=len(l),
            right_pad=50,
        )

    def _showMassLocHist(self, m, fig, ax, x, y, modInd):
        out = []
        outc = []

        for name, val in self._core_masses.items():
            ind = None
            for n in val["names"]:
                if n in m.hist.data:
                    ind = m.hist.data[n][modInd] > 0.0
            if ind is None:
                continue

            if np.count_nonzero(ind):
                ax.plot(
                    [x[ind][0], x[ind][0]],
                    ax.get_ylim(),
                    "--",
                    color=self.colors[val["color"]],
                    linewidth=2,
                )
                out.append(x[ind][0])
                outc.append(l)

        ax2 = ax.twiny()
        ax2.plot(ax.get_xlim(), ax.get_ylim())
        ax2.cla()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(out)
        ax2.set_xticklabels(outc)
        plt.sca(ax)

    def _findShockLoc(self, prof, ind):
        cs = prof.data["csound"][ind]
        vel = prof.data["velocity"][ind]
        # Find location of shock
        s = np.count_nonzero(cs)
        fs = False
        k = -1
        for k in range(0, s - 1):
            if vel[k + 1] >= cs[k] and vel[k] < cs[k]:
                fs = True
                break

        if not fs:
            for k in range(s - 1, 0):
                if vel[k + 1] >= -cs[k] and vel[k] < -cs[k]:
                    fs = True
                    break
        return fs, k

    def _showShockLoc(self, prof, fig, ax, xaxis, yrng, ind):
        fs, k = self._findShockLoc(prof, ind)
        # check we are either side of shock
        if fs:
            xx = [xaxis[ind][k], xaxis[ind][k]]
            ax.plot(xx, yrng, "--", color=self.colors["clr_DarkGray"], linewidth=2)

    def _getMassFrac(self, data, i, ind, log=False, prof=True):
        if prof:
            if "logdq" in data:
                scale = 10 ** (data.logdq[ind])
            elif "dq" in data:
                scale = data.dq[ind]
            elif "dm" in data:
                scale = data.dm[ind] / (self.msun * data.star_mass)
            else:
                raise AttributeError(
                    "No suitable mass co-ordinate available for _getMassFrac, need either logdq, dq or dm in profile"
                )
        else:
            scale = 1.0

        if log:
            x = np.sum(10 ** data.data[i][ind] * scale)
        else:
            x = np.sum(data.data[i][ind] * scale)

        return x

    def _getMassIso(self, data, i, massInd, log=False, prof=True):
        mcen = 0.0
        try:
            mcen = data.M_center
        except AttributeError:
            pass
        mass = data.star_mass - mcen / self.msun

        return self._getMassFrac(data, i, massInd, log=log, prof=prof) * mass

    def _addExtraLabelsToAxis(
        self,
        fig,
        labels,
        colors=None,
        num_left=0,
        num_right=0,
        left_pad=85,
        right_pad=85,
    ):

        total_num = len(labels)

        for idx, i in enumerate(labels):
            if type(i) is bytes:
                labels[idx] = i.decode()

        if colors is None:
            colors = ["k"]
            colors = colors * total_num

        scale = 2.5
        if num_left > 0:
            for i in range(num_left):
                axis = fig.add_subplot(num_left, 3, (i * 3) + 1)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.yaxis.set_major_locator(plt.NullLocator())
                axis.xaxis.set_major_locator(plt.NullLocator())
                axis.yaxis.set_minor_locator(plt.NullLocator())
                axis.xaxis.set_minor_locator(plt.NullLocator())
                axis.patch.set_facecolor("None")
                axis.plot(0, 0, color="w")
                scale = 2.0
                text = axis.set_ylabel(
                    labels[i], color=colors[i], labelpad=left_pad, fontsize=16
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=3, foreground="black"),
                        path_effects.Normal(),
                    ]
                )

        flip = False
        if num_right > 0:
            for j in range(num_right):
                i = num_left + j
                axis = fig.add_subplot(num_right, 3, ((j + 1) * 3))
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.yaxis.set_major_locator(plt.NullLocator())
                axis.xaxis.set_major_locator(plt.NullLocator())
                axis.yaxis.set_minor_locator(plt.NullLocator())
                axis.xaxis.set_minor_locator(plt.NullLocator())
                axis.patch.set_facecolor("None")
                axis.plot(0, 0, color="w")
                axis2 = axis.twinx()
                axis2.spines["top"].set_visible(False)
                axis2.spines["right"].set_visible(False)
                axis2.spines["bottom"].set_visible(False)
                axis2.spines["left"].set_visible(False)
                axis2.yaxis.set_major_locator(plt.NullLocator())
                axis2.xaxis.set_major_locator(plt.NullLocator())
                axis2.yaxis.set_minor_locator(plt.NullLocator())
                axis2.xaxis.set_minor_locator(plt.NullLocator())
                axis2.patch.set_facecolor("None")
                text = axis2.set_ylabel(
                    labels[i], color=colors[i], labelpad=right_pad, fontsize=16
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=3, foreground="black"),
                        path_effects.Normal(),
                    ]
                )

    def _addMixLabelsAxis(self, fig):
        self._setMixRegionsCol(kip=True)
        self._addExtraLabelsToAxis(
            fig,
            labels=self.mix_names,
            colors=self.mix_col,
            num_left=len(self.mix_names),
        )

    def setTitle(
        self,
        ax,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        name=None,
        model=None,
        age=None,
        age_units=None,
        fontCent=None,
        fontOther=None,
    ):

        if show_title_name:
            ax.set_title(name, loc="center", fontsize=fontCent)
        if show_title_model:
            ax.set_title("Model " + str(np.int(model)), loc="right", fontsize=fontOther)
        if show_title_age:
            s = "age " + "{:8.4e}".format(np.float(age))
            if age_units is None:
                s = s + " yrs"
            else:
                s = s + " " + age_units
            ax.set_title(s, loc="left", fontsize=fontOther)

    def _plotAnnotatedLine(
        self,
        ax,
        x,
        y,
        xmin,
        xmax,
        fy=None,
        ymin=None,
        ymax=None,
        annotate_line=False,
        label=None,
        points=False,
        xlog=False,
        ylog=False,
        xrev=False,
        yrev=False,
        linecol=None,
        linewidth=2,
        num_labels=5,
        linestyle="-",
        ind=None,
        fontsize=None,
        offset=None,
    ):
        if ind is not None:
            x = x[ind]
            y = y[ind]

        if xlog:
            ax.set_xscale("log", nonpositive="clip")
        if ylog:
            ax.set_yscale("log", nonpositive="clip")
            if np.all(y <= 0):
                return

        if fy is not None:
            y = fy(y)

        if ymin is None or ymax is None:
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)

            if ymin == ymax:
                ymin = 0.5 * ymin
                ymax = 1.5 * ymax

        if xmin is None or xmax is None:
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)

        if xmin > xmax:
            xrev = True
            tmp = xmin
            xmin = xmax
            xmax = tmp

        if label is not None and type(label) is bytes:
            label = label.decode()

        # y[np.logical_not(np.isfinite(y))]=ymin-(ymax-ymin)
        if linecol is None:
            (line,) = ax.plot(x, y, linestyle=linestyle, linewidth=linewidth)
        else:
            (line,) = ax.plot(
                x, y, linestyle=linestyle, color=linecol, linewidth=linewidth
            )
        if points:
            ax.scatter(x, y)

        if annotate_line:
            self._annotateLine(
                ax,
                x,
                y,
                num_labels,
                xmin,
                xmax,
                xlog=xlog,
                text=label,
                line=line,
                fontsize=fontsize,
                offset=offset,
            )

        ax.set_xlim(xmin, xmax)
        if xrev:
            ax.invert_xaxis()

        ax.set_ylim(ymin, ymax)
        if yrev:
            ax.invert_yaxis()

        # self._setTicks(ax)

        return x, y

    def _setupPlot(self, fig, ax):
        if fig is None:
            fig = plt.figure(figsize=(12, 12))
        if ax is None:
            ax = fig.add_subplot(111)
        return fig, ax

    def _setupHist(self, fig, ax, m, minMod, maxMod):

        if m.hist._loaded is False:
            raise ValueError("Must call loadHistory first")

        fig, ax = self._setupPlot(fig, ax)

        if maxMod < 0:
            maxMod = m.hist.model_number[-1]
        modelIndex = (m.hist.model_number >= minMod) & (m.hist.model_number <= maxMod)

        return fig, ax, modelIndex

    def _setupProf(self, fig, ax, m, model, label="plot"):

        if m.prof._loaded is False:
            raise ValueError("Must call loadProfile first")

        fig, ax = self._setupPlot(fig, ax)

        if model is not None:
            try:
                if m.prof.head["model_number"] != model:
                    m.loadProfile(num=int(model))
            except:
                m.loadProfile(num=int(model))
        return fig, ax

    def _setYLabel(self, fig, ax, ylabel, default=None, color="k"):
        ax.set_ylabel(self.safeLabel(ylabel, default), color=color)

    def _setXLabel(self, fig, ax, xlabel, default=None, color="k"):
        ax.set_xlabel(self.safeLabel(xlabel, default), color=color)

    def _plotY2(
        self,
        fig,
        ax,
        x,
        data,
        xrngL,
        xlog,
        xrev,
        mInd,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        points=False,
    ):

        if y2 is not None:
            ax2 = ax.twinx()
            ax2.set_label("abun_ax2")
            y = data[y2]
            px, py = self._plotAnnotatedLine(
                ax2,
                x,
                y,
                fy=fy2,
                xmin=xrngL[0],
                xmax=xrngL[1],
                ymin=y2rng[0],
                ymax=y2rng[1],
                annotate_line=False,
                label=self.safeLabel(y2label, y2),
                points=points,
                xlog=xlog,
                ylog=y2log,
                xrev=xrev,
                yrev=y2rev,
                linecol=y2col,
                ind=mInd,
            )

            if y2Textcol is None:
                y2labcol = y2col
            else:
                y2labcol = y2Textcol

            self._setYLabel(fig, ax2, y2label, y2, color=y2labcol)

        plt.sca(ax)

    def _decay2Stable(self, data, abun_list, ind, log=False, prefix="", prof=True):
        res = []
        for i, j, p in zip(self.stable_isos, self._stable_a, self._stable_charge):
            res.append({"name": i, "p": p, "a": j, "n": j - p, "mass": 0})

        msum = 0
        for i in abun_list:
            element, p, n = self._getIso(i, prefix=prefix)
            a = p + n
            massFrac = self._getMassFrac(data, i, ind, log=log, prof=prof)
            for idj, j in enumerate(res):
                if j["a"] != a:
                    continue
                if (
                    self._jcode[idj] == 0
                    or (p >= self._stable_charge[idj] and self._jcode[idj] == 1)
                    or (p <= self._stable_charge[idj] and self._jcode[idj] == 2)
                    or (p == self._stable_charge[idj] and self._jcode[idj] == 3)
                ):
                    res[idj]["mass"] = res[idj]["mass"] + massFrac
                    msum = msum + massFrac

        for idx, i in enumerate(res):
            res[idx]["mass"] = res[idx]["mass"] / msum

        return res

    def get_solar(self):
        self.is_solar_set()
        res = []
        for i, j, p, m in zip(
            self.stable_isos, self._stable_a, self._stable_charge, self.sol_comp
        ):
            res.append({"name": i, "p": p, "a": j, "mass": m})
        return res

    def _plotMultiProf(
        self,
        m,
        list_y=[],
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[None, None],
        y1scale=None,
        model=None,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        y1col="k",
        y1log=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        show_core_loc=False,
        colors=None,
        y1label=None,
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        _axlabel="",
    ):

        fig, ax = self._setupProf(fig, ax, m, model, label=_axlabel)

        if not type(xaxis) is np.ndarray:
            x = m.prof.data[xaxis]
        else:
            x = xaxis

        x, xrngL, mInd = self._setXAxis(x, xmin, xmax, fx)

        num_plots = len(list_y)
        # Helps when we have many elements not on the plot that stretch the colormap
        if rand_col:
            random.shuffle(list_y)

        if num_plots > 1:
            self._cycleColors(ax, colors, cmap, num_plots)
            linecol = None
        else:
            linecol = y1col

        if y1scale is not None:
            if y1rng[0] is not None:
                y1rng = np.array(y1rng) / y1scale

        for i in list_y:
            if type(i) is str:
                y = m.prof.data[i]
            else:
                y = i

            if y1scale is not None:
                y = y / y1scale

            self._plotAnnotatedLine(
                ax=ax,
                x=x,
                y=y,
                fy=fy,
                xmin=xrngL[0],
                xmax=xrngL[1],
                ymin=y1rng[0],
                ymax=y1rng[1],
                annotate_line=annotate_line,
                label=self.safeLabel(None, i),
                points=points,
                ylog=y1log,
                num_labels=num_labels,
                linestyle=linestyle,
                xrev=xrev,
                xlog=xlog,
                ind=mInd,
                linecol=linecol,
            )

        if show_burn:
            self._plotBurnRegions(
                m,
                ax,
                x,
                m.prof.data[i],
                show_line=show_burn_line,
                show_x=show_burn_x,
                ind=mInd,
            )

        if show_mix:
            self._plotMixRegions(
                m,
                ax,
                x,
                m.prof.data[i],
                show_line=show_mix_line,
                show_x=show_mix_x,
                ind=mInd,
            )

        if show_burn_labels or show_mix_labels:
            self._showBurnMixLegend(ax, burn=show_burn_labels, mix=show_mix_labels)

        if show_core_loc:
            self._plotCoreLoc(m.prof, ax, xaxis, x, ax.get_ylim()[0], ax.get_ylim()[1])

        if show_shock:
            self._showShockLoc(m.prof, fig, ax, x, ax.get_ylim(), mInd)

        if y2 is not None:
            self._plotY2(
                fig,
                ax,
                x,
                m.prof.data,
                xrngL,
                xlog,
                xrev,
                mInd,
                y2,
                y2rng,
                fy2,
                y2Textcol,
                y2label,
                y2rev,
                y2log,
                y2col,
                points,
            )

        self._setXLabel(fig, ax, xlabel, xaxis)

        if len(list_y):
            if type(list_y[0]) is str:
                self._setYLabel(fig, ax, y1label, list_y[0])

        if title is not None:
            ax.set_title(title)
        elif show_title_name or show_title_model or show_title_age:
            self.setTitle(
                ax,
                show_title_name,
                show_title_model,
                show_title_age,
                "",
                m.prof.head["model_number"],
                m.prof.head["star_age"],
            )

        if show:
            plt.show()

    def _plotMultiHist(
        self,
        m,
        list_y=[],
        model=None,
        show=True,
        ax=None,
        xaxis="model_number",
        xmin=None,
        xmax=None,
        y1rng=[None, None],
        y1log=False,
        y1scale=None,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        fig=None,
        fx=None,
        fy=None,
        minMod=-1,
        maxMod=-1,
        y1col="k",
        show_title_name=False,
        title=None,
        annotate_line=True,
        linestyle="-",
        colors=None,
        show_core=False,
        y1label=None,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
    ):

        fig, ax, modelIndex = self._setupHist(fig, ax, m, minMod, maxMod)

        if not type(xaxis) is np.ndarray:
            x = m.hist.data[xaxis]
        else:
            x = xaxis

        x, xrngL, mInd = self._setXAxis(x[modelIndex], xmin, xmax, fx)

        num_plots = len(list_y)
        if num_plots > 1:
            self._cycleColors(ax, colors, cmap, num_plots)
            linecol = None
        else:
            linecol = y1col

        self._cycleColors(ax, colors, cmap, len(list_y))

        if y1scale is not None:
            if y1rng[0] is not None:
                y1rng = np.array(y1rng) / y1scale

        for i in list_y:
            if type(i) is str:
                y = m.hist.data[i]
            else:
                y = i

            if y1scale is not None:
                y = y / y1scale

            self._plotAnnotatedLine(
                ax=ax,
                x=x,
                y=y,
                fy=fy,
                xmin=xrngL[0],
                xmax=xrngL[1],
                ymin=y1rng[0],
                ymax=y1rng[1],
                annotate_line=annotate_line,
                label=self.safeLabel(None, i),
                points=points,
                ylog=y1log,
                num_labels=num_labels,
                xlog=xlog,
                xrev=xrev,
                ind=mInd,
                linecol=linecol,
            )

        if y2 is not None:
            self._plotY2(
                fig,
                ax,
                x,
                m.hist.data,
                xrngL,
                xlog,
                xrev,
                mInd,
                y2,
                y2rng,
                fy2,
                y2Textcol,
                y2label,
                y2rev,
                y2log,
                y2col,
                points,
            )

        if show_core:
            self._showMassLocHist(m, fig, ax, x, list_y[0], mInd)

        self._setXLabel(fig, ax, xlabel, xaxis)

        if len(list_y) > 0:
            if type(list_y[0]) is str:
                self._setYLabel(fig, ax, y1label, i)

        if title is not None:
            ax.set_title(title)
        elif show_title_name:
            self.setTitle(
                ax, show_title_name, show_title_model=False, show_title_age=False
            )

        if show:
            plt.show()

    def plotAbun(
        self,
        m,
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[10**-3, 1.0],
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        abun=None,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_core_loc=False,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"Abundance",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
    ):

        if abun is None:
            abun_list = m.prof.listAbun()
            log = ""
        else:
            abun_list = abun
            log = ""

        if len(abun_list) == 0:
            raise ValueError("Found no isotopes in the data")

        abun_log = True
        if len(log) > 0:
            abun_log = False

        self._plotMultiProf(
            m,
            list_y=abun_list,
            y1log=abun_log,
            _axlabel="abun",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def _plotAbunByA(
        self,
        data=None,
        data2=None,
        prefix="",
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        abun=None,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        colors=None,
        abun_random=False,
        min_abun=10**-16,
        line_labels=True,
        y1rng=[None, None],
        ind=None,
        ind2=None,
        model_number=-1,
        age=-1,
        stable=False,
        prof=True,
        fontsize=None,
        offset=None,
        silent=False,
    ):

        fig, ax = self._setupPlot(fig, ax)

        if abun is None:
            abun_names = data.listAbun(prefix=prefix)
        else:
            abun_names = abun

        if len(abun_names) == 0:
            raise ValueError("Found no isotopes in the data")

        log_abun = False
        if "log" in prefix:
            log_abun = True

        if y1rng[0] is not None:
            min_abun = np.minimum(min_abun, np.min(y1rng))

        # Filter out low level iso's:
        remove = {}
        for i in abun_names:
            if np.all(data.data[i][ind] < min_abun):
                remove[i] = True
                if not silent:
                    print("Removing ", str(i))
                continue
            if data2 is not None:
                if np.all(data2.data[i][ind2] < min_abun):
                    remove[i] = True
                    if not silent:
                        print("Removing ", str(i))
        abun_names = [i for i in abun_names if i not in remove]

        if stable:
            abun_solar = self.get_solar()
            abun_data_stable = self._decay2Stable(
                data, abun_names, ind, log_abun, prefix=prefix, prof=prof
            )

            for idx, i in enumerate(abun_data_stable):
                if abun_solar[idx]["mass"] > 0:
                    abun_data_stable[idx]["mass"] = (
                        abun_data_stable[idx]["mass"] / abun_solar[idx]["mass"]
                    )
                else:
                    abun_data_stable[idx]["mass"] = -1

            if data2 is not None:
                abun_data_stable2 = self._decay2Stable(
                    data2, abun_names, ind, log_abun, prefix=prefix, prof=prof
                )
                for idx, i in enumerate(abun_data_stable):
                    if abun_solar[idx]["mass"] > 0:
                        abun_data_stable2[idx]["mass"] = (
                            abun_data_stable2[idx]["mass"] / abun_solar[idx]["mass"]
                        )
                    else:
                        abun_data_stable2[idx]["mass"] = -1

            # _decay2Stable gives you a new abun list to work with so don't use the old one
            abun_names = [i["name"] for i in abun_data_stable]
            prefix = ""

        ele_names = []
        iso_mass = []
        abun_mass = []
        for idx, i in enumerate(abun_names):
            name, mass = self._splitIso(i, prefix=prefix)
            if "neut" in name or "prot" in name:
                continue
            ele_names.append(name)
            iso_mass.append(mass)
            if stable:
                if abun_data_stable[idx] == 0:
                    abun_mass.append(-1)
                else:
                    abun_mass.append(abun_data_stable[idx]["mass"])
            else:
                abun_mass.append(self._getMassFrac(data, i, ind, log_abun, prof=prof))

            if data2 is not None:
                if stable:
                    if abun_data_stable2[idx]["mass"] <= 0:
                        abun_mass[-1] = -1
                    else:
                        abun_mass[-1] = abun_mass[-1] / abun_data_stable2[idx]["mass"]
                else:
                    xx = self._getMassFrac(data2, i, ind2, log_abun, prof=prof)
                    if xx == 0:
                        abun_mass[-1] = -1
                    else:
                        abun_mass[-1] = abun_mass[-1] / self._getMassFrac(
                            data2, i, ind2, log_abun, prof=prof
                        )

        uniq_names = set(i for i in ele_names)
        sorted_names = sorted(uniq_names, key=self.elements.index)

        self._cycleColors(ax, colors, cmap, len(uniq_names), abun_random)

        abun_mass = np.array(abun_mass)
        iso_mass = np.array(iso_mass)

        if xmin is None:
            xmin = np.min(iso_mass[abun_mass > 0])
        if xmax is None:
            xmax = np.max(iso_mass[abun_mass > 0])

        if y1rng[0] is None:
            ymin = np.nanmin(abun_mass[abun_mass > 0])
        else:
            ymin = y1rng[0]

        if y1rng[1] is None:
            ymax = np.nanmax(abun_mass[abun_mass > 0])
        else:
            ymax = y1rng[1]

        for i in sorted_names:
            x = []
            y = []
            for idx, j in enumerate(ele_names):
                if i == j:
                    x.append(iso_mass[idx])
                    y.append(abun_mass[idx])

            x = np.array(x)
            y = np.array(y)
            if np.size(x) == 0:
                continue

            ind = np.argsort(x)
            x = x[ind]
            y = y[ind]

            ind = (y >= ymin) & (y <= ymax)
            x = x[ind]
            y = y[ind]

            # print(i,x,y)
            self._plotAnnotatedLine(
                ax=ax,
                x=x,
                y=y,
                ylog=True,
                ymin=ymin,
                ymax=ymax,
                points=True,
                annotate_line=line_labels,
                label=i,
                num_labels=1,
                xmin=xmin,
                xmax=xmax,
                fontsize=fontsize,
                offset=offset,
            )

        ax.set_xlabel("A")
        if data2 is None:
            ax.set_ylabel(r"$\log_{10}$ Abundance")
        else:
            ax.set_ylabel(r"$\log_{10}\left(\frac{\rm{Abun}_1}{\rm{Abun}_2}\right)$")

        if title is not None:
            ax.set_title(title)
        elif show_title_name or show_title_model or show_title_age:
            self.setTitle(
                ax,
                show_title_name,
                show_title_model,
                show_title_age,
                "Production",
                model_number,
                age,
            )

        ax.set_xlim(xmin - 5, xmax + 5)

        diff = (np.log10(ymax) - np.log10(ymin)) / 10.0
        ax.set_ylim(ymin - diff, ymax + diff)

        if show:
            plt.show()

    def plotAbunByA(
        self,
        m,
        m2=None,
        plot_type="profile",
        prefix="",
        model=-1,
        model2=-1,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        mass_range=None,
        abun=None,
        min_abun=10**-16,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        colors=None,
        abun_random=False,
        line_labels=True,
        y1rng=[None, None],
        ind=None,
        ind2=None,
        mass_range2=None,
        stable=False,
        y2rng=None,
        fontsize=None,
        offset=None,
        silent=False,
    ):

        data, data2, ind, ind2, age, model, prof = self._abunPlotSetup(
            m, m2, plot_type, model, model2, ind, ind2, mass_range, mass_range2
        )

        self._plotAbunByA(
            data=data,
            data2=data2,
            prefix=prefix,
            show=show,
            ax=ax,
            xmin=xmin,
            xmax=xmax,
            abun=abun,
            fig=fig,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            title=title,
            min_abun=min_abun,
            cmap=cmap,
            colors=colors,
            abun_random=abun_random,
            line_labels=line_labels,
            y1rng=y1rng,
            stable=stable,
            ind=ind,
            ind2=ind2,
            age=age,
            model_number=model,
            prof=prof,
            fontsize=fontsize,
            offset=offset,
            silent=silent,
        )

    def _plotAbunByZ(
        self,
        data=None,
        data2=None,
        prefix="",
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        abun=None,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        colors=None,
        abun_random=False,
        min_abun=10**-16,
        line_labels=True,
        y1rng=[None, None],
        ind=None,
        ind2=None,
        model_number=-1,
        age=-1,
        stable=False,
        prof=True,
        fontsize=None,
        offset=None,
        silent=False,
    ):

        fig, ax = self._setupPlot(fig, ax)

        if abun is None:
            abun_names = data.listAbun(prefix=prefix)
        else:
            abun_names = abun

        if len(abun_names) == 0:
            raise ValueError("Found no isotopes in the data")

        log_abun = False
        if "log" in prefix:
            log_abun = True

        if y1rng[0] is not None:
            min_abun = np.minimum(min_abun, np.min(y1rng))

        # Filter out low level iso's:
        remove = {}
        for i in abun_names:
            if np.all(data.data[i][ind] < min_abun):
                remove[i] = True
                if not silent:
                    print("Removing ", str(i))
                continue
            if data2 is not None:
                if np.all(data2.data[i][ind2] < min_abun):
                    remove[i] = True
                    if not silent:
                        print("Removing ", str(i))
        abun_names = [i for i in abun_names if i not in remove]

        if stable:
            abun_solar = self.get_solar()
            abun_data_stable = self._decay2Stable(
                data, abun_names, ind, log_abun, prefix=prefix, prof=prof
            )

            for idx, i in enumerate(abun_data_stable):
                if abun_solar[idx]["mass"] > 0:
                    abun_data_stable[idx]["mass"] = (
                        abun_data_stable[idx]["mass"] / abun_solar[idx]["mass"]
                    )
                else:
                    abun_data_stable[idx]["mass"] = -1

            if data2 is not None:
                abun_data_stable2 = self._decay2Stable(
                    data2, abun_names, ind, log_abun, prefix=prefix, prof=prof
                )
                for idx, i in enumerate(abun_data_stable):
                    if abun_solar[idx]["mass"] > 0:
                        abun_data_stable2[idx]["mass"] = (
                            abun_data_stable2[idx]["mass"] / abun_solar[idx]["mass"]
                        )
                    else:
                        abun_data_stable2[idx]["mass"] = -1

            # _decay2Stable gives you a new abun list to work with so don't use the old one
            abun_names = [i["name"] for i in abun_data_stable]
            prefix = ""

        ele_names = []
        iso_mass = []
        abun_mass = []
        for idx, i in enumerate(abun_names):
            name, mass = self._splitIso(i, prefix=prefix)
            if "neut" in name or "prot" in name:
                continue
            ele_names.append(name)
            iso_mass.append(mass)
            if stable:
                if abun_data_stable[idx] == 0:
                    abun_mass.append(-1)
                else:
                    abun_mass.append(abun_data_stable[idx]["mass"])
            else:
                abun_mass.append(self._getMassFrac(data, i, ind, log_abun, prof=prof))

            if data2 is not None:
                if stable:
                    if abun_data_stable2[idx]["mass"] <= 0:
                        abun_mass[-1] = -1
                    else:
                        abun_mass[-1] = abun_mass[-1] / abun_data_stable2[idx]["mass"]
                else:
                    xx = self._getMassFrac(data2, i, ind2, log_abun, prof=prof)
                    if xx == 0:
                        abun_mass[-1] = -1
                    else:
                        abun_mass[-1] = abun_mass[-1] / self._getMassFrac(
                            data2, i, ind2, log_abun, prof=prof
                        )

        uniq_names = set(i for i in ele_names)
        sorted_names = sorted(uniq_names, key=self.elements.index)

        self._cycleColors(ax, colors, cmap, len(uniq_names), abun_random)

        abun_mass = np.array(abun_mass)
        iso_mass = np.array(iso_mass)

        if xmin is None:
            xmin = self.elements.index(ele_names[0])
        if xmax is None:
            xmax = self.elements.index(ele_names[-1])

        if y1rng[0] is None:
            ymin = np.nanmin(abun_mass[abun_mass > 0])
        else:
            ymin = y1rng[0]

        if y1rng[1] is None:
            ymax = np.nanmax(abun_mass[abun_mass > 0])
        else:
            ymax = y1rng[1]

        for i in sorted_names:
            x = []
            y = []
            for idx, j in enumerate(ele_names):
                if i == j:
                    x.append(iso_mass[idx])
                    y.append(abun_mass[idx])

            print(i, self.elements.index(i), y)
            x = self.elements.index(i)
            y = np.sum(y)

            ax.scatter(x, y, s=50, color="black")

        ax.set_xlabel("Z")
        if data2 is None:
            if stable:
                ax.set_ylabel("[X]")
            else:
                ax.set_ylabel("X")
        else:
            ax.set_ylabel(r"$\log_{10}\left(\frac{\rm{Abun}_1}{\rm{Abun}_2}\right)$")

        if title is not None:
            ax.set_title(title)
        elif show_title_name or show_title_model or show_title_age:
            self.setTitle(
                ax,
                show_title_name,
                show_title_model,
                show_title_age,
                "Production",
                model_number,
                age,
            )

        ax.set_xlim(xmin - 5, xmax + 5)

        ax.set_yscale("log")
        # diff=(np.log10(ymax)-np.log10(ymin))/10.0
        # ax.set_ylim(ymin-diff,ymax+diff)

        if show:
            plt.show()

    def plotAbunByZ(
        self,
        m,
        m2=None,
        plot_type="profile",
        prefix="",
        model=-1,
        model2=-1,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        mass_range=None,
        abun=None,
        min_abun=10**-16,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        colors=None,
        abun_random=False,
        line_labels=True,
        y1rng=[None, None],
        ind=None,
        ind2=None,
        mass_range2=None,
        stable=False,
        y2rng=None,
        fontsize=None,
        offset=None,
        silent=False,
    ):

        data, data2, ind, ind2, age, model, prof = self._abunPlotSetup(
            m, m2, plot_type, model, model2, ind, ind2, mass_range, mass_range2
        )

        self._plotAbunByZ(
            data=data,
            data2=data2,
            prefix=prefix,
            show=show,
            ax=ax,
            xmin=xmin,
            xmax=xmax,
            abun=abun,
            fig=fig,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            title=title,
            min_abun=min_abun,
            cmap=cmap,
            colors=colors,
            abun_random=abun_random,
            line_labels=line_labels,
            y1rng=y1rng,
            stable=stable,
            ind=ind,
            ind2=ind2,
            age=age,
            model_number=model,
            prof=prof,
            fontsize=fontsize,
            offset=offset,
            silent=silent,
        )

    def _abunPlotSetup(
        self, m, m2, plot_type, model, model2, ind, ind2, mass_range, mass_range2
    ):
        data = None
        data2 = None

        if plot_type == "history":
            if model > 0:
                data = m.hist.data[np.where(m.hist.model_number == model)][0]
            else:
                raise ValueError("Must set model")
            if m2 is not None and model2 > 0:
                data2 = m2.hist.data[np.where(m2.hist.model_number == model2)][0]
            else:
                data2 = None

            ind = None
            ind2 = None

            age = data.data["star_age"]
            prof = False

        else:
            data = m.prof

            if m2 is not None:
                data2 = m2.prof

            if mass_range is None:
                mass_range = [0.0, m.prof.star_mass]

            massInd = (m.prof.mass >= mass_range[0]) & (m.prof.mass <= mass_range[1])

            if mass_range2 is None:
                mass_range2 = mass_range

            if m2 is not None:
                massInd2 = (m2.prof.mass >= mass_range[0]) & (
                    m2.prof.mass <= mass_range[1]
                )

            if ind is not None:
                ind = massInd & ind
            else:
                ind = massInd

            if ind2 is not None and m2 is not None:
                ind2 = massInd2 & ind2
            elif m2 is not None:
                ind2 = massInd2

            age = m.prof.star_age
            model = m.prof.model_number
            prof = True

        return data, data2, ind, ind2, age, model, prof

    def plotAbunPAndN(
        self,
        m,
        plot_type="profile",
        model=-1,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        mass_range=None,
        abun=None,
        ind=None,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        mass_frac_rng=[10**-10, 1.0],
        prefix="",
        bounds=0,
        n_minus_p=False,
    ):
        """
        bounds option applies to the mass fractions for each isotope compared to the mass_frac_rng
        0: Cut, so anything above or below mass_frac_rng is white
        1: Truncate, so anything above or below mass_frac_rng is given the nearest color in the colormap
        2: Warn, anything below is white, above is black
        """
        data, data2, ind, ind2, age, model, prof = self._abunPlotSetup(
            m, None, plot_type, model, -1, ind, None, mass_range, None
        )

        self._plotAbunPAndN(
            data,
            show=show,
            ax=ax,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            abun=abun,
            ind=ind,
            fig=fig,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            title=title,
            cmap=cmap,
            mass_frac_rng=mass_frac_rng,
            model_number=model,
            age=age,
            prefix=prefix,
            prof=prof,
            bounds=bounds,
            n_minus_p=n_minus_p,
        )

    def _plotAbunPAndN(
        self,
        data,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        ind=None,
        abun=None,
        ymin=None,
        ymax=None,
        fig=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        title=None,
        cmap=plt.cm.gist_ncar,
        mass_frac_rng=[10**-10, 1.0],
        model_number=-1,
        age=-1,
        prefix="",
        element_names=True,
        prof=True,
        bounds=0,
        n_minus_p=False,
    ):

        fig, ax = self._setupPlot(fig, ax)

        if abun is None:
            abun_names = data.listAbun(prefix=prefix)
        else:
            abun_names = abun

        if len(abun_names) == 0:
            raise ValueError("Found no isotopes in the data")

        log_abun = False
        if "log" in prefix:
            log_abun = True

        name_all = []
        names = []
        proton = []
        neutron = []
        mass = []
        for i in abun_names:
            na, pr, ne = self._getIso(i, prefix=prefix)
            names.append(na)
            proton.append(int(pr))
            neutron.append(int(ne))
            mass.append(self._getMassFrac(data, i, ind, log_abun, prof=prof))

        proton = np.array(proton)
        neutron = np.array(neutron)
        mass = np.log10(np.array(mass))

        if n_minus_p:
            xdata = neutron - proton
        else:
            xdata = neutron

        if xmin is None:
            xmin = np.min(xdata)
        if xmax is None:
            xmax = np.max(xdata)

        if ymin is None:
            ymin = np.min(proton)

        if ymax is None:
            ymax = np.max(proton)

        min_col = np.log10(mass_frac_rng[0])
        max_col = np.log10(mass_frac_rng[1])

        for idx, i in enumerate(abun_names):
            if "neut" in i or "prot" in i:
                continue
            if (
                proton[idx] >= ymin
                and proton[idx] <= ymax
                and xdata[idx] >= xmin
                and xdata[idx] <= xmax
            ):
                loctuple = (float(xdata[idx] - 0.5), float(proton[idx] - 0.5))
                if mass[idx] >= min_col and mass[idx] <= max_col:
                    ax.add_patch(
                        mpl.patches.Rectangle(
                            loctuple,
                            1.0,
                            1.0,
                            facecolor=cmap((mass[idx] - min_col) / (max_col - min_col)),
                        )
                    )
                else:
                    if bounds == 0:
                        ax.add_patch(
                            mpl.patches.Rectangle(loctuple, 1.0, 1.0, fill=False)
                        )
                    elif bounds == 1:
                        if mass[idx] < min_col:
                            ax.add_patch(
                                mpl.patches.Rectangle(
                                    loctuple, 1.0, 1.0, facecolor=cmap(0.0)
                                )
                            )
                        else:
                            ax.add_patch(
                                mpl.patches.Rectangle(
                                    loctuple, 1.0, 1.0, facecolor=cmap(1.0)
                                )
                            )
                    elif bounds == 2:
                        if mass[idx] < min_col:
                            ax.add_patch(
                                mpl.patches.Rectangle(loctuple, 1.0, 1.0, fill=False)
                            )
                        else:
                            ax.add_patch(
                                mpl.patches.Rectangle(loctuple, 1.0, 1.0, facecolor="k")
                            )
                    else:
                        raise ValueError("Bad value for bounds option, either 0,1 or 2")

        label_loc = xmin - 2
        if element_names:
            uniq_names = set(names)
            for i in uniq_names:
                if "neut" in i or "prot" in i:
                    continue
                ax.text(
                    label_loc, self.elements.index(i) - 0.25, i.title(), fontsize=14
                ).set_clip_on(True)

        norm = mpl.colors.Normalize(vmin=min_col, vmax=max_col)

        outArr = np.zeros((10, 10))
        outArr[:] = np.nan
        im = ax.imshow(outArr, aspect="auto", cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax=ax)
        cb.solids.set_edgecolor("face")
        cb.set_label(r"$\log_{10}\,$ \rm{Mass Frac}")

        if n_minus_p:
            ax.set_xlabel("Neutrons - Protons")
        else:
            ax.set_xlabel("Neutrons")
        ax.set_ylabel("Protons")

        xminoffset = -5
        xmaxoffset = 1
        yminoffset = -1
        ymaxoffset = 1

        ax.set_xlim(xmin + xminoffset, xmax + xmaxoffset)
        ax.set_ylim(ymin + yminoffset, ymax + ymaxoffset)

        if title is not None:
            ax.set_title(title)
        elif show_title_name or show_title_model or show_title_age:
            self.setTitle(
                ax,
                show_title_name,
                show_title_model,
                show_title_age,
                "Network",
                model_number,
                age,
            )

        if show:
            plt.show()

    def plotAbunHist(
        self,
        m,
        prefix="center_",
        show=True,
        ax=None,
        xaxis="model_number",
        xmin=None,
        xmax=None,
        y1rng=[10**-5, 1.0],
        y1log=True,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        fig=None,
        fx=None,
        fy=None,
        minMod=-1,
        maxMod=-1,
        y1label="Abundance",
        show_title_name=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        show_core=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
    ):

        abun_list = self._listAbun(m.hist, prefix=prefix)

        self._plotMultiHist(
            m,
            list_y=abun_list,
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1log=y1log,
            y1label=y1label,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            fig=fig,
            fx=fx,
            fy=fy,
            minMod=minMod,
            maxMod=maxMod,
            show_title_name=show_title_name,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            show_core=show_core,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotNeu(
        self,
        m,
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[10**-3, 10**10.0],
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_core_loc=False,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"$\epsilon_{\nu}$",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
        y1log=True,
    ):

        neu_list = [
            "nonnucneu_brem",
            "nonnucneu_pair",
            "nonnucneu_phot",
            "nonnucneu_plas",
            "nonnucneu_reco",
        ]

        self._plotMultiProf(
            m,
            list_y=neu_list,
            y1log=y1log,
            _axlabel="neu",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotDynamo(
        self,
        m,
        xaxis="mass",
        model=None,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        xlabel=None,
        y1rng=None,
        y2rng=None,
        show_burn=False,
        show_mix=False,
        legend=True,
        annotate_line=True,
        fig=None,
        fx=None,
        fy=None,
        title=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        show_rotation=True,
        show_shock=False,
    ):

        fig, ax = self._setupProf(fig, ax, m, model)

        ax1_1 = fig.add_subplot(231)
        ax1_2 = fig.add_subplot(234)

        ax2_t1 = fig.add_subplot(233)
        ax2_t2 = fig.add_subplot(236)

        ax2_1 = ax2_t1.twinx()
        ax2_2 = ax2_t2.twinx()

        for i in [ax1_1, ax1_2, ax2_1, ax2_2, ax2_t1, ax2_t2]:
            i.spines["top"].set_visible(False)
            i.spines["right"].set_visible(False)
            i.spines["bottom"].set_visible(False)
            i.spines["left"].set_visible(False)
            i.yaxis.set_major_locator(plt.NullLocator())
            i.xaxis.set_major_locator(plt.NullLocator())
            i.yaxis.set_minor_locator(plt.NullLocator())
            i.xaxis.set_minor_locator(plt.NullLocator())
            i.patch.set_facecolor("None")

        ax1_1.plot(0, 0, color="w")
        ax1_2.plot(0, 0, color="w")
        ax2_1.plot(0, 0, color="w")
        ax2_2.plot(0, 0, color="w")

        ax2 = ax.twinx()

        x, xrngL, mInd = self._setXAxis(m.prof.data[xaxis], xmin, xmax, fx)

        # ind=(m.prof.data['dynamo_log_B_r']>-90)
        ax.plot(
            m.prof.data[xaxis],
            m.prof.data["dynamo_log_B_r"],
            label=r"$B_r$",
            linewidth=2,
            color="g",
        )
        # ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
        ax.plot(
            m.prof.data[xaxis],
            m.prof.data["dynamo_log_B_phi"],
            label=r"$B_{\phi}$",
            linewidth=2,
            color="b",
        )

        if show_rotation:
            ax2.plot(
                m.prof.data[xaxis],
                np.log10(m.prof.data["omega"]),
                "--",
                label=r"$\log_{10} \omega$",
                linewidth=2,
                color="r",
            )
            # ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
            ax2.plot(
                m.prof.data[xaxis],
                np.log10(m.prof.data["j_rot"]) - 20.0,
                "--",
                label=r"$\log_{10} j [10^{20}]$",
                linewidth=2,
                color="k",
            )

        scale = 2.1
        ax1_1.set_ylabel(
            r"$B_r$", color="g", labelpad=scale * mpl.rcParams["font.size"]
        )
        ax1_2.set_ylabel(
            r"$B_{\phi}$", color="b", labelpad=scale * mpl.rcParams["font.size"]
        )

        if show_rotation:
            ax2_1.set_ylabel(
                r"$\log_{10} \omega$",
                color="r",
                labelpad=scale * mpl.rcParams["font.size"],
            )
            ax2_2.set_ylabel(
                r"$\log_{10} j [10^{20}]$",
                color="k",
                labelpad=scale * mpl.rcParams["font.size"],
            )

        if show_burn:
            self._plotBurnRegions(
                m,
                ax,
                m.prof.data[xaxis],
                m.prof.data["dynamo_log_B_phi"],
                show_line=False,
                show_x=True,
                ind=mInd,
            )

        if show_mix:
            self._plotMixRegions(
                m,
                ax,
                m.prof.data[xaxis],
                m.prof.data["dynamo_log_B_phi"],
                show_line=False,
                show_x=True,
                ind=mInd,
            )

        if show_shock:
            self._showShockLoc(m.prof, fig, ax, x, ax.get_ylim(), mInd)

        self._setXLabel(fig, ax, xlabel, xaxis)
        self._setTicks(ax)
        self._setTicks(ax2)
        ax.set_xlim(xrngL)
        self._setYLim(ax, ax.get_ylim(), y1rng)
        self._setYLim(ax2, ax2.get_ylim(), y2rng)

        if title is not None:
            ax.set_title(title)
        elif show_title_name or show_title_model or show_title_age:
            self.setTitle(
                ax,
                show_title_name,
                show_title_model,
                show_title_age,
                "Dynamo",
                m.prof.head["model_number"],
                m.prof.head["star_age"],
            )

        if show:
            plt.show()

    def plotAngMom(
        self,
        m,
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[0.0, 10.0],
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"Angular Momentum",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
    ):

        list_y = []
        for i in m.prof:
            if "am_log_D" in i:
                list_y.append(i)

        self._plotMultiProf(
            m,
            list_y=list_y,
            y1log=True,
            _axlabel="angmom",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            show_burn_labels=show_burn_labels,
            show_mix_labels=show_mix_labels,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotBurn(
        self,
        m,
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[],
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_core_loc=False,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"$\epsilon$",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
    ):

        list_y = m.prof.listBurn()

        if not len(y1rng):
            maxy = 1.0
            for i in list_y:
                maxy = np.maximum(maxy, np.max(m.prof.data[i]))
            maxy = 10 ** (np.ceil(np.log10(maxy)))
            y1rng = [1.0, maxy]

        self._plotMultiProf(
            m,
            list_y=list_y,
            y1log=True,
            _axlabel="burn",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            show_core_loc=show_core_loc,
            show_burn_labels=show_burn_labels,
            show_mix_labels=show_mix_labels,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotMix(
        self,
        m,
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[1.0, 20],
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_core_loc=False,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"Mixing",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
    ):

        list_y = m.prof.listMix()

        self._plotMultiProf(
            m,
            list_y=list_y,
            _axlabel="mix",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            show_core_loc=show_core_loc,
            show_burn_labels=show_burn_labels,
            show_mix_labels=show_mix_labels,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotBurnHist(
        self,
        m,
        show=True,
        ax=None,
        xaxis="model_number",
        xmin=None,
        xmax=None,
        y1rng=[1, 10**10],
        y1log=True,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        fig=None,
        fx=None,
        fy=None,
        minMod=-1,
        maxMod=-1,
        y1label=r"$\epsilon$",
        show_title_name=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        show_core=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
    ):

        burn_list = m.hist.listBurn()

        self._plotMultiHist(
            m,
            list_y=burn_list,
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1log=y1log,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            fig=fig,
            fx=fx,
            fy=fy,
            minMod=minMod,
            maxMod=maxMod,
            y1label=y1label,
            show_title_name=show_title_name,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            show_core=show_core,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotLdivM(
        self,
        m,
        xaxis="log_Teff",
        y2=None,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        xlog=False,
        y1log=False,
        y1rng=[None, None],
        y2log=False,
        y1col="b",
        y2col="r",
        minMod=0,
        maxMod=-1,
        xrev=False,
        y1rev=False,
        y2rev=False,
        points=False,
        xlabel=None,
        y1label=None,
        y2label=None,
        fig=None,
        y2rng=[None, None],
        fx=None,
        fy1=None,
        fy2=None,
        show_core=False,
        y1Textcol=None,
        y2Textcol=None,
        show_title_name=False,
    ):

        list_y = [m.hist.log_L - np.log10(m.hist.star_mass)]

        self._plotMultiHist(
            m,
            list_y=list_y,
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1log=y1log,
            xlabel=xlabel,
            points=points,
            y1label=r"$\log_{10}\,\left(L/M\right)$",
            fig=fig,
            fx=fx,
            fy=fy1,
            minMod=minMod,
            maxMod=maxMod,
            annotate_line=False,
            show_title_name=show_title_name,
            show_core=show_core,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotProfile(
        self,
        m,
        model=None,
        xaxis="mass",
        y1="logT",
        y2=None,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        xlog=False,
        y1log=False,
        y2log=False,
        y1col="b",
        y2col="r",
        xrev=False,
        y1rev=False,
        y2rev=False,
        points=False,
        xlabel=None,
        y1label=None,
        y2label=None,
        show_burn=False,
        show_burn_x=False,
        show_burn_line=False,
        show_mix=False,
        show_mix_x=False,
        show_mix_line=False,
        y1Textcol=None,
        y2Textcol=None,
        fig=None,
        y1rng=[None, None],
        y2rng=[None, None],
        fx=None,
        fy1=None,
        fy2=None,
        show_burn_labels=False,
        show_mix_labels=False,
        show_title_name=False,
        title_name=None,
        show_title_model=False,
        show_title_age=False,
        y1linelabel=None,
        show_core_loc=False,
        show_shock=False,
        title=None,
    ):

        list_y = [y1]

        self._plotMultiProf(
            m,
            list_y=list_y,
            y1log=y1log,
            _axlabel="profile",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1col=y1col,
            xlabel=xlabel,
            points=points,
            annotate_line=False,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy1,
            show_core_loc=show_core_loc,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            show_burn_labels=show_burn_labels,
            show_mix_labels=show_mix_labels,
            show_mix_line=show_mix_line,
            show_burn_line=show_burn_line,
            show_mix_x=show_mix_x,
            show_burn_x=show_burn_x,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotHistory(
        self,
        m,
        xaxis="model_number",
        y1="star_mass",
        y2=None,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        xlog=False,
        y1log=False,
        y1rng=[None, None],
        y2log=False,
        y1col="b",
        y2col="r",
        minMod=0,
        maxMod=-1,
        xrev=False,
        y1rev=False,
        y2rev=False,
        points=False,
        xlabel=None,
        y1label=None,
        y2label=None,
        fig=None,
        y2rng=[None, None],
        fx=None,
        fy1=None,
        fy2=None,
        show_core=False,
        y1Textcol=None,
        y2Textcol=None,
        show_title_name=False,
    ):

        list_y = [y1]

        self._plotMultiHist(
            m,
            list_y=list_y,
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1log=y1log,
            xlabel=xlabel,
            points=points,
            fig=fig,
            fx=fx,
            fy=fy1,
            minMod=minMod,
            maxMod=maxMod,
            annotate_line=False,
            show_title_name=show_title_name,
            show_core=show_core,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotKip(
        self,
        m,
        show=True,
        reloadHistory=False,
        xaxis="num",
        ax=None,
        xrng=[None, None],
        mix=None,
        show_mix=True,
        cmin=None,
        cmax=None,
        burnMap=[mpl.cm.Purples_r, mpl.cm.hot_r],
        fig=None,
        yrng=None,
        ylabel=None,
        show_mass_loc=False,
        show_mix_labels=True,
        mix_alpha=1.0,
        step=1,
        y2=None,
        title=None,
        y2rng=None,
        zone_frac=1.0,
        mix_hatch=False,
        hatch_color="black",
        cbar_label=None,
        yaxis="mass",
        y1log=False,
    ):

        self.plotKip3(
            m,
            plot_type="history",
            xaxis="model_number",
            show=show,
            yaxis=yaxis,
            reloadHistory=reloadHistory,
            ax=ax,
            mod_min=xrng[0],
            mod_max=xrng[1],
            show_mix=show_mix,
            mix=mix,
            cmin=cmin,
            cmax=cmax,
            cmap=burnMap,
            fig=fig,
            yrng=yrng,
            ylabel=ylabel,
            show_mass_loc=show_mass_loc,
            show_mix_labels=show_mix_labels,
            mix_alpha=mix_alpha,
            xstep=step,
            y2=y2,
            title=title,
            y2rng=y2rng,
            zone_frac=zone_frac,
            mix_hatch=mix_hatch,
            hatch_color=hatch_color,
            cbar_label=cbar_label,
            y1log=y1log,
        )

    def plotKip2(
        self,
        m,
        show=True,
        reloadHistory=False,
        xaxis="num",
        yaxis="mass",
        ageZero=0.0,
        ax=None,
        xrng=[None, None],
        mix=None,
        cmin=None,
        cmax=None,
        burnMap=[mpl.cm.Purples_r, mpl.cm.hot_r],
        fig=None,
        yrng=None,
        show_mix=True,
        show_burn=True,
        show_mass_loc=False,
        show_mix_labels=True,
        mix_alpha=1.0,
        step=1,
        max_mass=99999.0,
        age_collapse=False,
        age_log=True,
        age_reverse=False,
        mod_out=None,
        xlabel=None,
        title=None,
        colorbar=True,
        burn=True,
        end_time=None,
        ylabel=None,
        age_zero=None,
        y2=None,
        y2rng=None,
        zone_frac=1.0,
        mix_hatch=False,
        hatch_color="black",
        cbar_label=None,
        y1log=False,
    ):

        self.plotKip3(
            m,
            plot_type="history",
            xaxis="star_age",
            show=show,
            yaxis=yaxis,
            reloadHistory=reloadHistory,
            ax=ax,
            mod_min=xrng[0],
            mod_max=xrng[1],
            show_mix=show_mix,
            mix=mix,
            cmin=cmin,
            cmax=cmax,
            cmap=burnMap,
            fig=fig,
            yrng=yrng,
            show_mass_loc=show_mass_loc,
            show_mix_labels=show_mix_labels,
            mix_alpha=mix_alpha,
            xstep=step,
            y2=y2,
            title=title,
            y2rng=y2rng,
            colorbar=colorbar,
            ylabel=ylabel,
            age_zero=age_zero,
            age_lookback=age_collapse,
            age_log=age_log,
            age_reverse=age_reverse,
            mod_index=mod_out,
            xlabel=xlabel,
            show_burn=burn,
            end_time=end_time,
            zone_frac=zone_frac,
            mix_hatch=mix_hatch,
            hatch_color=hatch_color,
            cbar_label=cbar_label,
            y1log=y1log,
        )

    # Will replace plotKip and plotKip2 when finished
    def plotKip3(
        self,
        m,
        plot_type="history",
        xaxis="model_number",
        yaxis="mass",
        zaxis="logT",
        xmin=None,
        xmax=None,
        mod_min=None,
        mod_max=None,
        yrng=None,
        xstep=1,
        xlabel=None,
        ylabel=None,
        title=None,
        show=True,
        reloadHistory=False,
        ax=None,
        fig=None,
        show_mix=True,
        mix=None,
        show_burn=True,
        show_outer_mass=True,
        cmin=None,
        cmax=None,
        cmap=[mpl.cm.Purples_r, mpl.cm.hot_r],
        colorbar=True,
        cbar_label=None,
        show_mass_loc=False,
        show_mix_labels=True,
        mix_alpha=1.0,
        cmap_merge=True,
        age_lookback=False,
        age_log=True,
        age_reverse=False,
        age_units="years",
        end_time=None,
        age_zero=None,
        y2=None,
        y2rng=None,
        mod_index=None,
        zlog=False,
        zone_frac=1.0,
        num_zones=None,
        mix_hatch=False,
        hatch_color="black",
        hatch_func=None,
        zaxis_norm=False,
        yaxis_norm=False,
        y2label=None,
        y2log=False,
        zaxis_contour=False,
        zaxis_levels=None,
        y1log=False,
        dbg=False,
        cbar_ax=None,
        cpad=0.0,
        cbar_extend="neither",
        zrng=None,
    ):

        if fig == None:
            fig = plt.figure(figsize=(12, 12))

        if title is not None:
            fig.suptitle(title)

        if plot_type == "history" and show_mix_labels:
            self._addMixLabelsAxis(fig)

        if ax == None:
            ax = fig.add_subplot(111)

        if y2 is not None:
            ax2 = ax.twinx()
            if y2log:
                ax2.set_yscale("log", nonpositive="clip")

        if y1log:
            ax.set_yscale("log", nonpositive="clip")

        if plot_type == "history":
            try:
                model_num = m.hist.model_number
            except (KeyError, AttributeError):
                raise ValueError("Must call loadHistory first")
        elif plot_type == "profile":
            try:
                model_num = m.prof.model_number
            except (KeyError, AttributeError):
                raise ValueError("Must load a profile file first")

            try:
                y = m.prof.data[yaxis]
            except (KeyError, AttributeError):
                raise ValueError("No value " + yaxis + " found")

        else:
            raise ValueError(
                "plot_type must be either history or profile, got " + plot_type
            )

        if not (xaxis == "model_number" or "age" in xaxis):
            raise ValueError(
                "kipenhan's can only plot model_number or age, got " + xaxis
            )

        radius = False
        if plot_type == "history":
            if yaxis == "mass":
                mix_prefix = "mix_"
                burn_prefix = "burn_"
                radius = False
            elif yaxis == "radius":
                mix_prefix = "mix_relr_"
                burn_prefix = "burn_relr_"
                radius = True
            else:
                raise ValueError(
                    "History based kipenhan's can only plot mass or radius, got "
                    + yaxis
                )

        # Extract Data
        data_x = []
        data_y = []
        data_z = []
        data_hatch = []
        burn = []
        mix_data = []
        hasdataz = False

        # Number of zones to plot

        if plot_type == "history":
            if num_zones is None:
                num_zones = np.max(m.hist.num_zones) * 1.0 / zone_frac

            if xaxis == "model_number":
                data_x = m.hist.model_number
            else:
                data_x = self._getSafeAgeHist(
                    m, age_lookback, age_zero, age_units, age_log, age_reverse, end_time
                )

            if dbg:
                print(data_x)

            modInd = self._getModInd(
                m, mod_index, mod_min, mod_max, xstep, xaxis, xmin, xmax
            )

            if radius:
                center = self._getSafeCenter(m, radius)
                if "radius" in m.hist:
                    data_y = np.linspace(
                        np.min(center[modInd]),
                        np.max(m.hist.data["radius"][modInd]),
                        int(num_zones),
                    )
                else:
                    data_y = np.linspace(
                        np.min(center[modInd]),
                        10 ** np.max(m.hist.data["log_R"][modInd]),
                        int(num_zones),
                    )
            else:
                center = self._getSafeCenter(m, radius)
                data_y = np.linspace(
                    np.min(center[modInd]),
                    np.max(m.hist.data["star_mass"][modInd]),
                    int(num_zones),
                )

            # May need to interpolate data:
            lin_x = np.linspace(
                data_x[modInd][0], data_x[modInd][-1], np.count_nonzero(data_x[modInd])
            )

            # Get burn data
            if show_burn:
                data_z = self._getHistBurnData(
                    m, data_x, data_y, modInd, burn_prefix, radius
                )
                data_z = self._rebinKipDataX(data_z, data_x[modInd], lin_x)
                if dbg:
                    print(np.nanmin(data_z), np.nanmax(data_z))
                hasdataz = True

            # Get mix data
            if show_mix:
                mix_data = self._getHistMixData(
                    m, data_x, data_y, modInd, mix, mix_prefix, radius
                )
                mix_data = self._rebinKipDataX(
                    mix_data, data_x[modInd], lin_x, nan=True, nan_value=1
                )

        else:
            show_mix = False
            show_burn = False
            if mod_min is None:
                mod_min = -1
                mod_max = -1
            ip = m.iterateProfiles(rng=[mod_min, mod_max], silent=True)
            count = 0
            zones = []
            for i in ip:
                data_x.append(m.prof.head[xaxis])
                data_y.append(m.prof.data[yaxis])

                if callable(zaxis):
                    zz = zaxis(m)
                else:
                    zz = m.prof.data[zaxis]

                if callable(hatch_func):
                    data_hatch.append(hatch_func(m))

                data_z.append(zz)

                if zaxis_norm:
                    data_z[-1] = data_z[-1] / np.max(data_z[-1])

                if yaxis_norm:
                    data_y[-1] = data_y[-1] / np.max(data_y[-1])

                zones.append(m.prof.head["num_zones"])
                count = count + 1

            data_x = np.hstack(data_x)
            hasdataz = True

            if xaxis == "star_age":
                data_x = self._getSafeAge(
                    data_x,
                    age_lookback,
                    age_zero,
                    age_units,
                    age_log,
                    age_reverse,
                    end_time,
                )

            if num_zones is None:
                num_zones = np.max(zones) * 1.0 / zone_frac

            if callable(hatch_func):
                data_hatch, _, _ = self._rebinKipDataXY(
                    data_hatch, data_x, data_y, count, num_zones
                )

            data_z, lin_x, data_y = self._rebinKipDataXY(
                data_z, data_x, data_y, count, num_zones
            )

        xmin = lin_x[0]
        xmax = lin_x[-1]

        ymin = data_y[0]
        ymax = data_y[-1]

        extent = (xmin, xmax, ymin, ymax)
        extent = np.double(np.array(extent))

        if cmap_merge:
            newCm = self.mergeCmaps(cmap, [[0.0, 0.5], [0.5, 1.0]])
        else:
            if not isinstance(cmap, str):
                newCm = cmap[-1]
            else:
                newCm = cmap

        if hasdataz:
            if zlog:
                # Get rid of warnigns about > nan's
                data_z[np.isnan(data_z)] = -1
                ind = data_z > 0
                data_z[ind] = np.log10(data_z[ind])
                data_z[~ind] = np.nan

            vmin = np.nanmin(data_z)
            vmax = np.nanmax(data_z)

            if vmin < 0 and vmax > 0:
                vmax = np.maximum(np.abs(vmax), np.abs(vmin))
                vmin = -vmax

            if cmin is not None:
                vmin = cmin

            if cmax is not None:
                vmax = cmax

            if zrng is not None:
                vmin = max([vmax - zrng, vmin])

            if not zaxis_contour:
                im1 = ax.imshow(
                    data_z.T,
                    cmap=newCm,
                    extent=extent,
                    interpolation="nearest",
                    origin="lower",
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                colorbar = False
                if zaxis_levels is None:
                    zaxis_levels = np.linspace(np.nanmin(data_z), np.nanmax(data_z), 20)
                ax.contour(lin_x, data_y, data_z.T, colors="black", levels=zaxis_levels)

        if show_mix:
            mixCmap, mixNorm = self._setMixRegionsCol(kip=True)
            if mix_hatch:
                plt.rcParams["hatch.color"] = hatch_color
                ax.contourf(
                    lin_x,
                    data_y,
                    mix_data.T,
                    colors="none",
                    alpha=0.0,
                    norm=mixNorm,
                    hatches=["//", "-", "x", "+", "\\", "/"],
                    antialiased=True,
                )
                ax.contour(lin_x, data_y, mix_data.T, colors=hatch_color, norm=mixNorm)
            else:
                ax.imshow(
                    mix_data.T,
                    cmap=mixCmap,
                    norm=mixNorm,
                    extent=extent,
                    interpolation="nearest",
                    origin="lower",
                    aspect="auto",
                    alpha=mix_alpha,
                )

        if callable(hatch_func):
            plt.rcParams["hatch.color"] = hatch_color
            ax.contourf(
                lin_x,
                data_y,
                data_hatch.T,
                colors="none",
                alpha=0.0,
                hatches=["//", "-", "x", "+", "\\", "/"],
                antialiased=True,
            )
            ax.contour(lin_x, data_y, data_hatch.T, colors=hatch_color)

        if plot_type == "history":
            if ylabel is None:
                lg = ""
                if y1log:
                    lg = r"$\log_{10}\,$"
                if radius:
                    ax.set_ylabel(lg + r"$\rm{Radius}\; [R_{\odot}]$")
                else:
                    ax.set_ylabel(lg + r"$\rm{Mass}\; [M_{\odot}]$")
            elif len(ylabel):
                ax.set_ylabel(self.safeLabel(ylabel, yaxis))
        else:
            ax.set_ylabel(self.safeLabel(ylabel, yaxis))

        # Add line at outer mass location
        if show_outer_mass and plot_type == "history":
            # f = interp1d(data_x[modInd], m.hist.data['star_mass'][modInd])
            # ax.plot(lin_x,f(lin_x),c='k')
            if radius:
                if "radius" in m.hist:
                    ax.plot(data_x[modInd], m.hist.data["radius"][modInd], c="k")
                else:
                    ax.plot(data_x[modInd], 10 ** m.hist.data["log_R"][modInd], c="k")
            else:
                ax.plot(data_x[modInd], m.hist.data["star_mass"][modInd], c="k")

        if y2 is not None:
            # Update axes 2 locations after ax1 is moved by the colorbar

            if not type(y2) is np.ndarray:
                y2 = m.hist.data[y2]

            # ax2.set_position(ax.get_position())
            if plot_type == "history":
                f = interp1d(data_x[modInd], y2[modInd])
            else:
                ind = m.hist.model_number > mod_min
                f = interp1d(data_x[modInd], y2[modInd])

            ax2.plot(lin_x, f(lin_x), c="k")
            if y2rng is not None:
                ax2.set_ylim(y2rng)

            if y2label:
                ax2.set_ylabel(y2label)

        if colorbar:
            if cbar_ax is not None:
                cb = fig.colorbar(im1, cax=cbar_ax, ax=ax, extend=cbar_extend)
            else:
                cb = fig.colorbar(im1, ax=ax, extend=cbar_extend)
            cb.solids.set_edgecolor("face")

            if cbar_label is None:
                cb.set_label(self._kip_cbar_label, labelpad=cpad)
            else:
                cb.set_label(self.safeLabel(cbar_label, zaxis), labelpad=cpad)

            cb.ax.minorticks_off()

        if xlabel is None:
            if xaxis == "model_number":
                ax.set_xlabel(r"$\rm{Model\, Number}$")
            else:
                self._setAgeLabel(ax, age_log, age_lookback, age_units)
        else:
            ax.set_xlabel(xlabel)

        if show_mass_loc:
            self._showMassLoc(
                m, fig, ax, np.linspace(xmin, xmax, np.count_nonzero(modInd)), modInd
            )

        self._setYLim(ax, ax.get_ylim(), yrng)

        self._setTicks(ax, y1log)

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

    def _setAgeLabel(self, ax, age_log, age_lookback, age_units):

        unit = ""
        if "sec" in age_units:
            unit = r"\rm{s}"
        elif "hour" in age_units:
            unit = r"\rm{Hrs}"
        elif "mega" in age_units:
            unit = r"\rm{Myr}"
        elif "year" in age_units:
            unit = r"\rm{yr}"

        if age_log:
            if age_lookback:
                ax.set_xlabel(
                    r"$\log_{10}\; \left(\rm{\tau_{cc}-\tau}\right)/" + unit + r"$"
                )
            else:
                ax.set_xlabel(r"$\log_{10}\; \left(\tau/" + unit + r"\right)$")
        else:
            if age_lookback:
                ax.set_xlabel(r"$\left(\rm{\tau_{cc}-\tau}\right)\; [" + unit + r"]$")
            else:
                ax.set_xlabel(r"$\tau\; \left(" + unit + r"\right)$")

    def _getHistBurnData(self, m, data_x, data_y, modInd, burn_prefix, radius):
        z = np.zeros((np.count_nonzero(data_x[modInd]), np.size(data_y)))
        z = self._rebinKipqData(m, burn_prefix, z, data_y, modInd, radius)

        z[z < -100.0] = 0.0
        return z

    def _getHistMixData(self, m, data_x, data_y, modInd, mix, mix_prefix, radius):
        z = np.zeros((np.count_nonzero(data_x[modInd]), np.size(data_y)))
        if mix is None or np.size(mix) > 1:
            z = self._rebinKipqData(m, mix_prefix, z, data_y, modInd, radius)
            if np.size(mix) > 1:
                z[np.isin(z, mix, invert=True)] = 0

        return z

    def _rebinKipqData(self, m, qtype, z, y, modInd, radius):
        if np.all(modInd):
            z = self._rebinKipqDataNoModInd(m, qtype, z, y, radius)
        else:
            # Much slower if we want to index over the x array (~factor 10), even if all modInd elements are True
            z = self._rebinKipqDataWithModInd(m, qtype, z, y, modInd, radius)
        return z

    def _rebinKipqDataNoModInd(self, m, qtype, z, y, radius):
        if radius:
            qtop = qtype + "top_"
            add = "add mix_relr_regions 40 and burn_relr_regions 40"
        else:
            qtop = qtype + "qtop_"
            add = "add mixing_regions 40 and burning_regions 40"
        qtyp = qtype + "type_"

        try:
            x = m.hist.data[qtop + "1"]
        except ValueError:
            raise KeyError(
                "No field", qtop + "* found,", add, "to your history_columns.list"
            )

        numBurnZones = int([xx.split("_")[-1] for xx in m.hist if qtop in xx][-1])

        if radius:
            if "radius" in m.hist:
                scaler = m.hist.data["radius"]
            else:
                scaler = 10 ** m.hist.data["log_R"]
        else:
            scaler = m.hist.data["star_mass"]

        for i in range(numBurnZones, 0, -1):
            mass = np.abs(m.hist.data[qtop + str(i)] * scaler)
            ind = np.searchsorted(y, mass, side="left")
            for j in range(np.size(ind)):
                z[j, 0 : ind[j]] = m.hist.data[qtyp + str(i)][j]

        return z

    def _rebinKipqDataWithModInd(self, m, qtype, z, y, modInd, radius):
        if radius:
            qtop = qtype + "top_"
            add = "add mix_relr_regions 40 and burn_relr_regions 40"
        else:
            qtop = qtype + "qtop_"
            add = "add mixing_regions 40 and burning_regions 40"
        qtyp = qtype + "type_"

        try:
            x = m.hist.data[qtop + "1"]
        except ValueError:
            raise KeyError(
                "No field", qtop + "* found,", add, "to your history_columns.list"
            )

        numBurnZones = int([xx.split("_")[-1] for xx in m.hist if qtop in xx][-1])

        if radius:
            if "radius" in m.hist:
                scaler = m.hist.data["radius"][modInd]
            else:
                scaler = 10 ** m.hist.data["log_R"][modInd]
        else:
            scaler = m.hist.data["star_mass"][modInd]

        for i in range(numBurnZones, 0, -1):
            mass = np.abs(m.hist.data[qtop + str(i)][modInd] * scaler)
            ind = np.searchsorted(y, mass, side="left")
            for j in range(np.size(ind)):
                z[j, 0 : ind[j]] = m.hist.data[qtyp + str(i)][modInd][j]

        return z

    def _getSafeAgeHist(
        self,
        m,
        age_lookback=False,
        age_zero=None,
        age_units="sec",
        age_log=False,
        age_reverse=False,
        end_time=None,
    ):
        if "star_age" in m.hist:
            age = m.hist.star_age
        else:
            age = m.hist.star_age_sec / self.secyear

        if "log_dt" in m.hist:
            # Age in years does not have enough digits to be able to distinguish the final models in pre-sn progenitors
            age_yr = age[0]

            age = np.zeros(np.shape(m.hist.model_number))
            age[0] = age_yr
            dt = 10 ** np.longdouble(m.hist.log_dt)
            age[1:] = dt[1:]
            age = np.cumsum(age)

        if "sec" in age_units:
            age = age * self.secyear
        elif "hour" in age_units:
            age = (age * self.secyear) / (3600.0)
        elif "mega" in age_units:
            age = age / (10**6)
        elif "year" in age_units:
            pass
        else:
            raise ValueError("Bad age unit: " + str(age_units))

        return self._getSafeAge(
            age, age_lookback, age_zero, age_units, age_log, age_reverse, end_time
        )

    def _getSafeAge(
        self,
        age,
        age_lookback=False,
        age_zero=None,
        age_units="sec",
        age_log=False,
        age_reverse=False,
        end_time=None,
    ):

        if age_lookback:
            xx = age[-1]
            if end_time is not None:
                xx = end_time
            age = xx - age

        if age_zero is not None:
            age = age - age_zero

        if age_log:
            if age_lookback:
                # Fudge the first value not to be exactly 0.0
                age[-1] = age[-2] / 2.0
            age[0] = age[1] / 2.0
            age = np.log10(age)

        if age_reverse:
            age = age[::-1]

        return age

    def _getSafeCenter(self, m, radius):
        center = None
        if radius:
            if "r_center" in m.hist:
                if np.any(m.hist.data["r_center"] > 10**m.hist.log_R):
                    # r_center in cm
                    center = m.hist.data["r_center"] / self.rsun
                else:
                    # Solar units
                    center = m.hist.data["r_center"]
        else:
            if "m_center" in m.hist:
                if np.any(m.hist.data["m_center"] > m.hist.star_mass):
                    # m_center in grams
                    center = m.hist.data["m_center"] / self.msun
                else:
                    # Solar units
                    center = m.hist.data["m_center"]

        if center is None:
            center = np.zeros(np.size(m.hist.data["model_number"]))
            center = center + 10**-4

        return center

    def _getModInd(
        self,
        m,
        mod_index=None,
        mod_min=None,
        mod_max=None,
        step=1,
        xaxis="",
        xmin=None,
        xmax=None,
    ):

        modInd = np.zeros(np.size(m.hist.model_number), dtype="bool")
        modInd[:] = True

        modInd[::step] = False
        modInd[:] = np.logical_not(modInd)

        # Mod_index is user supplied index routine
        if mod_index is not None:
            modInd = modInd & mod_index

        if mod_min is None:
            mod_min = m.hist.model_number[modInd][0]

        if mod_max is None:
            mod_max = m.hist.model_number[modInd][-1]

        modInd = (
            modInd & (m.hist.model_number >= mod_min) & (m.hist.model_number <= mod_max)
        )

        if len(xaxis) > 0 and xaxis != "model_number":
            if xmin is None:
                xmin = np.nanmin(m.hist.data[xaxis][modInd])

            if xmax is None:
                xmax = np.nanmax(m.hist.data[xaxis][modInd])

            modInd = (
                modInd & (m.hist.data[xaxis] >= xmin) & (m.hist.data[xaxis] <= xmax)
            )

        return modInd

    def _rebinKipDataX(self, data, x, lin_x, nan=False, nan_value=1):
        sorter = np.argsort(x)
        ind = np.searchsorted(x, lin_x, sorter=sorter, side="left")

        s_ind = sorter[ind]

        # When flipping the ages we may end up with points at the edge
        # so just make sure we dont go out of bound (searcgsoretd returns N if no suitable match found)
        s_ind[s_ind == len(data)] = len(data) - 1

        data = data[s_ind, :]
        if nan:
            data[data < nan_value] = np.nan
        data = np.array(data)
        return data

    def _rebinKipDataXY(self, data_z, data_x, data_y, num_x_zones, num_y_zones):
        ymin = np.nanmin([np.nanmin(y) for y in data_y])
        ymax = np.nanmax([np.nanmax(y) for y in data_y])
        data_y2 = np.linspace(ymin, ymax, int(num_y_zones))
        data_z2 = np.zeros((np.size(data_x), int(num_y_zones)))
        for i in range(len(data_x)):
            f = interp1d(data_y[i], data_z[i], bounds_error=False, fill_value=np.nan)
            data_z2[i, :] = f(data_y2)

        lin_x = np.linspace(np.nanmin(data_x), np.nanmax(data_x), int(num_x_zones))
        data_z = self._rebinKipDataX(data_z2, data_x, lin_x)
        return data_z, lin_x, data_y2

    def plotTRho(
        self,
        m,
        model=None,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        fig=None,
        yrng=[None, None],
        show_burn=False,
        show_mix=False,
        show_burn_labels=False,
        show_mix_labels=False,
        showAll=False,
        showBurn=False,
        showPgas=False,
        showDegeneracy=False,
        showGamma=False,
        showEOS=False,
        logT=False,
        logRho=False,
        title=None,
        ycol="k",
        points=False,
    ):

        fig, ax = self._setupProf(fig, ax, m, model)

        try:
            x = m.prof.logRho
            xname = "logRho"
            xlog = False
        except:
            x = m.prof.Rho
            xname = "rho"
            xlog = True

        try:
            y = m.prof.logT
            yname = "logT"
            ylog = False
        except:
            y = m.prof.temperature
            yname = "temperature"
            ylog = True

        self.plotProfile(
            m,
            xaxis=xname,
            y1=yname,
            y1log=ylog,
            xlog=xlog,
            model=model,
            show=False,
            show_mix=show_mix,
            show_burn=show_burn,
            show_mix_line=True,
            show_burn_line=True,
            show_mix_x=False,
            show_burn_x=False,
            xmin=xmin,
            xmax=xmax,
            ax=ax,
            y1label=self.labels("teff", log=True),
            xlabel=self.labels("rho", log=True),
            fig=fig,
            y1rng=yrng,
            y2rng=None,
            y1col=ycol,
            show_burn_labels=show_burn_labels,
            show_mix_labels=show_mix_labels,
            title=title,
            points=points,
        )

        if showBurn or showAll:
            self._showBurnData(ax)

        if showPgas or showAll:
            self._showPgas(ax)

        if showDegeneracy or showAll:
            self._showDegeneracy(ax)

        if showGamma or showAll:
            self._showGamma4(ax)

        if showEOS or showAll:
            self._showEOS(ax)

        if show:
            plt.show()

    def plotHR(
        self,
        m,
        minMod=0,
        maxMod=-1,
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        fig=None,
        points=None,
    ):
        self.plotHistory(
            m,
            xaxis="log_Teff",
            y1="log_L",
            y1log=False,
            minMod=minMod,
            maxMod=maxMod,
            show=show,
            xmin=xmin,
            xmax=xmax,
            xrev=True,
            y1rev=False,
            ax=ax,
            y1col="k",
            xlabel=self.labels("teff", log=True),
            y1label=self.labels("lum", log=True),
            fig=fig,
            points=points,
        )

    def mergeCmaps(self, cmaps, rng=[[0.0, 0.5], [0.5, 1.0]]):
        """
        Creates a diverging colomap

        cmaps: list of colormaps (ie [cm.Purples_r,cm.hot_r])
        rng: list of list with the rng to define the colormaps over ie [[0.0,0.5],[0.5,1.0]]
                to have a diverging colormap centered at the mid point

        Returns:
        LinearSegmentedColormap
        """
        cdict = {"red": [], "blue": [], "green": []}
        for i in range(len(cmaps)):
            cmap = cmaps[i]
            minX = rng[i][0]
            maxX = rng[i][1]
            cmapseg = cmap._segmentdata
            for key in ("red", "green", "blue"):
                for j in range(len(cmapseg[key])):
                    cdict[key].append(
                        [
                            minX + (maxX - minX) * cmapseg[key][j][0],
                            cmapseg[key][j][1],
                            cmapseg[key][j][2],
                        ]
                    )

        return mpl.colors.LinearSegmentedColormap("colormap", cdict, 1024)

    def stackedPlots(
        self,
        m,
        typ="profile",
        num=1,
        model=None,
        xaxis="mass",
        show=True,
        fig=None,
        ax=None,
        xmin=None,
        xmax=None,
        xlog=False,
        xlabel=None,
        xrev=False,
        y1rev=[],
        y2rev=[],
        points=False,
        minMod=0,
        maxMod=-1,
        y1=[],
        y2=[],
        y1log=[],
        y2log=[],
        y1col=[],
        y2col=[],
        y1label=[],
        y2label=[],
    ):
        if num < 2:
            raise ValueError("num must be >=2")

        empty = [None] * len(y1)
        f = [False] * len(y1)
        if len(y1) > 0:
            if not y2:
                y2 = empty
            if not y1log:
                y1L = f
            if not y2log:
                y2L = f
            if not y1rev:
                y1rev = f
            if not y2rev:
                y2rev = f
            if not y1col:
                y1col = ["r"] * len(y1)
            if not y2col:
                y2col = ["b"] * len(y1)
            if not y1label:
                y1label = empty * len(y1)
            if not y2label:
                y2label = empty * len(y1)

        f, axis = plt.subplots(num, sharex=True)
        f.subplots_adjust(hspace=0)

        for i in range(num):
            if typ == "profile":
                self.plotProfile(
                    m=m,
                    model=model,
                    xaxis=xaxis,
                    show=False,
                    ax=axis[i],
                    xmin=xmin,
                    xmax=xmax,
                    xlog=xlog,
                    xlabel=xlabel,
                    xrev=xrev,
                    y1rev=y1rev[i],
                    y2rev=y2rev[i],
                    points=points,
                    y1=y1[i],
                    y2=y2[i],
                    y1log=y1log[i],
                    y2log=y2log[i],
                    y1col=y1col[i],
                    y2col=y2col[i],
                    y1label=y1label[i],
                    y2label=y2label[i],
                )
            else:
                self.plotHistory(
                    m=m,
                    xaxis=xaxis,
                    show=False,
                    ax=axis[i],
                    xmin=xmin,
                    xmax=xmax,
                    xlog=xlog,
                    xlabel=xlabel,
                    xrev=xrev,
                    y1rev=y1rev[i],
                    y2rev=y2rev[i],
                    points=points,
                    y1=y1[i],
                    y2=y2[i],
                    y1log=y1log[i],
                    y2log=y2log[i],
                    y1col=y1col[i],
                    y2col=y2col[i],
                    y1label=y1label[i],
                    y2label=y2label[i],
                    minMod=minMod,
                    maxMod=maxMod,
                )

        if show:
            plt.show()

    def plotMultiProfiles(
        self,
        m,
        mods=None,
        index=None,
        xaxis="mass",
        y1="",
        show=True,
        ax=None,
        xmin=None,
        xmax=None,
        xlog=False,
        y1log=False,
        cmap=plt.cm.gist_ncar,
        xrev=False,
        y1rev=False,
        points=False,
        xlabel=None,
        y1label=None,
        fig=None,
        show_mix=False,
        show_burn=True,
    ):
        """Plots mulitple profiles either given as a list of mod numbers or an index over the history data"""
        if fig == None:
            fig = plt.figure(figsize=(12, 12))
        if ax == None:
            ax = fig.add_subplot(111)

        if mods is not None:
            cm = [cmap(i) for i in np.linspace(0.0, 0.9, len(mods))]
            for i in range(len(mods)):
                model = mods[i]
                self.plotProfile(
                    m,
                    model=model,
                    xaxis=xaxis,
                    show=False,
                    ax=ax,
                    fig=fig,
                    xmin=xmin,
                    xmax=xmax,
                    xlog=xlog,
                    xlabel=xlabel,
                    xrev=xrev,
                    y1rev=y1rev,
                    points=points,
                    y1=y1,
                    y1log=y1log,
                    y1col="k",
                    y1label=y1label,
                    show_mix=show_mix,
                    show_burn=show_burn,
                    show_mix_line=True,
                    show_burn_line=True,
                )
        elif index is not None:
            cm = [cmap(i) for i in np.linspace(0.0, 0.9, np.count_nonzero(index))]
            for i in m.hist.model_number[index]:
                model = m.hist.model_number[index][i]
                self.plotProfile(
                    m,
                    model=model,
                    xaxis=xaxis,
                    show=False,
                    ax=ax,
                    xmin=xmin,
                    xmax=xmax,
                    xlog=xlog,
                    xlabel=xlabel,
                    xrev=xrev,
                    y1rev=y1rev,
                    points=points,
                    y1=y1,
                    y1log=y1log,
                    y1col=cm[i],
                    y1label=y1label,
                    fig=fig,
                )

        ax.legend(loc=0, fontsize=12)

        if show:
            plt.show()

    def plotGrid2(self, m, show=True):
        """Why not grid1? trying to copy mesa's grids and grid2 is easier for now"""
        fig = plt.figure(figsize=(12, 12))
        fig.subplots_adjust(wspace=0.5)
        fig.subplots_adjust(hspace=0.5)
        ax = plt.subplot(2, 2, 1)
        self.plotTRho(m, ax=ax, show=False)

        ax = plt.subplot(2, 4, 5)
        self.plotHR(m, ax=ax, maxMod=m.prof.head["model_number"], show=False)

        ax = plt.subplot(2, 4, 6)
        self.plotHistory(
            m,
            ax=ax,
            show=False,
            xaxis="log_center_T",
            y1="log_center_Rho",
            y1log="linear",
            minMod=0,
            maxMod=m.prof.head["model_number"],
            y1col="k",
            xlabel=self.labels("teff", log=True, center=True),
            y1label=self.labels("rho", log=True, center=True),
        )

        ax = plt.subplot(1, 2, 2)
        self.plotAbun(m, ax=ax, show=False, xlabel=self.labels("mass"))

        if show == True:
            plt.show()

    def plotSliderProf(self, m, func, *args, **kwargs):
        from matplotlib.widgets import Button

        class Index(object):
            def __init__(self, models):
                self.ind = 1
                self.models = models

            def safe_ind(self):
                if self.ind < 0:
                    return self.models[0]
                elif self.ind > len(self.models):
                    return self.models[-1]
                else:
                    return self.models[self.ind]

            def next(self, event):
                self.ind += 1
                update(self.safe_ind())

            def prev(self, event):
                self.ind -= 1
                update(self.safe_ind())

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        f = getattr(self, func)

        models = m.prof_ind["model"]

        m.loadProfile(num=models[0])
        f(m, fig=fig, ax=ax, show=False, show_title_model=True, *args, **kwargs)

        def update(val):
            m.loadProfile(num=val)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ymin2 = None
            ymax2 = None
            for i in fig.axes:
                if "_ax2" in i.get_label():
                    fig.delaxes(i)
                    ymin2, ymax2 = i.get_ylim()
            plt.sca(ax)
            plt.cla()
            try:
                kwargs.pop("xmax")
            except KeyError:
                pass
            try:
                kwargs.pop("xmin")
            except KeyError:
                pass
            try:
                kwargs.pop("y1rng")
            except KeyError:
                pass
            f(
                m,
                fig=fig,
                ax=ax,
                xmin=xmin,
                xmax=xmax,
                y1rng=[ymin, ymax],
                y2rng=[ymin2, ymax2],
                show_title_model=True,
                show=False,
                *args,
                **kwargs
            )
            fig.canvas.draw()

        callback = Index(models)
        axprev = plt.axes([0.7, 0.05, 0.05, 0.055])
        axnext = plt.axes([0.81, 0.05, 0.05, 0.055])
        bnext = Button(axnext, "Next")
        bnext.label.set_fontsize("12")
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, "Previous")
        bprev.on_clicked(callback.prev)
        bprev.label.set_fontsize("12")

        plt.show()

    def accretionSum(self, m):
        for idx, i in enumerate(self._listAbun(m.prof)):
            if i == "neut" or i == "prot":
                continue
            print("accretion_species_id(" + str(idx) + ') = "' + str(i) + '"')
            print(
                "accretion_species_xa(" + str(idx) + ") = " + str(self._abunSum(m, i))
            )
        print("num_accretion_species =" + str(idx))

    def plotRates(
        self,
        m,
        prefix="raw_rates_",
        show=True,
        ax=None,
        xaxis="mass",
        xmin=None,
        xmax=None,
        y1rng=[None, None],
        y1log=False,
        y1scale=None,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        abun=None,
        rand_col=False,
        show_burn=False,
        show_mix=False,
        fig=None,
        fx=None,
        fy=None,
        show_core_loc=False,
        show_title_name=False,
        show_title_model=False,
        show_title_age=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        y1label=r"Rates",
        title=None,
        show_shock=False,
        show_burn_labels=False,
        show_mix_labels=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
        show_burn_line=False,
        show_burn_x=True,
        show_mix_line=False,
        show_mix_x=True,
    ):

        rate_list = self._listRates(m.prof, prefix)

        if len(rate_list) == 0:
            raise ValueError("Found no rates in the data")

        self._plotMultiProf(
            m,
            list_y=rate_list,
            y1log=y1log,
            y1scale=y1scale,
            _axlabel="rates",
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            show_burn=show_burn,
            show_mix=show_mix,
            fig=fig,
            fx=fx,
            fy=fy,
            show_title_name=show_title_name,
            show_title_model=show_title_model,
            show_title_age=show_title_age,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            y1label=y1label,
            title=title,
            show_shock=show_shock,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )

    def plotRatesHist(
        self,
        m,
        prefix="raw_rates_",
        show=True,
        ax=None,
        xaxis="model_number",
        xmin=None,
        xmax=None,
        y1rng=[None, None],
        y1log=False,
        y1scale=None,
        cmap=plt.cm.gist_ncar,
        num_labels=3,
        xlabel=None,
        points=False,
        rand_col=False,
        fig=None,
        fx=None,
        fy=None,
        minMod=-1,
        maxMod=-1,
        y1label="Rates",
        show_title_name=False,
        annotate_line=True,
        linestyle="-",
        colors=None,
        show_core=False,
        y2=None,
        y2rng=[None, None],
        fy2=None,
        y2Textcol=None,
        y2label=None,
        y2rev=False,
        y2log=False,
        y2col="k",
        xlog=False,
        xrev=False,
    ):

        rates_list = self._listRates(m.hist, prefix=prefix)

        self._plotMultiHist(
            m,
            list_y=rates_list,
            show=show,
            ax=ax,
            xaxis=xaxis,
            xmin=xmin,
            xmax=xmax,
            y1rng=y1rng,
            y1log=y1log,
            y1scale=y1scale,
            y1label=y1label,
            cmap=cmap,
            num_labels=num_labels,
            xlabel=xlabel,
            points=points,
            rand_col=rand_col,
            fig=fig,
            fx=fx,
            fy=fy,
            minMod=minMod,
            maxMod=maxMod,
            show_title_name=show_title_name,
            annotate_line=annotate_line,
            linestyle=linestyle,
            colors=colors,
            show_core=show_core,
            y2=y2,
            y2rng=y2rng,
            fy2=fy2,
            y2Textcol=y2Textcol,
            y2label=y2label,
            y2rev=y2rev,
            y2log=y2log,
            y2col=y2col,
            xlog=xlog,
            xrev=xrev,
        )
