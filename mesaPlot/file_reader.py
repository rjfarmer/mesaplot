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
import os
import pickle
import bisect
import subprocess
import hashlib
from io import BytesIO
import pandas

from distutils.version import StrictVersion

msun = 1.9892 * 10 ** 33

# Conviently the index of this list is the proton number
_elementsPretty = [
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
_elements = [x.lower() for x in _elementsPretty]

_PICKLE_VERSION = 6


def _hash(fname):
    hash_md5 = hashlib.md5()
    if not os.path.exists(fname):
        return None

    try:
        x = subprocess.check_output(["md5sum", fname])
        return x.split()[0].decode()
    except (FileNotFoundError, subprocess.CalledProcessError):  # (no md5sum, no file)
        pass

    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class data(object):
    def __init__(self):
        self.data = {}
        self.head = {}
        self._loaded = False
        self._mph = ""
        self._type = ""

    def __getattr__(self, name):
        if "data" in self.__dict__:
            if len(self.data) == 0:
                raise AttributeError("Must load data first")

        return self.__getitem__(name)

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        try:
            return list(self.data.keys()) + list(self.head.keys())
        except AttributeError:
            return list(self.data.dtype.names) + list(self.head.dtype.names)

    def __dir__(self):
        return self.keys() + list(self.__dict__.keys())

    def __getitem__(self, key):
        if "data" in self.__dict__:
            if key in self.data.dtype.names:
                return self.data[key]
            if key in self.head.dtype.names:
                return self.head[key][0]

        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError("No attribute " + str(key))

    def __iter__(self):
        if len(self.data):
            for i in self.keys():
                yield i

    def loadFile(
        self,
        filename,
        max_num_lines=-1,
        cols=[],
        final_lines=-1,
        _dbg=False,
        use_pickle=True,
        reload_pickle=False,
        silent=False,
        is_mod=False,
    ):

        if is_mod:
            self._loadMod(filename)
            return

        pickname = filename + ".pickle"
        if (
            use_pickle
            and os.path.exists(pickname)
            and not reload_pickle
            and final_lines < 0
        ):
            if self._loadPickle(pickname, filename):
                return

        # Not using pickle
        self._loadFile(filename, max_num_lines, cols, final_lines)

    def _loadPickle(self, pickname, filename):
        if not os.path.exists(pickname):
            return False

        with open(pickname, "rb") as f:
            # Get checksum
            filehash = _hash(filename)
            try:
                pickhash = pickle.load(f)
            except:
                raise ValueError("Pickle file corrupted please delete it and try again")
            if len(str(pickhash)) == len(str(_PICKLE_VERSION)):
                if int(pickhash) != int(_PICKLE_VERSION):
                    # Not a hash but a version number/ or wrong version number:
                    return False

            try:
                pickhash = pickle.load(f)
            except TypeError:
                return False

            if (
                os.path.exists(filename) and pickhash == filehash
            ) or not os.path.exists(filename):
                # Data has not changed
                # Get Data
                self.data = pickle.load(f)
                self.head = pickle.load(f)
                self._loaded = True
                return True
        return False

    def _loadFile(self, filename, max_num_lines=-1, cols=[], final_lines=-1):
        if not os.path.exists(filename):
            raise FileNotFoundError("No file " + str(filename) + " found")

        head = pandas.read_csv(filename, delim_whitespace=True, header=1, nrows=1)

        if max_num_lines > 0:
            data = pandas.read_csv(
                filename, delim_whitespace=True, header=4, nrows=max_num_lines
            )
        else:
            data = pandas.read_csv(filename, delim_whitespace=True, header=4)

        if final_lines > 0:
            data = self.data[-final_lines:]

        # Convert from pandas to numpy
        dtype = np.dtype(
            [
                (data.dtypes.index[idx], data.dtypes[idx].name)
                for idx, i in enumerate(data.dtypes)
            ]
        )
        self.data = np.zeros(np.size(data[dtype.names[0]]), dtype=dtype)
        for i in data:
            self.data[i] = data[i].to_numpy()

        dtype = np.dtype(
            [
                (head.dtypes.index[idx], head.dtypes[idx].name)
                for idx, i in enumerate(head.dtypes)
            ]
        )
        self.head = np.zeros(1, dtype=dtype)
        for i in head:
            self.head[i] = head[i].to_numpy()

        self._loaded = True
        self._saveFile(filename)

    def _saveFile(self, filename):
        filehash = _hash(filename)
        pickname = filename + ".pickle"
        with open(filename + ".pickle", "wb") as f:
            pickle.dump(_PICKLE_VERSION, f)
            pickle.dump(filehash, f)
            pickle.dump(self.data, f)
            pickle.dump(self.head, f)

    def _loadMod(self, filename):
        from io import BytesIO, StringIO

        count = 0
        with open(filename, "r") as f:
            for l in f:
                count = count + 1
                if "!" not in l:
                    break
            head = []
            head_names = []
            head.append(str(l.split()[0]))
            head_names.append("mod_version")
            # Blank line
            f.readline()
            count = count + 1
            # Gap between header and main data
            for l in f:
                count = count + 1
                if l == "\n":
                    break
                head.append(str(l.split()[1]))
                head_names.append(l.split()[0])
            data_names = []
            l = f.readline()
            count = count + 1
            data_names.append("zone")
            data_names.extend(l.split())
            # Make a dictionary of converters

        # Replace MMsun with star_mass
        head_names[head_names.index("M/Msun")] = "star_mass"

        d = {k: self._fds2f for k in range(len(head_names))}
        self.head = np.genfromtxt(
            StringIO(" ".join(head)),
            names=head_names,
            dtype=None,
            encoding="ascii",
            converters=d,
        )

        d = {k: self._fds2f for k in range(len(data_names))}

        data = np.genfromtxt(
            filename,
            skip_header=count,
            names=data_names,
            skip_footer=5,
            dtype=None,
            converters=d,
            encoding="ascii",
        )

        # Add mass co-ord

        mass = np.cumsum(data["dq"][::-1])[::-1] * self.head["star_mass"]

        olddt = data.dtype
        newdt = np.dtype(olddt.descr + [("mass", ("<f8"))])

        newarr = np.zeros(np.shape(data), dtype=newdt)
        for i in olddt.names:
            newarr[i] = data[i]

        newarr["mass"] = mass
        self.data = newarr

        self._loaded = True

    def _fds2f(self, x):
        if isinstance(x, str):
            f = x.replace("'", "").replace("D", "E")
        else:
            f = x.decode().replace("'", "").replace("D", "E")
        try:
            f = np.float(f)
        except ValueError:
            pass

        return f

    def listAbun(self, prefix=""):
        abun_list = []
        for j in self.keys():
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

    def splitIso(self, iso, prefix=""):
        name = ""
        mass = ""
        iso = iso[len(prefix) :]
        for i in iso:
            if i.isdigit():
                mass += i
            else:
                name += i
        if "neut" in name or "prot" in name:
            mass = 1
        return name, int(mass)

    def getIso(self, iso, prefix=""):
        name, mass = self.splitIso(iso, prefix)
        if "prot" in name:
            p = 1
            n = 0
        else:
            p = _elements.index(name)
            n = mass - p
        return name, p, n

    def listBurn(self):
        burnList = []
        ignore = ["qtop", "type", "min"]
        extraBurn = [
            "pp",
            "cno",
            "tri_alfa",
            "c12_c12",
            "c12_O16",
            "o16_o16",
            "pnhe4",
            "photo",
            "other",
        ]
        for i in self.keys():
            if ("burn_" in i or i in extraBurn) and not any(j in i for j in ignore):
                burnList.append(str(i))
        return burnList

    def listMix(self):
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
        for i in self.keys():
            if i in mixList:
                mixListOut.append(str(i))
        return mixListOut

    def abunSum(self, iso, mass_min=0.0, mass_max=9999.0):
        if "mass" in self.keys():
            ind = (self.data["mass"] >= mass_min) & (self.data["mass"] <= mass_max)
            mdiff = self._getMdiff()
            return np.sum(self.data[iso][ind] * mdiff[ind])
        else:
            return self._getMassHist(iso)

    def eleSum(self, element, mass_min=0.0, mass_max=9999.0, prefix=""):
        la = self.listAbun(prefix)
        x = 0.0
        for i in la:
            if element == self.splitIso(i)[0]:
                x = x + self.abunSum(i, mass_min, mass_max)
        return x

    def getMassFrac(self, iso, ind=None, log=False, prof=True):
        if prof:
            if "logdq" in self.keys():
                scale = 10 ** (self.data["logdq"][ind])
            elif "dq" in self.keys():
                scale = self.data["dq"][ind]
            elif "dm" in self.keys():
                scale = self.data["dm"][ind] / (msun * self.mass_star())
            else:
                raise AttributeError(
                    "No suitable mass co-ordinate available for getMassFrac, need either logdq, dq or dm in profile"
                )
        else:
            scale = 1.0

        if log:
            x = np.sum(10 ** self.data[i][ind] * scale)
        else:
            x = np.sum(self.data[i][ind] * scale)

        return x

    @property
    def mass_star(self):
        if "star_mass" in self.data.dtype.names:
            return self.data["star_mass"]
        elif "star_mass" in self.head_names:
            return self.head["star_mass"]
        else:
            raise ValueError("No star_mass available")

    def _getMdiff(self):
        sm = self.head["star_mass"]
        if "M_center" in self.head_names:
            sm = sm - self.head["M_center"] / msun

        if "logdq" in self.keys():
            return 10 ** (self.data["logdq"]) * sm
        elif "dq" in self.keys():
            return self.data["dq"] * sm
        elif "dm" in self.keys():
            return self.data["dm"]
        else:
            raise AttributeError(
                "No suitable mass co-ordinate available for _getMdiff, need either logdq, dq or dm in data"
            )

    def _getMassHist(self, iso):
        if "log_total_mass_" + iso in self.keys():
            return 10 ** self.data["log_total_mass_" + iso]
        elif "total_mass_" + iso in self.keys():
            return self.data["log_total_mass_" + iso]
        else:
            return None

    def core_mass(
        self,
        element,
        prev_element=None,
        core_boundary_fraction=0.01,
        min_boundary_fraction=0.1,
    ):
        if prev_element is None:
            if element == "he4":
                prev_element = "h1"
            elif element == "c12":
                prev_element = "he4"
            elif element == "o16":
                prev_element = "c12"
            elif element == "si28":
                prev_element = "o16"
            elif element == "fe56":
                prev_element = "si28"

        ind = self.data[element] >= core_boundary_fraction
        ind = ind & (self.data[prev_element] <= min_boundary_fraction)
        return np.max(self.data["mass"][ind])

    def mergeHist(self,*new):
        self.hist.data = np.concatenate((self.hist.data,
                                        [i.hist.data for i in new]),
                                        dtype=self.hist.data.dtype)


class MESA(object):
    def __init__(self):
        self.hist = data()
        self.prof = data()
        self.binary = data()
        self.prof_ind = ""
        self.log_fold = ""
        self.clearProfCache()
        self.cache_limit = 100
        self._cache_wd = ""

        self.hist._mph = "history"
        self.prof._mph = "profile"
        self.hist._type = "loadHistory"
        self.prof._type = "loadProfile"

        self.hist._mph = "binary"
        self.hist._mph = "loadBinary"

    def loadHistory(
        self,
        f="",
        filename_in=None,
        max_model=-1,
        max_num_lines=-1,
        cols=[],
        final_lines=-1,
        _dbg=False,
        use_pickle=True,
        reload_pickle=False,
    ):
        """
        Reads a MESA history file.

        Optional:
        f: Folder in which history.data exists, if not present uses self.log_fold, if that is
        not set try the current working directory.
        filename_in: Reads the file given by name
        max_model: Maximum model to read into, may help when having to clean files with many retires, backups and restarts by not processing data beyond max_model
        max_num_lines: Maximum number of lines to read from the file, maps ~maximum model number but not quite (retires, backups and restarts effect this)
        cols: If none returns all columns, else if set as a list only stores those columns, will always add model_number to the list
        final_lines: Reads number of lines from end of the file if > 0


        Returns:
        self.hist.head: The header data in the history file as a structured dtype
        self.hist.data:  The data in the main body of the history file as a structured dtype
        self.hist.head_names: List of names of the header fields
        self.hist.data_names: List of names of the data fields

        Note it will clean the file up of backups,retries and restarts, preferring to use
        the newest data line.
        """
        if len(f) == 0:
            if len(self.log_fold) == 0:
                self.log_fold = "LOGS/"
            f = self.log_fold
        else:
            self.log_fold = f + "/"

        if filename_in is None:
            filename = os.path.join(self.log_fold, "history.data")
        else:
            filename = filename_in

        self.hist.loadFile(
            filename,
            max_num_lines,
            cols,
            final_lines=final_lines,
            _dbg=_dbg,
            use_pickle=use_pickle,
            reload_pickle=reload_pickle,
        )
        if max_model > 0:
            self.hist.data = self.hist.data[self.hist.model_number <= max_model]

        # Reverse model numbers, we want the unique elements
        # but keeping the last not the first.
        if np.unique(np.diff(self.hist.model_number)).size == 1:
            # Return early if all step sizes are constant
            return

        # Fix case where we have at end of file numbers:
        # 1 2 3 4 5 3, without this we get the extra 4 and 5
        if np.size(self.hist.model_number) > 1:
            self.hist.data = self.hist.data[
                self.hist.model_number <= self.hist.model_number[-1]
            ]
            mod_rev = self.hist.model_number[::-1]
            _, mod_ind = np.unique(mod_rev, return_index=True)
            self.hist.data = self.hist.data[
                np.size(self.hist.model_number) - mod_ind - 1
            ]

    def scrubHistory(self, f="", fileOut="LOGS/history.data.scrubbed"):
        self.loadHistory(f)
        with open(fileOut, "w") as f:
            print(
                " ".join([str(i) for i in range(1, np.size(self.hist.head_names) + 1)]),
                file=f,
            )
            print(" ".join([str(i) for i in self.hist.head_names]), file=f)
            print(
                " ".join([str(self.hist.head[i]) for i in self.hist.head_names]), file=f
            )
            print(" ", file=f)
            print(
                " ".join([str(i) for i in range(1, np.size(self.hist.data_names) + 1)]),
                file=f,
            )
            print(" ".join([str(i) for i in self.hist.data_names]), file=f)
            for j in range(np.size(self.hist.data)):
                print(
                    " ".join([str(self.hist.data[i][j]) for i in self.hist.data_names]),
                    file=f,
                )

    def loadProfile(
        self,
        f="",
        num=None,
        prof=None,
        mode="nearest",
        silent=False,
        cache=True,
        cols=[],
        use_pickle=True,
        reload_pickle=False,
    ):
        if num is None and prof is None:
            self._readProfile(f)  # f is a filename
            return

        if len(f) == 0:
            if len(self.log_fold) == 0:
                self.log_fold = "LOGS/"
            f = self.log_fold
        else:
            self.log_fold = f

        self._loadProfileIndex(f)  # Assume f is a folder
        prof_nums = np.atleast_1d(self.prof_ind["profile"]).astype("int")

        if prof is not None:
            pos = np.where(prof_nums == prof)[0][0]
        else:
            if np.count_nonzero(self.prof_ind) == 1:
                pos = 0
            else:
                if num <= 0:
                    pos = num
                else:
                    # Find profile with mode 'nearest','upper','lower','first','last'
                    pos = bisect.bisect_left(self.prof_ind["model"], num)
                    if pos == 0 or mode == "first":
                        pos = 0
                    elif pos == np.size(self.prof_ind["profile"]) or mode == "last":
                        pos = -1
                    elif mode == "lower":
                        pos = pos - 1
                    elif mode == "upper":
                        pos = pos
                    elif mode == "nearest":
                        if (
                            self.prof_ind["model"][pos] - num
                            < num - self.prof_ind["model"][pos - 1]
                        ):
                            pos = pos
                        else:
                            pos = pos - 1
                    else:
                        raise ValueError("Invalid mode")

        profile_num = np.atleast_1d(self.prof_ind["profile"])[pos]
        filename = f + "/profile" + str(int(profile_num)) + ".data"
        if not silent:
            print(filename)
        self._readProfile(
            filename,
            cache=cache,
            cols=cols,
            use_pickle=use_pickle,
            reload_pickle=reload_pickle,
            silent=silent,
        )
        return

    def loadMod(self, filename=None):
        """
        Read a MESA .mod file.
        """
        self.mod = data()

        self.mod.loadFile(filename, is_mod=True)

    def iterateProfiles(
        self, f="", priority=None, rng=[-1.0, -1.0], step=1, cache=True, silent=False
    ):
        if len(f) == 0:
            if len(self.log_fold) == 0:
                self.log_fold = "LOGS/"
            f = self.log_fold
        else:
            self.log_fold = f
        # Load profiles index file
        self._loadProfileIndex(f)
        for x in self.prof_ind:
            if priority != None:
                if type(priority) is not list:
                    priority = [priority]
                if x["priority"] in priority or 0 in priority:
                    self.loadProfile(
                        f=f + "/profile" + str(int(x["profile"])) + ".data",
                        cache=cache,
                        silent=silent,
                    )
                    yield
            if len(rng) == 2 and rng[0] > 0:
                if (
                    x["model"] >= rng[0]
                    and x["model"] <= rng[1]
                    and np.remainder(x["model"] - rng[0], step) == 0
                ):
                    self.loadProfile(
                        f=f + "/profile" + str(int(x["profile"])) + ".data",
                        cache=cache,
                        silent=silent,
                    )
                    yield
                elif x["model"] > rng[1]:
                    return
            elif len(rng) > 2 and rng[0] > 0:
                if x["model"] in rng:
                    self.loadProfile(
                        f=f + "/profile" + str(int(x["profile"])) + ".data",
                        cache=cache,
                        silent=silent,
                    )
                    yield
            else:
                self.loadProfile(
                    f=f + "/profile" + str(int(x["profile"])) + ".data",
                    cache=cache,
                    silent=silent,
                )
                yield
        return

    def _loadProfileIndex(self, f):
        self.prof_ind = np.genfromtxt(
            f + "/profiles.index", skip_header=1, names=["model", "priority", "profile"]
        )

    def _readProfile(
        self,
        filename,
        cache=True,
        cols=[],
        use_pickle=True,
        reload_pickle=False,
        silent=False,
    ):
        """
        Reads a MESA profile file.

        Required:
        filename: Path to profile to read

        Optional:
        cache: If true caches the profile data so multiple profile loads do not need to reread the data
        cols: cols: If none returns all columns, else if set as a list only storing those columns, it will always add zone to the list of columns

        Returns:
        self.prof.head: The header data in the profile as a structured dtype
        self.prof.data:  The data in the main body of the profile file as a structured dtype
        self.prof.head_names: List of names of the header fields
        self.prof.data_names: List of names of the data fields
        """

        # Handle cases where we change directories inside the python session
        if self._cache_wd != os.getcwd():
            self.clearProfCache()
            self._cache_wd = os.getcwd()

        if filename in self._cache_prof_name and cache:
            self.prof = self._cache_prof[self._cache_prof_name.index(filename)]
        else:
            x = data()
            x.loadFile(
                filename,
                cols=cols,
                use_pickle=use_pickle,
                reload_pickle=reload_pickle,
                silent=silent,
            )
            if cache:
                if len(self._cache_prof_name) == self.cache_limit:
                    self._cache_prof.pop(0)
                    self._cache_prof_name.pop(0)
                self._cache_prof.append(x)
                self._cache_prof_name.append(filename)
            self.prof = x

    def clearProfCache(self):
        self._cache_prof = []
        self._cache_prof_name = []

    def abun(self, element):
        xx = 0
        for ii in range(0, 1000):
            try:
                xx = xx + np.sum(
                    self.prof.data[element + str(ii)] * 10 ** self.prof.logdq
                )
            except:
                pass
        return xx

    def loadBinary(
        self, f="", filename_in=None, max_model=-1, max_num_lines=-1, cols=[]
    ):
        """
        Reads a MESA binary history file.

        Optional:
        f: Folder in which binary_history.data exists, if not present uses self.log_fold, if that is
        not set try the current working directory.
        filename_in: Reads the file given by name
        max_model: Maximum model to read into, may help when having to clean files with many retries, backups and restarts by not processing data beyond max_model
        max_num_lines: Maximum number of lines to read from the file, maps ~maximum model number but not quite (retries, backups and restarts effect this)
        cols: If none returns all columns, else if set as a list only stores those columns, it will always add model_number to the list


        Returns:
        self.binary.head: The header data in the history file as a structured dtype
        self.binary.data:  The data in the main body of the history file as a structured dtype
        self.binary.head_names: List of names of the header fields
        self.binary.data_names: List of names of the data fields

        Note it will clean the file up of backups, retries and restarts, preferring to use
        the newest data line.
        """
        if len(f) == 0:
            if len(self.log_fold) == 0:
                self.log_fold = "./"
            f = self.log_fold
        else:
            self.log_fold = f + "/"

        if filename_in is None:
            filename = os.path.join(self.log_fold, "binary_history.data")
        else:
            filename = filename_in

        self.binary.loadFile(filename, max_num_lines, cols)

        if max_model > 0:
            self.binary.data = self.binary.data[self.binary.model_number <= max_model]

        # Reverse model numbers, we want the unique elements
        # but keeping the last not the first.

        # Fix case where we have at end of file numbers:
        # 1 2 3 4 5 3, without this we get the extra 4 and 5
        self.binary.data = self.binary.data[
            self.binary.model_number <= self.binary.model_number[-1]
        ]

        mod_rev = self.binary.model_number[::-1]
        _, mod_ind = np.unique(mod_rev, return_index=True)
        self.binary.data = self.binary.data[
            np.size(self.binary.model_number) - mod_ind - 1
        ]


class inlist(object):
    def __init__(self):
        pass

    def read(self, filename):
        res = {}
        with open(filename, "r") as f:
            for l in f:
                l = l.strip()
                if l.startswith("!") or not len(l.strip()):
                    continue
                if "=" in l:
                    line = l.split("=")
                    res[line[0].strip()] = line[1].strip()
        return res
