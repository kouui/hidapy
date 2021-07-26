

_GROUP_ = {
    "kind" : (
        "spectro-polarimetry",
        "spectroscopy",
    ),
}

_DATASET_ = {
    "target" : (
        "prominence",
        "sunspot",
        "filament",
        "spicule",
        "diskcenter",
        "quietregion",
        "activeregion",
        "lamp"
    ),
}

#-----------------------------------------------------------------------------
# hdf5 io
import h5py
#-----------------------------------------------------------------------------

def make_group(fname, name, description, kind="spectro-polarimetry"):
    r""" """

    assert kind in _GROUP_["kind"], f"kind : {_GROUP_['kind']}"


    with h5py.File(fname, 'a') as f:
        if name in f.keys():
            return 0
        group = f.create_group(name)
        group.attrs["description"] = description
        group.attrs["kind"] = kind

        return 1

def make_dataset(fname, gname, name, data, description, exposure, dtime, wave, period, target, angle=None):
    r""" """

    assert target in _DATASET_['target'], f"target : {_DATASET_['target']}"

    with h5py.File(fname, 'a') as f:
        group = f[gname]
        if name in group.keys():
            return 0
        dataset = group.create_dataset( name=name, data=data )
        dataset.attrs["description"] = description
        dataset.attrs["exposure[msec]"] = exposure
        dataset.attrs["datetime"] = dtime
        dataset.attrs["wavelength[AA]"] = wave
        dataset.attrs["period[sec]"] = period
        dataset.attrs["target"] = target

        if angle is not None:
            dataset.attrs["angle[deg]"] = angle

        return 1

def is_in_group(fname, gname, name):

    with h5py.File(fname, 'a') as f:
        group = f[gname]
        if name in group.keys():
            return 1
        else:
            return 0
