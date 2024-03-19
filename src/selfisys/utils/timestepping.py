#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# The text of the license is located in the root directory of the source package.
# -------------------------------------------------------------------------------------


"""Wrapper aroiund `pysbmy.timestepping.read_timestepping` to merge several timestepping
files into a single one. This allows to easily use linearly spaced timesteps with
respect to the scale factor while chosing at what redshifts the snapshots are saved.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def merge_nTS(ts_path_list, merged_path):
    from h5py import File
    from numpy import concatenate
    from pysbmy.timestepping import read_timestepping

    ts = [read_timestepping(ts_path) for ts_path in ts_path_list]
    with File(merged_path, "w") as hf:
        hf.attrs["/info/scalars/nsteps"] = sum([tsi.nsteps for tsi in ts])
        hf.attrs["/info/scalars/nkicks"] = sum([tsi.nkicks for tsi in ts])
        hf.attrs["/info/scalars/ndrifts"] = sum([tsi.ndrifts for tsi in ts])
        hf.attrs["/info/scalars/ai"] = ts[0].ai
        hf.attrs["/info/scalars/af"] = ts[-1].af
        hf.create_dataset(
            "/scalars/snapshots", data=concatenate([tsi.snapshots for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/aKickBeg", data=concatenate([tsi.aKickBeg for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/aKickEnd", data=concatenate([tsi.aKickEnd for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/aDriftBeg", data=concatenate([tsi.aDriftBeg for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/aDriftEnd", data=concatenate([tsi.aDriftEnd for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/aiKick", data=concatenate([tsi.aiKick for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/afKick", data=concatenate([tsi.afKick for tsi in ts])
        )
        data = concatenate(
            [
                [ts[0].aDrift[0]],
                concatenate(
                    [concatenate([tsi.aDrift[1:], [tsi.aDrift[-1]]]) for tsi in ts[:-1]]
                ),
                ts[-1].aDrift[1:],
            ]
        )
        hf.create_dataset("/scalars/aDrift", data=data)
        hf.create_dataset(
            "/scalars/aDriftSave",
            data=concatenate(
                [
                    ts[0].aDriftSave,
                    concatenate([tsi.aDriftSave[1:] for tsi in ts[1:]]),
                ]
            ),
        )
        hf.create_dataset(
            "/scalars/aiDrift", data=concatenate([tsi.aiDrift for tsi in ts])
        )
        hf.create_dataset(
            "/scalars/afDrift", data=concatenate([tsi.afDrift for tsi in ts])
        )
        hf.create_dataset("/scalars/aKick", data=concatenate([tsi.aKick for tsi in ts]))
