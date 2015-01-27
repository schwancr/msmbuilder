# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import absolute_import, print_function, division

import sys
import os
import re
import glob
from os.path import join, exists, expanduser
import socket
import getpass
import itertools
from datetime import datetime
from collections import Sequence

import tables
import mdtraj as md
from mdtraj.core.trajectory import _parse_topology
import numpy as np
from . import version

_PYTABLES_DISABLE_COMPRESSION = tables.Filters(complevel=0)


__all__ = ['dataset']


def dataset(path, mode='r', fmt=None, verbose=False, **kwargs):
    """Open a dataset object

    MSMBuilder supports several dataset 'formats' for storing
    lists of sequences on disk.

    This function can also be used as a context manager.

    Parameters
    ----------
    path : str
        The path to the dataset on the filesystem
    mode : {'r', 'w'}
        Open a dataset for reading or writing
    fmt : {'dir-npy', 'hdf5', 'mdtraj'}
        The format of the data on disk

        ``dir-npy``
            A directory of binary numpy files, one file per sequence

        ``hdf5``
            A single hdf5 file with each sequence as an array node

        ``mdtraj``
            A read-only set of trajectory files that can be loaded
            with mdtraj

        ``dir-npy-union`` or ``hdf5-union``
            Several datasets of the respective type which will have
            their features union-ed together.

    verbose : bool
        Whether to print information about the dataset

    """

    if mode == 'r' and fmt is None:
        fmt = _guess_format(path)
    elif mode == 'w' and fmt is None:
        raise ValueError('mode="w", but no fmt. fmt=%s' % fmt)

    if fmt == 'dir-npy':
        return NumpyDirDataset(path, mode=mode, verbose=verbose)
    elif fmt == 'mdtraj':
        return MDTrajDataset(path, mode=mode, verbose=verbose, **kwargs)
    elif fmt == 'hdf5':
        return HDF5Dataset(path, mode=mode, verbose=verbose)
    elif fmt.endswith("-union"):
        sub_fmt = fmt[:-len('-union')]
        return UnionDataset(path, fmt=sub_fmt, mode=mode, verbose=verbose)
    else:
        raise NotImplementedError("unknown fmt: %s" % fmt)


def _guess_format(path):
    if isinstance(path, (list, tuple)):
        fmt = _guess_format(path[0])
        err = "Only the union of 'dir-npy' and 'hdf5' formats is supported"
        assert fmt in ['dir-npy', 'hdf5'], err
        err = "All datasets must be the same format"
        for p in path[1:]:
            assert _guess_format(p) == fmt, err
        return "{}-union".format(fmt)

    if os.path.isdir(path):
        return 'dir-npy'

    if path.endswith('.h5') or path.endswith('.hdf5'):
        # TODO: Check for mdtraj .h5 file
        return 'hdf5'

    # TODO: What about a list of trajectories, e.g. from command line nargs='+'
    return 'mdtraj'


class _BaseDataset(Sequence):
    _PROVENANCE_TEMPLATE = '''MSMBuilder Dataset:
  MSMBuilder:\t{version}
  Command:\t{cmdline}
  Path:\t\t{path}
  Username:\t{user}
  Hostname:\t{hostname}
  Date:\t\t{date}
  Comments:\t\t{comments}
'''
    _PREV_TEMPLATE = '''
== Derived from ==
{previous}
'''

    def __init__(self, path, mode='r', verbose=False):
        self.path = path
        self.mode = mode
        self.verbose = verbose

        if mode not in ('r', 'w'):
            raise ValueError('mode must be one of "r", "w"')
        if mode == 'w':
            if exists(path):
                raise ValueError('File exists: %s' % path)
            os.makedirs(path)
            self._write_provenance()

    def create_derived(self, out_path, comments='', fmt=None):
        if fmt is None:
            out_dataset = self.__class__(out_path, mode='w', verbose=self.verbose)
        else:
            out_dataset = dataset(out_path, mode='w', verbose=self.verbose, fmt=fmt)
        out_dataset._write_provenance(previous=self.provenance, comments=comments)
        return out_dataset

    def apply(self, fn):
        for key in self.keys():
            yield fn(self.get(key))

    def _build_provenance(self, previous=None, comments=''):
        val = self._PROVENANCE_TEMPLATE.format(
            version=version.full_version,
            cmdline=' '.join(sys.argv),
            user=getpass.getuser(),
            hostname=socket.gethostname(),
            path=self.path,
            comments=comments,
            date=datetime.now().strftime("%B %d, %Y %I:%M %p"))
        if previous:
            val += self._PREV_TEMPLATE.format(previous=previous)
        return val

    @property
    def provenance(self):
        raise NotImplementedError('implemented in subclass')

    def _write_provenance(self, previous, comments=''):
        raise NotImplementedError('implemented in subclass')

    def __len__(self):
        return sum(1 for xx in self.keys())

    def __getitem__(self, i):
        return self.get(i)

    def __setitem__(self, i, x):
        return self.set(i, x)

    def __iter__(self):
        for key in sorted(self.keys(), key=_keynat):
            yield self.get(key)

    def keys(self):
        raise NotImplementedError('implemeneted in subclass')

    def get(self, i):
        raise NotImplementedError('implemeneted in subclass')

    def set(self, i, x):
        raise NotImplementedError('implemeneted in subclass')

    def close(self):
        pass

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class NumpyDirDataset(_BaseDataset):
    """Mixtape dataset container

    Parameters
    ----------
    path : str
    mode : {'r', 'w'}

    Examples
    --------
    for X in Dataset('path/to/dataset'):
        print X
    """

    _ITEM_FORMAT = '%08d.npy'
    _ITEM_RE = re.compile('(\d{8}).npy')
    _PROVENANCE_FILE = 'PROVENANCE.txt'

    def get(self, i, mmap=False):
        if isinstance(i, slice):
            items = []
            start, stop, step = i.indices(len(self))
            for ii in itertools.islice(itertools.count(), start, stop, step):
                items.append(self.get(ii))
            return items

        mmap_mode = 'r' if mmap else None

        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] loading %s' % filename)
        try:
            return np.load(filename, mmap_mode)
        except IOError as e:
            raise IndexError(e)

    def set(self, i, x):
        if 'w' not in self.mode:
            raise IOError('Dataset not opened for writing')
        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] saving %s' % filename)
        return np.save(filename, x)

    def keys(self):
        for fn in sorted(os.listdir(os.path.expanduser(self.path)), key=_keynat):
            match = self._ITEM_RE.match(fn)
            if match:
                yield int(match.group(1))

    @property
    def provenance(self):
        try:
            with open(join(self.path, self._PROVENANCE_FILE), 'r') as f:
                return f.read()
        except IOError:
            return 'No available provenance'

    def _write_provenance(self, previous=None, comments=''):
        with open(join(self.path, self._PROVENANCE_FILE), 'w') as f:
            p = self._build_provenance(previous=previous, comments=comments)
            f.write(p)


class HDF5Dataset(_BaseDataset):
    _ITEM_FORMAT = 'arr_%d'
    _ITEM_RE = re.compile('arr_(\d+)')

    def __init__(self, path, mode='r', verbose=False):
        if mode not in ('r', 'w'):
            raise ValueError('mode must be one of "r", "w"')
        if mode == 'w':
            if exists(path):
                raise ValueError('File exists: %s' % path)

        filters = tables.Filters(complevel=0)
        self._handle = tables.open_file(path, mode=mode,
                                        filters=_PYTABLES_DISABLE_COMPRESSION)
        self.path = path
        self.mode = mode
        self.verbose = verbose

        if mode == 'w':
            self._write_provenance()

    def __getstate__(self):
        # pickle does not like to pickle the pytables handle, so...
        # self.flush()
        return {'path': self.path, 'mode': self.mode, 'verbose': self.verbose}

    def __setstate__(self, state):
        self.path = state['path']
        self.mode = state['mode']
        self.verbose = state['verbose']
        self._handle = tables.open_file(self.path, mode=self.mode,
                                        filters=_PYTABLES_DISABLE_COMPRESSION)

    def get(self, i, mmap=False):
        if isinstance(i, slice):
            items = []
            start, stop, step = i.indices(len(self))
            for ii in itertools.islice(itertools.count(), start, stop, step):
                items.append(self.get(ii))
            return items

        return self._handle.get_node('/', self._ITEM_FORMAT % i)[:]

    def keys(self):
        for node in self._handle.iter_nodes('/'):
            match = self._ITEM_RE.match(node.name)
            if match:
                yield int(match.group(1))

    def set(self, i, x):
        if 'w' not in self.mode:
            raise IOError('Dataset not opened for writing')

        try:
            self._handle.create_carray('/', self._ITEM_FORMAT % i, obj=x)
        except tables.exceptions.NodeError:
            self._handle.remove_node('/', self._ITEM_FORMAT % i)
            self.set(i, x)

    @property
    def provenance(self):
        try:
            return self._handle.root._v_attrs['provenance']
        except KeyError:
            return 'No available provenance'

    def _write_provenance(self, previous=None, comments=''):
        p = self._build_provenance(previous=previous, comments=comments)
        self._handle.root._v_attrs['provenance'] = p

    def close(self):
        if hasattr(self, '_handle'):
            self._handle.close()

    def flush(self):
        self._handle.flush()

    def __del__(self):
        self.close()


class MDTrajDataset(_BaseDataset):
    _PROVENANCE_TEMPLATE = '''MDTraj dataset:
  path:\t\t{path}
  topology:\t{topology}
  stride:\t{stride}
  atom_indices\t{atom_indices}
'''

    def __init__(self, path, mode='r', topology=None, stride=1,
                 atom_indices=None, verbose=False):
        if mode != 'r':
            raise ValueError('mode must be "r"')
        self.path = path
        self.topology = topology
        self.stride = stride
        self.atom_indices = atom_indices
        self.verbose = verbose

        if isinstance(path, list):
            self.glob_matches = [expanduser(fn) for fn in path]
        else:
            self.glob_matches = sorted(glob.glob(expanduser(path)), key=_keynat)

        if topology is None:
            self._topology = None
        else:
            self._topology = _parse_topology(os.path.expanduser(topology))

    def get(self, i):
        if self.verbose:
            print('[MDTraj dataset] loading %s' % self.filename(i))

        if self._topology is None:
            t = md.load(self.filename(i), stride=self.stride,
                        atom_indices=self.atom_indices)
        else:
            t = md.load(self.filename(i), stride=self.stride,
                        atom_indices=self.atom_indices, top=self._topology)
        return t

    def filename(self, i):
        return self.glob_matches[i]

    def iterload(self, i, chunk):
        if self.verbose:
            print('[MDTraj dataset] iterloading %s' % self.filename(i))

        if self._topology is None:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices)
        else:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices, top=self._topology)

    def keys(self):
        return iter(range(len(self.glob_matches)))

    @property
    def provenance(self):
        return self._PROVENANCE_TEMPLATE.format(
            path=self.path, topology=self.topology,
            atom_indices=self.atom_indices, stride=self.stride)


def _dim_match(arr):
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


class UnionDataset(_BaseDataset):
    def __init__(self, paths, mode, fmt='dir-npy', verbose=False):
        # Check mode
        if mode != 'r':
            raise ValueError("Union datasets are read only")

        # Check format
        supported_subformats = ['dir-npy', 'hdf5']
        if fmt not in supported_subformats:
            err = "Format must be one of {}. You gave {}"
            err = err.format(supported_subformats, fmt)
            raise ValueError(err)

        # Save parameters
        self.verbose = verbose
        self.datasets = [dataset(path, mode, fmt, verbose)
                         for path in paths]

        # Sanity check
        self._check_same_length()

    def _check_same_length(self):
        """Check that the datasets are the same length"""
        lens = []
        for ds in self.datasets:
            lens.append(
                sum(1 for _ in ds.keys())
            )
        if len(set(lens)) > 1:
            err = "Each dataset must be the same length. You gave: {}"
            err = err.format(lens)
            raise ValueError(err)

    def keys(self):
        return self.datasets[0].keys()

    def get(self, i):
        return np.concatenate([_dim_match(ds.get(i))
                               for ds in self.datasets], axis=1)

    def close(self):
        for ds in self.datasets:
            ds.close()

    def flush(self):
        for ds in self.datasets:
            ds.close()

    @property
    def provenance(self):
        return "\n\n".join(ds.provenance for ds in self.datasets)


def _keynat(string):
    """A natural sort helper function for sort() and sorted()
    without using regular expression.
    """
    r = []
    for c in string:
        if c.isdigit():
            if r and isinstance(r[-1], int):
                r[-1] = r[-1] * 10 + int(c)
            else:
                r.append(int(c))
        else:
            r.append(9 + ord(c))
    return r
