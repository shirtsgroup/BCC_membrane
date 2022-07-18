#!/usr/bin/env python

import sys
sys.path.append('lib/scripts')
import argparse
from bcc_build import BicontinuousCubicBuild
from genmdp import SimulationMdp
from gentop import SystemTopology
import transform, gromacs
import topology, restrain, file_rw
import warnings
import mdtraj as md
import os
import numpy as np
import subprocess
import yaml

#warnings.simplefilter('error', UserWarning)


def initialize():

    parser = argparse.ArgumentParser(description='Equilibrate bicontinuous cubic phase unit cell')

    parser.add_argument('-y', '--yaml', type=str, help='A .yaml configuration file. This is preferred over using'
                                                       'argparse. It leads to better reproducibility.')

    # Build parameters
    parser.add_argument('-p', '--space_group', default='gyroid', type=str, help='Liquid crystal phase to build (HII, QI '
                                                                              'etc.)')
    parser.add_argument('-b', '--build_monomer', default='Dibrpyr14', type=str, help='Name of single monomer'
                        'structure file (.gro format) used to build full system')
    parser.add_argument('-o', '--out', default='initial.gro', help='Name of output .gro file for full system')
    parser.add_argument('-seed', '--random_seed', default=False, type=int, help='Random seed for column shift. Set this'
                                                                                'to reproduce results')
    parser.add_argument('-box', '--box_length', type=float, default=10., help='Length of box vectors [x y z]')
    parser.add_argument('-g', '--grid', default=50, type=int, help='Number of sections to break grid into when '
                                                                'approximating the chosen implicit function')
    parser.add_argument('-dens', '--density', default=1.1, type=float, help='Density of system (g/cm3)')
    parser.add_argument('-c', '--curvature', default=1, type=float,
                        help='mean curvature of the system. A value of 0, < 0 and > 0 correspond to zero, negative and'
                             'positive mean curvature respectively.')
    parser.add_argument('-wt', '--weight_percent', default=77.1, type=float, help='Weight %% of monomer in membrane')
    parser.add_argument('-sol', '--solvent', default='GLY', type=str, help='Name of solvent residue mixed with monomer')
    parser.add_argument('-shift', '--shift', default=0, type=float, help='Shift position of head group shift units in'
                        'the direction opposite of the normal vector at that point')
    parser.add_argument('-res', '--residue', default='MOL', help='Name of residue corresponding to build monomer')
    parser.add_argument('-resd', '--dummy_residue', default='MOLd', help='Name of residue to be cross-linked with '
                                                                         'dummy atoms included in the topology')


    return parser


class EnsembleError(Exception):
    """ Raised if invalid thermodynamic ensemble is passed """

    def __init__(self, message):

        super().__init__(message)


class EquilibrateBCC(topology.LC):

    def __init__(self, build_monomer, space_group, box_length, weight_percent, density, restraints, restraint_residue,
                 shift=0, curvature=0, mpi=False, nprocesses=4):
        """
        :param build_monomer: name of monomer with which to build phase
        :param space_group: name of space group into which monomers are arranged (i.e. gyroid, schwarzD etc.)
        :param box_length: length of each edge of the unit cell (nm). The box is cubic, so they must all be the same (or
        just one length specified)
        :param weight_percent: percent by weight of monomer in membrane
        :param density: experimentally derived density of membrane
        :param restraints: while scaling and shrinking unit cell, add restraints to atoms annotated with 'Rb' in the
        monomer topology file. 3x1 tuple or list
        :param restraint_residue: name of resiude to be restrained
        :param shift: translate monomer along vector perpendicular to space group surface by this amount (nm). This
        parameter effectively controls the pore size
        :param curvature: determines whether the phase is normal or inverted. {-1 : QI phase, 1: QII phase}'
        :param mpi: parallelize GROMACS commands using MPI
        :param nprocesses: number of MPI processes if `mpi` is true

        :type monomer: str
        :type space_group: str
        :type dimensions: float, list of floats
        :type weight_percent: int, float
        :type density: float
        :type restraints: tuple or list
        :type restraint_residue: str
        :type shift: float
        :type curvature: int
        :type mpi: bool
        :type nprocesses: int
        """

        super().__init__(build_monomer)

        self.build_monomer = build_monomer
        self.space_group = space_group
        self.period = box_length
        self.weight_percent = weight_percent
        self.density = density
        self.shift = shift
        self.curvature = curvature
        self.restraints = restraints
        self.restraint_residue = restraint_residue
        if self.restraint_residue not in self.residues:
            warnings.warn('Specified restraint residue, %s, is not consistent with passed monomer residues (%s).'
                          % (self.restraint_residue, ', '.join(self.residues)))

        # names of files (hard coded for now since it's not really important
        self.em_mdp = 'em.mdp'
        self.nvt_mdp = 'nvt.mdp'
        self.npt_mdp = 'npt.mdp'
        self.top_name = 'topol.top'

        self.topology = None
        self.system = None
        self.gro_name = None  # name of .gro file
        self.mdp = None
        self.solvent = None
        self.nsolvent = 0

        # parallelization
        self.mpi = mpi
        self.nprocesses = nprocesses

    def build_initial_config(self, grid_points=50, r=0.4, name='initial.gro'):
        """ Build an initial configuration with parameters specified in the __init__ function

        :param grid_points: number of grid points to use when approximating implicit surface
        :param r: distance between monomer placement points (no monomers will be placed within a sphere of radius, r
        from a given monomer.
        :param name: name of output configuration

        :type grid_points: int
        :type r: float
        :type name: str
        """

        self.system = BicontinuousCubicBuild(self.build_monomer, self.space_group, self.period, self.weight_percent,
                                             self.density)

        self.system.gen_grid(grid_points, self.curvature)

        self.system.determine_monomer_placement(r=r)

        self.system.place_monomers(shift=self.shift)

        self.system.reorder()

        self.system.write_final_configuration(name=name)

        self.gro_name = name

    def generate_topology(self, name='topol.top', xlinked=False, restrained=False):
        """ Generate a topology for the current unit cell

        :param name: name of topology file
        :param xlinked: True if the system has been cross-linked
        :param restrained: add restraints to topology

        :type name: str
        :type xlinked: bool
        :type restrained: bool
        """

        if restrained:
            r = restrain.RestrainedTopology(self.gro_name, self.restraint_residue, self.build_restraints, com=False,
                                            xlink=xlinked, vparams=None)
            r.add_position_restraints('xyz', self.restraints)
            r.write_topology()

        self.topology = SystemTopology(self.gro_name, xlink=xlinked, restraints=False)
        self.topology.write_top(name=name)


if __name__ == "__main__":

    args = initialize().parse_args()

    if args.yaml:
        with open(args.yaml, 'r') as yml:
            cfg = yaml.load(yml, Loader=yaml.FullLoader)

            build_params = cfg['build_parameters']

    else:
        sys.exit('Using argparse for most arguments in this script is no longer supported. Please make a .yaml file')

    os.environ["GMX_MAXBACKUP"] = "-1"  # stop GROMACS from making backups
    if not os.path.isdir("./intermediates"):
        os.mkdir('intermediates')

    equil = EquilibrateBCC(build_params['build_monomer'], build_params['space_group'], build_params['box_length'],
                           build_params['weight_percent'], build_params['density'], build_params['restraints'],
                           build_params['restraint_residue'], shift=build_params['shift'],
                           curvature=build_params['curvature'], mpi=False, nprocesses=1)

    equil.build_initial_config(grid_points=build_params['grid'], r=0.4)


    equil.generate_topology(name='topol.top', restrained=False)  # creates an output file

    



