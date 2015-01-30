# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 12:28:15 2014

@author: Nate
"""

import numpy, scipy.sparse
import networkx, matplotlib.pyplot
import pdb

    
class lattice():
    """
    object representation of a lattice, its structure, and tunnelling matrices
    """
    def __init__(self,options={}):
        default_ops={"t":1, "U":1, "mu":1, "dim":[10], "periodic":True}
        default_ops.update(options)
        for op in default_ops:
            setattr(self,op,default_ops[op])
        self.square_connectivity(self.dim, periodic=default_ops["periodic"])
        self.hilbert_space=self.hilbert_space_factory(self)
        self.hamiltonian=self.hamiltonian_factory(self)
        
    def square_connectivity(self,dim, periodic=True):
        """
        generates a square-lattice connectivity map in n-dimensions given a list of dimensions
        """
        self.dim=dim
        self.graph=networkx.grid_graph(dim=dim, periodic=periodic)

    def show(self):
        """
        displays lattice as a graph
        """
        networkx.draw(self.graph)
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    class hilbert_space_factory():
        """
        represents the hilbert space for quantum states defined on this lattice
        """
        def __init__(self,parent, allowed_occupancies=numpy.arange(2+1,dtype=numpy.uint8)):
            self.allowed_occupancies=allowed_occupancies
            if parent.graph!=None:
                self.fock_basis=self.generate_fock_basis(parent.graph,self.allowed_occupancies)
                
        def generate_fock_basis(self,graph,allowed_occupancies):
            """
            generates a set of basis states according to a list of allowed occupancies
            for a number of n_sites
            """
            n_sites=graph.number_of_nodes()
            b=numpy.rollaxis(numpy.indices((len(allowed_occupancies),)*n_sites,dtype=numpy.uint8),0,n_sites+1)
            b=b.reshape((len(allowed_occupancies)**n_sites,n_sites))
            b1=numpy.copy(b)
            for k,v in zip(range(len(allowed_occupancies)),allowed_occupancies):
                b1[b==k]=v
            return b1
            
    class hamiltonian_factory():
        """
        represents the hamiltonian for atoms on this lattice
        """
        def __init__(self,parent):
            for param in ["t","U","mu","hilbert_space"]:
                setattr(self,param,getattr(parent,param))
            self.sparse_connectivity=networkx.to_scipy_sparse_matrix(parent.graph)
            #self.calculate_BH_matrix_elements()
                
        def calculate_BH_matrix_elements(self):
            """
            calculates matrix elements for the bose hubbard hamiltonian for two states
            state_i, state_j in fock basis
            """
            # first handle diagonal (site number-operator commuting) elements
            self.sum_nnm1=(self.hilbert_space.fock_basis*(self.hilbert_space.fock_basis-1)).sum(axis=1)
            self.sum_n=self.hilbert_space.fock_basis.sum(axis=1)
            self.muU=self.U*self.sum_nnm1 - self.mu*self.sum_n
            # now handle off-diagonal tunneling elements - tunneling preserves (commutes with) number
            for n in numpy.unique(self.sum_n):
                # pick out the subspace at occupancy n
                n_subspace=self.hilbert_space.fock_basis[self.sum_n==n].astype(numpy.int32)
                # difference all fock basis states by local number, sum their absolute values across sites
                # tunnel-coupled states must have one site raised and one site lowered, thus
                # the sum of absolute value of differences must be 2 - use these to mark possible
                # tunnel-coupled sites in a list of basis state i <-> basis state j , (i,j) pairs
                fock_diffs=n_subspace[:,numpy.newaxis]-n_subspace
                single_swap_subspace=(numpy.abs(fock_diffs).sum(axis=2)==2).nonzero()
                # for each candidate tunnel-coupled pair, check the connectivity graph
                # what we want is essentially a graph to graph mapping; the connection 
                # of basis state i to basis state j is a lowering of site s and raising of site t
                # so potential matrix element (i,j) depends on lattice connectivity (s,t)
                # start by finding raised and lowered sites for each (i,j) basis state pair
                raised=(fock_diffs[single_swap_subspace]==1).nonzero()[1]
                lowered=(fock_diffs[single_swap_subspace]==-1).nonzero()[1]
                pdb.set_trace()
                
            self.K=numpy.empty((self.hilbert_space.fock_basis.shape[0],)*2,dtype=numpy.float64)            
            pass