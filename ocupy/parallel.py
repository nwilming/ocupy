#!/usr/bin/env python
"""A set of classes that allow to run embarassingly 
parallel tasks over a grid"""

import math
import cPickle

import xmlrpclib

import numpy as np

from twisted.internet import reactor 
from twisted.web import xmlrpc


class TaskManager(xmlrpc.XMLRPC):
    """
    A server that distributes tasks to connecting clients and collects results.
    
    This class is a generic implementation of an XML-RPC server that iterates
    over a task_store object and distributes tasks which are stored in the 
    task_store. 
    
    A client that connects to this server via XML-RPC is expected to carry 
    out the following steps: 
    
    1. It calls get_task upon which the server returns a tuple that 
    contains an id and a serialized task_store object that contains a 
    subset of all tasks. The task store object is serialized via
    task_store.to_dict()
    2. The client iterates over all tasks in the task_store it received
    and returns the results as a tuple that contains the task_store
    id received from get task and the results.
    3. If get_task returns false, the client exits.
    
    To collect status updates and results of the computations a client
    can connect and call get_status or return_results. 
    """
    
    def __init__(self, task_store):
        self._start_server()
        self.task_store = task_store
        self.task_iterator = enumerate(task_store.partition())
        self.scheduled_tasks = {}
        self.results = 0
        self.reschedule = []
    
    def _start_server(self):
        """
        Starts the XMLRPC interface
        """
        xmlrpc.XMLRPC.__init__(self)

    def xmlrpc_exit(self):
        """ 
        Terminates server
        """
        reactor.stop()
        return True
    
    def xmlrpc_reschedule(self):
        """
        Reschedule all running tasks. 
        """
        if not len(self.scheduled_tasks) == 0:
            self.reschedule = list(self.scheduled_tasks.iteritems())
            self.scheduled_tasks = {}
        return True 

    def xmlrpc_get_task(self):
        """
        Return a new task description: ID and necessary parameters, 
        all are given in a dictionary
        """
        try:
            if len(self.reschedule) == 0:
                (task_id, cur_task) = self.task_iterator.next()
            else:
                (task_id, cur_task) = self.reschedule.pop()
            self.scheduled_tasks.update({task_id: cur_task})
            return (task_id, cur_task.to_dict())
        except StopIteration:
            print 'StopIteration: No more tasks'
            return False
        except Exception as err:
            print 'Some other error'
            print err
            return False                        

    def xmlrpc_task_done(self, result):
        """
        Take the results of a computation and put it into the results list.
        """
        (task_id, task_results) = result
        del self.scheduled_tasks[task_id]
        self.task_store.update_results(task_id, task_results)
        self.results += 1
        return True
    
    def xmlrpc_status(self):
        """
        Return a status message
        """
        return ("""
        %i Jobs are still wating for execution
        %i Jobs are being processed
        %i Jobs are done
        """ %(self.task_store.partitions - 
                self.results - 
                len(self.scheduled_tasks),
              len(self.scheduled_tasks),
              self.results))

    def xmlrpc_save2file(self, filename):
        """
        Save results and own state into file.
        """
        savefile = open(filename,'wb')
        try:
            cPickle.dump({'scheduled':self.scheduled_tasks,
                          'reschedule':self.reschedule},savefile)
        except cPickle.PicklingError:
            return -1
        savefile.close()
        return 1

 
class Worker(object):
    """
    A base for XML-RPC clients that do work for a TaskManager.

    The client works as follows: It connects to a server and calls
    get_task. It then configures it's own task object by calling
    task_store.from_dict(dict) and iterates over the task in it's
    own task_store.
    
    Each iteration returns one task_description that is used as
    an argument for the compute method. Whatever is returned as
    a result from compute is returned to the TaskManager. 

    To implement a specific worker for your own task, the only
    thing to do is to implement the compute method. 
    If the worker needs to load data or other things that are 
    needed for each task, the setup method can be used. Setup is
    called when the Worker inits.  
    """
    def __init__(self, url, task_store):
        self.server = xmlrpclib.Server(url)
        self.setup()   
        self.results = [] 
        self.task_store = task_store
   
    def setup(self):
        """Called before compute is called for the first time.
           Can be used to set up data and so forth"""
        pass
     
    def run(self):
        """This function needs to be called to start the computation."""
        (task_id, tasks) = self.server.get_task()
        self.task_store.from_dict(tasks)
        for (index, task) in self.task_store:
            result = self.compute(index, task)
            self.results.append(result)
        self.server.task_done((task_id, self.results))
        
    def compute(self, index, task_description):
        """ The compute function returns the results for the task 
            described by task_description. Task description contains
            a dictionary that refers to parameters and values are the
            corresponding values."""
        raise NotImplementedError(
                'Function needs to implemented by specific worker')

class TaskStore(object):
    """
    A TaskStore manages a set of tasks.
    
    A task in a task store consists of a dictionary that has uses keys
    that are meaningfull to a worker's compute method. I.e. the keys name
    parameters for the compute method and the values are the values of these
    parameters. 

    A task store can partition itself into groups of tasks via the
    partition function. It returns task store objects that only access a subset 
    of all tasks. This is achieved by creating the task store object with a 
    linear index of tasks.
    
    A task store object can also iterate over tasks directly and return
    the dictionary that is associated with each task. Consequently a 
    task store that has received task indices upon creation will only 
    iterate over the tasks indexed by the indices.

    The current implementation and concepts used imply that a task_store
    object is fully defined by the number of partitions it creates and
    the indices it posses. It is de facto stateless. 

    To send a task_store via a XML-RPC call it is necessary to serialize
    it into a dictionary which is performed by the functions
    from_dict and to_dict. 
    """

    def __init__(self, ident, 
                       num_partitions = 100, 
                       indices = None):
        self.ident = ident
        self.partitions = num_partitions
        self.indices = indices
    
    def to_dict(self):
        """Returns a dictionary representation that allows to fully 
            recreate the task store"""
        return {'partitions' : self.partitions, 
                'indices' : self.indices, 'ident' : self.ident}
    
    def from_dict(self, description):
        """Configures the task store to be the task_store described 
            in description"""
        assert(self.ident == description['ident'])
        self.partitions = description['partitions']
        self.indices    = description['indices']
         
    def __iter__(self):
        for index in self.indices:       
            params = self.ind2sub(index)
            yield (index, self.get(index, *params))

    def partition(self):
        """Partitions all tasks into groups of tasks. A group is
           represented by a task_store object that indexes a sub-
           set of tasks."""
        step = int(math.ceil(self.num_tasks / float(self.partitions)))
        if self.indices == None:
            slice_ind = range(0, self.num_tasks, step)
            for start in slice_ind:
                yield self.__class__(self.partitions, 
                                     range(start, start + step))
        else:
            slice_ind = range(0, len(self.indices), step)
            for start in slice_ind:
                if start + step <= len(self.indices):
                    yield self.__class__(self.partitions, 
                                         self.indices[start: start + step])
                else:
                    yield self.__class__(self.partitions, self.indices[start:])


    def update_results(self, task_id, task_description):
        """User implemented method that organizes results into some
           structure and takes care of saving it"""
        raise NotImplementedError

    def get(self, index, *params):
        """User implemented method that returns a task description
           for a set of parameters. This allows to create complex
           task, i.e. this function can add information to a task
           description that is dependent on the parameters. """
        raise NotImplementedError 
    
    def sub2ind(self, *params):
        """ Map a set of parameters to linear index."""
        raise NotImplementedError

    def ind2sub(self, index):
        """ Map index to a set of parameters. """
        raise NotImplementedError

def ind2sub(ind, dimensions):
    """
    Calculates subscripts for indices into regularly spaced matrixes.
    """
    # check that the index is within range
    if ind >= np.prod(dimensions):
        raise RuntimeError("ind2sub: index exceeds array size")
    cum_dims = list(dimensions)
    cum_dims.reverse()
    m = 1
    mult = []
    for d in cum_dims:
        m = m*d
        mult.append(m)
    mult.pop()
    mult.reverse()
    mult.append(1)
    indices = []
    for d in mult:
        indices.append((ind/d)+1)
        ind = ind - (ind/d)*d
    return indices


def sub2ind(indices, dimensions):
    """
    An exemplary sub2ind implementation to create randomization 
    scripts. 

    This function calculates indices from subscripts into regularly spaced
    matrixes.
    """
    # check that none of the indices exceeds the size of the array
    if any([i > j for i, j in zip(indices, dimensions)]):
        raise RuntimeError("sub2ind:an index exceeds its dimension's size")
    dims = list(dimensions)
    dims.append(1)
    dims.remove(dims[0])
    dims.reverse()
    ind  = list(indices)
    ind.reverse()
    idx = 0
    mult = 1
    for (cnt, dim) in zip(ind, dims):
        mult = dim*mult
        idx = idx + (cnt-1)*mult    
    return idx

def RestoreTaskStoreFactory(store_class, chunk_size, restore_file, save_file):
    """
    Restores a task store from file.
    """
    intm_results = np.load(restore_file)
    intm = intm_results[intm_results.files[0]]
    idx = np.isnan(intm).flatten().nonzero()[0]
    partitions = math.ceil(len(idx) / float(chunk_size))
    task_store = store_class(partitions, idx.tolist(), save_file)
    task_store.num_tasks = len(idx)
    # Also set up matrices for saving results
    for f in intm_results.files:
        task_store.__dict__[f] = intm_results[f]
    return task_store

