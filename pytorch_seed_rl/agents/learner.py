"""Agent that runs inference and learning in parallel via multiple threads.

Consists of:
    #. n inference threads
    #. 1 training thread (python main thread)
    #. l data prefetching threads
    #. 1 reporting/logging object
"""

class Learner():
    """Agent that runs inference and learning in parallel via multiple threads.

    #. Runs inference for observations received from :py:class:`~pytorch_seed_rl.agents.actors`.
    #. Puts incomplete trajectories to :py:class:`~pytorch_seed_rl.data_structures.trajectory_store`
    #. Trains global model from trajectories received from a data prefetching thread.
    """

    def __init__(self):
        pass

    def infer(self):
        """Runs inference as rpc.
        """

    def train(self):
        """Trains on sampled, prefetched trajecotries.
        """

    def prefetch_data(self):
        """prefetches data from inference thread
        """
