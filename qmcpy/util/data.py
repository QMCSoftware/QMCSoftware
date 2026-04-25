import gzip
import pickle
try:
    import dill as _serializer
except ImportError:
    _serializer = pickle

from ..util import _univ_repr


class Data(object):

    def __init__(self, parameters):
        self.parameters = parameters

    def save(self, path, compress=False):
        """Save this Data object to disk using pickle.

        Warning:
            Pickle and dill files are not secure against untrusted input. Only
            save and later load checkpoint files that you created or fully
            trust.

        Args:
            path (str or pathlib.Path): File path to save to. If
                ``compress=True``, a ``.gz`` suffix is appended automatically
                when not already present.
            compress (bool, optional): Gzip-compress the saved file. Defaults
                to False.
        """
        path = str(path)
        if compress and not path.endswith(".gz"):
            path = path + ".gz"
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "wb") as f:
            _serializer.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load a Data object from disk.

        Warning:
            Loading pickle or dill files can execute arbitrary code. Only load
            checkpoint files that you created or fully trust.

        Args:
            path (str or pathlib.Path): Path to the saved file. Files ending
                in ``.gz`` are decompressed automatically.

        Returns:
            Data: The loaded Data object.
        """
        path = str(path)
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as f:
            return _serializer.load(f)

    def __repr__(self):
        string = _univ_repr(self, "Data", self.parameters + ["time_integrate"])
        if hasattr(self, "stopping_crit") and self.stopping_crit:
            string += "\n" + str(self.stopping_crit)
        if hasattr(self, "integrand") and self.integrand:
            string += "\n" + str(self.integrand)
        if hasattr(self, "true_measure") and self.true_measure:
            string += "\n" + str(self.true_measure)
        if hasattr(self, "discrete_distrib") and self.discrete_distrib:
            string += "\n" + str(self.discrete_distrib)
        return string
