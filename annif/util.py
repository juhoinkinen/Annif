"""Utility functions for Annif"""

import glob
import os
import tempfile
from annif import logger


def atomic_save(obj, dirname, filename, method=None):
    """Save the given object (which must have a .save() method, unless the
    method parameter is given) into the given directory with the given
    filename, using a temporary file and renaming the temporary file to the
    final name."""

    tempfd, tempfilename = tempfile.mkstemp(prefix=filename, dir=dirname)
    os.close(tempfd)
    logger.debug('saving %s to temporary file %s', str(obj), tempfilename)
    if method is not None:
        method(obj, tempfilename)
    else:
        obj.save(tempfilename)
    for fn in glob.glob(tempfilename + '*'):
        newname = fn.replace(tempfilename, os.path.join(dirname, filename))
        logger.debug('renaming temporary file %s to %s', fn, newname)
        os.rename(fn, newname)


def localname(uri):
    """return the local name extracted from a URI, i.e. the part after the
    last slash or hash character"""

    return uri.split('/')[-1].split('#')[-1]


def cleanup_uri(uri):
    """remove angle brackets from a URI, if any"""
    if uri.startswith('<') and uri.endswith('>'):
        return uri[1:-1]
    return uri
