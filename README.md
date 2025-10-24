MoBILAB
=======
![figure](https://github.com/aojeda/mobilab/blob/master/data/Ms_browser.png)

MoBILAB is an open source toolbox for analysis and visualization of mobile brain/body imaging data. Read our Frontiers paper [here](https://www.frontiersin.org/articles/10.3389/fnhum.2014.00121/full).

[Documentation.](https://sccn.ucsd.edu/wiki/MoBILAB)

MoBILAB no longer compatible with recent MATLAB versions
===========
MoBILAB relies on complex MATLAB objects that are no longer fully supported in recent MATLAB versions, and would be too difficult to port (MoBILAB is known to work on MATLAB 2017a if you are committed -- and maybe some later version as well). Therefore, it is no longer maintained. Instead, use the XDF plugins in EEGLAB, which, as of 2024, support importing multiple LSL/XDF data streams into EEGLAB.

Release notes
=============
- Got to the dependency/xdf/xdf folder
- Compile mex file for OSX and Windows

Versions
======
v20200220 - first official revision as EEGLAB plugin

v20201223 - create error when the wrong EEGLAB options are present and remove warning for property grid

v20210311 - additional error if the wrong EEGLAB options are selected. Remove redundant functions with EEGLAB

v20210924 - fix path issue
