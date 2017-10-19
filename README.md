# BEHR PSM Gridding

This repository contains a modified version of the omi gridding package 
(https://github.com/gkuhl/omi) that works with BEHR data, as well as the
interface code to let it be called directly from the Matlab side of BEHR.

## A note on the no-he5 branch

Like the other BEHR repositories, we will follow the Git Flow paradigm
(http://nvie.com/posts/a-successful-git-branching-model/) here. This means
that the `master` branch should always be the "production" code (that is 
used to produce the published BEHR data) and `develop` is the branch that
is actively changing. 

In this repo, we essentially have two "master" branches: `master` itself and
`no-he5`. `no-he5` is a version with all of the omi package code dealing with
reading and saving HDF5 files removed, which eliminates the need to have the
Python h5py package installed.

To keep the branching model somewhat simple, the `develop` branch will be based
off `master`, meaning that it will include the HDF5 code. Then, when ready to 
deploy new code to `master`, first, any changes (from a release or hotfix branch)
into `master`, then merge from `master` into `no-he5`, taking care to remove any
references to HDF5 read/write code.
