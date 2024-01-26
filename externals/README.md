### Externals

This folder contains snapshot of GitHub repos, which are used to create virtual environment in an editable mode. 
In case a repo releases an update, a new snapshot can be taken. 
The naming convention of the snapshots is `<name_repo>_<version_package>_<first 7 digits of commit hash>`.

Each study uses its own conda environment (look for the `environment.yml` file).
Different conda enviornments are used to work on different snapshots of a repo.

When cloning a repo here, it is important to remove both `.git` and `.github` hidden folders, so as to simply commit the snapshot as part of the current project. 

In other words, the snapshot in this folder are a workaround to using `git submodules`.

### Changelog and repo edits

As there the history of each cloned repo is deleted, it is important to keep a changelog of the modifications to the current snapshot.
At the very least, on should commit the edits alone (without other work), starting the commit message with the name of the snapshot.


