# Gradient

Project bootstrapped using Python `3.8.17` (default, Jun  6 2023, 20:10:50) 
[GCC 11.3.0] on linux WSL.

Virtual environment and dependency management provided by `venv` and `pipenv`

Project bootstrap details below:

```console
$ python3.8 -m venv env
$ source ./env/bin/activate
$ pipenv install
$ git lfs install
```

Large files management provided by `git-lfs`

# Rules for commit and branch names

Run the following command to inject our custom hooks into your local repo:

```bash
cp -r ./githooks/. .git/hooks/
```

These hooks control the naming of our branches and commits:

- Branch: `(task|story|bugfix|improvement|research)/GRA-[0-9]+ (your message)`

- Commit msg: `GRA-[0-9]+ (your message)`

**Please verify that your local git hooks work**: try to create a branch with random name or try to commit with a random message in terminal and check if you receive an error message and commit does not occur.