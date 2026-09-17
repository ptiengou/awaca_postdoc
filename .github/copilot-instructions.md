# Repository Agent Rules

- Do not create a git commit, push to a remote, create or delete a branch, or rebase without the user's explicit approval for that specific operation.
- Treat files outside this repository as read-only, including in the code you write.
Do not modify, overwrite, delete, rename, or generate files in external folders, especially `/bdd/`.
- If completing a task would require changing an external file, stop and ask the user for explicit approval before making that change.
- git diff commands can be run without approval, but any changes to files must be approved by the user.