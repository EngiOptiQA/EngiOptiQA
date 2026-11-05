# Contributing

Thanks for contributing!

## Branch naming

Create a branch for each change using one of these naming patterns:

| Change type | Branch pattern | Example |
|---|---|---|
| New feature | `feature/<short-description>` | `feature/new-problem` |
| Bug fix | `fix/<short-description>` | `fix/problem-formulation` |
| Documentation | `docs/<short-description>` | `docs/new-solver-backend` |
| Examples | `example/<short-description>` | `example/new-problem-2d` |

Use lowercase words separated by hyphens.

## Pull requests

1. Create a branch from `main`.
2. Make and test your changes.
3. Open a pull request targeting `main`.
4. Include a concise description of the change.
5. Link related issues, if applicable.

### Pull request checklist

Before submitting a pull request, confirm that:

- [ ] Tests pass locally.
- [ ] New or changed behavior is covered by an example.
- [ ] Documentation has been updated where needed.
- [ ] The branch name follows the repository convention.

## Commit messages

Write short, descriptive commit messages. For example:

```text
Add new problem formulation
Fix variable ordering in problem formulation
Document new solver backend
Add 2D example for new problem