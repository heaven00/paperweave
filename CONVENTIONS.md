# Engineering Conventions

## Conventions
> When writing code, follow these conventions.

- treat tests as documentation, if they don't exist create them
- always update documentation, tests after making along with the changes
- follow coding functional standard inspired by "grokking simplicity"
- write test function names describing the behaviour you are testing
- when building tests, don't mock anything until necessary or mentioned specifically
- chainlit is not chainlit, so avoid writing a lot of logic in chainlit functions


## Project Structure

- `src/app` - The UI that and logic that enable interaction with the user
- `src/paperweave` - The main module that contains the to generate the podcast and react to it
- `tests/app` - tests for the UI interaction and logic
- `tests/paperweave` - tests regarding the core module
