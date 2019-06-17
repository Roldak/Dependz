- Fix following case which typechecks:
    ```
    A : random_type
    A = B
    ```
  This is due to missing domain equations.

- Fix `free_symbols` in the following case:
    `t:X -> (t -> t) -> Y`
  This should return `[t]` (assuming `X, Y` are constants), but
  it returns `[]` currently because binder is not taken into
  account and `(t -> t)` is deeper.

- Make following code typecheck:
    ```
    filled : n:Nat -> t -> Vec t n

    test : n:Nat -> Nat -> Vec Nat n
    test = \x. \y. filled x Z
    ```
- And the following not typecheck:
    ```
    test : n:Nat -> Nat -> Vec Nat n
    test = \x. \n. filled n Z
    ```

- Also make the following not typecheck:
    ```
    test : n:Nat -> Nat -> Vec Nat n
    test = \x. \y. filled n Z
    ```

