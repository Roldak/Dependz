
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

- Address the following:
    ```
    n:Nat -> (n:Nat -> Vec Nat n)
    ```
    Where the second occurrence of `n:Nat` is _constraining_ but
    is currently not considered so. Solutions:
    - Forbid this pattern
    - Fix `has_constaining_binder` (implem or callers)

- Re-type arrows that have unconstraining binders to remove the
  binder altogether. E.g:
  ```
  foo : x:Nat -> Nat
  test : Nat -> Nat
  test = foo
  ```
  Should typecheck and `foo` should have type `Nat -> Nat`
  (and not `x$XX:Nat -> Nat`)

- Implement totality checking.
