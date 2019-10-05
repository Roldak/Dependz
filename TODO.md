
- Make the following not typecheck:
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

- Fix rigid-rigid first order unify equation to make it recurse in the
  arrow-arrow in the case where it currently doesn't, which is when
  one of the arrows has a constraining binder but the other has
  no binder. Indeed, it is necessary to handle higher order unification
  where we are trying to unify `x:Nat -> P x` with `Nat -> Nat`.
  Idea: recurse anyway and add a Predicate equation that asserts that the
  resulting arrow type has no constraining binder?
