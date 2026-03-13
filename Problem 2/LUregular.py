"""
LU Decomposition — Regular Case  (Theorem 1.3, p.18)
======================================================
A REGULAR matrix A can be factored as:

    A = L · U

where:
  L = lower unitriangular matrix  (1's on diagonal, entries below diagonal
      are the multipliers used during elimination)
  U = upper triangular matrix  (pivots sit on the diagonal)

WHY is this useful?
-------------------
To solve  A x = b,  substitute  A = LU:

    LU x = b

Let  c = U x.  Then the problem splits into TWO easy triangular solves:

  Step 1 — Forward Substitution:   L c = b   (solve for c, top to bottom)
  Step 2 — Back Substitution:      U x = c   (solve for x, bottom to top)

Both triangular systems are easy because each step isolates exactly one unknown.

HOW to build L and U:
----------------------
Run regular Gaussian elimination on A.
  - U is the resulting upper triangular matrix.
  - L is built by recording the multiplier used at each elimination step:
      multiplier  lij = M[i,j] / M[j,j]
    This multiplier goes into position (i, j) of L.  Everything else on
    the diagonal of L is 1, and everything above the diagonal is 0.

Example (textbook, p.19):
    A = [[2,  1,  1],       L = [[1, 0, 0],     U = [[2,  1,  1],
         [4,  5,  2],            [2, 1, 0],           [0,  3,  0],
         [2, -2,  0]]            [1,-1, 1]]            [0,  0, -1]]
"""

import numpy as np


# ── STEP 0 — LU FACTORIZATION ─────────────────────────────────────────────────

def lu_factorize(A):
    """
    Decompose a REGULAR matrix A into L and U such that  A = L · U.

    L is lower unitriangular (1's on diagonal).
    U is upper triangular (pivots on diagonal).

    The multiplier  lij = M[i,j] / M[j,j]  is recorded directly into L
    as elimination proceeds — this is the 'L-shortcut' (eq. 1.23, p.18).

    Parameters
    ----------
    A : n×n regular matrix (list of lists or numpy array)

    Returns
    -------
    L : n×n lower unitriangular matrix
    U : n×n upper triangular matrix
    """
    n = A.shape[0]

    # Start: L = identity (will fill in multipliers), U = copy of A
    L = np.eye(n)          # eye(n) = n×n identity matrix
    U = A.copy()           # we'll reduce U in place to upper triangular form

    for j in range(n):     # j = current pivot column

        if U[j, j] == 0:
            raise ValueError(
                f"Zero pivot at position ({j},{j}). "
                f"Matrix is NOT regular. Use the permuted LU solver instead."
            )

        for i in range(j + 1, n):    # i = rows below the pivot

            # Multiplier: how many times do we subtract row j from row i
            # to zero out U[i, j]?
            lij = U[i, j] / U[j, j]

            L[i, j] = lij            # store multiplier in L (below diagonal)
            U[i, :] -= lij * U[j, :] # subtract lij × pivot row from row i
            # After this, U[i,j] = 0 (as intended)

    return L, U


# ── STEP 1 — FORWARD SUBSTITUTION: solve L c = b ─────────────────────────────

def forward_substitution(L, b):
    """
    Solve  L c = b  where L is lower unitriangular.

    Works top-to-bottom. Since L has 1's on the diagonal, each step
    directly gives the next component of c without any division.

    Equation (1.26):
      c[1] = b[1]
      c[i] = b[i]  −  sum( L[i,j] * c[j]  for j < i )

    Parameters
    ----------
    L : n×n lower unitriangular matrix
    b : right-hand side vector of length n

    Returns
    -------
    c : solution vector
    """
    n = len(b)
    c = np.zeros(n)

    for i in range(n):
        # Start with the right-hand side value for row i
        c[i] = b[i]
        # Subtract the contributions of already-known c values to the left of c[i]
        for j in range(i):     # j runs over columns BEFORE position i
            c[i] -= L[i, j] * c[j]
        # Note: L[i,i] = 1 always, so no division needed here

    return c


# ── STEP 2 — BACK SUBSTITUTION: solve U x = c ────────────────────────────────

def back_substitution(U, c):
    """
    Solve  U x = c  where U is upper triangular.

    Works bottom-to-top. The last equation has only one unknown, so we
    solve directly, then work upward using already-found values.

    Equation (1.28):
      x[n] = c[n] / U[n,n]
      x[i] = ( c[i]  −  sum( U[i,j] * x[j]  for j > i ) )  /  U[i,i]

    Parameters
    ----------
    U : n×n upper triangular matrix
    c : right-hand side vector of length n

    Returns
    -------
    x : solution vector
    """
    n = len(c)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):   # start from last row, move upward
        x[i] = c[i]
        for j in range(i + 1, n):    # j runs over columns AFTER position i
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]              # divide by the diagonal pivot

    return x


# ── FULL PIPELINE: A x = b via LU ────────────────────────────────────────────

def solve_lu(A, b, verbose=True):
    """
    Solve  A x = b  for a REGULAR matrix A using LU decomposition.

    Full pipeline:
      1. Factorize:          A  →  L, U    (A = LU)
      2. Forward sub:        L c = b       (find intermediate vector c)
      3. Back sub:           U x = c       (find solution x)

    Parameters
    ----------
    A : n×n regular coefficient matrix
    b : right-hand side vector of length n

    Returns
    -------
    x : solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # ── Factorization ──
    L, U = lu_factorize(A)

    if verbose:
        print("L (lower unitriangular — stores the multipliers):")
        print_matrix(L)
        print("\nU (upper triangular — stores the pivots on diagonal):")
        print_matrix(U)
        print("\nVerification  L · U = A :")
        print_matrix(L @ U)

    # ── Forward substitution: L c = b ──
    c = forward_substitution(L, b)

    if verbose:
        print(f"\nForward substitution  L c = b  →  c = {np.round(c, 6)}")

    # ── Back substitution: U x = c ──
    x = back_substitution(U, c)

    return x


# ── HELPER: pretty-print a matrix ────────────────────────────────────────────

def print_matrix(M):
    for row in M:
        print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 62)
    print("  LU DECOMPOSITION — REGULAR CASE  (A = LU)")
    print("=" * 62)

    # ── Example 1: from textbook p.19–21 ──────────────────────────────────────
    print()
    print("Example 1 (textbook p.19–21):")
    print("   2x +  y +  z = 1")
    print("   4x + 5y + 2z = 2")
    print("   2x − 2y + 0z = 2")
    print()

    A1 = np.array([
        [ 2,  1,  1],
        [ 4,  5,  2],
        [ 2, -2,  0]
    ], dtype=float)
    b1 = np.array([1, 2, 2], dtype=float)

    x1 = solve_lu(A1, b1)
    print(f"\nSolution x = {np.round(x1, 6)}")

    print("\nVerification  A · x = b :")
    for i, (val, expected) in enumerate(zip(A1 @ x1, b1)):
        print(f"  Row {i+1}: {val:.6f}  (expected {expected})")

    # ── Example 2: textbook p.12 (the classic 3×3 from section 1.3) ──────────
    print()
    print("─" * 62)
    print()
    print("Example 2 (textbook p.12 — the running example):")
    print("    x + 2y +  z = 2")
    print("   2x + 6y +  z = 7")
    print("    x +  y + 4z = 3")
    print()

    A2 = np.array([
        [1, 2, 1],
        [2, 6, 1],
        [1, 1, 4]
    ], dtype=float)
    b2 = np.array([2, 7, 3], dtype=float)

    x2 = solve_lu(A2, b2)
    print(f"\nSolution: x={x2[0]:.4f}, y={x2[1]:.4f}, z={x2[2]:.4f}")

    print("\nVerification  A · x = b :")
    for i, (val, expected) in enumerate(zip(A2 @ x2, b2)):
        print(f"  Row {i+1}: {val:.6f}  (expected {expected})")

    # ── What if we try a non-regular matrix? ─────────────────────────────────
    print()
    print("─" * 62)
    print()
    print("Example 3 — attempt LU on a NON-regular matrix (zero on diagonal):")
    A3 = np.array([[0, 2, 1], [2, 6, 1], [1, 1, 4]], dtype=float)
    b3 = np.array([2, 7, 3], dtype=float)
    try:
        solve_lu(A3, b3)
    except ValueError as e:
        print(f"  ERROR: {e}")
