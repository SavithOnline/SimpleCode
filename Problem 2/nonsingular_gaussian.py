"""
Gaussian Elimination — Nonsingular Case (with Pivoting)
=========================================================
A matrix is NONSINGULAR if it can be reduced to upper triangular form
with all nonzero pivots — but it may need ROW SWAPS along the way.

When a diagonal entry is zero, we look BELOW it in the same column
and swap in a nonzero entry. This is called "pivoting."

If no nonzero entry can be found anywhere below (entire column is zero),
the matrix is SINGULAR — no unique solution exists.

Every REGULAR matrix is automatically nonsingular.
But a nonsingular matrix is NOT necessarily regular (it might need swaps).

Pipeline:
  1. Build augmented matrix M = [A | b]
  2. Forward elimination WITH pivoting → upper triangular form [U | c]
  3. Back substitution → solution vector x

Example system (from textbook, p.22 — needs a row swap):
  2y +  z = 2        ← note: no x in this equation!
  2x + 6y +  z = 7
  x  +  y + 4z = 3
"""

import numpy as np


# ── 1. FORWARD ELIMINATION (Nonsingular Case — with pivoting) ─────────────────

def nonsingular_gaussian_elimination(M):
    """
    Reduces augmented matrix M = [A | b] to upper triangular form [U | c],
    swapping rows when a zero appears on the diagonal (pivoting).

    If an entire column below (and including) the diagonal is zero,
    the matrix is SINGULAR and we raise an error.

    Parameters
    ----------
    M : numpy array of shape (n, n+1)

    Returns
    -------
    M       : array in upper triangular form
    swaps   : number of row swaps performed (useful for determinant sign)
    """
    n = M.shape[0]
    swaps = 0       # track row swaps (each swap flips the sign of the determinant)

    for j in range(n):      # j = current pivot column / row

        # ── PIVOTING: find a nonzero entry at or below the diagonal ──────────
        if M[j, j] == 0:

            # Search rows below row j for a nonzero entry in column j
            pivot_row = None
            for k in range(j + 1, n):
                if M[k, j] != 0:
                    pivot_row = k
                    break   # take the first nonzero we find

            if pivot_row is None:
                # Every entry in this column is zero — matrix is singular
                raise ValueError(
                    f"Singular matrix: entire column {j} is zero from row {j} downward. "
                    f"No unique solution exists."
                )

            # Swap row j with the pivot row we found
            print(f"  [Pivoting] Swapping row {j} ↔ row {pivot_row}")
            M[[j, pivot_row]] = M[[pivot_row, j]]   # Python swap trick
            swaps += 1

        # ── ELIMINATION: zero out everything below the pivot ─────────────────
        pivot = M[j, j]     # now guaranteed nonzero

        for i in range(j + 1, n):
            multiplier = M[i, j] / pivot
            M[i, :] = M[i, :] - multiplier * M[j, :]

    return M, swaps


# ── 2. BACK SUBSTITUTION (identical to regular case) ─────────────────────────

def back_substitution(M):
    """
    Given [U | c] in upper triangular form, solve for x working backwards.

    Parameters
    ----------
    M : numpy array of shape (n, n+1) in upper triangular form

    Returns
    -------
    x : solution vector of length n
    """
    n = M.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        rhs = M[i, n]
        for j in range(i + 1, n):
            rhs -= M[i, j] * x[j]
        x[i] = rhs / M[i, i]

    return x


# ── 3. FULL SOLVE PIPELINE ────────────────────────────────────────────────────

def solve_nonsingular(A, b):
    """
    Solve Ax = b assuming A is a NONSINGULAR (but possibly non-regular) matrix.

    Steps:
      1. Form augmented matrix [A | b]
      2. Forward elimination with pivoting → [U | c]
      3. Back substitution → x

    Parameters
    ----------
    A : n×n coefficient matrix
    b : right-hand side vector of length n

    Returns
    -------
    x : solution vector
    """
    n = len(b)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(n, 1)
    M = np.hstack([A, b])

    print("Augmented matrix [A | b]:")
    print_matrix(M, n)

    print("\nForward elimination (with pivoting if needed):")
    M, swaps = nonsingular_gaussian_elimination(M)

    print(f"\nAfter elimination [U | c] — upper triangular form  ({swaps} row swap(s)):")
    print_matrix(M, n)

    # Report the pivots (the diagonal entries of U)
    pivots = [M[i, i] for i in range(n)]
    print(f"\nPivots: {[round(p, 6) for p in pivots]}")
    print(f"det(A) ≈ {round((-1)**swaps * np.prod(pivots), 6)}  "
          f"  (sign flips once per row swap)")

    x = back_substitution(M)
    return x


# ── HELPER: Pretty-print the augmented matrix ─────────────────────────────────

def print_matrix(M, n):
    for row in M:
        left  = "  ".join(f"{v:8.4f}" for v in row[:n])
        right = f"{row[n]:8.4f}"
        print(f"  [ {left}  |  {right} ]")


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Example 1: Matrix that needs a row swap (nonsingular, not regular) ────
    print("=" * 60)
    print("  GAUSSIAN ELIMINATION — NONSINGULAR CASE (with pivoting)")
    print("=" * 60)
    print()
    print("Example 1 — needs a row swap (from textbook, p.22):")
    print("   0x + 2y +  z = 2    ← zero coefficient on x!")
    print("   2x + 6y +  z = 7")
    print("   x  +  y + 4z = 3")
    print()

    A1 = [
        [0, 2, 1],   # zero in the (1,1) position → needs a swap!
        [2, 6, 1],
        [1, 1, 4]
    ]
    b1 = [2, 7, 3]

    solution1 = solve_nonsingular(A1, b1)

    print("\nSolution:")
    labels = ["x", "y", "z"]
    for label, val in zip(labels, solution1):
        print(f"  {label} = {val:.6f}")

    print()
    print("Verification  A·x  =  b :")
    Ax1 = np.array(A1) @ solution1
    for i, (computed, expected) in enumerate(zip(Ax1, b1)):
        print(f"  Row {i+1}: {computed:.6f}  (expected {expected})")

    # ── Example 2: Regular matrix (no swap needed — works fine here too) ─────
    print()
    print("-" * 60)
    print()
    print("Example 2 — regular matrix (no swap needed, just to show it works):")
    print("   x  + 2y +  z = 2")
    print("   2x + 6y +  z = 7")
    print("   x  +  y + 4z = 3")
    print()

    A2 = [
        [1, 2, 1],
        [2, 6, 1],
        [1, 1, 4]
    ]
    b2 = [2, 7, 3]

    solution2 = solve_nonsingular(A2, b2)

    print("\nSolution:")
    for label, val in zip(labels, solution2):
        print(f"  {label} = {val:.6f}")

    # ── Example 3: Singular matrix — to show the error ────────────────────────
    print()
    print("-" * 60)
    print()
    print("Example 3 — SINGULAR matrix (no solution — to show the error):")
    print("   x +  y = 1")
    print("   2x + 2y = 3   ← just 2× the first equation, can't equal 3!")
    print()

    A3 = [
        [1, 1],
        [2, 2]
    ]
    b3 = [1, 3]

    try:
        solve_nonsingular(A3, b3)
    except ValueError as e:
        print(f"\n  ERROR caught: {e}")
