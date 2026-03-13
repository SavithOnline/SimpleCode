"""
Gaussian Elimination — Regular Case
=====================================
A matrix is REGULAR if, at every step of elimination,
the diagonal entry (the "pivot") is already nonzero.
No row swapping is needed. This is the simplest case.

Pipeline:
  1. Build augmented matrix M = [A | b]
  2. Forward elimination → upper triangular form [U | c]
  3. Back substitution → solution vector x

Example system (from textbook p.12):
  x  + 2y +  z = 2
  2x + 6y +  z = 7
  x  +  y + 4z = 3
"""

import numpy as np


# ── 1. FORWARD ELIMINATION (Regular Case) ─────────────────────────────────────

def regular_gaussian_elimination(M):
    """
    Takes the augmented matrix M = [A | b] and reduces it to
    upper triangular form [U | c] using only row scaling + row addition.

    If a diagonal pivot turns out to be zero, the matrix is NOT regular
    and we stop immediately.

    Parameters
    ----------
    M : numpy array of shape (n, n+1)  — the augmented matrix

    Returns
    -------
    M : the same array, now in upper triangular form [U | c]
    """
    n = M.shape[0]   # number of equations (= number of unknowns for a square system)

    for j in range(n):          # j is the current pivot COLUMN (and pivot ROW)

        pivot = M[j, j]         # the diagonal entry — must be nonzero for regular matrices

        if pivot == 0:
            raise ValueError(
                f"Zero pivot encountered at position ({j},{j}). "
                f"Matrix is NOT regular. Try the nonsingular solver instead."
            )

        # Eliminate the entries BELOW the pivot in column j
        for i in range(j + 1, n):      # i walks down the rows below the pivot

            # multiplier = (entry to eliminate) / (pivot)
            # This tells us "how many times row j do I need to subtract
            # from row i to make M[i,j] = 0?"
            multiplier = M[i, j] / pivot

            # Subtract (multiplier × pivot row) from row i
            # This zeros out M[i,j] and adjusts all entries to the right
            M[i, :] = M[i, :] - multiplier * M[j, :]

    return M   # now in upper triangular form


# ── 2. BACK SUBSTITUTION ──────────────────────────────────────────────────────

def back_substitution(M):
    """
    Given an augmented matrix [U | c] in upper triangular form,
    solve for x by working backwards from the last equation upward.

    The last equation has only one unknown → solve directly.
    Each earlier equation uses already-found values to isolate the next unknown.

    Parameters
    ----------
    M : numpy array of shape (n, n+1) in upper triangular form

    Returns
    -------
    x : solution vector of length n
    """
    n = M.shape[0]
    x = np.zeros(n)         # pre-allocate solution vector

    # Start from the LAST equation and work upward
    for i in range(n - 1, -1, -1):

        # Right-hand side for this equation
        rhs = M[i, n]

        # Subtract contributions from unknowns we've already solved
        # (these sit to the RIGHT of the current unknown on row i)
        for j in range(i + 1, n):
            rhs -= M[i, j] * x[j]

        # Divide by the diagonal coefficient to isolate x[i]
        x[i] = rhs / M[i, i]

    return x


# ── 3. FULL SOLVE PIPELINE ────────────────────────────────────────────────────

def solve_regular(A, b):
    """
    Solve Ax = b assuming A is a REGULAR matrix.

    Steps:
      1. Form augmented matrix [A | b]
      2. Forward elimination → [U | c]
      3. Back substitution → x

    Parameters
    ----------
    A : n×n coefficient matrix (list of lists or numpy array)
    b : right-hand side vector of length n

    Returns
    -------
    x : solution vector
    """
    n = len(b)

    # Build augmented matrix [A | b] as floats
    # We add a new column to A containing b
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(n, 1)
    M = np.hstack([A, b])    # [A | b], shape (n, n+1)

    print("Augmented matrix [A | b]:")
    print_matrix(M, n)

    # Step 1: Forward elimination
    M = regular_gaussian_elimination(M)
    print("\nAfter forward elimination [U | c] (upper triangular form):")
    print_matrix(M, n)

    # Step 2: Back substitution
    x = back_substitution(M)
    return x


# ── HELPER: Pretty-print the augmented matrix ─────────────────────────────────

def print_matrix(M, n):
    """Print augmented matrix with a vertical bar separating A from b."""
    for row in M:
        left  = "  ".join(f"{v:8.4f}" for v in row[:n])
        right = f"{row[n]:8.4f}"
        print(f"  [ {left}  |  {right} ]")


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  GAUSSIAN ELIMINATION — REGULAR CASE")
    print("=" * 60)
    print()
    print("System (from textbook, p.12):")
    print("   x  + 2y +  z = 2")
    print("   2x + 6y +  z = 7")
    print("   x  +  y + 4z = 3")
    print()

    # Coefficient matrix A
    A = [
        [1, 2, 1],
        [2, 6, 1],
        [1, 1, 4]
    ]

    # Right-hand side b
    b = [2, 7, 3]

    solution = solve_regular(A, b)

    print("\nSolution:")
    labels = ["x", "y", "z"]
    for label, val in zip(labels, solution):
        print(f"  {label} = {val:.6f}")

    print()
    print("Verification  A·x  =  b :")
    Ax = np.array(A) @ solution
    for i, (computed, expected) in enumerate(zip(Ax, b)):
        print(f"  Row {i+1}: {computed:.6f}  (expected {expected})")
