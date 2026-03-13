"""
Matrix Inverse via Gauss–Jordan Elimination  (Section 1.5, p.36–39)
=====================================================================
GOAL: Find A⁻¹ for a nonsingular (invertible) square matrix A.

THE KEY IDEA  (p.36)
---------------------
Finding A⁻¹ means solving  A X = I  for the matrix X.

Writing I column by column as  [e₁ | e₂ | ... | eₙ],
this is actually n separate systems:

    A x₁ = e₁,  A x₂ = e₂,  ...,  A xₙ = eₙ

All n systems share the SAME coefficient matrix A.
Running Gaussian elimination n times would be wasteful —
we'd repeat identical row operations on each system.

SMARTER: stack all the right-hand sides into one big augmented matrix

    M = [ A | I ]   (size n × 2n)

and apply row operations ONCE to both halves simultaneously.

THE THREE PHASES  (p.37–39)
-----------------------------
Phase 1 — Forward Elimination (with pivoting for nonsingular matrices):
    Apply row operations of Type #1 (add a multiple of one row to another)
    and Type #2 (swap rows) to reduce the LEFT half to upper triangular form U.
    The right half transforms from I into some matrix C.
    Result:  [ U | C ]

Phase 2 — Scale rows (Type #3 row operation):
    Divide each row i by its pivot U[i,i].
    This makes all diagonal entries of U equal to 1.
    The left half becomes upper UNItriangular (1's on diagonal).
    Result:  [ V | B ]   where V is upper unitriangular

Phase 3 — Back-Elimination (upward pass):
    Use Type #1 row operations, but now working UPWARD to zero out
    entries ABOVE the diagonal as well.
    When the left half becomes the identity I,
    the right half is exactly A⁻¹.
    Result:  [ I | A⁻¹ ]

WHY does this give the inverse?  (p.36)
----------------------------------------
Every row operation is left-multiplication by an elementary matrix E.
Applying a sequence of row operations to [ A | I ] is the same as
computing  E_k ... E_2 E_1 [ A | I ] = [ E_k...E_1 A | E_k...E_1 I ].
When the left half becomes I, we have  E_k...E_1 A = I,
which means  E_k...E_1 = A⁻¹.
And since those same operations were also applied to the right half (which started as I),
the right half becomes  E_k...E_1 · I = A⁻¹.

SINGULAR matrices: if at any pivot position both the diagonal entry
and every entry below it are zero, A is singular — it has no inverse.

Example (textbook p.36–39):
    A = [[0, 2, 1],          A⁻¹ = [[-23/18,  7/18,  2/9],
         [2, 6, 1],                  [  7/18,  1/18, -1/9],
         [1, 1, 4]]                  [  2/9,  -1/9,   2/9]]
"""

import numpy as np


def gauss_jordan_inverse(A, verbose=True):
    """
    Compute A⁻¹ using Gauss–Jordan Elimination on the augmented matrix [A | I].

    Three phases:
      1. Forward elimination (with pivoting)  →  [ U | C ]
      2. Scale rows by pivot                  →  [ V | B ]   (V upper unitriangular)
      3. Upward elimination                   →  [ I | A⁻¹ ]

    Parameters
    ----------
    A       : n×n square matrix (list of lists or numpy array)
    verbose : if True, prints each phase of the augmented matrix

    Returns
    -------
    A_inv   : n×n inverse matrix
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to have an inverse.")

    # ── Build augmented matrix  M = [ A | I ]  of shape n × 2n ───────────────
    I = np.eye(n)
    M = np.hstack([A, I])   # stack A and I side by side

    if verbose:
        print("Starting augmented matrix  [ A | I ]:")
        print_augmented(M, n)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Forward Elimination  (reduce left half to upper triangular U)
    # Uses Type #1 row ops (add multiple of one row to another)
    # Uses Type #2 row ops (swap rows) when a zero diagonal is encountered
    # ═════════════════════════════════════════════════════════════════════════

    for j in range(n):          # j = current pivot column

        # ── Pivoting: if diagonal is zero, swap with a row below that has a
        #    nonzero entry in column j ────────────────────────────────────────
        if M[j, j] == 0:
            pivot_row = None
            for k in range(j + 1, n):
                if M[k, j] != 0:
                    pivot_row = k
                    break
            if pivot_row is None:
                raise ValueError(
                    f"Matrix is SINGULAR (zero column at position {j}). "
                    "No inverse exists."
                )
            M[[j, pivot_row]] = M[[pivot_row, j]]   # swap entire rows
            if verbose:
                print(f"\n  [Pivot] Swapped row {j} ↔ row {pivot_row}")

        # ── Eliminate entries BELOW the pivot in column j ────────────────────
        for i in range(j + 1, n):
            multiplier = M[i, j] / M[j, j]
            M[i, :] -= multiplier * M[j, :]
            # This zeros out M[i,j] and updates the right half (the growing inverse)

    if verbose:
        print("\nAfter Phase 1 — forward elimination  [ U | C ]:")
        print_augmented(M, n)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Scale Each Row  (divide by its pivot to get 1's on diagonal)
    # Uses Type #3 row ops (multiply a row by a nonzero scalar)
    # After this step, the left half is upper UNItriangular (1's on diagonal)
    # ═════════════════════════════════════════════════════════════════════════

    for i in range(n):
        pivot = M[i, i]     # the diagonal entry — guaranteed nonzero now
        M[i, :] /= pivot    # scale the entire row (affects both halves)

    if verbose:
        print("\nAfter Phase 2 — scale rows by pivot  [ V | B ]  (1's on diagonal):")
        print_augmented(M, n)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Upward (Back) Elimination  (zero out entries ABOVE diagonal)
    # Works from the LAST column upward, eliminating above each 1 on diagonal.
    # When done, the left half is the identity I and the right half is A⁻¹.
    # ═════════════════════════════════════════════════════════════════════════

    for j in range(n - 1, 0, -1):      # j goes from last column down to column 1
        for i in range(j):              # i = rows ABOVE row j
            multiplier = M[i, j]        # entry above the diagonal 1 in column j
            M[i, :] -= multiplier * M[j, :]
            # After this, M[i,j] = 0

    if verbose:
        print("\nAfter Phase 3 — upward elimination  [ I | A⁻¹ ]:")
        print_augmented(M, n)

    # ── Extract A⁻¹ from the right half of M ─────────────────────────────────
    A_inv = M[:, n:]    # columns n through 2n-1
    return A_inv


# ── HELPER: print augmented matrix with a dividing bar ───────────────────────

def print_augmented(M, n):
    """Print [ A | I ] with a vertical bar separating the two halves."""
    for row in M:
        left  = "  ".join(f"{v:8.4f}" for v in row[:n])
        right = "  ".join(f"{v:8.4f}" for v in row[n:])
        print(f"  [ {left}  |  {right} ]")


# ── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 72)
    print("  MATRIX INVERSE via GAUSS–JORDAN ELIMINATION")
    print("=" * 72)

    # ── Example 1: Textbook p.36–39 — needs a row swap  ──────────────────────
    print()
    print("Example 1 (textbook p.36–39):")
    print("   A = [[0, 2, 1],")
    print("        [2, 6, 1],")
    print("        [1, 1, 4]]")
    print()

    A1 = np.array([
        [0, 2, 1],
        [2, 6, 1],
        [1, 1, 4]
    ], dtype=float)

    inv1 = gauss_jordan_inverse(A1)

    print("\nComputed A⁻¹:")
    for row in inv1:
        print("  [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")

    print("\nVerification  A · A⁻¹  (should be identity):")
    product = A1 @ inv1
    for row in product:
        print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")
    print(f"  Is identity? {'YES ✓' if np.allclose(product, np.eye(3)) else 'NO ✗'}")

    # ── Example 2: textbook p.19 regular matrix (no swap needed) ─────────────
    print()
    print("─" * 72)
    print()
    print("Example 2 — regular matrix from textbook p.19:")
    print("   A = [[2,  1,  1],")
    print("        [4,  5,  2],")
    print("        [2, -2,  0]]")
    print()

    A2 = np.array([
        [2,  1,  1],
        [4,  5,  2],
        [2, -2,  0]
    ], dtype=float)

    inv2 = gauss_jordan_inverse(A2)

    print("\nComputed A⁻¹:")
    for row in inv2:
        print("  [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")

    print("\nVerification  A · A⁻¹  (should be identity):")
    product2 = A2 @ inv2
    for row in product2:
        print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")
    print(f"  Is identity? {'YES ✓' if np.allclose(product2, np.eye(3)) else 'NO ✗'}")

    # ── Example 3: 4×4 matrix ─────────────────────────────────────────────────
    print()
    print("─" * 72)
    print()
    print("Example 3 — 4×4 matrix:")
    print("   A = [[ 1,  2, -1,  0],")
    print("        [ 2,  4, -2, -1],   ← note: row 2 is almost 2× row 1!")
    print("        [-3, -5,  6,  1],")
    print("        [-1,  2,  8, -2]]")
    print()

    A3 = np.array([
        [ 1,  2, -1,  0],
        [ 2,  4, -2, -1],
        [-3, -5,  6,  1],
        [-1,  2,  8, -2]
    ], dtype=float)

    inv3 = gauss_jordan_inverse(A3)

    print("\nComputed A⁻¹:")
    for row in inv3:
        print("  [" + "  ".join(f"{v:9.4f}" for v in row) + " ]")

    print("\nVerification  A · A⁻¹  (should be identity):")
    product3 = A3 @ inv3
    for row in product3:
        print("  [" + "  ".join(f"{v:8.4f}" for v in row) + " ]")
    print(f"  Is identity? {'YES ✓' if np.allclose(product3, np.eye(4)) else 'NO ✗'}")

    # ── Example 4: singular matrix — should raise an error ───────────────────
    print()
    print("─" * 72)
    print()
    print("Example 4 — SINGULAR matrix (no inverse exists):")
    print("   A = [[1, 2],")
    print("        [2, 4]]   ← row 2 = 2 × row 1: linearly dependent!")
    print()

    A4 = np.array([[1, 2], [2, 4]], dtype=float)
    try:
        gauss_jordan_inverse(A4)
    except ValueError as e:
        print(f"  ERROR: {e}")

    # ── Bonus: use A⁻¹ to solve a system ─────────────────────────────────────
    print()
    print("─" * 72)
    print()
    print("Bonus — once we have A⁻¹, solving  A x = b  is just a multiplication:")
    print("  x = A⁻¹ · b")
    print()
    print("  Using Example 1 matrix with b = [2, 7, 3]:")

    b = np.array([2, 7, 3], dtype=float)
    x = inv1 @ b
    print(f"  x = A⁻¹ · b = {np.round(x, 6)}")
    print(f"  Verify A · x = {np.round(A1 @ x, 6)}  (expected {b})")
    print()
    print("  Note (textbook p.51): computing A⁻¹ then multiplying is about")
    print("  3× more work than direct Gaussian elimination + back substitution.")
    print("  Use the inverse only when you need to solve MANY systems with same A.")
